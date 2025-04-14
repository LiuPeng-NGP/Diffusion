import torch
from torchvision.utils import make_grid, save_image
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.amp

import numpy as np
import logging
import yaml
import argparse
import os

from ddpm import DDPM
from transformer import Transformer

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])
    def items(self):
        # Return the items of the dictionary used to initialize the class
        return self.__dict__.items()
def gather_tensor(tensor):
    tensor_list = [tensor.clone() for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list, tensor)
    tensor_list = torch.cat(tensor_list, dim=0)
    return tensor_list

def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/eval.log")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

# ===== sampling =====

def sample(args):
    """
    Sampling
    """
    assert torch.cuda.is_available(), "Sampling currently requires at least one GPU."

    # Configuration:
    yaml_path = args.config
    with open(yaml_path, 'r') as f:
        args = yaml.full_load(f)
    args = Config(args)
    steps = args.steps
    batches = args.sampling_batch_size
    epoch_checkpoint = args.epoch_checkpoint
    
    # Setup DDP:
    dist.init_process_group("nccl")
    local_rank = dist.get_rank()
    device = local_rank % torch.cuda.device_count()
    local_seed = args.global_seed + local_rank
    torch.cuda.set_device(device)
    
    # Log:
    if local_rank == 0:
        logger = create_logger(args.save_dir)
        logger.info(f"Experiment directory created at {args.save_dir}")
    else:
        logger = create_logger(None)
    
    logger.info("########## Configuration ##########")
    for key, value in args.items():
        logger.info(f"{key}: {value}")

    # Seed:
    device = "cuda:%d" % local_rank
    logger.info("local_rank = {}, seed = {}".format(local_rank, local_seed))
    np.random.seed(seed=local_seed)
    torch.manual_seed(seed=local_seed)
    torch.cuda.manual_seed_all(seed=local_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    nn_model=Transformer(**args.network)

    if epoch_checkpoint == -1:
        epoch_checkpoint = args.n_epoch - 1
    target = os.path.join(args.save_dir, "ckpts", f"model_{epoch_checkpoint}.pth")
    logger.info(f"loading model at{target}")
    checkpoint = torch.load(target, map_location=device)
    
    nn_model.load_state_dict(checkpoint['ema'], strict=False)
    model = DDP(nn_model.to(device), device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    diffusion = DDPM(model)
    diffusion.model.eval()

    if local_rank == 0:
        gen_dir = os.path.join(args.save_dir, f"EMAgenerated_ep{epoch_checkpoint}_ddpm_steps{steps}")
        os.makedirs(gen_dir, exist_ok=True)
        gen_dir_png = os.path.join(gen_dir, "pngs")
        os.makedirs(gen_dir_png, exist_ok=True)
    res = []

    for batch in range(batches):
        with torch.no_grad():
            assert 400 % dist.get_world_size() == 0
            samples_per_process = 400 // dist.get_world_size()
            noise = torch.randn([samples_per_process,3,32, 32]).to(device)
            with torch.no_grad():
                x_gen = diffusion.sample(noise)
        dist.barrier()
        x_gen = gather_tensor(x_gen).cpu()
        if local_rank == 0:
            res.append(x_gen)
            grid = make_grid(x_gen, nrow=20)
            png_path = os.path.join(gen_dir, f"grid_{batch}.png")
            save_image(grid, png_path, nrow=20, normalize=True, value_range=(-1, 1))

    if local_rank == 0:
        res = torch.cat(res)
        for no, img in enumerate(res):
            png_path = os.path.join(gen_dir_png, f"{no}.png")
            save_image(img, png_path, normalize=True, value_range=(-1, 1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--use_amp", action='store_true', default=False)
    args = parser.parse_args()
    sample(args)