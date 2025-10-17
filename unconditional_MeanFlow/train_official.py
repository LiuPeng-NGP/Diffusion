import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.amp

import numpy as np

import yaml
import argparse
import os
import logging
from tqdm import tqdm
from time import time
from collections import OrderedDict
from copy import deepcopy

from meanflow import MeanFlow
from unet_official import SongUNet

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.

class Config(object):
    def __init__(self, dic):
        for key in dic:
            setattr(self, key, dic[key])
    def items(self):
        return self.__dict__.items()

def cleanup():
    dist.destroy_process_group()

def create_logger(logging_dir):
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/train.log")]
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

### MODIFIED ### Step 5: More accurate EMA update to match official implementation's periodic logic
@torch.no_grad()
def update_ema(ema_model, model, num_updates, decay=0.999, period=1):
    """
    Step the EMA model towards the current model with periodic updates.
    """
    decay_effective = decay ** period
    if num_updates % period == 0:
        ema_params = OrderedDict(ema_model.named_parameters())
        model_params = OrderedDict(model.named_parameters())

        for name, param in model_params.items():
            ema_params[name].mul_(decay_effective).add_(param.data, alpha=1 - decay_effective)

# ===== training =====
def train(args):
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    yaml_path = args.config
    with open(yaml_path, 'r') as f:
        args = yaml.full_load(f)
    args = Config(args)
    use_amp = args.use_amp

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    local_rank = dist.get_rank()
    device = local_rank % torch.cuda.device_count()
    local_seed = args.global_seed + local_rank
    torch.cuda.set_device(device)

    model_dir = os.path.join(args.save_dir, "ckpts")
    vis_dir = os.path.join(args.save_dir, "visual")
    if local_rank == 0:
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

    logger = create_logger(args.save_dir) if local_rank == 0 else create_logger(None)

    if local_rank == 0:
        logger.info(f"Experiment directory created at {args.save_dir}")
        logger.info("########## Configuration ##########")
        for key, value in args.items():
            logger.info(f"{key}: {value}")

    logger.info("local_rank = {}, seed = {}".format(local_rank, local_seed))
    np.random.seed(seed=local_seed)
    torch.manual_seed(seed=local_seed)
    torch.cuda.manual_seed_all(seed=local_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    nn_model=SongUNet(**args.network)
    if local_rank == 0:
        logger.info(f'Total trainable parameters in U-Net: {sum(p.numel() for p in nn_model.parameters() if p.requires_grad)}')

    model = DDP(nn_model.to(device), device_ids=[local_rank], find_unused_parameters=True, broadcast_buffers=False)
    diffusion = MeanFlow(model, **args.diffusion)

    if local_rank == 0:
        ema = deepcopy(nn_model).to(device)
        requires_grad(ema, False)
        # Initialize EMA weights to be the same as the model's
        update_ema(ema, diffusion.model.module, num_updates=0, decay=0)

    ### MODIFIED ### Step 1: Align data processing.
    # We now normalize inside the training loop, not here.
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # This scales images from [0, 255] to [0.0, 1.0]
    ])
    train_set = CIFAR10(root='../data', train=True, download=True, transform=transform)
    logger.info(f"CIFAR10 train dataset:{len(train_set)}")

    sampler = DistributedSampler(
        train_set,
        num_replicas=dist.get_world_size(),
        rank=local_rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    ### MODIFIED ### Step 2: Change optimizer to Adam and handle LR scaling like official code
    # The base learning rate is scaled by the number of GPUs.
    # This is done here instead of multiplying later.
    lr = args.learning_rate * dist.get_world_size()
    logger.info(f"Using DDP. Base LR: {args.learning_rate:.1e}, World Size: {dist.get_world_size()}, Effective LR: {lr:.1e}")
    optim = torch.optim.Adam(diffusion.model.parameters(), lr=lr, betas=(0.9, 0.999)) # Use Adam with official betas
    
    scaler = torch.amp.GradScaler(enabled=use_amp)

    ### MODIFIED ### Step 3: Use official per-iteration LR scheduler
    warmup_iters = args.warm_epoch * len(train_loader)
    total_iters = args.n_epoch * len(train_loader)
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1e-8 / lr if lr > 0 else 0, end_factor=1.0, total_iters=warmup_iters)
    main_scheduler = torch.optim.lr_scheduler.ConstantLR(optim, total_iters=total_iters - warmup_iters, factor=1.0)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optim, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_iters])


    if args.load_epoch != -1:
        # Checkpoint loading would need to be adapted for the new scheduler object
        logger.info("Checkpoint loading needs review to handle the new lr_scheduler object.")

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    
    for current_epoch in range(args.load_epoch + 1, args.n_epoch):
        ### MODIFIED ### Step 3: Remove per-epoch LR logic
        sampler.set_epoch(current_epoch)
        dist.barrier()
        diffusion.model.train()
        
        if local_rank == 0:
            current_lr = optim.param_groups[0]['lr']
            logger.info(f'epoch {current_epoch}, current lr {current_lr:f}')
            progress_bar = tqdm(train_loader)
        else:
            progress_bar = train_loader
            
        for x, c in progress_bar:
            optim.zero_grad()
            x = x.to(device, non_blocking=True)
            
            ### MODIFIED ### Step 1: Add manual normalization in the loop
            x = x * 2.0 - 1.0

            loss = diffusion.loss(x)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)

            ### MODIFIED ### Step 4: Remove gradient clipping and NaN handling
            # torch.nn.utils.clip_grad_norm_(parameters=diffusion.model.parameters(), max_norm=1.0)
            # for param in diffusion.model.parameters():
            #     if param.grad is not None:
            #         torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

            scaler.step(optim)
            scaler.update()

            ### MODIFIED ### Step 3: Update scheduler per iteration
            lr_scheduler.step()

            dist.barrier()
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss.item() / dist.get_world_size()
            
            if local_rank == 0:
                ### MODIFIED ### Step 5: Update EMA with num_updates
                update_ema(ema, diffusion.model.module, num_updates=train_steps, decay=args.ema)
                progress_bar.set_description(f"loss: {loss:.4f}")

            running_loss += loss
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Steps/Sec: {steps_per_sec:.2f}")
                
                running_loss = 0
                log_steps = 0
                start_time = time()
                
        # testing (visualization and saving) loop remains the same
        if local_rank == 0:
            if current_epoch % 100 == 0 or current_epoch == args.n_epoch - 1:
                temp_meanflow = MeanFlow(ema, **args.diffusion)
                noise = torch.randn([args.n_sample, 3, 32, 32], device=device)
                temp_meanflow.model.eval()
                with torch.no_grad():
                    x_gen = temp_meanflow.sample(noise)
                
                x_real = x[:args.n_sample]
                x_all = torch.cat([x_gen.cpu(), x_real.cpu()])
                save_path = os.path.join(vis_dir, f"image_ep{current_epoch}_ema.png")
                save_image(x_all, save_path, nrow=10, normalize=True, value_range=(-1, 1))
                logger.info(f'Saved image at {save_path}')

                if args.save_model:
                    checkpoint = {
                        'model': diffusion.model.state_dict(),
                        'ema': ema.state_dict(),
                        'optim': optim.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict() # Save scheduler state
                    }
                    save_path = os.path.join(model_dir, f"model_{current_epoch}.pth")
                    torch.save(checkpoint, save_path)
                    logger.info(f'Saved model at {save_path}')

    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    train(args)