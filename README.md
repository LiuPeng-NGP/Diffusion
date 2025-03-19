# Diffusion
My replication for diffusion model.

# Environment
conda env create -f environment.yml
conda activate diffusion

# Data
run the code in extract_data.ipynb to download CIFAR-10 data.

# Unconditional DDPM
cd unconditional_DDPM
## Train
torchrun --nnodes=1 --nproc_per_node=2 train.py --config config.yaml --use_amp
source activate diffusion && cd /liupeng/Diffusion/unconditional_DDPM && torchrun --nnodes=1 --nproc_per_node=8 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_DDPM/2000_unconditional_DDPM/EMAgenerated_ep1999_ddpm_steps1000/pngs data/cifar10-pngs

FID:  3.2444050218018106


# Class-condiitonal DDPM
cd Class_conditional_DDPM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55000 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55500 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid Class_conditional_DDPM/2000_conditional_DDPM/EMAgenerated_ep1999_ddpm_steps1000/pngs data/cifar10-pngs

FID:  2.9473849136342665

# Unconditional EDM
cd unconditional_EDM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55001 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55501 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_EDM/1200_unconditional_EDM/EMAgenerated_ep1199_edm_steps18/pngs data/cifar10-pngs

FID:  4.234571648031874

# Class-condiitonal EDM
cd Class_conditional_EDM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55002 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55502 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid Class_conditional_EDM/1200_conditional_EDM/EMAgenerated_ep1199_edm_steps18/pngs data/cifar10-pngs

FID:  3.707334678054849

# Unconditional Consistency Model
cd unconditional_CM
cd /home/liupeng/Diffusion/unconditional_CM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55002 train.py --config config.yaml --use_amp
export CUDA_VISIBLE_DEVICES=1
torchrun --nnodes=1 --nproc_per_node=1 --rdzv_endpoint=localhost:55002 train.py --config config.yaml --use_amp


source activate diffusion && cd /liupeng/Diffusion/unconditional_CM && torchrun --nnodes=1 --nproc_per_node=8 train.py --config config.yaml --use_amp
source activate diffusion && cd /liupeng/Diffusion/unconditional_CM && torchrun --nnodes=1 --nproc_per_node=8 train.py --config config_128.yaml --use_amp
## Sample
source activate diffusion && cd /liupeng/Diffusion/unconditional_CM && torchrun --nnodes=1 --nproc_per_node=8 sample.py --config config.yaml --use_amp
torchrun --nnodes=1 --nproc_per_node=2 sample.py --config config.yaml --use_amp
## Evaluation
###  without pretrained diffusion
python3 -m pytorch_fid unconditional_CM/results/4000_unconditional_CM/EMAgenerated_ep3999_cm/pngs data/cifar10-pngs
source activate diffusion && cd /liupeng/Diffusion/unconditional_CM && python3 -m pytorch_fid unconditional_CM/results/4000_unconditional_CM/EMAgenerated_ep3999_cm/pngs data/cifar10-pngs

FID:  19.22231772112133

### without pretrained diffusion, modified group normalization
python3 -m pytorch_fid unconditional_CM/results/4000_unconditional_CM_modified_unet/EMAgenerated_ep2800_cm/pngs data/cifar10-pngs

FID: 20.945697441804384

python3 -m pytorch_fid unconditional_CM/results/4000_unconditional_CM_modified_unet/EMAgenerated_ep4800_cm/pngs data/cifar10-pngs

FID:  25.05543113984743
# Class-conditional Consistency Model