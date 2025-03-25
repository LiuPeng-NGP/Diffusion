# Diffusion
My replication for diffusion model.

# Environment
conda env create -f environment.yml
conda activate diffusion

# Data
python3 extract_data.py

# Unconditional DDPM
cd unconditional_DDPM
## Train
torchrun --nnodes=1 --nproc_per_node=2 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_DDPM/results/2000_unconditional_DDPM/EMAgenerated_ep1999_ddpm_steps1000/pngs data/cifar10-pngs

FID:  3.2444050218018106


# Class-condiitonal DDPM
cd Class_conditional_DDPM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55000 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55500 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid Class_conditional_DDPM/results/2000_conditional_DDPM/EMAgenerated_ep1999_ddpm_steps1000/pngs data/cifar10-pngs

FID:  2.9473849136342665

# Unconditional EDM
cd unconditional_EDM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55001 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55501 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_EDM/results/1200_unconditional_EDM/EMAgenerated_ep1199_edm_steps18/pngs data/cifar10-pngs

FID:  4.234571648031874

# Class-condiitonal EDM
cd Class_conditional_EDM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55002 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55502 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid Class_conditional_EDM/results/1200_conditional_EDM/EMAgenerated_ep1199_edm_steps18/pngs data/cifar10-pngs

FID:  3.707334678054849

# Unconditional Consistency Model
cd unconditional_CM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55003 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55003 sample.py --config config.yaml --use_amp

## Evaluation
