# Diffusion
Replication for diffusion models.

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
python3 -m pytorch_fid unconditional_CM/results/2000_unconditional_CM/EMAgenerated_ep1999_cm/pngs data/cifar10-pngs

FID:  13.528500369549818

# Unconditional Flow Matching
cd unconditional_FM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55004 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55014 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_FM/results/2000_unconditional_FM/EMAgenerated_ep1999_fm/pngs data/cifar10-pngs

50 steps FID:  7.1084244983728695

200 steps FID:  5.290798096032063

1000 steps FID:  4.914015652275225

# Unconditional Optimal Transport Conditional Flow Matching
cd unconditional_OTCFM
## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55005 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55015 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_OTCFM/results/2000_unconditional_FM/EMAgenerated_ep1999_fm/pngs data/cifar10-pngs

200 steps FID:  5.2403232585506885


# Unconditional DDPM transformer
cd unconditional_DDPM_transformer

## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55006 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55016 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_DDPM_transformer/results/2000_unconditional_DDPM/EMAgenerated_ep1999_ddpm_steps1000/pngs data/cifar10-pngs

FID:  8.94988141073594

# Unconditional DDPM u shape transformer
cd unconditional_DDPM_u_transformer

## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55008 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55018 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_DDPM_u_transformer/results/2000_unconditional_DDPM/EMAgenerated_ep1999_ddpm_steps1000/pngs data/cifar10-pngs

FID:  5.398042306693924

# Unconditional CM transformer
cd unconditional_CM_transformer

## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55007 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55017 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_CM_transformer/results/2000_unconditional_CM/EMAgenerated_ep1999_cm/pngs data/cifar10-pngs

FID:  18.990830834827193

# Unconditional CM u shape transformer
cd unconditional_CM_u_transformer

## Train
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55008 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=2 --rdzv_endpoint=localhost:55018 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_CM_u_transformer/results/2000_unconditional_CM/EMAgenerated_ep1999_cm/pngs data/cifar10-pngs

FID:  23.093370349372435

# unconditional MeanFlow

cd unconditional_MeanFlow
## Train
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:55009 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:55019 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_MeanFlow/results/2000_unconditional_MeanFlow/EMAgenerated_ep1999_meanflow/pngs data/cifar10-pngs
FID:  20.889072117867954

# unconditional MeanFlow with pretrained FM
cd uncondiitonal_MeanFlow_w_pretrainedFM
## Train
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:55010 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:55020 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_MeanFlow/results/2000_unconditional_MeanFlow/EMAgenerated_ep1999_meanflow/pngs data/cifar10-pngs
FID:  20.889072117867954

# unconditional MeanFlow multisteps
cd unconditional_MeanFlow_multisteps
## Train
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:55011 train.py --config config.yaml --use_amp
## Sample
torchrun --nnodes=1 --nproc_per_node=8 --rdzv_endpoint=localhost:55021 sample.py --config config.yaml --use_amp
## Evaluation
python3 -m pytorch_fid unconditional_MeanFlow_multisteps/results/2000_unconditional_MeanFlow/EMAgenerated_ep1999_meanflow/pngs data/cifar10-pngs
### 2 steps
FID:  14.809913632763653
### 3 steps
FID:  14.028773281908798
### 5 steps
FID:  13.799752407995243
### 10 steps
FID:  13.829525196642294
### 30 steps
FID:  14.005572469598405