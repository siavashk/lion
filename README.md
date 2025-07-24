# Simplified VAE Training Script

This is a concise PyTorch Lightning implementation that replaces the bloated original training script (`train_vae.sh`). It captures the essential VAE training functionality in a much more readable and maintainable format.

## What this script does

The original script trains a hierarchical VAE for point cloud generation with:
- **Global style encoder**: Captures shape-level features (128-dim)
- **Local point encoder**: Captures point-wise features (1-dim per point + coordinates) 
- **AdaIN-based decoder**: Reconstructs point clouds using style conditioning
- **Hierarchical latent space**: Both global and local representations
- **KL annealing**: Gradually increases KL loss weight during training

## Key simplifications

1. **Removed complex abstractions**: No more nested config files, complex trainer hierarchies, or distributed training complexity
2. **Self-contained**: All model components are in one file
3. **PyTorch Lightning**: Modern training loop with automatic logging, checkpointing, etc.
4. **Synthetic data**: Includes placeholder dataset (replace with your actual data loading)
5. **Cleaner architecture**: Simplified but functionally equivalent model components

## Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Basic training
python train_vae_lightning.py

# With custom parameters (matching original script)
python train_vae_lightning.py \
    --batch_size 2 \
    --num_points 2048 \
    --max_epochs 100 \
    --learning_rate 1e-3 \
    --kl_weight 0.5 \
    --gpus 1
```

## Key parameters from original script

The script preserves all the important parameters from the original:
- Learning rate: 1e-3
- KL weight: 0.5  
- Latent dimension: 1
- Skip weight: 0.01
- Sigma offset: 6.0
- Number of points: 2048
- Loss type: L1 sum
- KL annealing: enabled

## To use with real data

Replace the `PointCloudDataset` class with your actual data loading logic. The original script used ShapeNet point clouds with category 'c1'.

## Model architecture

- **Style encoder**: PointNet-style global feature extractor
- **Point encoder**: Point-wise VAE encoder with style conditioning  
- **Decoder**: Point-wise decoder with AdaIN normalization
- **Loss**: L1 reconstruction + KL divergence with annealing

This captures the essence of the original `models.vae_adain` model in a much simpler form.
