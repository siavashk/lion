#!/usr/bin/env python3
"""
Simplified PyTorch Lightning VAE training script for point clouds.
This is a concise equivalent of the original complex training setup.

Requirements:
- pytorch-lightning
- torch
- numpy
- PyTorchEMD (for loss computation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, Any, Tuple
import math
import os
from argparse import ArgumentParser

# Try to import PyTorch3D for advanced loss computation (optional)
try:
    from pytorch3d.loss import chamfer_distance
    HAS_PYTORCH3D = True
except ImportError:
    HAS_PYTORCH3D = False
    print("Note: pytorch3d not found. Using L1 loss (which matches original script anyway).")


class PointCloudDataset(Dataset):
    """Simple point cloud dataset - replace with your actual data loading"""
    
    def __init__(self, split='train', num_points=2048, num_samples=1000):
        self.split = split
        self.num_points = num_points
        self.num_samples = num_samples
        
        # Generate synthetic data for demo - replace with actual data loading
        # In real usage, load ShapeNet data here
        np.random.seed(42 if split == 'train' else 123)
        self.data = []
        for i in range(num_samples):
            # Generate random point clouds as placeholder
            points = np.random.randn(num_points, 3).astype(np.float32)
            # Normalize to unit sphere
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            self.data.append(points)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        points = torch.from_numpy(self.data[idx]).float()
        return {'points': points}


class PointNetEncoder(nn.Module):
    """Simplified PointNet-style encoder for global features"""
    
    def __init__(self, input_dim=3, output_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, output_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(output_dim)
        
    def forward(self, x):
        # x: (B, N, 3) -> (B, 3, N)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        # Global max pooling
        x = torch.max(x, dim=2)[0]  # (B, output_dim)
        return x


class AdaIN(nn.Module):
    """Adaptive Instance Normalization layer"""
    
    def __init__(self, num_features, style_dim):
        super().__init__()
        self.norm = nn.InstanceNorm1d(num_features, affine=False)
        self.style_scale = nn.Linear(style_dim, num_features)
        self.style_shift = nn.Linear(style_dim, num_features)
        
    def forward(self, x, style):
        # x: (B, C, N), style: (B, style_dim)
        normalized = self.norm(x)
        scale = self.style_scale(style).unsqueeze(2)  # (B, C, 1)
        shift = self.style_shift(style).unsqueeze(2)  # (B, C, 1)
        return scale * normalized + shift


class PointEncoder(nn.Module):
    """Point-wise encoder with style conditioning"""
    
    def __init__(self, input_dim=3, latent_dim=1, style_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        # Point-wise encoding layers
        self.conv1 = nn.Conv1d(input_dim + style_dim, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, (latent_dim + input_dim) * 2, 1)  # mu and log_var
        
        self.adain1 = AdaIN(64, style_dim)
        self.adain2 = AdaIN(128, style_dim)
        
    def forward(self, points, style):
        # points: (B, N, 3), style: (B, style_dim)
        B, N, _ = points.shape
        
        # Broadcast style to all points
        style_expanded = style.unsqueeze(1).expand(B, N, -1)  # (B, N, style_dim)
        x = torch.cat([points, style_expanded], dim=2)  # (B, N, 3 + style_dim)
        x = x.transpose(2, 1)  # (B, 3 + style_dim, N)
        
        x = F.relu(self.adain1(self.conv1(x), style))
        x = F.relu(self.adain2(self.conv2(x), style))
        x = self.conv3(x)  # (B, (latent_dim + input_dim) * 2, N)
        
        x = x.transpose(2, 1)  # (B, N, (latent_dim + input_dim) * 2)
        
        # Split into mu and log_var
        mu = x[:, :, :(self.latent_dim + self.input_dim)]
        log_var = x[:, :, (self.latent_dim + self.input_dim):]
        
        return mu.reshape(B, -1), log_var.reshape(B, -1)


class PointDecoder(nn.Module):
    """Point decoder with style conditioning"""
    
    def __init__(self, latent_dim=1, input_dim=3, style_dim=128, output_points=2048):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.output_points = output_points
        
        self.conv1 = nn.Conv1d(latent_dim + input_dim + style_dim, 128, 1)
        self.conv2 = nn.Conv1d(128, 128, 1)
        self.conv3 = nn.Conv1d(128, input_dim, 1)
        
        self.adain1 = AdaIN(128, style_dim)
        self.adain2 = AdaIN(128, style_dim)
        
    def forward(self, latent_points, style):
        # latent_points: (B, N * (latent_dim + input_dim))
        # style: (B, style_dim)
        B = latent_points.shape[0]
        latent_points = latent_points.reshape(B, self.output_points, self.latent_dim + self.input_dim)
        
        # Broadcast style
        style_expanded = style.unsqueeze(1).expand(B, self.output_points, -1)
        x = torch.cat([latent_points, style_expanded], dim=2)
        x = x.transpose(2, 1)  # (B, latent_dim + input_dim + style_dim, N)
        
        x = F.relu(self.adain1(self.conv1(x), style))
        x = F.relu(self.adain2(self.conv2(x), style))
        x = self.conv3(x)  # (B, input_dim, N)
        
        return x.transpose(2, 1)  # (B, N, input_dim)


class HierarchicalVAE(pl.LightningModule):
    """Simplified Hierarchical VAE for Point Clouds"""
    
    def __init__(
        self,
        input_dim: int = 3,
        style_dim: int = 128,
        latent_dim: int = 1,
        num_points: int = 2048,
        learning_rate: float = 1e-3,
        kl_weight: float = 0.5,
        sigma_offset: float = 6.0,
        skip_weight: float = 0.01,
        anneal_kl: bool = True,
        max_epochs: int = 8000,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.input_dim = input_dim
        self.style_dim = style_dim
        self.latent_dim = latent_dim
        self.num_points = num_points
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.sigma_offset = sigma_offset
        self.skip_weight = skip_weight
        self.anneal_kl = anneal_kl
        self.max_epochs = max_epochs
        
        # Model components
        self.style_encoder = PointNetEncoder(input_dim, style_dim * 2)  # mu and log_var
        self.point_encoder = PointEncoder(input_dim, latent_dim, style_dim)
        self.decoder = PointDecoder(latent_dim, input_dim, style_dim, num_points)
        
        # Style MLP
        self.style_mlp = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim)
        )
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def kl_divergence(self, mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1)
    
    def get_kl_weight(self):
        if not self.anneal_kl:
            return self.kl_weight
        
        # KL annealing
        step = self.global_step
        total_steps = self.max_epochs * 100  # Approximate steps per epoch
        anneal_portion = 0.5  # Anneal for first 50% of training
        
        if step < anneal_portion * total_steps:
            weight = self.kl_weight * (step / (anneal_portion * total_steps))
        else:
            weight = self.kl_weight
        
        return weight
    
    def forward(self, points):
        batch_size = points.shape[0]
        
        # Encode global style
        style_params = self.style_encoder(points)  # (B, style_dim * 2)
        style_mu = style_params[:, :self.style_dim]
        style_log_var = style_params[:, self.style_dim:]
        
        # Sample style
        style = self.reparameterize(style_mu, style_log_var)
        style = self.style_mlp(style)
        
        # Encode points
        point_mu, point_log_var = self.point_encoder(points, style)
        
        # Apply sigma offset
        point_log_var = point_log_var - self.sigma_offset
        
        # Sample point latents
        point_latents = self.reparameterize(point_mu, point_log_var)
        
        # Add skip connection to coordinate part
        latent_coords = point_latents[:, :self.num_points * self.input_dim].reshape(
            batch_size, self.num_points, self.input_dim
        )
        latent_coords = self.skip_weight * latent_coords + points
        
        # Reconstruct latent points tensor
        if self.latent_dim > 0:
            latent_features = point_latents[:, self.num_points * self.input_dim:]
            latent_coords_flat = latent_coords.reshape(batch_size, -1)
            point_latents = torch.cat([latent_coords_flat, latent_features], dim=1)
        else:
            point_latents = latent_coords.reshape(batch_size, -1)
        
        # Decode
        reconstructed = self.decoder(point_latents, style)
        
        return {
            'reconstructed': reconstructed,
            'style_mu': style_mu,
            'style_log_var': style_log_var,
            'point_mu': point_mu,
            'point_log_var': point_log_var,
            'style': style
        }
    
    def compute_loss(self, points, output):
        batch_size = points.shape[0]
        
        # Reconstruction loss (L1)
        recon_loss = F.l1_loss(
            output['reconstructed'].reshape(-1, self.input_dim),
            points.reshape(-1, self.input_dim),
            reduction='sum'
        ) / batch_size
        
        # KL losses
        style_kl = self.kl_divergence(output['style_mu'], output['style_log_var']).mean()
        point_kl = self.kl_divergence(output['point_mu'], output['point_log_var']).mean()
        
        # Get current KL weight
        kl_weight = self.get_kl_weight()
        
        # Total loss
        total_loss = recon_loss + kl_weight * (style_kl + point_kl)
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'style_kl': style_kl,
            'point_kl': point_kl,
            'kl_weight': kl_weight
        }
    
    def training_step(self, batch, batch_idx):
        points = batch['points']
        output = self(points)
        losses = self.compute_loss(points, output)
        
        # Log metrics
        self.log('train_loss', losses['loss'], prog_bar=True)
        self.log('train_recon_loss', losses['recon_loss'])
        self.log('train_style_kl', losses['style_kl'])
        self.log('train_point_kl', losses['point_kl'])
        self.log('kl_weight', losses['kl_weight'])
        
        return losses['loss']
    
    def validation_step(self, batch, batch_idx):
        points = batch['points']
        output = self(points)
        losses = self.compute_loss(points, output)
        
        self.log('val_loss', losses['loss'], prog_bar=True)
        self.log('val_recon_loss', losses['recon_loss'])
        
        return losses['loss']
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.99))
        return optimizer
    
    def sample(self, num_samples=1, device=None):
        """Generate new point clouds"""
        if device is None:
            device = next(self.parameters()).device
        
        with torch.no_grad():
            # Sample from prior
            style = torch.randn(num_samples, self.style_dim).to(device)
            style = self.style_mlp(style)
            
            # Sample point latents
            latent_size = self.num_points * (self.latent_dim + self.input_dim)
            point_latents = torch.randn(num_samples, latent_size).to(device)
            
            # Decode
            generated = self.decoder(point_latents, style)
            
        return generated


def main():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_points', type=int, default=2048)
    parser.add_argument('--max_epochs', type=int, default=100)  # Reduced for demo
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--kl_weight', type=float, default=0.5)
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use (0 for CPU)')
    args = parser.parse_args()
    
    # Create datasets
    train_dataset = PointCloudDataset('train', args.num_points, 1000)
    val_dataset = PointCloudDataset('val', args.num_points, 200)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Create model
    model = HierarchicalVAE(
        num_points=args.num_points,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        max_epochs=args.max_epochs
    )
    
    # Create trainer
    if args.gpus > 0:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator='gpu',
            devices=args.gpus,
            log_every_n_steps=10,
            check_val_every_n_epoch=10
        )
    else:
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator='cpu',
            log_every_n_steps=10,
            check_val_every_n_epoch=10
        )
    
    # Train
    trainer.fit(model, train_loader, val_loader)
    
    # Generate samples
    print("Generating samples...")
    samples = model.sample(num_samples=5)
    print(f"Generated samples shape: {samples.shape}")
    
    # Save model
    torch.save(model.state_dict(), 'vae_model.pt')
    print("Model saved to vae_model.pt")


if __name__ == '__main__':
    main() 