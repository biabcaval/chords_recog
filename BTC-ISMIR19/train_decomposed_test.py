# encoding: utf-8
"""
Quick training test with synthetic data for decomposed model.

This script trains the decomposed model for a few iterations using randomly generated data
to verify the training pipeline works without needing real audio data.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).parent))

from models.btc_model_decomposed import BTC_model_decomposed
from utils.chord_decomposition import COMPONENT_NAMES, get_vocab_sizes
from utils.hparams import HParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticDataset(Dataset):
    """Synthetic dataset for quick testing without real audio."""
    
    def __init__(self, num_samples=100, seq_len=50, feature_size=192):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.feature_size = feature_size
        self.vocab_sizes = get_vocab_sizes()
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Random features
        features = torch.randn(self.seq_len, self.feature_size)
        
        # Random labels for each component
        labels = {
            component: torch.randint(0, self.vocab_sizes[component], (self.seq_len,))
            for component in COMPONENT_NAMES
        }
        
        return features, labels


def collate_fn(batch):
    """Custom collate for synthetic data."""
    features_list = []
    labels_dict = {component: [] for component in COMPONENT_NAMES}
    
    for features, labels in batch:
        features_list.append(features)
        for component in COMPONENT_NAMES:
            labels_dict[component].append(labels[component])
    
    # Stack features: (batch_size, seq_len, feature_size)
    features = torch.stack(features_list)
    
    # Stack labels: (batch_size, seq_len)
    for component in COMPONENT_NAMES:
        labels_dict[component] = torch.stack(labels_dict[component])
    
    return features, labels_dict


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        # Forward pass
        optimizer.zero_grad()
        predictions, loss, _ = model(features, labels=labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % 5 == 0:
            logger.info(f"  Batch {batch_idx + 1}/{len(dataloader)}: loss={loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Train decomposed chord model with synthetic data')
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of synthetic samples')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seq_len', type=int, default=50, help='Sequence length')
    parser.add_argument('--feature_size', type=int, default=192, help='Feature size')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create synthetic dataset
    logger.info("Creating synthetic dataset...")
    train_dataset = SyntheticDataset(
        num_samples=args.num_samples,
        seq_len=args.seq_len,
        feature_size=args.feature_size
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    logger.info(f"Dataset created: {len(train_dataset)} samples")
    logger.info(f"Dataloader: {len(train_loader)} batches")
    
    # Create model config
    config = {
        'feature_size': args.feature_size,
        'hidden_size': 128,
        'num_layers': 2,
        'num_heads': 8,
        'total_key_depth': 64,
        'total_value_depth': 64,
        'filter_size': 512,
        'timestep': args.seq_len,
        'input_dropout': 0.1,
        'layer_dropout': 0.1,
        'attention_dropout': 0.1,
        'relu_dropout': 0.1,
        'output_dropout': 0.1,
        'probs_out': False,
        'use_decomposition': True,
    }
    
    # Create model
    logger.info("Creating model...")
    model = BTC_model_decomposed(config)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {total_params:,} total params, {trainable_params:,} trainable")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training loop
    logger.info(f"\nStarting training for {args.num_epochs} epochs...\n")
    
    for epoch in range(args.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{args.num_epochs}")
        epoch_loss = train_epoch(model, train_loader, optimizer, device)
        logger.info(f"  Average loss: {epoch_loss:.4f}\n")
    
    logger.info("Training completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Prepare your real dataset")
    logger.info("2. Run: python train_decomposed.py --config run_config.yaml --num_epochs 50")
    logger.info("3. Monitor training with: tensorboard --logdir=runs/")


if __name__ == '__main__':
    main()
