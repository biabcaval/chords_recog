#!/usr/bin/env python
# encoding: utf-8
"""
Training script for chord recognition with 8-component decomposition.

Handles data splitting with k-fold cross-validation and trains the decomposed model.
Generates train/validation metrics during training.

Usage:
    python train_decomposed_full.py --config run_config.yaml --dataset billboard --kfold 4 --num_epochs 50
"""

import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import logging
from pathlib import Path
import json
from datetime import datetime
import os

from models.btc_model_decomposed import BTC_model_decomposed, MultiTaskLoss
from data.audio_dataset_structured import AudioDatasetStructured
from utils.decomposed_inference import DecomposedChordTrainer
from utils.hparams import HParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DecomposedDataLoader:
    """Wraps AudioDatasetStructured for k-fold cross-validation."""
    
    def __init__(self, config, dataset_name, root_dir, train_kfold=4, test_kfold=4):
        """
        Initialize loader with k-fold splits.
        
        Args:
            config: HParams configuration
            dataset_name: Name of dataset (e.g., 'billboard', 'dj_avan')
            root_dir: Root directory for decomposed data
            train_kfold: Fold index for validation (others used for training)
            test_kfold: Fold index for testing
        """
        self.config = config
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.train_kfold = train_kfold
        self.test_kfold = test_kfold
        
        # AudioDataset expects dataset_names as tuple
        dataset_names = (dataset_name,)
        
        # Create datasets with different kfold indices
        self.train_dataset = AudioDatasetStructured(
            config,
            root_dir=root_dir,
            train=True,
            kfold=train_kfold,
            decompose=True,
            dataset_names=dataset_names
        )
        
        self.val_dataset = AudioDatasetStructured(
            config,
            root_dir=root_dir,
            train=False,
            kfold=train_kfold,  # Same fold as validation set
            decompose=True,
            dataset_names=dataset_names
        )
        
        self.test_dataset = AudioDatasetStructured(
            config,
            root_dir=root_dir,
            train=False,
            kfold=test_kfold,
            decompose=True,
            dataset_names=dataset_names
        )
    
    def get_loaders(self, batch_size, num_workers=4):
        """Create PyTorch DataLoaders."""
        from torch.utils.data import DataLoader
        
        # Debug: Check dataset sizes
        logger.info(f"Train dataset size: {len(self.train_dataset)}")
        logger.info(f"Val dataset size: {len(self.val_dataset)}")
        logger.info(f"Test dataset size: {len(self.test_dataset)}")
        
        # Debug: Check if datasets are empty
        if len(self.train_dataset) == 0:
            logger.error("TRAIN DATASET IS EMPTY!")
            logger.error(f"Root dir: {self.root_dir}")
            logger.error(f"Dataset names: {self.train_dataset.dataset_names}")
            # Try to list the directory
            import os
            check_path = os.path.join(self.root_dir, "result_decomposed")
            if os.path.exists(check_path):
                logger.error(f"Contents of {check_path}:")
                for item in os.listdir(check_path):
                    logger.error(f"  - {item}")
                    # Check deeper
                    deeper_path = os.path.join(check_path, item)
                    if os.path.isdir(deeper_path):
                        logger.error(f"    Contents of {deeper_path}:")
                        for subitem in os.listdir(deeper_path):
                            logger.error(f"      - {subitem}")
                            # Check even deeper
                            deeper_path2 = os.path.join(deeper_path, subitem)
                            if os.path.isdir(deeper_path2):
                                logger.error(f"        Contents of {deeper_path2}:")
                                for subitem2 in os.listdir(deeper_path2)[:3]:
                                    logger.error(f"          - {subitem2}")
            else:
                logger.error(f"{check_path} does not exist!")
                # Try result instead
                check_path = os.path.join(self.root_dir, "result")
                if os.path.exists(check_path):
                    logger.error(f"Found result instead: {check_path}")
        
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
        
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
        
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=self.collate_fn
        )
        
        logger.info(f"Train samples: {len(self.train_dataset)}")
        logger.info(f"Val samples: {len(self.val_dataset)}")
        logger.info(f"Test samples: {len(self.test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for decomposed data."""
        # Simple implementation - can be enhanced
        return torch.utils.data._default_collate(batch)


def train_epoch(model, loader, optimizer, device, loss_fn, log_interval=10):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(loader):
        # Move data to device
        features = batch['feature'].to(device)
        targets = {k: v.to(device) for k, v in batch['targets'].items()}
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        
        # Compute loss
        loss = loss_fn(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / num_batches
            logger.info(f"  Batch {batch_idx+1}/{len(loader)}: Loss={avg_loss:.4f}")
    
    return total_loss / num_batches


def validate(model, loader, device, loss_fn):
    """Validate model."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            features = batch['feature'].to(device)
            targets = {k: v.to(device) for k, v in batch['targets'].items()}
            
            outputs = model(features)
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(
        description='Train decomposed chord recognition model'
    )
    parser.add_argument('--config', type=str, default='run_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str, default='dj_avan',
                       help='Dataset name (billboard, dj_avan, etc.)')
    parser.add_argument('--data_root', type=str, default='/home/daniel.melo/datasets',
                       help='Root directory for preprocessed data (will look in result_decomposed subdirectory)')
    parser.add_argument('--kfold', type=int, default=4,
                       help='K-fold index for validation (0-4). Others used for training.')
    parser.add_argument('--test_kfold', type=int, default=None,
                       help='K-fold index for testing (if None, same as kfold)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--output_dir', type=str, default='./checkpoints_decomposed',
                       help='Directory to save checkpoints')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                       help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Validation interval (epochs)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    test_kfold = args.test_kfold if args.test_kfold is not None else args.kfold
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config
    logger.info(f"Loading configuration from {args.config}")
    config = HParams.load(args.config)
    
    # Prepare data
    logger.info(f"Preparing {args.dataset} dataset from {args.data_root}...")
    data_loader = DecomposedDataLoader(
        config,
        dataset_name=args.dataset,
        root_dir=args.data_root,
        train_kfold=args.kfold,
        test_kfold=test_kfold
    )
    
    train_loader, val_loader, test_loader = data_loader.get_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    logger.info("Creating model...")
    model = BTC_model_decomposed(
        config=config,
        class_weights=None
    )
    
    model.to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss function (use the one from model)
    loss_fn = model.criterion
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'epochs': []
    }
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, loss_fn,
            log_interval=args.log_interval
        )
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_loss = validate(model, val_loader, device, loss_fn)
            logger.info(f"Val loss: {val_loss:.4f}")
            
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)
            metrics['epochs'].append(epoch + 1)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_path = output_dir / f"best_model_{args.dataset}_fold{args.kfold}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best model to {checkpoint_path}")
            
            scheduler.step(val_loss)
    
    # Test
    logger.info("\nEvaluating on test set...")
    test_loss = validate(model, test_loader, device, loss_fn)
    logger.info(f"Test loss: {test_loss:.4f}")
    
    # Save metrics
    metrics['test_loss'] = test_loss
    metrics_path = output_dir / f"metrics_{args.dataset}_fold{args.kfold}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
