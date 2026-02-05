#!/usr/bin/env python
# encoding: utf-8
"""
Training script for chord recognition with structure decomposition.

This script demonstrates how to train the decomposed chord recognition model
with the 8-component architecture.

Usage:
    python train_decomposed.py --config run_config.yaml --device cuda:0
"""

import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import logging
from pathlib import Path
import json
from datetime import datetime

from models.btc_model_decomposed import BTC_model_decomposed, MultiTaskLoss
from data.audio_dataset_structured import AudioDatasetStructured, AudioDataLoaderStructured
from utils.decomposed_inference import DecomposedChordTrainer, DecomposedChordInference, ChordMetrics
from utils.hparams import HParams

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Train chord recognition model with structure decomposition'
    )
    parser.add_argument('--config', type=str, default='run_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on (cuda, cpu, etc.)')
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay for optimizer')
    parser.add_argument('--gamma', type=float, default=0.5,
                       help='Class weighting gamma parameter')
    parser.add_argument('--w_max', type=float, default=10.0,
                       help='Class weighting maximum cap')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--val_interval', type=int, default=1,
                       help='Validation interval (epochs)')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = HParams.load(args.config)
    
    # Prepare datasets
    logger.info("Preparing datasets...")
    
    # Get data root and dataset names from config
    data_root = config.experiment.get('data_root', config.path.get('root_path', '/data/music/chord_recognition'))
    dataset_names = config.experiment.get('dataset_names', ['billboard'])
    
    logger.info(f"Data root: {data_root}")
    logger.info(f"Datasets: {dataset_names}")
    
    train_dataset = AudioDatasetStructured(
        config,
        root_dir=data_root,
        dataset_names=tuple(dataset_names),
        train=True,
        decompose=True
    )
    
    val_dataset = AudioDatasetStructured(
        config,
        root_dir=data_root,
        dataset_names=tuple(dataset_names),
        train=False,
        decompose=True
    )
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = AudioDataLoaderStructured(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4
    )
    
    val_loader = AudioDataLoaderStructured(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Compute class weights
    logger.info("Computing class weights...")
    class_weights = MultiTaskLoss.compute_class_weights(
        train_dataset,
        gamma=args.gamma,
        w_max=args.w_max,
        device=device
    )
    
    # Log class weights
    logger.info("Class weights computed:")
    for component, weights in class_weights.items():
        logger.info(f"  {component}: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}")
    
    # Initialize model
    logger.info("Initializing model...")
    model = BTC_model_decomposed(config, class_weights=class_weights)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    # Alternative: use ReduceLROnPlateau for adaptive scheduling
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    
    # Setup trainer and inference
    trainer = DecomposedChordTrainer(model, device=device, verbose=True)
    inference = DecomposedChordInference(model, device=device)
    metrics_fn = ChordMetrics()
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    best_epoch = 0
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")
        
        # Train
        train_loss, component_losses = trainer.train_epoch(train_loader, optimizer)
        logger.info(f"Train Loss: {train_loss:.4f}")
        training_history['train_loss'].append(train_loss)
        
        # Validate
        if (epoch + 1) % args.val_interval == 0:
            val_metrics = trainer.validate(val_loader)
            val_loss = val_metrics['val_loss']
            logger.info(f"Val Loss: {val_loss:.4f}")
            training_history['val_loss'].append(val_loss)
            
            # Save best checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                
                checkpoint_path = output_dir / f"model_best.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                }, checkpoint_path)
                logger.info(f"Saved best checkpoint to {checkpoint_path}")
        
        # Update learning rate
        scheduler.step()
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"model_epoch_{epoch + 1:03d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    
    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save({
        'epoch': args.num_epochs,
        'model_state_dict': model.state_dict(),
    }, final_path)
    logger.info(f"Saved final model to {final_path}")
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")
    
    logger.info(f"\n=== Training Complete ===")
    logger.info(f"Best validation loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    logger.info(f"Checkpoints saved to: {output_dir}")


if __name__ == '__main__':
    main()
