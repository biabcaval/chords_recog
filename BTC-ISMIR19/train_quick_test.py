# encoding: utf-8
"""
Quick training test for the decomposed chord model.
Runs only 2 epochs with a small subset of data.
"""
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, '.')

from utils.hparams import HParams
from data.audio_dataset_structured import AudioDatasetStructured, _collate_fn_structured
from models.btc_model_decomposed import BTC_model_decomposed, MultiTaskLoss
from utils.chord_decomposition import CHORD_VOCAB, COMPONENT_NAMES

def main():
    print("=" * 60)
    print("QUICK TRAINING TEST - Decomposed Chord Model")
    print("=" * 60)
    
    # Load config
    config = HParams.load("run_config.yaml")
    
    # Get paths
    data_root = config.experiment.get('data_root', '/home/daniel.melo/datasets')
    dataset_names = config.experiment.get('dataset_names', ['billboard'])
    
    logger.info(f"Data root: {data_root}")
    logger.info(f"Datasets: {dataset_names}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    logger.info("Loading datasets...")
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
    
    logger.info(f"Full train dataset: {len(train_dataset)} samples")
    logger.info(f"Full val dataset: {len(val_dataset)} samples")
    
    # Use small subset for quick test
    subset_size = min(500, len(train_dataset))
    val_subset_size = min(100, len(val_dataset))
    
    random.seed(42)
    train_indices = random.sample(range(len(train_dataset)), subset_size)
    val_indices = random.sample(range(len(val_dataset)), val_subset_size)
    
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    logger.info(f"Using subset: {len(train_subset)} train, {len(val_subset)} val samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=16,
        shuffle=True,
        num_workers=0,
        collate_fn=_collate_fn_structured
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_fn_structured
    )
    
    # Create model
    logger.info("Creating model...")
    model = BTC_model_decomposed(config=config.model).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss and optimizer
    criterion = MultiTaskLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Training loop
    num_epochs = 3
    logger.info(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(device)
            components = {k: v.to(device) for k, v in batch['components'].items()}
            
            optimizer.zero_grad()
            outputs = model(features)
            
            loss, loss_dict = criterion(outputs, components)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / num_batches
        
        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                components = {k: v.to(device) for k, v in batch['components'].items()}
                
                outputs = model(features)
                loss, _ = criterion(outputs, components)
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0
        
        logger.info(f"\n=== Epoch {epoch+1}/{num_epochs} Complete ===")
        logger.info(f"Train Loss: {avg_train_loss:.4f}")
        logger.info(f"Val Loss: {avg_val_loss:.4f}\n")
    
    # Quick inference test
    logger.info("Testing inference...")
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader))
        features = sample_batch['features'].to(device)
        outputs = model(features)
        
        # Get predictions for first sample
        logger.info("\nSample predictions (first frame of first sample):")
        for comp_name in COMPONENT_NAMES:
            logits = outputs[comp_name][0, 0]  # first sample, first frame
            pred_idx = torch.argmax(logits).item()
            pred_label = CHORD_VOCAB[comp_name][pred_idx]
            
            # Ground truth
            gt_idx = sample_batch['components'][comp_name][0, 0].item()
            gt_label = CHORD_VOCAB[comp_name][gt_idx]
            
            logger.info(f"  {comp_name}: pred='{pred_label}' (idx={pred_idx}), gt='{gt_label}' (idx={gt_idx})")
    
    print("\n" + "=" * 60)
    print("QUICK TEST COMPLETE!")
    print("=" * 60)

if __name__ == '__main__':
    main()
