# encoding: utf-8
"""
Training and inference utilities for decomposed chord recognition.

This module provides utility functions for training, validation, and inference
with the chord structure decomposition model.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from utils.chord_decomposition import ChordReassembler, ChordDecomposer, COMPONENT_NAMES
import logging

logger = logging.getLogger(__name__)


class DecomposedChordInference:
    """
    Handles inference with the decomposed chord model.
    
    Supports:
    - Getting predictions from model
    - Decoding component predictions to chord labels
    - Handling batch inference
    - Confidence computation
    """
    
    def __init__(self, model, device=None):
        """
        Args:
            model: The trained BTC_model_decomposed
            device: Device to run inference on (defaults to model's device)
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.reassembler = ChordReassembler()
    
    def predict_batch(self, x: torch.Tensor, return_probabilities=False):
        """
        Run inference on a batch of features.
        
        Args:
            x: Feature tensor (batch_size, seq_len, feature_size)
            return_probabilities: If True, return probability distributions
        
        Returns:
            If return_probabilities=True:
                probabilities: Dict mapping components to probability tensors
            Otherwise:
                predictions: Dict mapping components to predicted indices
        """
        self.model.eval()
        with torch.no_grad():
            if return_probabilities:
                probabilities = self.model.predict_probabilities(x)
                return probabilities
            else:
                # model(x) returns different things based on probs_out setting:
                # - If probs_out=True: returns logits dict
                # - If probs_out=False: returns (predictions, loss, weights_list) tuple
                output = self.model(x)
                
                if isinstance(output, dict):
                    # probs_out=True: got logits, need to get predictions
                    predictions = self.model.decomposer.get_predictions(output)
                elif isinstance(output, tuple):
                    # probs_out=False: got (predictions, loss, weights_list)
                    predictions = output[0]
                else:
                    raise ValueError(f"Unexpected model output type: {type(output)}")
                
                return predictions
    
    def decode_predictions(self, predictions: Dict[str, torch.Tensor], 
                          reshape_to_sequences=False) -> List[str]:
        """
        Convert component predictions to chord labels.
        
        Args:
            predictions: Dict mapping component names to predicted indices
                        Shape: (batch_size*seq_len,) or (batch_size, seq_len)
            reshape_to_sequences: If True, reshape from flat to sequences
        
        Returns:
            List of chord labels
        """
        # Get batch size and sequence length
        first_component = list(predictions.keys())[0]
        pred_tensor = predictions[first_component]
        
        if len(pred_tensor.shape) == 2:
            batch_size, seq_len = pred_tensor.shape
            total_steps = batch_size * seq_len
        else:
            total_steps = pred_tensor.shape[0]
        
        # Convert to indices dict
        indices_batch = {}
        for component in COMPONENT_NAMES:
            indices = predictions[component].cpu().numpy()
            if len(indices.shape) == 2:
                indices = indices.reshape(-1)
            indices_batch[component] = indices
        
        # Decode using reassembler
        chord_labels = self.reassembler.reassemble_batch(indices_batch)
        
        return chord_labels
    
    def predict_and_decode(self, x: torch.Tensor) -> List[str]:
        """
        Run inference and directly return chord labels.
        
        Args:
            x: Feature tensor
        
        Returns:
            List of chord labels
        """
        predictions = self.predict_batch(x, return_probabilities=False)
        chord_labels = self.decode_predictions(predictions)
        return chord_labels
    
    def get_confidence_scores(self, probabilities: Dict[str, torch.Tensor]) -> np.ndarray:
        """
        Compute overall confidence from component probabilities.
        
        Uses the maximum probability across components as the confidence metric.
        
        Args:
            probabilities: Dict mapping components to probability tensors
        
        Returns:
            Confidence scores array (batch_size*seq_len,)
        """
        first_component = list(probabilities.keys())[0]
        n_steps = probabilities[first_component].shape[0]
        
        confidences = np.ones(n_steps)
        
        # For each time step, compute the minimum max probability across components
        # This ensures all components are confident about their predictions
        for t in range(n_steps):
            step_confidences = []
            for component in COMPONENT_NAMES:
                prob = probabilities[component]
                if len(prob.shape) == 3:
                    max_prob = prob[t // prob.shape[1], t % prob.shape[1], :].max().item()
                else:
                    max_prob = prob[t, :].max().item()
                step_confidences.append(max_prob)
            
            # Use minimum confidence across components
            confidences[t] = min(step_confidences)
        
        return confidences


class DecomposedChordTrainer:
    """
    Trainer for the decomposed chord recognition model.
    
    Handles training loop, validation, and metrics computation.
    """
    
    def __init__(self, model, device=None, verbose=True):
        """
        Args:
            model: The BTC_model_decomposed to train
            device: Device to train on
            verbose: Print progress information
        """
        self.model = model
        self.device = device or next(model.parameters()).device
        self.verbose = verbose
    
    def train_epoch(self, train_loader, optimizer, scheduler=None):
        """
        Run one training epoch.
        
        Args:
            train_loader: DataLoader with training batches
            optimizer: Optimizer for training
            scheduler: Optional learning rate scheduler
        
        Returns:
            avg_loss: Average loss over the epoch
            component_losses: Dict with average per-component losses
        """
        self.model.train()
        total_loss = 0.0
        component_losses_sum = {component: 0.0 for component in COMPONENT_NAMES}
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Prepare batch
            features = batch['features'].to(self.device)
            components = {comp: batch['components'][comp].to(self.device) 
                         for comp in COMPONENT_NAMES}
            
            # Reshape labels to (batch_size, seq_len)
            batch_size = features.shape[0]
            seq_len = features.shape[3]  # features shape: (batch, 1, feature_size, seq_len)
            
            labels = {}
            for component in COMPONENT_NAMES:
                labels[component] = components[component].reshape(batch_size, seq_len)
            
            # Forward pass (now returns 4 values)
            predictions, loss, _, batch_component_losses = self.model(features, labels=labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Aggregate component losses
            if batch_component_losses:
                for comp, val in batch_component_losses.items():
                    component_losses_sum[comp] += val
            
            if self.verbose and (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate average component losses
        component_losses_avg = {comp: val / num_batches if num_batches > 0 else 0.0 
                                for comp, val in component_losses_sum.items()}
        
        return avg_loss, component_losses_avg
    
    def validate(self, val_loader):
        """
        Run validation.
        
        Args:
            val_loader: DataLoader with validation batches
        
        Returns:
            metrics: Dict with validation metrics including per-component losses
        """
        self.model.eval()
        total_loss = 0.0
        component_losses_sum = {component: 0.0 for component in COMPONENT_NAMES}
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                components = {comp: batch['components'][comp].to(self.device) 
                             for comp in COMPONENT_NAMES}
                
                batch_size = features.shape[0]
                seq_len = features.shape[3]
                
                labels = {}
                for component in COMPONENT_NAMES:
                    labels[component] = components[component].reshape(batch_size, seq_len)
                
                # Forward pass (now returns 4 values)
                predictions, loss, _, batch_component_losses = self.model(features, labels=labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Aggregate component losses
                if batch_component_losses:
                    for comp, val in batch_component_losses.items():
                        component_losses_sum[comp] += val
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate average component losses
        component_losses_avg = {comp: val / num_batches if num_batches > 0 else 0.0 
                                for comp, val in component_losses_sum.items()}
        
        return {
            'val_loss': avg_loss,
            'component_losses': component_losses_avg
        }


class ChordMetrics:
    """
    Compute evaluation metrics for chord recognition.
    
    Supports:
    - Accuracy (for each component and overall)
    - Precision, recall, F1
    - Chord sequence metrics
    """
    
    def __init__(self):
        self.decomposer = ChordDecomposer()
        self.reassembler = ChordReassembler()
    
    def component_accuracy(self, predictions: Dict[str, np.ndarray], 
                          targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute per-component accuracy.
        
        Args:
            predictions: Dict mapping components to predicted indices
            targets: Dict mapping components to target indices
        
        Returns:
            accuracies: Dict mapping components to accuracy values
        """
        accuracies = {}
        for component in COMPONENT_NAMES:
            if component in predictions and component in targets:
                pred = predictions[component].reshape(-1)
                targ = targets[component].reshape(-1)
                
                correct = np.sum(pred == targ)
                total = len(targ)
                
                accuracy = correct / total if total > 0 else 0.0
                accuracies[component] = accuracy
        
        return accuracies
    
    def chord_accuracy(self, predictions: Dict[str, np.ndarray],
                      targets: Dict[str, np.ndarray]) -> float:
        """
        Compute overall chord accuracy (all 8 components must match).
        
        Args:
            predictions: Dict mapping components to predicted indices
            targets: Dict mapping components to target indices
        
        Returns:
            accuracy: Fraction of frames with all components correctly predicted
        """
        first_component = list(predictions.keys())[0]
        n_frames = len(predictions[first_component].reshape(-1))
        
        correct = 0
        for i in range(n_frames):
            all_match = True
            for component in COMPONENT_NAMES:
                pred_idx = predictions[component].reshape(-1)[i]
                targ_idx = targets[component].reshape(-1)[i]
                if pred_idx != targ_idx:
                    all_match = False
                    break
            if all_match:
                correct += 1
        
        return correct / n_frames if n_frames > 0 else 0.0
    
    def evaluate(self, predictions: Dict[str, np.ndarray],
                targets: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Comprehensive evaluation.
        
        Args:
            predictions: Predicted component indices
            targets: Target component indices
        
        Returns:
            metrics: Dict with all computed metrics
        """
        metrics = {}
        
        # Component-wise accuracy
        comp_acc = self.component_accuracy(predictions, targets)
        metrics.update({f'{comp}_accuracy': acc for comp, acc in comp_acc.items()})
        
        # Overall accuracy
        metrics['chord_accuracy'] = self.chord_accuracy(predictions, targets)
        
        # Mean component accuracy
        metrics['mean_component_accuracy'] = np.mean(list(comp_acc.values()))
        
        return metrics
