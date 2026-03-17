# encoding: utf-8
"""
Training and inference utilities for decomposed chord recognition.

This module provides utility functions for training, validation, and inference
with the chord structure decomposition model.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from utils.chord_decomposition import ChordReassembler, ChordDecomposer, COMPONENT_NAMES, CHORD_VOCAB
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
        self.last_component_raw_losses = {component: 0.0 for component in COMPONENT_NAMES}
        self.last_component_weighted_losses = {component: 0.0 for component in COMPONENT_NAMES}
        self.last_component_weights = {component: 1.0 for component in COMPONENT_NAMES}
        self.last_gradnorm_loss = 0.0
        self.last_gradnorm_inv_rate = {component: 1.0 for component in COMPONENT_NAMES}
        self.last_gradnorm_grad_norm = {component: 0.0 for component in COMPONENT_NAMES}
        self.last_gradnorm_target = {component: 0.0 for component in COMPONENT_NAMES}

    def _apply_gradnorm_update(self):
        criterion = getattr(self.model, 'criterion', None)
        if criterion is None or not getattr(criterion, 'gradnorm_enabled', False):
            return

        raw_losses = getattr(criterion, 'last_raw_loss_tensors', {})
        shared_features = getattr(self.model, 'last_shared_features', None)
        if not raw_losses or shared_features is None:
            return

        component_names = [c for c in criterion.component_names if c in raw_losses]
        if not component_names:
            return

        grad_norm_bases = []
        raw_loss_values = []
        eps = float(getattr(criterion, 'gradnorm_eps', 1e-8))

        for component in component_names:
            loss_i = raw_losses[component]
            grad_i = torch.autograd.grad(
                loss_i,
                shared_features,
                retain_graph=True,
                allow_unused=True,
            )[0]
            if grad_i is None:
                grad_norm = torch.tensor(0.0, device=loss_i.device)
            else:
                grad_norm = torch.norm(grad_i.detach(), p=2)
            grad_norm_bases.append(grad_norm)
            raw_loss_values.append(loss_i.detach().clamp_min(eps))

        grad_norm_bases = torch.stack(grad_norm_bases)
        raw_loss_values = torch.stack(raw_loss_values)

        if criterion.gradnorm_initial_losses is None:
            criterion.gradnorm_initial_losses = raw_loss_values.detach().clone()
        init_losses = criterion.gradnorm_initial_losses.to(raw_loss_values.device).clamp_min(eps)

        inv_loss_ratio = raw_loss_values / init_losses
        inv_rate = inv_loss_ratio / inv_loss_ratio.mean().clamp_min(eps)

        index_map = {c: i for i, c in enumerate(criterion.component_names)}
        selected_idx = torch.tensor([index_map[c] for c in component_names], device=raw_loss_values.device)
        w_selected = criterion.gradnorm_weights[selected_idx]
        grad_norm = w_selected * grad_norm_bases
        grad_norm_mean = grad_norm.mean()
        target = (grad_norm_mean * (inv_rate ** float(criterion.gradnorm_alpha))).detach()

        gradnorm_loss = torch.sum(torch.abs(grad_norm - target))
        if criterion.gradnorm_weights.grad is not None:
            criterion.gradnorm_weights.grad.zero_()
        gradnorm_loss.backward(retain_graph=True)

        with torch.no_grad():
            grad = criterion.gradnorm_weights.grad
            if grad is not None:
                criterion.gradnorm_weights.data -= float(criterion.gradnorm_lr) * grad
            criterion.gradnorm_weights.data.clamp_(min=float(criterion.gradnorm_w_min))
            criterion.gradnorm_weights.data.mul_(
                len(criterion.component_names) / criterion.gradnorm_weights.data.sum().clamp_min(eps)
            )

        self.last_gradnorm_loss = float(gradnorm_loss.detach().cpu().item())
        inv_rate_cpu = inv_rate.detach().cpu().tolist()
        grad_norm_cpu = grad_norm.detach().cpu().tolist()
        target_cpu = target.detach().cpu().tolist()
        self.last_gradnorm_inv_rate = {component: float(val) for component, val in zip(component_names, inv_rate_cpu)}
        self.last_gradnorm_grad_norm = {component: float(val) for component, val in zip(component_names, grad_norm_cpu)}
        self.last_gradnorm_target = {component: float(val) for component, val in zip(component_names, target_cpu)}
    
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
        weighted_component_losses_sum = {component: 0.0 for component in COMPONENT_NAMES}
        component_weights_sum = {component: 0.0 for component in COMPONENT_NAMES}
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
            self._apply_gradnorm_update()
            
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
            breakdown = getattr(getattr(self.model, 'criterion', None), 'last_forward_breakdown', None)
            if breakdown:
                for comp, values in breakdown.items():
                    weighted_component_losses_sum[comp] += values.get('weighted_loss', 0.0)
                    component_weights_sum[comp] += values.get('weight', 1.0)
            
            if self.verbose and (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # Calculate average component losses
        component_losses_avg = {comp: val / num_batches if num_batches > 0 else 0.0 
                                for comp, val in component_losses_sum.items()}
        self.last_component_raw_losses = component_losses_avg
        self.last_component_weighted_losses = {
            comp: val / num_batches if num_batches > 0 else 0.0
            for comp, val in weighted_component_losses_sum.items()
        }
        self.last_component_weights = {
            comp: val / num_batches if num_batches > 0 else 1.0
            for comp, val in component_weights_sum.items()
        }
        criterion = getattr(self.model, 'criterion', None)
        if criterion is not None:
            self.last_component_weights = criterion.get_weight_dict()
        
        return avg_loss, component_losses_avg
    
    def validate(self, val_loader, compute_class_distribution=True):
        """
        Run validation.
        
        Args:
            val_loader: DataLoader with validation batches
            compute_class_distribution: If True, collect per-head class
                distributions and per-class accuracy (recall).
        
        Returns:
            metrics: Dict with validation metrics including per-component losses
                and optionally class distribution data.
        """
        self.model.eval()
        total_loss = 0.0
        component_losses_sum = {component: 0.0 for component in COMPONENT_NAMES}
        weighted_component_losses_sum = {component: 0.0 for component in COMPONENT_NAMES}
        component_weights_sum = {component: 0.0 for component in COMPONENT_NAMES}
        num_batches = 0

        vocab_sizes = {comp: len(CHORD_VOCAB[comp]) for comp in COMPONENT_NAMES}
        if compute_class_distribution:
            pred_counts = {comp: np.zeros(vocab_sizes[comp], dtype=np.int64)
                          for comp in COMPONENT_NAMES}
            target_counts = {comp: np.zeros(vocab_sizes[comp], dtype=np.int64)
                            for comp in COMPONENT_NAMES}
            correct_per_class = {comp: np.zeros(vocab_sizes[comp], dtype=np.int64)
                                for comp in COMPONENT_NAMES}
            total_correct = {comp: 0 for comp in COMPONENT_NAMES}
            total_frames = 0
        
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
                
                predictions, loss, _, batch_component_losses = self.model(features, labels=labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                if batch_component_losses:
                    for comp, val in batch_component_losses.items():
                        component_losses_sum[comp] += val
                breakdown = getattr(getattr(self.model, 'criterion', None), 'last_forward_breakdown', None)
                if breakdown:
                    for comp, values in breakdown.items():
                        weighted_component_losses_sum[comp] += values.get('weighted_loss', 0.0)
                        component_weights_sum[comp] += values.get('weight', 1.0)

                if compute_class_distribution:
                    for comp in COMPONENT_NAMES:
                        p = predictions[comp].cpu().numpy().reshape(-1)
                        t = labels[comp].cpu().numpy().reshape(-1)
                        n_cls = vocab_sizes[comp]
                        pred_counts[comp] += np.bincount(p, minlength=n_cls)[:n_cls]
                        target_counts[comp] += np.bincount(t, minlength=n_cls)[:n_cls]
                        hits = (p == t)
                        for cls_idx in range(n_cls):
                            correct_per_class[comp][cls_idx] += hits[t == cls_idx].sum()
                        total_correct[comp] += hits.sum()
                    total_frames += p.shape[0]
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        component_losses_avg = {comp: val / num_batches if num_batches > 0 else 0.0 
                                for comp, val in component_losses_sum.items()}
        self.last_component_raw_losses = component_losses_avg
        self.last_component_weighted_losses = {
            comp: val / num_batches if num_batches > 0 else 0.0
            for comp, val in weighted_component_losses_sum.items()
        }
        self.last_component_weights = {
            comp: val / num_batches if num_batches > 0 else 1.0
            for comp, val in component_weights_sum.items()
        }
        
        result = {
            'val_loss': avg_loss,
            'component_losses': component_losses_avg,
        }

        if compute_class_distribution and total_frames > 0:
            class_dist = {}
            for comp in COMPONENT_NAMES:
                class_names = CHORD_VOCAB[comp]
                n_cls = vocab_sizes[comp]
                p_total = pred_counts[comp].sum()
                t_total = target_counts[comp].sum()
                comp_info = {
                    'class_names': class_names,
                    'pred_counts': pred_counts[comp],
                    'target_counts': target_counts[comp],
                    'pred_pct': (pred_counts[comp] / p_total * 100) if p_total > 0 else np.zeros(n_cls),
                    'target_pct': (target_counts[comp] / t_total * 100) if t_total > 0 else np.zeros(n_cls),
                    'per_class_recall': np.array([
                        correct_per_class[comp][i] / target_counts[comp][i]
                        if target_counts[comp][i] > 0 else 0.0
                        for i in range(n_cls)
                    ]),
                    'accuracy': total_correct[comp] / total_frames,
                }
                class_dist[comp] = comp_info
            result['class_distribution'] = class_dist

        return result


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
        Compute overall chord accuracy (all components must match).
        
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
