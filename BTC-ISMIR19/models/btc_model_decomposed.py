# encoding: utf-8
"""
Chord Structure Decomposition Model with Multi-Head Architecture.

This module implements the BTCDecomposed model that predicts 8 chord components
in parallel, following the Chord Structure Decomposition approach.
"""

import torch
import torch.nn as nn
import numpy as np
from utils.transformer_modules import (
    bi_directional_self_attention_layers,
    ConformerEncoder,
)
from utils.chord_decomposition import COMPONENT_NAMES, CHORD_VOCAB


class ComponentHead(nn.Module):
    """
    Individual output head for a single chord component.
    
    Args:
        hidden_size: Size of the hidden representation from feature extractor
        vocab_size: Number of classes for this component
        dropout: Dropout probability
    """
    
    def __init__(self, hidden_size: int, vocab_size: int, dropout: float = 0.0):
        super().__init__()
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        x = self.dropout(x)
        logits = self.linear(x)
        return logits


class MultiHeadChordDecomposer(nn.Module):
    """
    Multi-head architecture for chord structure decomposition.
    
    Implements 8 parallel output heads, one for each chord component:
    1. Root (13 classes)
    2. Bass (13 classes)
    3. Triad (7 classes)
    4. Misc/Power (2 classes)
    5. 7th (4 classes)
    6. 9th (4 classes)
    7. 11th (3 classes)
    8. 13th (3 classes)
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Get vocabulary sizes for each component
        self.vocab_sizes = {component: len(CHORD_VOCAB[component])
                          for component in COMPONENT_NAMES}
        
        # Create output head for each component
        self.heads = nn.ModuleDict({
            component: ComponentHead(hidden_size, self.vocab_sizes[component], dropout)
            for component in COMPONENT_NAMES
        })
        
        # Store component order for consistent access
        self.component_names = COMPONENT_NAMES
    
    def forward(self, x):
        """
        Forward pass through all component heads.
        
        Args:
            x: (batch_size, seq_len, hidden_size)
            
        Returns:
            logits: Dict mapping component names to logits tensors
                   Each logit tensor has shape (batch_size, seq_len, vocab_size)
        """
        logits = {}
        for component in self.component_names:
            logits[component] = self.heads[component](x)
        return logits
    
    def get_predictions(self, logits):
        """
        Get class predictions from logits using argmax.
        
        Args:
            logits: Dict mapping component names to logits
            
        Returns:
            predictions: Dict mapping component names to class indices
        """
        predictions = {}
        for component in self.component_names:
            predictions[component] = torch.argmax(logits[component], dim=-1)
        return predictions
    
    def get_probabilities(self, logits):
        """
        Get class probabilities from logits using softmax.
        
        Args:
            logits: Dict mapping component names to logits
            
        Returns:
            probabilities: Dict mapping component names to probability distributions
        """
        probabilities = {}
        for component in self.component_names:
            probabilities[component] = torch.softmax(logits[component], dim=-1)
        return probabilities


class BTC_model_decomposed(nn.Module):
    """
    BTC model with chord structure decomposition.
    
    Extends the standard BTC architecture to predict 8 chord components in parallel
    instead of a single monolithic chord label. This follows the Chord Structure
    Decomposition technique.
    
    Args:
        config: Configuration dictionary with model hyperparameters
        class_weights: Optional dict mapping component names to class weight tensors
    """
    
    def __init__(self, config, class_weights=None, component_weights=None):
        super().__init__()
        
        # Handle both dict-like and HParams-like config objects
        if hasattr(config, 'model'):
            # It's an HParams object
            cfg = config.model
        else:
            # It's a dict
            cfg = config
        
        self.timestep = cfg['timestep']
        self.probs_out = cfg.get('probs_out', False)
        self.use_decomposition = cfg.get('use_decomposition', True)
        
        # Feature extractor (bi-directional self-attention)
        params = (
            cfg['feature_size'],
            cfg['hidden_size'],
            cfg['num_layers'],
            cfg['num_heads'],
            cfg['total_key_depth'],
            cfg['total_value_depth'],
            cfg['filter_size'],
            cfg['timestep'],
            cfg['input_dropout'],
            cfg['layer_dropout'],
            cfg['attention_dropout'],
            cfg['relu_dropout']
        )
        
        self.self_attn_layers = bi_directional_self_attention_layers(*params)
        
        # Multi-head chord decomposition
        self.decomposer = MultiHeadChordDecomposer(
            hidden_size=cfg['hidden_size'],
            dropout=cfg.get('output_dropout', 0.0)
        )
        
        # Loss function
        self.criterion = MultiTaskLoss(
            vocab_sizes=self.decomposer.vocab_sizes,
            class_weights=class_weights,
            gamma=cfg.get('class_weight_gamma', 0.5),
            w_max=cfg.get('class_weight_max', 10.0),
            component_weights=component_weights
        )
        
        # Store component names for reference
        self.component_names = COMPONENT_NAMES
    
    def forward(self, x, labels=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input features. Can be:
               - (batch_size, seq_len, feature_size): Standard 3D format
               - (batch_size, 1, feature_size, seq_len): Image-like 4D format (gets transposed)
            labels: Optional dict mapping component names to target indices
                   Each label tensor should have shape (batch_size, seq_len)
        
        Returns:
            If probs_out=True:
                logits: Dict of logit tensors for each component
            If probs_out=False:
                predictions: Dict of predicted class indices
                loss: Scalar loss value (None if labels not provided)
                weights_list: Attention weights (from self_attn_layers)
                component_losses: Dict of per-component loss values (None if labels not provided)
        """
        # Handle different input shapes
        if x.dim() == 4:
            # Image-like format: (batch, 1, feature_size, seq_len)
            # Reshape to: (batch, seq_len, feature_size)
            x = x.squeeze(1)  # (batch, feature_size, seq_len)
            x = x.permute(0, 2, 1)  # (batch, seq_len, feature_size)
        
        # Feature extraction
        self_attn_output, weights_list = self.self_attn_layers(x)
        
        # Get logits from all heads
        logits = self.decomposer(self_attn_output)
        
        # Return mode: logits only (for inference with CRF, etc.)
        if self.probs_out:
            return logits
        
        # Get predictions
        predictions = self.decomposer.get_predictions(logits)
        
        # Calculate loss if labels provided
        loss = None
        component_losses = None
        if labels is not None:
            loss, component_losses = self.criterion(logits, labels)
        
        return predictions, loss, weights_list, component_losses
    
    def predict_probabilities(self, x):
        """
        Get probability distributions for all components.
        
        Args:
            x: Input features. Can be:
               - (batch_size, seq_len, feature_size): Standard 3D format
               - (batch_size, 1, feature_size, seq_len): Image-like 4D format (gets transposed)
        
        Returns:
            probabilities: Dict mapping component names to probability tensors
        """
        # Handle different input shapes
        if x.dim() == 4:
            # Image-like format: (batch, 1, feature_size, seq_len)
            # Reshape to: (batch, seq_len, feature_size)
            x = x.squeeze(1)  # (batch, feature_size, seq_len)
            x = x.permute(0, 2, 1)  # (batch, seq_len, feature_size)
        
        self_attn_output, _ = self.self_attn_layers(x)
        logits = self.decomposer(self_attn_output)
        probabilities = self.decomposer.get_probabilities(logits)
        return probabilities


class ChordFormer_model_decomposed(nn.Module):
    """
    ChordFormer model with chord structure decomposition.

    Uses a Conformer encoder and predicts decomposed chord components in parallel
    through MultiHeadChordDecomposer.

    Args:
        config: Configuration dictionary with model hyperparameters
        class_weights: Optional dict mapping component names to class weight tensors
    """

    def __init__(self, config, class_weights=None, component_weights=None):
        super().__init__()

        # Handle both dict-like and HParams-like config objects
        if hasattr(config, 'model'):
            cfg = config.model
        else:
            cfg = config

        self.timestep = cfg['timestep']
        self.probs_out = cfg.get('probs_out', False)
        self.use_decomposition = cfg.get('use_decomposition', True)

        # Conformer encoder (ChordFormer backbone)
        self.conformer_encoder = ConformerEncoder(
            embedding_size=cfg['feature_size'],
            hidden_size=cfg['hidden_size'],
            num_layers=cfg['num_layers'],
            num_heads=cfg['num_heads'],
            conv_kernel_size=cfg.get('conv_kernel_size', 31),
            ff_expansion_factor=cfg.get('ff_expansion_factor', 4),
            conv_expansion_factor=cfg.get('conv_expansion_factor', 2),
            max_length=cfg['timestep'],
            input_dropout=cfg.get('input_dropout', 0.2),
            layer_dropout=cfg.get('layer_dropout', 0.2),
            attention_map=True,
        )

        # Multi-head chord decomposition
        self.decomposer = MultiHeadChordDecomposer(
            hidden_size=cfg['hidden_size'],
            dropout=cfg.get('output_dropout', 0.0)
        )

        # Loss function
        self.criterion = MultiTaskLoss(
            vocab_sizes=self.decomposer.vocab_sizes,
            class_weights=class_weights,
            gamma=cfg.get('class_weight_gamma', 0.5),
            w_max=cfg.get('class_weight_max', 10.0),
            component_weights=component_weights
        )

        self.component_names = COMPONENT_NAMES

    def _prepare_input(self, x):
        """Normalize model input shape to (batch_size, seq_len, feature_size)."""
        if x.dim() == 4:
            x = x.squeeze(1)
            x = x.permute(0, 2, 1)
        return x

    def forward(self, x, labels=None):
        """
        Forward pass through Conformer + decomposed output heads.

        Args:
            x: Input features. Can be:
               - (batch_size, seq_len, feature_size): Standard 3D format
               - (batch_size, 1, feature_size, seq_len): Image-like 4D format
            labels: Optional dict mapping component names to target indices

        Returns:
            If probs_out=True:
                logits: Dict of logit tensors for each component
            If probs_out=False:
                predictions: Dict of predicted class indices
                loss: Scalar loss value (None if labels not provided)
                weights_list: Attention weights from Conformer blocks
                component_losses: Dict of per-component loss values (None if labels not provided)
        """
        x = self._prepare_input(x)

        # Feature extraction with Conformer encoder
        encoder_output, weights_list = self.conformer_encoder(x)

        # Get logits from all decomposition heads
        logits = self.decomposer(encoder_output)

        if self.probs_out:
            return logits

        predictions = self.decomposer.get_predictions(logits)

        loss = None
        component_losses = None
        if labels is not None:
            loss, component_losses = self.criterion(logits, labels)

        return predictions, loss, weights_list, component_losses

    def predict_probabilities(self, x):
        """
        Get probability distributions for all components.

        Args:
            x: Input features in 3D or 4D format

        Returns:
            probabilities: Dict mapping component names to probability tensors
        """
        x = self._prepare_input(x)
        encoder_output, _ = self.conformer_encoder(x)
        logits = self.decomposer(encoder_output)
        probabilities = self.decomposer.get_probabilities(logits)
        return probabilities


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for chord component prediction with class re-weighting.
    
    Implements separate CrossEntropyLoss for each component with optional
    class weighting to handle imbalanced data (e.g., rare chords).
    
    The weighting formula is:
        w_m^(j) = min((n_m^(j) / max(n_m'^(j)))^(-gamma), w_max)
    
    where:
        - n_m^(j) is the count of class m in component j
        - gamma is the weighting exponent
        - w_max is the maximum weight cap
    
    Args:
        vocab_sizes: Dict mapping component names to vocabulary sizes
        class_weights: Optional dict with pre-computed class weights
        gamma: Weighting exponent (default: 0.5)
        w_max: Maximum weight cap (default: 10.0)
        component_weights: Optional dict of loss weights for each component
    """
    
    def __init__(self, vocab_sizes, class_weights=None, gamma=0.5, w_max=10.0,
                 component_weights=None):
        super().__init__()
        
        self.vocab_sizes = vocab_sizes
        self.gamma = gamma
        self.w_max = w_max
        self.component_names = list(vocab_sizes.keys())
        
        # Initialize loss functions for each component
        self.losses = nn.ModuleDict()
        
        for component in self.component_names:
            weight = None
            if class_weights is not None and component in class_weights:
                weight = class_weights[component]
            
            self.losses[component] = nn.CrossEntropyLoss(
                weight=weight,
                reduction='mean'
            )
        
        # Component-level loss weights (for weighted combination)
        if component_weights is None:
            # Default: equal weight for all components
            component_weights = {component: 1.0 for component in self.component_names}
        self.component_weights = component_weights
        self.last_forward_breakdown = {}
    
    def forward(self, logits, labels):
        """
        Calculate multi-task loss.
        
        Args:
            logits: Dict mapping component names to logit tensors
                   Shape: (batch_size, seq_len, vocab_size)
            labels: Dict mapping component names to target indices
                   Shape: (batch_size, seq_len)
        
        Returns:
            total_loss: Weighted sum of component losses (tensor)
            loss_dict: Dict mapping component names to individual loss values (floats)
        """
        total_loss = None
        loss_dict = {}
        self.last_forward_breakdown = {}
        
        for component in self.component_names:
            if component not in logits or component not in labels:
                continue
            
            logits_c = logits[component]
            labels_c = labels[component]
            
            # Reshape for CrossEntropyLoss
            # Input: (batch_size, seq_len, vocab_size) -> (batch_size*seq_len, vocab_size)
            # Target: (batch_size, seq_len) -> (batch_size*seq_len,)
            batch_size, seq_len, vocab_size = logits_c.shape
            
            logits_flat = logits_c.reshape(-1, vocab_size)
            labels_flat = labels_c.reshape(-1)
            
            # Calculate loss for this component
            component_loss = self.losses[component](logits_flat, labels_flat)
            
            # Apply component weight
            weight = self.component_weights.get(component, 1.0)
            weighted_loss = weight * component_loss
            self.last_forward_breakdown[component] = {
                'raw_loss': float(component_loss.detach().cpu().item()),
                'weight': float(weight),
                'weighted_loss': float(weighted_loss.detach().cpu().item()),
            }
            
            if total_loss is None:
                total_loss = weighted_loss
            else:
                total_loss = total_loss + weighted_loss
            
            loss_dict[component] = component_loss.item()
        
        # Return tensor (if no losses, return 0 tensor)
        if total_loss is None:
            total_loss = torch.tensor(0.0, requires_grad=True)
        
        return total_loss, loss_dict
    
    @staticmethod
    def compute_class_weights(train_dataset, gamma=0.5, w_max=10.0, device=None, return_counts=False):
        """
        Compute class weights from training data using the Class Re-weighting formula.
        
        Formula:
            w_m^(j) = min((n_m^(j) / max(n_m'^(j)))^(-gamma), w_max)
        
        where:
            - n_m^(j) is the count of class m in component j
            - max(n_m'^(j)) is the maximum count among all classes in component j
            - gamma is the weighting exponent (default 0.5)
            - w_max is the maximum weight cap (default 10.0)
        
        This gives higher weights to rare classes to handle the long tail distribution.
        
        Args:
            train_dataset: Training dataset with 'components' field. Each sample should
                          have a 'components' dict mapping component names to index arrays.
                          Dataset should support __len__ and __getitem__.
            gamma: Weighting exponent (lower = more smoothing). Default: 0.5
            w_max: Maximum weight cap to prevent extreme weights. Default: 10.0
            device: Device to place tensors on (defaults to CPU)
        
        Returns:
            class_weights: Dict mapping component names to weight tensors
            class_counts (optional): Dict mapping component names to class counts (if return_counts=True)
        """
        import logging
        logger = logging.getLogger(__name__)
        
        class_weights = {}
        class_counts = {
            component: np.zeros(len(CHORD_VOCAB[component]), dtype=np.float64)
            for component in COMPONENT_NAMES
        }

        # Single dataset pass for all components using vectorized bincount.
        n_samples = len(train_dataset)
        for i in range(n_samples):
            sample = train_dataset[i]
            if 'components' not in sample:
                continue

            sample_components = sample['components']
            for component in COMPONENT_NAMES:
                vocab_size = len(CHORD_VOCAB[component])
                component_data = sample_components.get(component, None)
                if component_data is None:
                    continue

                if isinstance(component_data, torch.Tensor):
                    component_indices = component_data.detach().cpu().numpy().reshape(-1)
                elif isinstance(component_data, np.ndarray):
                    component_indices = component_data.reshape(-1)
                else:
                    component_indices = np.asarray(component_data).reshape(-1)

                if component_indices.size == 0:
                    continue

                component_indices = component_indices.astype(np.int64, copy=False)
                valid = component_indices[(component_indices >= 0) & (component_indices < vocab_size)]
                if valid.size == 0:
                    continue

                class_counts[component] += np.bincount(valid, minlength=vocab_size).astype(np.float64)

        for component in COMPONENT_NAMES:
            counts = class_counts[component]
            vocab_size = len(CHORD_VOCAB[component])
            max_count = float(np.max(counts))

            if max_count == 0:
                logger.warning(f"No data found for component '{component}'. Using uniform weights.")
                weights = np.ones(vocab_size, dtype=np.float64)
            else:
                # Start with max cap for all classes (including unseen classes).
                weights = np.full(vocab_size, w_max, dtype=np.float64)
                nonzero = counts > 0
                ratios = counts[nonzero] / max_count
                weights_nonzero = np.minimum(np.power(ratios, -gamma), w_max)
                weights[nonzero] = weights_nonzero

            weights_tensor = torch.tensor(weights, dtype=torch.float32)
            if device is not None:
                weights_tensor = weights_tensor.to(device)

            class_weights[component] = weights_tensor

            logger.debug(
                f"Component '{component}': "
                f"counts range [{counts.min():.0f}, {counts.max():.0f}], "
                f"weights range [{weights.min():.3f}, {weights.max():.3f}]"
            )

        if return_counts:
            return class_weights, class_counts
        return class_weights
