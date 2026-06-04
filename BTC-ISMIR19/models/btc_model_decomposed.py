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
    
    Supports two modes:
      - Simple: Dropout -> Linear(hidden_size, vocab_size)
      - FFN bottleneck: Linear(hidden_size, ffn_dim) -> ReLU -> Dropout
                        -> Linear(ffn_dim, ffn_dim//2) -> ReLU
                        -> Linear(ffn_dim//2, vocab_size)
    
    Args:
        hidden_size: Size of the hidden representation from feature extractor
        vocab_size: Number of classes for this component
        dropout: Dropout probability
        use_ffn: If True, use FFN bottleneck instead of direct projection
        ffn_dim: Hidden dimension of the first FFN layer (default: hidden_size // 2)
    """
    
    def __init__(self, hidden_size: int, vocab_size: int, dropout: float = 0.0,
                 use_ffn: bool = False, ffn_dim: int = None):
        super().__init__()
        self.use_ffn = use_ffn
        
        if use_ffn:
            if ffn_dim is None:
                ffn_dim = hidden_size // 2
            bottleneck_dim = ffn_dim // 2
            self.ffn = nn.Sequential(
                nn.Linear(hidden_size, ffn_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(ffn_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, vocab_size),
            )
        else:
            self.linear = nn.Linear(hidden_size, vocab_size)
            self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            
        Returns:
            logits: (batch_size, seq_len, vocab_size)
        """
        if self.use_ffn:
            return self.ffn(x)
        x = self.dropout(x)
        return self.linear(x)


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
    
    def __init__(self, hidden_size: int, dropout: float = 0.0,
                 use_ffn: bool = False, ffn_dim: int = None):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Get vocabulary sizes for each component
        self.vocab_sizes = {component: len(CHORD_VOCAB[component])
                          for component in COMPONENT_NAMES}
        
        # Create output head for each component
        self.heads = nn.ModuleDict({
            component: ComponentHead(hidden_size, self.vocab_sizes[component], dropout,
                                     use_ffn=use_ffn, ffn_dim=ffn_dim)
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
        self.last_shared_features = None
        
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
            component_weights=component_weights,
            gradnorm_enabled=cfg.get('gradnorm_enabled', False),
            gradnorm_alpha=cfg.get('gradnorm_alpha', 1.5),
            gradnorm_lr=cfg.get('gradnorm_lr', 0.025),
            gradnorm_eps=cfg.get('gradnorm_eps', 1e-8),
            gradnorm_w_min=cfg.get('gradnorm_w_min', 1e-3),
            gradnorm_w_max=cfg.get('gradnorm_w_max', 10.0),
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
        self.last_shared_features = self_attn_output
        
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
        self.last_shared_features = None

        # Conformer encoder (ChordFormer backbone)
        # Two new flags allow toggling between ChordMax-style (sinusoidal pos.
        # encoding ON, BatchNorm in conv) and ChordFormer-style (no pos. encoding,
        # BatchNorm still ON per Tabela 2). Both default to the current ChordMax
        # behavior so existing runs are unaffected.
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
            use_positional_encoding=cfg.get('use_positional_encoding', True),
            use_batchnorm_in_conv=cfg.get('use_batchnorm_in_conv', True),
        )

        # Multi-head chord decomposition (with optional FFN bottleneck)
        use_head_ffn = cfg.get('use_head_ffn', False)
        head_ffn_dim = cfg.get('head_ffn_dim', None)
        self.decomposer = MultiHeadChordDecomposer(
            hidden_size=cfg['hidden_size'],
            dropout=cfg.get('output_dropout', 0.0),
            use_ffn=use_head_ffn,
            ffn_dim=head_ffn_dim,
        )

        # Loss function
        self.criterion = MultiTaskLoss(
            vocab_sizes=self.decomposer.vocab_sizes,
            class_weights=class_weights,
            gamma=cfg.get('class_weight_gamma', 0.5),
            w_max=cfg.get('class_weight_max', 10.0),
            component_weights=component_weights,
            gradnorm_enabled=cfg.get('gradnorm_enabled', False),
            gradnorm_alpha=cfg.get('gradnorm_alpha', 1.5),
            gradnorm_lr=cfg.get('gradnorm_lr', 0.025),
            gradnorm_eps=cfg.get('gradnorm_eps', 1e-8),
            gradnorm_w_min=cfg.get('gradnorm_w_min', 1e-3),
            gradnorm_w_max=cfg.get('gradnorm_w_max', 10.0),
            focal_gamma=cfg.get('focal_gamma', 0.0),
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
        self.last_shared_features = encoder_output

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
    
    def __init__(
        self,
        vocab_sizes,
        class_weights=None,
        gamma=0.5,
        w_max=10.0,
        component_weights=None,
        gradnorm_enabled=False,
        gradnorm_alpha=1.5,
        gradnorm_lr=0.025,
        gradnorm_eps=1e-8,
        gradnorm_w_min=1e-3,
        gradnorm_w_max=10.0,
        focal_gamma=0.0,
    ):
        super().__init__()
        
        self.vocab_sizes = vocab_sizes
        self.gamma = gamma
        self.w_max = w_max
        self.focal_gamma = float(focal_gamma)
        self.component_names = list(vocab_sizes.keys())
        
        # Initialize loss functions for each component.
        # When focal loss is active, class weights are applied as alpha
        # (gathered per-sample) instead of being baked into CrossEntropyLoss,
        # because weight inside F.cross_entropy would distort pt = exp(-ce).
        self.losses = nn.ModuleDict()
        
        for component in self.component_names:
            weight = None
            if class_weights is not None and component in class_weights:
                weight = class_weights[component]
            
            if self.focal_gamma > 0.0:
                self.losses[component] = nn.CrossEntropyLoss(reduction='none')
                if weight is not None:
                    self.register_buffer(f'focal_alpha_{component}', weight)
            else:
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
        self.last_raw_loss_tensors = {}

        # GradNorm configuration/state.
        self.gradnorm_enabled = bool(gradnorm_enabled)
        self.gradnorm_alpha = float(gradnorm_alpha)
        self.gradnorm_lr = float(gradnorm_lr)
        self.gradnorm_eps = float(gradnorm_eps)
        self.gradnorm_w_min = float(gradnorm_w_min)
        self.gradnorm_w_max = float(gradnorm_w_max)
        self.gradnorm_initial_losses = None
        self.last_gradnorm_info = {}

        init_weights = torch.tensor(
            [float(self.component_weights.get(c, 1.0)) for c in self.component_names],
            dtype=torch.float32,
        )
        if self.gradnorm_enabled:
            self.gradnorm_weights = nn.Parameter(init_weights.clone())
        else:
            self.register_parameter('gradnorm_weights', None)

    def _get_weight_tensor(self, device):
        if self.gradnorm_enabled and self.gradnorm_weights is not None:
            return self.gradnorm_weights.to(device)
        return torch.tensor(
            [float(self.component_weights.get(c, 1.0)) for c in self.component_names],
            dtype=torch.float32,
            device=device,
        )

    def get_weight_dict(self):
        if self.gradnorm_enabled and self.gradnorm_weights is not None:
            with torch.no_grad():
                return {
                    c: float(v)
                    for c, v in zip(self.component_names, self.gradnorm_weights.detach().cpu().tolist())
                }
        return {c: float(self.component_weights.get(c, 1.0)) for c in self.component_names}
    
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
        self.last_raw_loss_tensors = {}
        weight_tensor = None
        
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
            if self.focal_gamma > 0.0:
                ce_per_sample = self.losses[component](logits_flat, labels_flat)
                pt = torch.exp(-ce_per_sample)
                focal_term = (1 - pt) ** self.focal_gamma
                alpha_buf = getattr(self, f'focal_alpha_{component}', None)
                if alpha_buf is not None:
                    alpha_t = alpha_buf.to(labels_flat.device).gather(0, labels_flat)
                    component_loss = (alpha_t * focal_term * ce_per_sample).mean()
                else:
                    component_loss = (focal_term * ce_per_sample).mean()
            else:
                component_loss = self.losses[component](logits_flat, labels_flat)
            self.last_raw_loss_tensors[component] = component_loss
            
            # Apply component weight
            if weight_tensor is None:
                weight_tensor = self._get_weight_tensor(component_loss.device)
            weight_idx = self.component_names.index(component)
            weight = weight_tensor[weight_idx]
            # GradNorm updates weights separately; model update should not backprop into w_i.
            weighted_loss = weight.detach() * component_loss if self.gradnorm_enabled else weight * component_loss
            self.last_forward_breakdown[component] = {
                'raw_loss': float(component_loss.detach().cpu().item()),
                'weight': float(weight.detach().cpu().item()),
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

        All counting and weight arithmetic runs on `device` (GPU when available),
        using `torch.bincount` and vectorized torch ops. Only the final
        `class_counts` dict is materialized as float64 numpy arrays on CPU to
        preserve the existing serialization API in callers.

        Args:
            train_dataset: Training dataset with 'components' field. Each sample should
                          have a 'components' dict mapping component names to index arrays.
                          Dataset should support __len__ and __getitem__.
            gamma: Weighting exponent (lower = more smoothing). Default: 0.5
            w_max: Maximum weight cap to prevent extreme weights. Default: 10.0
            device: Device to run computation and place output tensors on. Defaults
                to CUDA when available, otherwise CPU.
            return_counts: If True, also return per-component class counts.

        Returns:
            class_weights: Dict mapping component names to weight tensors on `device`.
            class_counts (optional): Dict mapping component names to numpy float64
                                     arrays of counts on CPU.
        """
        import logging
        logger = logging.getLogger(__name__)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif not isinstance(device, torch.device):
            device = torch.device(device)

        # All counting accumulators live on `device` for GPU-resident bincount.
        # int64 to allow safe accumulation across very large datasets.
        counts_by_component = {
            component: torch.zeros(
                len(CHORD_VOCAB[component]),
                dtype=torch.long,
                device=device,
            )
            for component in COMPONENT_NAMES
        }
        vocab_sizes = {component: len(CHORD_VOCAB[component]) for component in COMPONENT_NAMES}

        # Single dataset pass for all components using GPU bincount.
        n_samples = len(train_dataset)
        for i in range(n_samples):
            sample = train_dataset[i]
            if 'components' not in sample:
                continue

            sample_components = sample['components']
            for component in COMPONENT_NAMES:
                vocab_size = vocab_sizes[component]
                component_data = sample_components.get(component, None)
                if component_data is None:
                    continue

                if isinstance(component_data, torch.Tensor):
                    indices = component_data.detach().to(
                        device=device, dtype=torch.long, non_blocking=True
                    ).reshape(-1)
                else:
                    indices = torch.as_tensor(
                        component_data, dtype=torch.long, device=device
                    ).reshape(-1)

                if indices.numel() == 0:
                    continue

                # Filter out-of-range indices on-device before bincount.
                valid_mask = (indices >= 0) & (indices < vocab_size)
                if not bool(valid_mask.any()):
                    continue
                valid = indices[valid_mask]

                counts_by_component[component] += torch.bincount(valid, minlength=vocab_size)

        class_weights = {}
        class_counts = {}
        for component in COMPONENT_NAMES:
            counts = counts_by_component[component]
            vocab_size = vocab_sizes[component]
            max_count_t = counts.max()
            max_count = float(max_count_t.item())

            if max_count == 0:
                logger.warning(f"No data found for component '{component}'. Using uniform weights.")
                weights_tensor = torch.ones(vocab_size, dtype=torch.float32, device=device)
            else:
                # Start with max cap for all classes (covers unseen classes with count==0).
                weights_tensor = torch.full(
                    (vocab_size,), float(w_max), dtype=torch.float32, device=device
                )
                nonzero = counts > 0
                ratios = counts[nonzero].to(torch.float32) / float(max_count)
                weights_nonzero = torch.clamp(torch.pow(ratios, -float(gamma)), max=float(w_max))
                weights_tensor[nonzero] = weights_nonzero

            class_weights[component] = weights_tensor

            # Preserve API: class_counts items as numpy float64 on CPU.
            class_counts[component] = counts.detach().cpu().numpy().astype(np.float64)

            logger.debug(
                f"Component '{component}': "
                f"counts range [{int(counts.min().item())}, {int(counts.max().item())}], "
                f"weights range [{weights_tensor.min().item():.3f}, "
                f"{weights_tensor.max().item():.3f}]"
            )

        if return_counts:
            return class_weights, class_counts
        return class_weights
