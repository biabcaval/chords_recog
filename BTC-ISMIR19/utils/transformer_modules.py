from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


def compute_class_weights(train_dataset, num_classes=170, gamma=0.5, w_max=10.0, device=None):
    """
    Compute class weights inversely proportional to class frequency.
    
    Args:
        train_dataset: Dataset to compute weights from (must have __getitem__ returning dict with 'chord' key)
        num_classes: Number of chord classes
        gamma: Exponent for weight calculation (lower = more smoothing)
        w_max: Maximum weight cap to prevent extreme weights
        device: Device to put weights on (defaults to cuda if available)
    
    Returns:
        torch.Tensor: Class weights of shape (num_classes,)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class_counts = torch.zeros(num_classes)
    
    for i in range(len(train_dataset)):
        data = train_dataset[i]
        chords = data['chord']
        if isinstance(chords, np.ndarray):
            chords = torch.from_numpy(chords)
        elif not isinstance(chords, torch.Tensor):
            chords = torch.tensor(chords)
        
        # Count occurrences of each chord class
        for c in chords.view(-1):
            if 0 <= c < num_classes:
                class_counts[c] += 1
    
    # Avoid division by zero
    class_counts = torch.clamp(class_counts, min=1)
    
    # Weight inversely proportional to frequency: (N_max / N_class)^gamma
    N_max = class_counts.max()
    weights = torch.clamp((N_max / class_counts) ** gamma, max=w_max)
    
    return weights.to(device)


def compute_structured_class_weights(train_dataset, num_roots=13, num_qualities=16, num_bass=13,
                                      gamma=0.5, w_max=10.0, device=None):
    """
    Compute per-class weights for structured output (Root, Quality, Bass).
    
    This function calculates class weights inversely proportional to frequency
    for each component separately, following the ChordFormer reweighted loss approach.
    
    Args:
        train_dataset: Dataset with 'root', 'quality', 'bass' keys
        num_roots: Number of root classes (12 pitches + no chord = 13)
        num_qualities: Number of quality classes (14 qualities + no chord + unknown = 16)
        num_bass: Number of bass classes (12 pitches + no bass = 13)
        gamma: Exponent for weight calculation (lower = more smoothing)
        w_max: Maximum weight cap to prevent extreme weights
        device: Device to put weights on
    
    Returns:
        tuple: (root_weights, quality_weights, bass_weights) - each tensor of respective shape
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    root_counts = torch.zeros(num_roots)
    quality_counts = torch.zeros(num_qualities)
    bass_counts = torch.zeros(num_bass)
    
    for i in range(len(train_dataset)):
        data = train_dataset[i]
        
        # Get structured labels if available
        if 'root' in data and 'quality' in data and 'bass' in data:
            roots = data['root']
            qualities = data['quality']
            basses = data['bass']
            
            # Convert to tensors if needed
            if isinstance(roots, np.ndarray):
                roots = torch.from_numpy(roots)
            elif not isinstance(roots, torch.Tensor):
                roots = torch.tensor(roots)
                
            if isinstance(qualities, np.ndarray):
                qualities = torch.from_numpy(qualities)
            elif not isinstance(qualities, torch.Tensor):
                qualities = torch.tensor(qualities)
                
            if isinstance(basses, np.ndarray):
                basses = torch.from_numpy(basses)
            elif not isinstance(basses, torch.Tensor):
                basses = torch.tensor(basses)
            
            # Count occurrences
            for r in roots.view(-1):
                if 0 <= r < num_roots:
                    root_counts[r] += 1
            for q in qualities.view(-1):
                if 0 <= q < num_qualities:
                    quality_counts[q] += 1
            for b in basses.view(-1):
                if 0 <= b < num_bass:
                    bass_counts[b] += 1
    
    # Check if we found structured data
    if root_counts.sum() == 0:
        return None, None, None
    
    # Avoid division by zero
    root_counts = torch.clamp(root_counts, min=1)
    quality_counts = torch.clamp(quality_counts, min=1)
    bass_counts = torch.clamp(bass_counts, min=1)
    
    # Weight inversely proportional to frequency: (N_max / N_class)^gamma
    root_weights = torch.clamp((root_counts.max() / root_counts) ** gamma, max=w_max)
    quality_weights = torch.clamp((quality_counts.max() / quality_counts) ** gamma, max=w_max)
    bass_weights = torch.clamp((bass_counts.max() / bass_counts) ** gamma, max=w_max)
    
    return root_weights.to(device), quality_weights.to(device), bass_weights.to(device)

def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)
    return torch_mask.unsqueeze(0).unsqueeze(1)

def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            (float(num_timescales) - 1))
    inv_timescales = min_timescale * np.exp(
        np.arange(num_timescales).astype(np.float64) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)

class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class OutputLayer(nn.Module):
    """
    Abstract base class for output layer.
    Handles projection to output labels
    """
    def __init__(self, hidden_size, output_size, probs_out=False, class_weights=None):
        super(OutputLayer, self).__init__()
        self.output_size = output_size
        self.output_projection = nn.Linear(hidden_size, output_size)
        self.probs_out = probs_out
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=int(hidden_size/2), batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size
        # Register class weights as a buffer (not a parameter, but moves with model)
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights)
        else:
            self.class_weights = None

    def loss(self, hidden, labels):
        raise NotImplementedError('Must implement {}.loss'.format(self.__class__.__name__))


class SoftmaxOutputLayer(OutputLayer):
    """
    Implements a softmax based output layer with optional class reweighting.
    """
    def __init__(self, hidden_size, output_size, probs_out=False, class_weights=None):
        super(SoftmaxOutputLayer, self).__init__(hidden_size, output_size, probs_out, class_weights)

    def forward(self, hidden):
        logits = self.output_projection(hidden)
        probs = F.softmax(logits, -1)
        # _, predictions = torch.max(probs, dim=-1)
        topk, indices = torch.topk(probs, 2)
        predictions = indices[:,:,0]
        second = indices[:,:,1]
        if self.probs_out is True:
            return logits
            # return probs
        return predictions, second

    def loss(self, hidden, labels):
        logits = self.output_projection(hidden)
        log_probs = F.log_softmax(logits, -1)
        # Ensure labels are on the same device as log_probs
        labels = labels.to(log_probs.device)
        if self.class_weights is not None:
            return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1), weight=self.class_weights.to(log_probs.device))
        else:
            return F.nll_loss(log_probs.view(-1, self.output_size), labels.view(-1))


class StructuredOutputLayer(nn.Module):
    """
    Implements structured output layer for chord recognition with separate heads for Root, Quality, and Bass.
    This follows the ChordFormer approach of layered classification with per-class reweighting.
    """
    def __init__(self, hidden_size, num_roots=13, num_qualities=16, num_bass=13, 
                 probs_out=False, root_weight=1.0, quality_weight=1.0, bass_weight=1.0,
                 root_class_weights=None, quality_class_weights=None, bass_class_weights=None):
        """
        Args:
            hidden_size: Size of hidden representation
            num_roots: Number of root classes (12 pitches + no chord = 13)
            num_qualities: Number of quality classes (14 qualities + no chord + unknown = 16)
            num_bass: Number of bass classes (12 pitches + no bass = 13)
            probs_out: If True, return logits instead of predictions
            root_weight: Global weight for root loss component
            quality_weight: Global weight for quality loss component
            bass_weight: Global weight for bass loss component
            root_class_weights: Per-class weights for root classes (tensor of shape [num_roots])
            quality_class_weights: Per-class weights for quality classes (tensor of shape [num_qualities])
            bass_class_weights: Per-class weights for bass classes (tensor of shape [num_bass])
        """
        super(StructuredOutputLayer, self).__init__()
        
        self.num_roots = num_roots
        self.num_qualities = num_qualities
        self.num_bass = num_bass
        self.probs_out = probs_out
        
        # Global loss weights for each component
        self.root_weight = root_weight
        self.quality_weight = quality_weight
        self.bass_weight = bass_weight
        
        # Register per-class weights as buffers (moves with model to GPU)
        if root_class_weights is not None:
            self.register_buffer('root_class_weights', root_class_weights)
        else:
            self.root_class_weights = None
            
        if quality_class_weights is not None:
            self.register_buffer('quality_class_weights', quality_class_weights)
        else:
            self.quality_class_weights = None
            
        if bass_class_weights is not None:
            self.register_buffer('bass_class_weights', bass_class_weights)
        else:
            self.bass_class_weights = None
        
        # Separate projection heads for each component
        self.root_projection = nn.Linear(hidden_size, num_roots)
        self.quality_projection = nn.Linear(hidden_size, num_qualities)
        self.bass_projection = nn.Linear(hidden_size, num_bass)
        
        # Optional LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=int(hidden_size/2), 
                           batch_first=True, bidirectional=True)
        self.hidden_size = hidden_size
    
    def forward(self, hidden):
        """
        Forward pass through structured output layer.
        
        Args:
            hidden: Hidden representation [batch_size, seq_len, hidden_size]
            
        Returns:
            If probs_out=True: (root_logits, quality_logits, bass_logits)
            If probs_out=False: (root_pred, quality_pred, bass_pred, root_second, quality_second, bass_second)
        """
        # Get logits for each component
        root_logits = self.root_projection(hidden)
        quality_logits = self.quality_projection(hidden)
        bass_logits = self.bass_projection(hidden)
        
        if self.probs_out:
            return root_logits, quality_logits, bass_logits
        
        # Get predictions and second-best predictions
        root_probs = F.softmax(root_logits, -1)
        quality_probs = F.softmax(quality_logits, -1)
        bass_probs = F.softmax(bass_logits, -1)
        
        # Get top-2 predictions for each component
        root_topk, root_indices = torch.topk(root_probs, 2)
        quality_topk, quality_indices = torch.topk(quality_probs, 2)
        bass_topk, bass_indices = torch.topk(bass_probs, 2)
        
        root_pred = root_indices[:, :, 0]
        root_second = root_indices[:, :, 1]
        
        quality_pred = quality_indices[:, :, 0]
        quality_second = quality_indices[:, :, 1]
        
        bass_pred = bass_indices[:, :, 0]
        bass_second = bass_indices[:, :, 1]
        
        return root_pred, quality_pred, bass_pred, root_second, quality_second, bass_second
    
    def loss(self, hidden, root_labels, quality_labels, bass_labels):
        """
        Calculate weighted sum of losses for all three components with per-class reweighting.
        
        Args:
            hidden: Hidden representation [batch_size, seq_len, hidden_size]
            root_labels: Ground truth root labels [batch_size, seq_len]
            quality_labels: Ground truth quality labels [batch_size, seq_len]
            bass_labels: Ground truth bass labels [batch_size, seq_len]
            
        Returns:
            total_loss: Weighted sum of component losses with per-class weights
        """
        # Get logits for each component
        root_logits = self.root_projection(hidden)
        quality_logits = self.quality_projection(hidden)
        bass_logits = self.bass_projection(hidden)
        
        # Calculate log probabilities
        root_log_probs = F.log_softmax(root_logits, -1)
        quality_log_probs = F.log_softmax(quality_logits, -1)
        bass_log_probs = F.log_softmax(bass_logits, -1)
        
        # Calculate individual losses with per-class weights
        root_loss = F.nll_loss(
            root_log_probs.view(-1, self.num_roots), 
            root_labels.view(-1),
            weight=self.root_class_weights
        )
        quality_loss = F.nll_loss(
            quality_log_probs.view(-1, self.num_qualities), 
            quality_labels.view(-1),
            weight=self.quality_class_weights
        )
        bass_loss = F.nll_loss(
            bass_log_probs.view(-1, self.num_bass), 
            bass_labels.view(-1),
            weight=self.bass_class_weights
        )
        
        # Weighted sum of losses (global component weights)
        total_loss = (self.root_weight * root_loss + 
                     self.quality_weight * quality_loss + 
                     self.bass_weight * bass_loss)
        
        return total_loss

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0, attention_map=False):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))

        self.attention_map = attention_map

        self.num_heads = num_heads
        self.query_scale = (total_key_depth // num_heads) ** -0.5
        self.bias_mask = bias_mask

        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)

        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2] // self.num_heads).permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3] * self.num_heads)

    def forward(self, queries, keys, values):

        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)

        # Scale queries
        queries *= self.query_scale

        # Combine queries and keys
        logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))

        # Add bias to mask future values
        if self.bias_mask is not None:
            logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # Combine with values to get context
        contexts = torch.matmul(weights, values)

        # Merge heads
        contexts = self._merge_heads(contexts)
        # contexts = torch.tanh(contexts)

        # Linear to get output
        outputs = self.output_linear(contexts)

        if self.attention_map is True:
            return outputs, weights

        return outputs


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data_loader),
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (kernel_size - 1, 0) if pad_type == 'left' else (kernel_size // 2, (kernel_size - 1) // 2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size, kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data_loader),
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)] * (len(layer_config) - 2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


# ============================================================================
# Conformer Components (ChordFormer Architecture)
# ============================================================================

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self, dim):
        super(GLU, self).__init__()
        self.dim = dim

    def forward(self, x):
        outputs, gate = x.chunk(2, dim=self.dim)
        return outputs * torch.sigmoid(gate)


class DepthwiseConv1d(nn.Module):
    """Depthwise 1D Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, groups=in_channels, bias=bias
        )

    def forward(self, x):
        return self.conv(x)


class PointwiseConv1d(nn.Module):
    """Pointwise 1D Convolution (1x1 conv)"""
    def __init__(self, in_channels, out_channels, stride=1, padding=0, bias=True):
        super(PointwiseConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=padding, bias=bias)

    def forward(self, x):
        return self.conv(x)


class ConformerConvModule(nn.Module):
    """
    Conformer Convolution Module.
    
    Structure:
        LayerNorm -> Pointwise Conv -> GLU -> Depthwise Conv -> BatchNorm -> Swish -> Pointwise Conv -> Dropout
    """
    def __init__(self, hidden_size, kernel_size=31, expansion_factor=2, dropout=0.1):
        super(ConformerConvModule, self).__init__()
        
        self.layer_norm = LayerNorm(hidden_size)
        self.pointwise_conv1 = PointwiseConv1d(hidden_size, hidden_size * expansion_factor, bias=True)
        self.glu = GLU(dim=1)
        
        # Depthwise conv with same padding
        padding = (kernel_size - 1) // 2
        self.depthwise_conv = DepthwiseConv1d(hidden_size, hidden_size, kernel_size, padding=padding, bias=False)
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.swish = Swish()
        self.pointwise_conv2 = PointwiseConv1d(hidden_size, hidden_size, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # LayerNorm
        x = self.layer_norm(x)
        
        # Transpose for convolutions: [batch, seq, hidden] -> [batch, hidden, seq]
        x = x.transpose(1, 2)
        
        # First pointwise conv + GLU
        x = self.pointwise_conv1(x)
        x = self.glu(x)
        
        # Depthwise conv
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        
        # Second pointwise conv
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        
        # Transpose back: [batch, hidden, seq] -> [batch, seq, hidden]
        x = x.transpose(1, 2)
        
        return x


class ConformerFeedForward(nn.Module):
    """
    Conformer Feed Forward Module with Swish activation.
    
    Structure:
        LayerNorm -> Linear -> Swish -> Dropout -> Linear -> Dropout
    """
    def __init__(self, hidden_size, expansion_factor=4, dropout=0.1):
        super(ConformerFeedForward, self).__init__()
        
        inner_size = hidden_size * expansion_factor
        
        self.layer_norm = LayerNorm(hidden_size)
        self.linear1 = nn.Linear(hidden_size, inner_size)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(inner_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return x


class ConformerMultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention for Conformer with relative positional encoding.
    """
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(ConformerMultiHeadAttention, self).__init__()
        
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.layer_norm = LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_weights=False):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            return_weights: If True, also return attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Layer normalization
        x_norm = self.layer_norm(x)
        
        # Project to Q, K, V
        q = self.query(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x_norm).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        if return_weights:
            return output, attn_weights
        return output


class ConformerBlock(nn.Module):
    """
    Single Conformer Block.
    
    Structure (Macaron-style):
        x + 0.5 * FFN(x) -> x + MHSA(x) -> x + Conv(x) -> x + 0.5 * FFN(x) -> LayerNorm
    """
    def __init__(self, hidden_size, num_heads, conv_kernel_size=31, 
                 ff_expansion_factor=4, conv_expansion_factor=2, dropout=0.1, attention_map=False):
        super(ConformerBlock, self).__init__()
        
        self.attention_map = attention_map
        
        # First half feed-forward module
        self.ff1 = ConformerFeedForward(hidden_size, ff_expansion_factor, dropout)
        
        # Multi-head self-attention module
        self.self_attn = ConformerMultiHeadAttention(hidden_size, num_heads, dropout)
        
        # Convolution module
        self.conv_module = ConformerConvModule(hidden_size, conv_kernel_size, conv_expansion_factor, dropout)
        
        # Second half feed-forward module
        self.ff2 = ConformerFeedForward(hidden_size, ff_expansion_factor, dropout)
        
        # Final layer normalization
        self.layer_norm = LayerNorm(hidden_size)

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
            (Optional) Attention weights if attention_map=True
        """
        # First feed-forward (half-step)
        x = x + 0.5 * self.ff1(x)
        
        # Multi-head self-attention
        if self.attention_map:
            attn_output, weights = self.self_attn(x, return_weights=True)
        else:
            attn_output = self.self_attn(x)
        x = x + attn_output
        
        # Convolution module
        x = x + self.conv_module(x)
        
        # Second feed-forward (half-step)
        x = x + 0.5 * self.ff2(x)
        
        # Final layer normalization
        x = self.layer_norm(x)
        
        if self.attention_map:
            return x, weights
        return x


class ConformerEncoder(nn.Module):
    """
    Conformer Encoder: Stack of Conformer blocks with input projection and positional encoding.
    """
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, 
                 conv_kernel_size=31, ff_expansion_factor=4, conv_expansion_factor=2,
                 max_length=1000, input_dropout=0.1, layer_dropout=0.1, attention_map=False):
        super(ConformerEncoder, self).__init__()
        
        self.attention_map = attention_map
        
        # Input projection
        self.input_projection = nn.Linear(embedding_size, hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        
        # Positional encoding
        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        
        # Stack of Conformer blocks
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                ff_expansion_factor=ff_expansion_factor,
                conv_expansion_factor=conv_expansion_factor,
                dropout=layer_dropout,
                attention_map=attention_map
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch_size, seq_len, embedding_size]
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
            (Optional) List of attention weights from each layer
        """
        weights_list = []
        
        # Input dropout
        x = self.input_dropout(x)
        
        # Project to hidden size
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.timing_signal[:, :x.shape[1], :].type_as(x)
        
        # Pass through Conformer blocks
        for block in self.conformer_blocks:
            if self.attention_map:
                x, weights = block(x)
                weights_list.append(weights)
            else:
                x = block(x)
        
        return x, weights_list