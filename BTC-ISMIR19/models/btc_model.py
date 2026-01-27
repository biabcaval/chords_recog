from utils.transformer_modules import *
from utils.transformer_modules import _gen_timing_signal, _gen_bias_mask
from utils.hparams import HParams

use_cuda = torch.cuda.is_available()

class self_attention_block(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, attention_map=False):
        super(self_attention_block, self).__init__()

        self.attention_map = attention_map
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,hidden_size, num_heads, bias_mask, attention_dropout, attention_map)
        self.positionwise_convolution = PositionwiseFeedForward(hidden_size, filter_size, hidden_size, layer_config='cc', padding='both', dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs

        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        if self.attention_map is True:
            y, weights = self.multi_head_attention(x_norm, x_norm, x_norm)
        else:
            y = self.multi_head_attention(x_norm, x_norm, x_norm)

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_convolution(x_norm)

        # Dropout and residual
        y = self.dropout(x + y)

        if self.attention_map is True:
            return y, weights
        return y

class bi_directional_self_attention(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads, max_length,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):

        super(bi_directional_self_attention, self).__init__()

        self.weights_list = list()

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(max_length),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.attn_block = self_attention_block(*params)

        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  torch.transpose(_gen_bias_mask(max_length), dim0=2, dim1=3),
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  True)

        self.backward_attn_block = self_attention_block(*params)

        self.linear = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, inputs):
        x, list = inputs

        # Forward Self-attention Block
        encoder_outputs, weights = self.attn_block(x)
        # Backward Self-attention Block
        reverse_outputs, reverse_weights = self.backward_attn_block(x)
        # Concatenation and Fully-connected Layer
        outputs = torch.cat((encoder_outputs, reverse_outputs), dim=2)
        y = self.linear(outputs)

        # Attention weights for Visualization
        self.weights_list = list
        self.weights_list.append(weights)
        self.weights_list.append(reverse_weights)
        return y, self.weights_list

class bi_directional_self_attention_layers(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0):
        super(bi_directional_self_attention_layers, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  max_length,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)
        self.self_attn_layers = nn.Sequential(*[bi_directional_self_attention(*params) for l in range(num_layers)])
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)

    def forward(self, inputs):
        # Add input dropout
        x = self.input_dropout(inputs)

        # Project to hidden size
        x = self.embedding_proj(x)

        # Add timing signal
        x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)

        # A Stack of Bi-directional Self-attention Layers
        y, weights_list = self.self_attn_layers((x, []))

        # Layer Normalization
        y = self.layer_norm(y)
        return y, weights_list

class BTC_model(nn.Module):
    def __init__(self, config, class_weights=None):
        super(BTC_model, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config['probs_out']
        self.feature_type = config.get('feature_type', 'cqt')  # Default to 'cqt'

        if self.feature_type == 'hcqt':
            input_channels = config.get('n_harmonics')  # Number of harmonics in HCQT
        else:
            input_channels = 1  # Single channel for CQT

        # Add a convolutional layer for HCQT
        self.conv1 = nn.Conv2d(input_channels, config['hidden_size'], kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        params = (config['hidden_size'],
                  config['hidden_size'],
                  config['num_layers'],
                  config['num_heads'],
                  config['total_key_depth'],
                  config['total_value_depth'],
                  config['filter_size'],
                  config['timestep'],
                  config['input_dropout'],
                  config['layer_dropout'],
                  config['attention_dropout'],
                  config['relu_dropout'])

        self.self_attn_layers = bi_directional_self_attention_layers(*params)
        self.output_layer = SoftmaxOutputLayer(
            hidden_size=config['hidden_size'], 
            output_size=config['num_chords'], 
            probs_out=config['probs_out'],
            class_weights=class_weights
        )

    def forward(self, x, labels):
        if self.feature_type == 'hcqt':
            # HCQT input: (batch_size, n_harmonics, n_bins, time_frames)
            x = self.conv1(x)  # Apply convolution
            x = self.relu(x)
            x = self.pool(x)  # Downsample
            x = x.flatten(2).transpose(1, 2)  # Reshape to (batch_size, time_frames, feature_size)

        # Output of Bi-directional Self-attention Layers
        self_attn_output, weights_list = self.self_attn_layers(x)

        # return logit values for CRF
        if self.probs_out is True:
            logits = self.output_layer(self_attn_output)
            return logits

        # Output layer and Soft-max
        prediction, second = self.output_layer(self_attn_output)
        prediction = prediction.view(-1)
        second = second.view(-1)

        # Loss Calculation
        loss = self.output_layer.loss(self_attn_output, labels)
        return prediction, loss, weights_list, second


class BTC_model_structured(nn.Module):
    """
    BTC model with structured output (Root, Quality, Bass) following ChordFormer approach.
    """
    def __init__(self, config, class_weights=None, 
                 root_class_weights=None, quality_class_weights=None, bass_class_weights=None):
        super(BTC_model_structured, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config.get('probs_out', False)
        self.use_structured = config.get('use_structured_output', True)

        params = (config['feature_size'],
                  config['hidden_size'],
                  config['num_layers'],
                  config['num_heads'],
                  config['total_key_depth'],
                  config['total_value_depth'],
                  config['filter_size'],
                  config['timestep'],
                  config['input_dropout'],
                  config['layer_dropout'],
                  config['attention_dropout'],
                  config['relu_dropout'])

        self.self_attn_layers = bi_directional_self_attention_layers(*params)
        
        # Use structured output layer with per-class weights
        from utils.transformer_modules import StructuredOutputLayer
        self.output_layer = StructuredOutputLayer(
            hidden_size=config['hidden_size'],
            num_roots=config.get('num_roots', 13),
            num_qualities=config.get('num_qualities', 16),
            num_bass=config.get('num_bass', 13),
            probs_out=self.probs_out,
            root_weight=config.get('root_weight', 1.0),
            quality_weight=config.get('quality_weight', 1.0),
            bass_weight=config.get('bass_weight', 1.0),
            root_class_weights=root_class_weights,
            quality_class_weights=quality_class_weights,
            bass_class_weights=bass_class_weights
        )

    def forward(self, x, labels=None, root_labels=None, quality_labels=None, bass_labels=None):
        """
        Forward pass with structured output.
        
        Args:
            x: Input features [batch_size, seq_len, feature_size]
            labels: Legacy chord labels (optional, for compatibility)
            root_labels: Root labels [batch_size, seq_len]
            quality_labels: Quality labels [batch_size, seq_len]
            bass_labels: Bass labels [batch_size, seq_len]
        """
        # Output of Bi-directional Self-attention Layers
        self_attn_output, weights_list = self.self_attn_layers(x)

        # return logit values for CRF or other purposes
        if self.probs_out is True:
            root_logits, quality_logits, bass_logits = self.output_layer(self_attn_output)
            return root_logits, quality_logits, bass_logits

        # Output layer predictions
        root_pred, quality_pred, bass_pred, root_second, quality_second, bass_second = self.output_layer(self_attn_output)
        
        # Flatten predictions
        root_pred = root_pred.view(-1)
        quality_pred = quality_pred.view(-1)
        bass_pred = bass_pred.view(-1)
        
        root_second = root_second.view(-1)
        quality_second = quality_second.view(-1)
        bass_second = bass_second.view(-1)

        # Loss Calculation
        if root_labels is not None and quality_labels is not None and bass_labels is not None:
            root_labels = root_labels.view(-1, self.timestep)
            quality_labels = quality_labels.view(-1, self.timestep)
            bass_labels = bass_labels.view(-1, self.timestep)
            loss = self.output_layer.loss(self_attn_output, root_labels, quality_labels, bass_labels)
        else:
            loss = None
        
        return {
            'root_pred': root_pred,
            'quality_pred': quality_pred,
            'bass_pred': bass_pred,
            'root_second': root_second,
            'quality_second': quality_second,
            'bass_second': bass_second,
            'loss': loss,
            'weights_list': weights_list
        }


class ChordFormer_model(nn.Module):
    """
    ChordFormer: Conformer-based architecture for large-vocabulary chord recognition.
    
    This model combines:
    - Conformer encoder (CNN + Transformer hybrid) for capturing local patterns and global dependencies
    - Structured output (Root, Quality, Bass) for handling large vocabularies
    - Per-class reweighted loss for addressing class imbalance
    
    Based on: "ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition"
    """
    def __init__(self, config, class_weights=None,
                 root_class_weights=None, quality_class_weights=None, bass_class_weights=None):
        super(ChordFormer_model, self).__init__()

        self.timestep = config['timestep']
        self.probs_out = config.get('probs_out', False)
        
        # Import Conformer encoder
        from utils.transformer_modules import ConformerEncoder, StructuredOutputLayer
        
        # Conformer Encoder
        self.conformer_encoder = ConformerEncoder(
            embedding_size=config['feature_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            num_heads=config['num_heads'],
            conv_kernel_size=config.get('conv_kernel_size', 31),
            ff_expansion_factor=config.get('ff_expansion_factor', 4),
            conv_expansion_factor=config.get('conv_expansion_factor', 2),
            max_length=config['timestep'],
            input_dropout=config['input_dropout'],
            layer_dropout=config['layer_dropout'],
            attention_map=True  # Always collect attention weights for visualization
        )
        
        # Structured output layer with per-class weights
        self.output_layer = StructuredOutputLayer(
            hidden_size=config['hidden_size'],
            num_roots=config.get('num_roots', 13),
            num_qualities=config.get('num_qualities', 16),
            num_bass=config.get('num_bass', 13),
            probs_out=self.probs_out,
            root_weight=config.get('root_weight', 1.0),
            quality_weight=config.get('quality_weight', 1.0),
            bass_weight=config.get('bass_weight', 1.0),
            root_class_weights=root_class_weights,
            quality_class_weights=quality_class_weights,
            bass_class_weights=bass_class_weights
        )

    def forward(self, x, labels=None, root_labels=None, quality_labels=None, bass_labels=None):
        """
        Forward pass with Conformer encoder and structured output.
        
        Args:
            x: Input features [batch_size, seq_len, feature_size]
            labels: Legacy chord labels (optional, for compatibility)
            root_labels: Root labels [batch_size, seq_len]
            quality_labels: Quality labels [batch_size, seq_len]
            bass_labels: Bass labels [batch_size, seq_len]
            
        Returns:
            dict: Contains predictions, losses, and attention weights
        """
        # Conformer Encoder
        encoder_output, weights_list = self.conformer_encoder(x)

        # Return logit values for CRF or other purposes
        if self.probs_out is True:
            root_logits, quality_logits, bass_logits = self.output_layer(encoder_output)
            return root_logits, quality_logits, bass_logits

        # Output layer predictions
        root_pred, quality_pred, bass_pred, root_second, quality_second, bass_second = self.output_layer(encoder_output)
        
        # Flatten predictions
        root_pred = root_pred.view(-1)
        quality_pred = quality_pred.view(-1)
        bass_pred = bass_pred.view(-1)
        
        root_second = root_second.view(-1)
        quality_second = quality_second.view(-1)
        bass_second = bass_second.view(-1)

        # Loss Calculation
        if root_labels is not None and quality_labels is not None and bass_labels is not None:
            root_labels = root_labels.view(-1, self.timestep)
            quality_labels = quality_labels.view(-1, self.timestep)
            bass_labels = bass_labels.view(-1, self.timestep)
            loss = self.output_layer.loss(encoder_output, root_labels, quality_labels, bass_labels)
        else:
            loss = None
        
        return {
            'root_pred': root_pred,
            'quality_pred': quality_pred,
            'bass_pred': bass_pred,
            'root_second': root_second,
            'quality_second': quality_second,
            'bass_second': bass_second,
            'loss': loss,
            'weights_list': weights_list
        }


if __name__ == "__main__":
    config = HParams.load("run_config.yaml")
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = 2
    timestep = 108
    feature_size = 144
    num_chords = 25

    features = torch.randn(batch_size,timestep,feature_size,requires_grad=True).to(device)
    chords = torch.randint(25,(batch_size*timestep,)).to(device)

    model = BTC_model(config=config.model).to(device)

    prediction, loss, weights_list, second = model(features, chords)
    print(prediction.size())
    print(loss)


