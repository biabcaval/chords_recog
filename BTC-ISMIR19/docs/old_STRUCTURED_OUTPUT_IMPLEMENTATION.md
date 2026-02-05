# ChordFormer Structured Output Implementation

This document describes the implementation of structured output (Root, Quality, Bass) in the BTC chord recognition model, inspired by the ChordFormer approach.

## Overview

The structured output approach decomposes chord prediction into three independent classification tasks:
1. **Root**: Predicting the root note (0-11 for C through B, 12 for "no chord")
2. **Quality**: Predicting the chord quality (0-13 for different qualities, 14 for "no chord", 15 for unknown)
3. **Bass**: Predicting the bass note relative to root (0-11 for intervals, 12 for "no bass")

## Changes Made

### 1. Data Preprocessing (`utils/chords.py`)

**Added:**
- `quality_to_id()`: Maps quality strings (maj, min, dim, etc.) to integer IDs (0-15)
- Updated `get_converted_chord_voca()`: Now returns structured components (root, quality, bass) in addition to the original chord_id

**Quality Mapping:**
```python
'min': 0, 'maj': 1, 'dim': 2, 'aug': 3, 'min6': 4, 'maj6': 5,
'min7': 6, 'minmaj7': 7, 'maj7': 8, '7': 9, 'dim7': 10,
'hdim7': 11, 'sus2': 12, 'sus4': 13, 'N': 14 (no chord)
```

### 2. Feature Generation (`utils/preprocess.py`)

**Updated `generate_labels_features_voca()`:**
- Creates separate lists for root, quality, and bass alongside chord_list
- Applies pitch shifting correctly:
  - Root and bass are shifted modulo 12
  - Quality remains unchanged
- Saves all four components in the .pt files:
  ```python
  result = {
      'feature': feature,
      'chord': chord_list,      # Original flat representation
      'root': root_list,        # NEW
      'quality': quality_list,  # NEW
      'bass': bass_list,        # NEW
      'etc': etc
  }
  ```

### 3. Dataset Loading (`data/audio_dataset.py`)

**Updated:**
- `__getitem__()`: Loads structured targets if available in data files
- `_collate_fn()`: Batches structured targets and returns 9 values instead of 6:
  - Original: (features, input_percentages, chords, collapsed_chords, chord_lens, boundaries)
  - New: adds (roots, qualities, basses) when available

### 4. Model Architecture (`utils/transformer_modules.py`)

**Added `StructuredOutputLayer` class:**
- Three separate linear projection heads:
  - `root_projection`: hidden_size → num_roots (13)
  - `quality_projection`: hidden_size → num_qualities (16)
  - `bass_projection`: hidden_size → num_bass (13)

**Forward pass returns:**
- If `probs_out=True`: (root_logits, quality_logits, bass_logits)
- If `probs_out=False`: (root_pred, quality_pred, bass_pred, root_second, quality_second, bass_second)

**Loss calculation:**
- Weighted sum of NLL losses for each component:
  ```python
  total_loss = root_weight * root_loss + quality_weight * quality_loss + bass_weight * bass_loss
  ```

### 5. BTC Model (`models/btc_model.py`)

**Added `BTC_model_structured` class:**
- Uses `StructuredOutputLayer` instead of `SoftmaxOutputLayer`
- Forward pass accepts structured labels:
  ```python
  forward(x, root_labels, quality_labels, bass_labels)
  ```
- Returns dictionary with all predictions:
  ```python
  {
      'root_pred': root_pred,
      'quality_pred': quality_pred,
      'bass_pred': bass_pred,
      'root_second': root_second,
      'quality_second': quality_second,
      'bass_second': bass_second,
      'loss': loss,
      'weights_list': weights_list
  }
  ```

### 6. Training Script (`train_curriculum.py`)

**Updated to support both models:**
- Added `btc_structured` as a model option
- Training loop handles both flat and structured data
- Computes accuracy for each component separately
- Logs structured metrics to TensorBoard and Weights & Biases

**Metrics tracked for structured model:**
- Overall accuracy (all components must match)
- Root accuracy
- Quality accuracy
- Bass accuracy

### 7. Configuration (`run_config.yaml`)

**Added parameters:**
```yaml
model:
  use_structured_output: False  # Set to True to enable
  num_roots: 13
  num_qualities: 16
  num_bass: 13
  root_weight: 1.0
  quality_weight: 1.0
  bass_weight: 1.0
```

## Usage

### 1. Preprocess data with structured labels

First, ensure you have the updated code, then reprocess your datasets:

```bash
cd /home/daniel.melo/BTC_ORIGINAL/chords_recog/BTC-ISMIR19
python scripts/preprocess_datasets.py
```

This will create new .pt files with the structured components.

### 2. Train with structured output

```bash
python train_curriculum.py \
    --model btc_structured \
    --dataset1 billboard \
    --dataset2 jaah \
    --test_dataset rwc \
    --kfold 0 \
    --voca \
    --index 1
```

### 3. Configuration

Edit `run_config.yaml` to adjust structured output settings:

```yaml
model:
  use_structured_output: True
  num_roots: 13        # 12 pitches + no chord
  num_qualities: 16    # 14 qualities + no chord + unknown
  num_bass: 13         # 12 pitches + no bass
  root_weight: 1.0     # Adjust to emphasize root prediction
  quality_weight: 1.0  # Adjust to emphasize quality prediction
  bass_weight: 1.0     # Adjust to emphasize bass prediction
```

## Benefits of Structured Output

1. **Interpretability**: Each component can be analyzed separately
2. **Modularity**: Can train different components with different weights
3. **Flexibility**: Can use predictions for different evaluation metrics (root-only, triad, seventh chords, etc.)
4. **Debugging**: Easier to identify which component is causing errors
5. **Transfer Learning**: Components can be pretrained separately

## Backward Compatibility

The original `BTC_model` class remains unchanged and can still be used with:
```bash
python train_curriculum.py --model btc --voca ...
```

The structured model only activates when:
1. Data files contain structured labels (root, quality, bass)
2. Model is set to `btc_structured`

## Next Steps

To fully replicate ChordFormer:
1. Consider adding class-specific weights for each component (not just global)
2. Implement evaluation metrics for structured output
3. Add conformer blocks (ChordFormer's main architectural contribution)
4. Experiment with different loss weights for root, quality, and bass

## Files Modified

1. `utils/chords.py` - Added quality_to_id() and structured output support
2. `utils/preprocess.py` - Save structured labels in .pt files
3. `data/audio_dataset.py` - Load and batch structured targets
4. `utils/transformer_modules.py` - Added StructuredOutputLayer class
5. `models/btc_model.py` - Added BTC_model_structured class
6. `train_curriculum.py` - Support training with structured output
7. `run_config.yaml` - Added configuration parameters
