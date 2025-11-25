# Curriculum Learning - Quick Start Guide

## What is Curriculum Learning?

Curriculum learning trains your model progressively from easy to hard examples, just like how humans learn. For chord recognition, this means:

- **Starting** with simple songs (major/minor chords, few changes)
- **Progressing** to complex songs (7th chords, many changes)
- **Result**: Better convergence and often higher accuracy

## Quick Start (3 Steps)

### Step 1: Enable Curriculum Learning

Edit `run_config.yaml`:

```yaml
curriculum:
  enabled: True  # Change this from False to True
```

### Step 2: Run Training

```bash
python train_curriculum.py --curriculum True --index 1 --kfold 0 \
    --model btc --dataset1 billboard --dataset2 jaah --test_dataset rwc
```

### Step 3: Monitor Progress

Watch the console output for curriculum stats:

```
==== Curriculum Learning Stats (Epoch 10) ====
  Data Ratio: 45.00% (3150/7000 samples)
  Mean Difficulty: 0.2847
  Difficulty Range: [0.0000, 0.4500]
```

## Testing Before Training

Want to see how it works without full training?

```bash
python test_curriculum.py
```

This will:
- Compute difficulty scores for your dataset
- Generate visualization plots
- Show how samples are selected at different epochs
- Save plots to `assets/curriculum_learning/`

## Configuration Options (Optional)

### Default Settings (Good for Most Cases)
```yaml
curriculum:
  enabled: True
  strategy: 'mixed'      # Considers all difficulty factors
  pacing: 'linear'       # Steady progression
  start_ratio: 0.3       # Start with easiest 30%
  pace_epochs: 30        # Reach 100% after 30 epochs
  warmup_epochs: 5       # First 5 epochs use only easy samples
```

### For Faster Training
```yaml
curriculum:
  enabled: True
  strategy: 'chord_complexity'  # Focus on chord types
  pacing: 'exponential'         # Faster progression
  start_ratio: 0.4              # Start with more data
  pace_epochs: 20               # Reach 100% faster
  warmup_epochs: 3              # Shorter warmup
```

### For Better Generalization
```yaml
curriculum:
  enabled: True
  strategy: 'mixed'      # All factors
  pacing: 'quadratic'    # Gentle progression
  start_ratio: 0.2       # Start with less data
  pace_epochs: 40        # Longer curriculum
  warmup_epochs: 10      # Longer warmup
```

## Difficulty Strategies

1. **`chord_complexity`** - Based on chord types (maj/min easier than 7th/9th)
2. **`change_frequency`** - Based on how often chords change
3. **`unique_chords`** - Based on number of different chords
4. **`mixed`** - Combination of all three (recommended)

## Pacing Functions

1. **`linear`** - Steady, predictable increase (recommended)
2. **`quadratic`** - Slow start, faster later
3. **`exponential`** - Very slow start, very fast later
4. **`step`** - Discrete jumps in difficulty

## Expected Results

With curriculum learning, you should see:
- ✓ Faster initial convergence (first 10-15 epochs)
- ✓ More stable training (less fluctuation in loss)
- ✓ Potentially higher final accuracy (0.5-2% improvement)
- ✓ Better performance on complex chord types

## Troubleshooting

**Q: Training seems slower than normal**
- A: Increase `start_ratio` to 0.4-0.5 or decrease `pace_epochs` to 20

**Q: Accuracy drops when harder samples are introduced**
- A: Increase `pace_epochs` to 40 or use `pacing: 'quadratic'`

**Q: No improvement over baseline**
- A: Try different `strategy` values or verify your dataset has varying difficulty

**Q: Import errors**
- A: Make sure all required packages are installed:
  ```bash
  pip install torch numpy sortedcontainers matplotlib
  ```

## File Structure

```
BTC-ISMIR19/
├── curriculum_learning.py       # Main implementation
├── train_curriculum.py          # Training script with curriculum support
├── test_curriculum.py           # Testing/visualization script
├── CURRICULUM_LEARNING.md       # Detailed documentation
├── CURRICULUM_QUICKSTART.md     # This file
└── run_config.yaml              # Configuration file
```

## Example Training Commands

### Basic (Small Vocabulary)
```bash
python train_curriculum.py --curriculum True --index 1 --kfold 0 \
    --model btc --dataset1 billboard --dataset2 jaah --test_dataset rwc
```

### Large Vocabulary
```bash
python train_curriculum.py --curriculum True --voca True --index 2 --kfold 0 \
    --model btc --dataset1 billboard --dataset2 jaah --test_dataset rwc
```

### Different K-Fold
```bash
python train_curriculum.py --curriculum True --index 3 --kfold 1 \
    --model btc --dataset1 billboard --dataset2 jaah --test_dataset rwc
```

### With CNN/CRNN Models
```bash
# CNN
python train_curriculum.py --curriculum True --model cnn --index 4 --kfold 0 \
    --dataset1 billboard --dataset2 jaah --test_dataset rwc

# CRNN
python train_curriculum.py --curriculum True --model crnn --index 5 --kfold 0 \
    --dataset1 billboard --dataset2 jaah --test_dataset rwc
```

## Monitoring with TensorBoard

Launch TensorBoard to see curriculum learning metrics:

```bash
tensorboard --logdir=assets/tensorboard
```

Curriculum metrics available:
- `curriculum/ratio` - Percentage of data being used
- `curriculum/mean_difficulty` - Average difficulty of current samples
- `curriculum/num_samples` - Number of samples in current epoch

## Next Steps

1. **Run baseline**: Train without curriculum learning for comparison
   ```bash
   python train.py --index 1 --kfold 0 --model btc \
       --dataset1 billboard --dataset2 jaah --test_dataset rwc
   ```

2. **Run with curriculum**: Train with curriculum learning
   ```bash
   python train_curriculum.py --curriculum True --index 2 --kfold 0 \
       --model btc --dataset1 billboard --dataset2 jaah --test_dataset rwc
   ```

3. **Compare results**: Check test scores and training curves

For more details, see `CURRICULUM_LEARNING.md`

