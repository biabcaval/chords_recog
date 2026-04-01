#!/usr/bin/env bash
set -euo pipefail

# Merge balanced_v1_train + balanced_v1_test into balanced_v1_full.
# Run on the VM where data_root points to /home/daniel.melo/datasets.

DATA_ROOT="${1:-/home/daniel.melo/datasets}"
BASE_DIR="$DATA_ROOT/personalized_datasets"

TRAIN_DIR="$BASE_DIR/balanced_v1_train"
TEST_DIR="$BASE_DIR/balanced_v1_test"
FULL_DIR="$BASE_DIR/balanced_v1_full"

echo "=== Merging balanced datasets ==="
echo "  Train source: $TRAIN_DIR"
echo "  Test  source: $TEST_DIR"
echo "  Output:       $FULL_DIR"
echo

for src in "$TRAIN_DIR" "$TEST_DIR"; do
    if [ ! -d "$src" ]; then
        echo "ERROR: source directory not found: $src"
        exit 1
    fi
done

if [ -d "$FULL_DIR" ]; then
    echo "WARNING: $FULL_DIR already exists."
    read -rp "Overwrite? [y/N] " answer
    if [[ ! "$answer" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
    rm -rf "$FULL_DIR"
fi

mkdir -p "$FULL_DIR/audio" "$FULL_DIR/annotations"

n_train=0
n_test=0

echo "Copying train files..."
for f in "$TRAIN_DIR/audio/"*; do
    [ -e "$f" ] || continue
    cp -v "$f" "$FULL_DIR/audio/"
    n_train=$((n_train + 1))
done
for f in "$TRAIN_DIR/annotations/"*; do
    [ -e "$f" ] || continue
    cp "$f" "$FULL_DIR/annotations/"
done

echo "Copying test files..."
for f in "$TEST_DIR/audio/"*; do
    [ -e "$f" ] || continue
    basename=$(basename "$f")
    if [ -e "$FULL_DIR/audio/$basename" ]; then
        echo "  SKIP (duplicate): $basename"
    else
        cp -v "$f" "$FULL_DIR/audio/"
        n_test=$((n_test + 1))
    fi
done
for f in "$TEST_DIR/annotations/"*; do
    [ -e "$f" ] || continue
    basename=$(basename "$f")
    if [ -e "$FULL_DIR/annotations/$basename" ]; then
        echo "  SKIP (duplicate): $basename"
    else
        cp "$f" "$FULL_DIR/annotations/"
    fi
done

# Merge manifest.json files if they exist
python3 -c "
import json, os, sys

full_dir = '$FULL_DIR'
train_manifest = '$TRAIN_DIR/manifest.json'
test_manifest = '$TEST_DIR/manifest.json'

songs = []
seen = set()

for mpath, split_label in [(train_manifest, 'train'), (test_manifest, 'test')]:
    if not os.path.exists(mpath):
        print(f'  No manifest at {mpath}, skipping')
        continue
    with open(mpath) as f:
        data = json.load(f)
    for s in data.get('songs', []):
        key = s.get('name', s.get('audio_file', ''))
        if key not in seen:
            s['original_split'] = split_label
            songs.append(s)
            seen.add(key)

manifest = {
    'dataset_name': 'balanced_v1_full',
    'split': 'full',
    'num_songs': len(songs),
    'note': 'Merged from balanced_v1_train + balanced_v1_test',
    'songs': songs,
}

out = os.path.join(full_dir, 'manifest.json')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(manifest, f, indent=2, ensure_ascii=False)
print(f'  Wrote manifest with {len(songs)} songs to {out}')
"

total=$((n_train + n_test))
echo
echo "=== Done ==="
echo "  Train songs: $n_train"
echo "  Test  songs: $n_test"
echo "  Total:       $total audio files in $FULL_DIR/audio/"
echo
echo "Next steps:"
echo "  1. Preprocess:  python scripts/preprocess_decomposed.py"
echo "  2. Weights:     python scripts/precompute_class_weights_decomposed.py"
echo "  3. Normalize:   python scripts/compute_normalization.py"
echo "  4. Train:       python train_decomposed.py --kfold 0"
