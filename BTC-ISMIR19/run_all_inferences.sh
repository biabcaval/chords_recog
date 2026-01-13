#!/bin/bash
# Run all inferences for experiments in BTC_ORIGINAL/BTC-ISMIR19/assets
# Output folders named: inference_{exp_num}_{train_datasets}_test_{test_dataset}

cd /home/daniel.melo/BTC_ORIGINAL/chords_recog/BTC-ISMIR19

ASSETS_DIR="/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/assets"
OUTPUT_BASE="/home/daniel.melo/BTC_ORIGINAL/BTC-ISMIR19/inferences"
CONFIG="/home/daniel.melo/BTC_ORIGINAL/chords_recog/BTC-ISMIR19/run_config.yaml"

echo "===== Starting Batch Inference ====="
echo "Output base: $OUTPUT_BASE"
echo ""

# exp1: trained on billboard, jaah, rwc - test on dj_avan (unseen)
echo "=== Experiment 1: BiJaRw -> test Dj ==="
python run_inference_batch.py \
    --checkpoint "$ASSETS_DIR/exp1_btc_billboard_jaah_rwc_voca_curriculum/kfold_0/model_040.pth.tar" \
    --test_dataset dj_avan \
    --train_name "BiJaRw" \
    --exp_num "1" \
    --output_base "$OUTPUT_BASE" \
    --config "$CONFIG" \
    --kfold 0

# exp2: trained on billboard, jaah, dj_avan, rwc (curriculum) - test on queen/robbiewilliams (unseen)
echo ""
echo "=== Experiment 2: BiJaDjRw (curriculum) -> test Qu ==="
python run_inference_batch.py \
    --checkpoint "$ASSETS_DIR/exp2_btc_billboard_jaah_dj_avan_rwc_voca_curriculum/kfold_0/model_040.pth.tar" \
    --test_dataset queen \
    --train_name "BiJaDjRw_curr" \
    --exp_num "2" \
    --output_base "$OUTPUT_BASE" \
    --config "$CONFIG" \
    --kfold 0

echo ""
echo "=== Experiment 2: BiJaDjRw (curriculum) -> test Ro ==="
python run_inference_batch.py \
    --checkpoint "$ASSETS_DIR/exp2_btc_billboard_jaah_dj_avan_rwc_voca_curriculum/kfold_0/model_040.pth.tar" \
    --test_dataset robbiewilliams \
    --train_name "BiJaDjRw_curr" \
    --exp_num "2" \
    --output_base "$OUTPUT_BASE" \
    --config "$CONFIG" \
    --kfold 0

# exp3: trained on billboard, jaah, dj_avan, rwc (standard) - test on queen/robbiewilliams (unseen)
echo ""
echo "=== Experiment 3: BiJaDjRw (standard) -> test Qu ==="
python run_inference_batch.py \
    --checkpoint "$ASSETS_DIR/exp3_btc_billboard_jaah_dj_avan_rwc_voca_standard/kfold_0/model_040.pth.tar" \
    --test_dataset queen \
    --train_name "BiJaDjRw" \
    --exp_num "3" \
    --output_base "$OUTPUT_BASE" \
    --config "$CONFIG" \
    --kfold 0

echo ""
echo "=== Experiment 3: BiJaDjRw (standard) -> test Ro ==="
python run_inference_batch.py \
    --checkpoint "$ASSETS_DIR/exp3_btc_billboard_jaah_dj_avan_rwc_voca_standard/kfold_0/model_040.pth.tar" \
    --test_dataset robbiewilliams \
    --train_name "BiJaDjRw" \
    --exp_num "3" \
    --output_base "$OUTPUT_BASE" \
    --config "$CONFIG" \
    --kfold 0

# exp4: trained on jaah, dj_avan, rwc - test on billboard/queen/robbiewilliams (unseen)
echo ""
echo "=== Experiment 4: JaDjRw -> test Bi ==="
python run_inference_batch.py \
    --checkpoint "$ASSETS_DIR/exp4_btc_jaah_dj_avan_rwc_voca_standard/kfold_2/model_040.pth.tar" \
    --test_dataset billboard \
    --train_name "JaDjRw" \
    --exp_num "4" \
    --output_base "$OUTPUT_BASE" \
    --config "$CONFIG" \
    --kfold 2

# exp5: robbie, queen, jaah, djavan -> test rwc
echo ""
echo "=== Experiment 5: RoQuJaDj -> test Rw ==="
BEST_EXP5=$(ls -v "$ASSETS_DIR/exp5_btc_robbie_queen_jaah_djavan_test_rwcvoca_standard"/idx_5_*.pth.tar 2>/dev/null | tail -1)
if [ -n "$BEST_EXP5" ]; then
    python run_inference_batch.py \
        --checkpoint "$BEST_EXP5" \
        --test_dataset rwc \
        --train_name "RoQuJaDj" \
        --exp_num "5" \
        --output_base "$OUTPUT_BASE" \
        --config "$CONFIG" \
        --kfold 0
fi

# exp6: bill, robbie, queen, jaah, djavan -> test rwc
echo ""
echo "=== Experiment 6: BiRoQuJaDj -> test Rw ==="
BEST_EXP6=$(ls -v "$ASSETS_DIR/exp6_btc_bill_robbie_queen_jaah_djavan_test_rwcvoca_standard"/idx_6_*.pth.tar 2>/dev/null | tail -1)
if [ -n "$BEST_EXP6" ]; then
    python run_inference_batch.py \
        --checkpoint "$BEST_EXP6" \
        --test_dataset rwc \
        --train_name "BiRoQuJaDj" \
        --exp_num "6" \
        --output_base "$OUTPUT_BASE" \
        --config "$CONFIG" \
        --kfold 0
fi

# exp7: bill, robbie, queen, jaah -> test rwc
echo ""
echo "=== Experiment 7: BiRoQuJa -> test Rw ==="
BEST_EXP7=$(ls -v "$ASSETS_DIR/exp7_btc_bill_robbie_queen_jaah_test_rwcvoca_standard"/idx_7_*.pth.tar 2>/dev/null | tail -1)
if [ -n "$BEST_EXP7" ]; then
    python run_inference_batch.py \
        --checkpoint "$BEST_EXP7" \
        --test_dataset rwc \
        --train_name "BiRoQuJa" \
        --exp_num "7" \
        --output_base "$OUTPUT_BASE" \
        --config "$CONFIG" \
        --kfold 0
fi

# exp8: bill, rwc, robbie, queen, jaah -> test djavan
echo ""
echo "=== Experiment 8: BiRwRoQuJa -> test Dj ==="
BEST_EXP8=$(ls -v "$ASSETS_DIR/exp8_btc_bill_rwc_robbie_queen_jaah_test_djavanvoca_standard"/idx_8_*.pth.tar 2>/dev/null | tail -1)
if [ -n "$BEST_EXP8" ]; then
    python run_inference_batch.py \
        --checkpoint "$BEST_EXP8" \
        --test_dataset dj_avan \
        --train_name "BiRwRoQuJa" \
        --exp_num "8" \
        --output_base "$OUTPUT_BASE" \
        --config "$CONFIG" \
        --kfold 0
fi

echo ""
echo "===== All inferences complete ====="
echo "Results saved to: $OUTPUT_BASE"

