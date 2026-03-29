#!/bin/bash
# ============================================================
# SpectralAI Zero-Matrix — Train remaining 11 layers for 16/16
# ============================================================
# Run AFTER regenerate_all.sh completes (5 layers: 0,4,8,12,15).
# This trains the remaining 11 layers to achieve full 16/16 replacement.
#
# Prerequisites:
#   - OLMoE model at /mnt/j/Proyectos/models/olmoe-1b-7b
#   - regenerate_all.sh completed successfully (5 layers validated)
#   - Python venv activated
#
# Usage:
#   cd /tmp/spectral
#   source .venv_wsl/bin/activate
#   bash scripts/train_remaining_layers.sh
# ============================================================

set -e

echo "============================================================"
echo "  SpectralAI — FASE 3: Remaining 11 Layers (16/16 target)"
echo "============================================================"

MODEL_DIR="/mnt/j/Proyectos/models/olmoe-1b-7b"
REMAINING_LAYERS="1 2 3 5 6 7 9 10 11 13 14"

# ── Step 1: Extract hidden states ────────────────────────────
echo ""
echo ">>> STEP 1: Extracting hidden states for layers: $REMAINING_LAYERS"
echo "    (~10 min per layer, ~856 MB each, ~9.4 GB total)"
echo ""

for L in $REMAINING_LAYERS; do
    OUTPUT="data/real_hiddens_layer${L}.pt"
    if [ -f "$OUTPUT" ]; then
        echo "  Layer $L: already exists, skipping"
    else
        echo "  Layer $L: extracting..."
        python python/extract_real_hiddens.py \
            --model-dir "$MODEL_DIR" \
            --layer "$L" \
            --output "$OUTPUT"
        echo "  Layer $L: done"
    fi
done

echo ""
echo ">>> STEP 1 COMPLETE — all 16 layers extracted"

# ── Step 2: Train routers ────────────────────────────────────
echo ""
echo ">>> STEP 2: Training BVH routers for remaining layers"
echo "    (~5-10 min per layer)"
echo ""

for L in $REMAINING_LAYERS; do
    SAVE_DIR="checkpoints/olmoe_distill_layer${L}"
    CKPT="${SAVE_DIR}/bvh_router_best.pt"

    if [ -f "$CKPT" ]; then
        echo "  Layer $L: checkpoint exists, skipping"
    else
        echo "  Layer $L: training..."
        python python/olmoe_bvh_distill.py \
            --layer "$L" \
            --real-data "data/real_hiddens_layer${L}.pt" \
            --epochs 50 \
            --save-dir "$SAVE_DIR" \
            --device cuda
        echo "  Layer $L: training done"
    fi
done

echo ""
echo ">>> STEP 2 COMPLETE"

# ── Step 3: Calibrate all remaining routers ──────────────────
echo ""
echo ">>> STEP 3: Linear calibration (4160 params per layer)"
echo "    (~20s per layer on CPU)"
echo ""

for L in $REMAINING_LAYERS; do
    SAVE_DIR="checkpoints/olmoe_distill_layer${L}"

    echo "  Layer $L: calibrating..."
    python python/calibrate_router.py \
        --mode linear \
        --epochs 100 \
        --real-data "data/real_hiddens_layer${L}.pt" \
        --router-checkpoint "${SAVE_DIR}/bvh_router_best.pt" \
        --device cpu
    echo "  Layer $L: calibration done"
done

echo ""
echo ">>> STEP 3 COMPLETE"

# ── Step 4: Verify 16/16 PPL ─────────────────────────────────
echo ""
echo ">>> STEP 4: Full 16/16 layer evaluation (target: PPL < 7.0)"
echo ""

# Build the multi-layer argument string for all 16 layers
MULTI_LAYER=""
for L in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    SAVE_DIR="checkpoints/olmoe_distill_layer${L}"
    if [ "$L" = "8" ]; then
        SAVE_DIR="checkpoints/olmoe_distill"
    fi
    CKPT="${SAVE_DIR}/bvh_router_best.pt"

    if [ -n "$MULTI_LAYER" ]; then
        MULTI_LAYER="${MULTI_LAYER},"
    fi
    MULTI_LAYER="${MULTI_LAYER}${L}:${CKPT}"
done

echo "  Evaluating with all 16 layers replaced..."
echo "  Checkpoints: $MULTI_LAYER"
echo ""

python python/olmoe_e2e_eval.py \
    --model-dir "$MODEL_DIR" \
    --multi-layer "$MULTI_LAYER" \
    --max-tokens 50000

echo ""
echo "============================================================"
echo "  16/16 LAYER REPLACEMENT COMPLETE"
echo "  Expected: PPL ~7.0 (~15% degradation from baseline 6.11)"
echo "  If PPL < 7.0: SUCCESS — full geometric routing viable!"
echo "============================================================"
