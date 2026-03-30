#!/bin/bash
# ============================================================
# SpectralAI Zero-Matrix — Regenerate ALL checkpoints and data
# ============================================================
# Run this in WSL after a data loss to get back to FASE 3 state.
#
# Prerequisites:
#   - OLMoE model at /mnt/j/Proyectos/models/olmoe-1b-7b
#   - Python venv with: torch transformers accelerate safetensors datasets scikit-learn
#
# Usage:
#   cd /tmp/spectral
#   source .venv_wsl/bin/activate
#   bash scripts/regenerate_all.sh
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "  SpectralAI — Full Pipeline Regeneration"
echo "============================================================"

MODEL_DIR="/mnt/j/Proyectos/models/olmoe-1b-7b"
LAYERS="0 4 8 12 15"

# ── Step 1: Extract hidden states ────────────────────────────
echo ""
echo ">>> STEP 1: Extracting hidden states for layers: $LAYERS"
echo "    (~10 min per layer, ~856 MB each)"
echo ""

for L in $LAYERS; do
    OUTPUT="data/real_hiddens_layer${L}.pt"
    if [ -f "$OUTPUT" ]; then
        echo "  Layer $L: already exists, skipping"
    else
        echo "  Layer $L: extracting..."
        python3 python/extract_real_hiddens.py \
            --model-dir "$MODEL_DIR" \
            --layer "$L" \
            --output "$OUTPUT"
        echo "  Layer $L: done"
    fi
done

echo ""
echo ">>> STEP 1 COMPLETE"

# ── Step 2: Train routers ────────────────────────────────────
echo ""
echo ">>> STEP 2: Training BVH routers for each layer"
echo "    (~5-10 min per layer)"
echo ""

for L in $LAYERS; do
    SAVE_DIR="checkpoints/olmoe_distill_layer${L}"
    # Layer 8 uses the default save dir
    if [ "$L" = "8" ]; then
        SAVE_DIR="checkpoints/olmoe_distill"
    fi

    CKPT="${SAVE_DIR}/bvh_router_best.pt"
    if [ -f "$CKPT" ]; then
        echo "  Layer $L: checkpoint exists, skipping"
    else
        echo "  Layer $L: training..."
        python3 python/olmoe_bvh_distill.py \
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

# ── Step 3: Calibrate all routers ────────────────────────────
echo ""
echo ">>> STEP 3: Linear calibration (4160 params per layer)"
echo "    (~20s per layer on CPU)"
echo ""

for L in $LAYERS; do
    SAVE_DIR="checkpoints/olmoe_distill_layer${L}"
    if [ "$L" = "8" ]; then
        SAVE_DIR="checkpoints/olmoe_distill"
    fi

    echo "  Layer $L: calibrating..."
    python3 python/calibrate_router.py \
        --mode linear \
        --epochs 100 \
        --real-data "data/real_hiddens_layer${L}.pt" \
        --router-checkpoint "${SAVE_DIR}/bvh_router_best.pt" \
        --device cpu
    echo "  Layer $L: calibration done"
done

echo ""
echo ">>> STEP 3 COMPLETE"

# ── Step 4: Verify single-layer PPL ─────────────────────────
echo ""
echo ">>> STEP 4: Verifying single-layer PPL (should be ~6.16)"
echo ""

python3 python/olmoe_e2e_eval.py \
    --model-dir "$MODEL_DIR" \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt \
    --max-tokens 50000

# ── Step 5: Verify multi-layer PPL ──────────────────────────
echo ""
echo ">>> STEP 5: Verifying 5-layer PPL (should be ~6.40)"
echo ""

python3 python/olmoe_e2e_eval.py \
    --model-dir "$MODEL_DIR" \
    --multi-layer "0:checkpoints/olmoe_distill_layer0/bvh_router_best.pt,4:checkpoints/olmoe_distill_layer4/bvh_router_best.pt,8:checkpoints/olmoe_distill/bvh_router_best.pt,12:checkpoints/olmoe_distill_layer12/bvh_router_best.pt,15:checkpoints/olmoe_distill_layer15/bvh_router_best.pt" \
    --max-tokens 50000

echo ""
echo "============================================================"
echo "  REGENERATION COMPLETE"
echo "  Expected results:"
echo "    Single layer (L8):        PPL ~6.16 (+0.8%)"
echo "    Multi-layer (0,4,8,12,15): PPL ~6.40 (+4.8%)"
echo "============================================================"
