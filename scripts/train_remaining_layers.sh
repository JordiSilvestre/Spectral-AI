#!/bin/bash
# ============================================================
# SpectralAI Zero-Matrix — Retrain ALL layers with Spectral Techniques
# ============================================================
# Estado actual (2026-03-30):
#   - 16/16 capas entrenadas
#   - L11 reentrenada con --spectral: 81.8% → 93.3% top-8
#   - Weak (< 85%): L3(80.5%), L5(81.9%), L6(84.3%), L7(84.3%)
#   - NEW: topk_matching_loss integrada (weight=0.3) — optimiza top-8 set
#     directamente, no solo KL divergence. "THE key missing piece."
#   - FORCE_RETRAIN=true por defecto para re-entrenar todas las capas
#
# Este script:
#   FASE 0: Copiar datos a disco local (/tmp) para I/O rápido
#   FASE A: Retrain capas débiles con --spectral (máximo impacto en PPL)
#   FASE A2: Retrain capas fuertes con --spectral (mejora incremental)
#   FASE B: Calibrar todas las capas
#   FASE C: Evaluar PPL 16/16
#
# Usage (en WSL):
#   cd /path/to/spectral-ai
#   source .venv_wsl/bin/activate
#   bash scripts/train_remaining_layers.sh
# ============================================================

set -eo pipefail

# Auto-detect python command
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: No python3 or python found in PATH"
    exit 1
fi

MODEL_DIR="${MODEL_DIR:-allenai/OLMoE-1B-7B-0924}"
EPOCHS="${EPOCHS:-200}"
DEVICE="${DEVICE:-cuda}"
FORCE_RETRAIN="${FORCE_RETRAIN:-true}"   # Force retrain even if spectral checkpoint exists (to use new topk_matching_loss)
DATA_SRC="data"
DATA_LOCAL="/tmp/spectral_hiddens"

echo "============================================================"
echo "  SpectralAI — Retrain ALL layers with Spectral Techniques"
echo "  Model: $MODEL_DIR"
echo "  Python: $PY"
echo "  Epochs: $EPOCHS | Device: $DEVICE"
echo "============================================================"

# ── FASE 0: Copiar datos a disco local para I/O rápido ──────
echo ""
echo ">>> FASE 0: Copiando datos a disco local ($DATA_LOCAL)..."
mkdir -p "$DATA_LOCAL"

for L in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    SRC="${DATA_SRC}/real_hiddens_layer${L}.pt"
    DST="${DATA_LOCAL}/real_hiddens_layer${L}.pt"
    if [ -f "$SRC" ]; then
        if [ ! -f "$DST" ] || [ "$SRC" -nt "$DST" ]; then
            echo "  Copiando layer $L → $DST..."
            cp "$SRC" "$DST"
        else
            echo "  Layer $L: ya en disco local"
        fi
    else
        echo "  Layer $L: SIN DATOS ($SRC no existe)"
    fi
done
echo ">>> FASE 0 COMPLETE — datos en disco local"

# ── Helper: retrain a layer with --spectral if needed ───────
retrain_layer() {
    local L=$1
    local SAVE_DIR="checkpoints/olmoe_distill_layer${L}"
    if [ "$L" = "8" ]; then
        SAVE_DIR="checkpoints/olmoe_distill"
    fi
    local CKPT="${SAVE_DIR}/bvh_router_best.pt"
    local DATA_FILE="${DATA_LOCAL}/real_hiddens_layer${L}.pt"

    if [ ! -f "$DATA_FILE" ]; then
        echo "  Layer $L: NO DATA, skipping"
        return
    fi

    local NEEDS_SPECTRAL=true
    if [ -f "$CKPT" ] && [ "$FORCE_RETRAIN" != "true" ]; then
        local HAS_SPECTRAL
        HAS_SPECTRAL=$($PY -c "
import torch
c = torch.load('$CKPT', map_location='cpu', weights_only=False)
print('true' if c.get('spectral_mode', c.get('lyra_mode', False)) else 'false')
" 2>&1 || echo "false")
        if [ "$HAS_SPECTRAL" = "true" ]; then
            echo "  Layer $L: already has Spectral Techniques, skipping (use FORCE_RETRAIN=true to override)"
            NEEDS_SPECTRAL=false
        else
            echo "  Layer $L: needs Spectral retrain"
        fi
    elif [ "$FORCE_RETRAIN" = "true" ]; then
        echo "  Layer $L: FORCE_RETRAIN=true, retraining with topk_matching_loss"
    fi

    if [ "$NEEDS_SPECTRAL" = "true" ]; then
        echo "  Layer $L: retraining with --spectral (epochs=$EPOCHS)..."
        mkdir -p "$SAVE_DIR"
        $PY python/olmoe_bvh_distill.py \
            --layer "$L" \
            --real-data "$DATA_FILE" \
            --epochs "$EPOCHS" \
            --save-dir "$SAVE_DIR" \
            --device "$DEVICE" \
            --spectral \
            --spectral-dim 256
        echo "  Layer $L: done"
    fi
}

# ── FASE A: Retrain capas débiles primero (máximo impacto) ──
PRIORITY_WEAK="3 5 6 7 2"

echo ""
echo ">>> FASE A: Retraining WEAK layers with --spectral"
echo "    Layers (priority order): $PRIORITY_WEAK"
echo ""

for L in $PRIORITY_WEAK; do
    retrain_layer "$L"
done

echo ""
echo ">>> FASE A COMPLETE"

# ── FASE A2: Retrain capas fuertes (mejora incremental) ─────
PRIORITY_STRONG="0 1 4 8 9 10 12 13 14 15"

echo ""
echo ">>> FASE A2: Retraining STRONG layers with --spectral"
echo "    Layers: $PRIORITY_STRONG"
echo ""

for L in $PRIORITY_STRONG; do
    retrain_layer "$L"
done

echo ""
echo ">>> FASE A2 COMPLETE"

# ── FASE B: Calibrar todas las capas ─────────────────────────
echo ""
echo ">>> FASE B: Linear calibration (all 16 layers)"
echo ""

for L in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    SAVE_DIR="checkpoints/olmoe_distill_layer${L}"
    if [ "$L" = "8" ]; then
        SAVE_DIR="checkpoints/olmoe_distill"
    fi
    CKPT="${SAVE_DIR}/bvh_router_best.pt"

    if [ ! -f "$CKPT" ]; then
        echo "  Layer $L: checkpoint missing, skipping calibration"
        continue
    fi

    echo "  Layer $L: calibrating (topk_preserving, 100 epochs)..."
    $PY python/calibrate_router.py \
        --mode topk_preserving \
        --epochs 100 \
        --real-data "${DATA_LOCAL}/real_hiddens_layer${L}.pt" \
        --router-checkpoint "$CKPT" \
        --device cpu
    echo "  Layer $L: calibration done"
done

echo ""
echo ">>> FASE B COMPLETE"

# ── FASE C: Evaluar PPL 16/16 ─────────────────────────────────
echo ""
echo ">>> FASE C: Full 16/16 PPL evaluation"
echo "    Baseline (gate lineal OLMoE): PPL 6.11"
echo "    Objetivo: PPL < 7.0 (<15% degradacion)"
echo ""

MULTI_LAYER=""
for L in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    SAVE_DIR="checkpoints/olmoe_distill_layer${L}"
    if [ "$L" = "8" ]; then
        SAVE_DIR="checkpoints/olmoe_distill"
    fi
    CKPT="${SAVE_DIR}/bvh_router_best.pt"
    if [ -n "$MULTI_LAYER" ]; then MULTI_LAYER="${MULTI_LAYER},"; fi
    MULTI_LAYER="${MULTI_LAYER}${L}:${CKPT}"
done

echo "--- PPL: PURE mode (BVH does everything) ---"
$PY python/olmoe_e2e_eval.py \
    --model-dir "$MODEL_DIR" \
    --multi-layer "$MULTI_LAYER" \
    --weight-mode softmax \
    --max-tokens 50000

echo ""
echo "--- PPL: HYBRID RESIDUAL mode (BVH selects, gate weights) ---"
$PY python/olmoe_e2e_eval.py \
    --model-dir "$MODEL_DIR" \
    --multi-layer "$MULTI_LAYER" \
    --weight-mode hybrid_residual \
    --max-tokens 50000

# ── Cleanup ─────────────────────────────────────────────────
echo ""
echo ">>> Limpiando datos locales ($DATA_LOCAL)..."
rm -rf "$DATA_LOCAL"

echo ""
echo "============================================================"
echo "  ALL PHASES COMPLETE — FASE F"
echo "  Retrained 16/16 layers: 200 epochs + spectral + topk_loss"
echo "  Evaluated BOTH pure and hybrid_residual modes"
echo "============================================================"
