#!/bin/bash
# ============================================================
# SpectralAI Zero-Matrix — Retrain weak layers with Lyra
# ============================================================
# Estado actual (2026-03-29):
#   - 16/16 capas entrenadas (sin Lyra excepto L1)
#   - L1: 81.9% top-8 (con Lyra) — referencia
#   - Weak (< 85%): L3(80.5%), L5(81.9%), L6(84.3%), L7(84.3%), L11(81.8%,16ep)
#
# Este script:
#   FASE A: Retrain capas débiles con --lyra (máximo impacto en PPL)
#   FASE B: Calibrar todas las capas
#   FASE C: Evaluar PPL 16/16
#
# Prerequisites:
#   - OLMoE model en MODEL_DIR
#   - data/real_hiddens_layer*.pt disponibles (todos presentes)
#   - Python venv activado (con torch, transformers)
#
# Usage (en WSL):
#   cd /mnt/j/Proyectos/SPECTRAL\ AI
#   source .venv_wsl/bin/activate    # o .venv/Scripts/activate en Windows
#   bash scripts/train_remaining_layers.sh
# ============================================================

set -eo pipefail

MODEL_DIR="${MODEL_DIR:-/mnt/j/Proyectos/models/olmoe-1b-7b}"
EPOCHS="${EPOCHS:-100}"
DEVICE="${DEVICE:-cuda}"

echo "============================================================"
echo "  SpectralAI — Retrain weak layers with Lyra (FASE 3)"
echo "  Model: $MODEL_DIR"
echo "  Epochs: $EPOCHS | Device: $DEVICE"
echo "============================================================"

# ── Orden de prioridad (de más débil a más fuerte) ───────────
# L11 primero: 81.8% Y solo 16 epochs (training incompleto!)
# L3: 80.5% (el más débil sin Lyra)
# L5: 81.9%
# L6: 84.3%
# L7: 84.3%
# L2: 84.7% (borderline)
PRIORITY_LYRA="11 3 5 6 7 2"

# Capas buenas sin Lyra — no re-entrenar por ahora (>85% top-8)
# L0(89.5%), L4(86.6%), L8(90.1%), L9(88.3%), L10(89.3%),
# L12(88.8%), L13(92.4%), L14(93.4%), L15(89.3%)

# ── FASE A: Retrain capas débiles con --lyra ─────────────────
echo ""
echo ">>> FASE A: Retraining weak layers with --lyra"
echo "    Layers (priority order): $PRIORITY_LYRA"
echo ""

for L in $PRIORITY_LYRA; do
    SAVE_DIR="checkpoints/olmoe_distill_layer${L}"
    CKPT="${SAVE_DIR}/bvh_router_best.pt"

    # Verificar si ya tiene Lyra (no re-entrenar si ya está)
    NEEDS_LYRA=true
    if [ -f "$CKPT" ]; then
        HAS_LYRA=$(python -c "
import torch
c = torch.load('$CKPT', map_location='cpu', weights_only=False)
print('true' if c.get('lyra_mode', False) else 'false')
" 2>&1 || echo "false")
        if [ "$HAS_LYRA" = "true" ]; then
            echo "  Layer $L: already has Lyra, skipping"
            NEEDS_LYRA=false
        else
            echo "  Layer $L: top-8=$(python -c "
import torch
c = torch.load('$CKPT', map_location='cpu', weights_only=False)
print(f\"{c.get('topk_accuracy', 0):.1%}\")
" 2>/dev/null || echo '?'), needs Lyra retrain"
        fi
    fi

    if [ "$NEEDS_LYRA" = "true" ]; then
        echo "  Layer $L: retraining with --lyra (epochs=$EPOCHS)..."
        mkdir -p "$SAVE_DIR"
        python python/olmoe_bvh_distill.py \
            --layer "$L" \
            --real-data "data/real_hiddens_layer${L}.pt" \
            --epochs "$EPOCHS" \
            --save-dir "$SAVE_DIR" \
            --device "$DEVICE" \
            --lyra
        echo "  Layer $L: done"
    fi
done

echo ""
echo ">>> FASE A COMPLETE"

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

    echo "  Layer $L: calibrating (linear, 100 epochs)..."
    python python/calibrate_router.py \
        --mode linear \
        --epochs 100 \
        --real-data "data/real_hiddens_layer${L}.pt" \
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

# Build --multi-layer argument
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

python python/olmoe_e2e_eval.py \
    --model-dir "$MODEL_DIR" \
    --multi-layer "$MULTI_LAYER" \
    --max-tokens 50000

echo ""
echo "============================================================"
echo "  FASE 3 COMPLETE"
echo "  Con Lyra en capas débiles, PPL esperado: 8.29 -> ~7.5-7.8"
echo "  Si < 7.0: excelente — train capas buenas con Lyra tambien"
echo "  Si > 8.0: escalar epochs o revisar capas problemáticas"
echo "============================================================"
