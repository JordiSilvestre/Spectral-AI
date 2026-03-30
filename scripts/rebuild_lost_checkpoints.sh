#!/usr/bin/env bash
# ==============================================================
#  rebuild_lost_checkpoints.sh
#  Recrea checkpoints perdidos el 2026-03-28
#
#  Perdidos:
#    1. checkpoints/gpt2_baseline_best.pt    (PPL ~187.4)
#    2. checkpoints/inception_best.pt        (PPL ~191.3)
#    3. checkpoints/orchestrator_multidomain_best.pt (100% routing)
#
#  Uso: bash scripts/rebuild_lost_checkpoints.sh [--skip-baseline] [--skip-inception] [--skip-multidomain]
# ==============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

SKIP_BASELINE=false
SKIP_INCEPTION=false
SKIP_MULTIDOMAIN=false

for arg in "$@"; do
    case "$arg" in
        --skip-baseline)     SKIP_BASELINE=true ;;
        --skip-inception)    SKIP_INCEPTION=true ;;
        --skip-multidomain)  SKIP_MULTIDOMAIN=true ;;
    esac
done

echo "============================================================"
echo "  SpectralAI — Rebuild Lost Checkpoints"
echo "  Lost on 2026-03-28, recreating from recovered scripts"
echo "============================================================"
echo ""

# Ensure checkpoints dir exists
mkdir -p checkpoints data

# ─────────────────────────────────────────────────────────────
# FASE 1: GPT-2 Baseline (target: PPL ~187.4)
# ─────────────────────────────────────────────────────────────
if [ "$SKIP_BASELINE" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  FASE 1: GPT-2 Baseline (target PPL ~187.4)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "checkpoints/gpt2_baseline_best.pt" ]; then
        echo "  [SKIP] checkpoints/gpt2_baseline_best.pt ya existe"
    else
        echo "  Entrenando GPT-2 baseline (~20-45 min)..."
        python3 python/gpt2_baseline.py \
            --epochs 10 \
            --batch-size 32 \
            --lr 5e-4 \
            --device cuda
        echo "  ✅ GPT-2 baseline completado"
    fi
    echo ""
else
    echo "  [SKIP] GPT-2 baseline (--skip-baseline)"
fi

# ─────────────────────────────────────────────────────────────
# FASE 2: Inception v4.0 (target: PPL ~191.3)
# ─────────────────────────────────────────────────────────────
if [ "$SKIP_INCEPTION" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  FASE 2: Inception v4.0 (target PPL ~191.3)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "checkpoints/inception_best.pt" ]; then
        echo "  [SKIP] checkpoints/inception_best.pt ya existe"
    else
        echo "  Entrenando Inception v4.0 (~30-60 min)..."
        python3 python/train_inception.py \
            --epochs 10 \
            --batch-size 32 \
            --lr 5e-4 \
            --alpha-spatial 0.05 \
            --device cuda
        echo "  ✅ Inception v4.0 completado"
    fi
    echo ""
else
    echo "  [SKIP] Inception v4.0 (--skip-inception)"
fi

# ─────────────────────────────────────────────────────────────
# FASE 3: Multi-Domain Orchestrator (target: 100% routing)
# ─────────────────────────────────────────────────────────────
if [ "$SKIP_MULTIDOMAIN" = false ]; then
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  FASE 3: Multi-Domain Orchestrator (target 100% routing)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ -f "checkpoints/orchestrator_multidomain_best.pt" ]; then
        echo "  [SKIP] checkpoints/orchestrator_multidomain_best.pt ya existe"
    else
        echo "  Entrenando Multi-Domain (~45-90 min)..."
        python3 python/train_multi_domain.py \
            --epochs 10 \
            --batch_size 32 \
            --lr 3e-4 \
            --device cuda
        echo "  ✅ Multi-Domain completado"
    fi
    echo ""
else
    echo "  [SKIP] Multi-Domain (--skip-multidomain)"
fi

# ─────────────────────────────────────────────────────────────
# Verificación final
# ─────────────────────────────────────────────────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  VERIFICACIÓN FINAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

check_file() {
    if [ -f "$1" ]; then
        SIZE=$(du -h "$1" | cut -f1)
        echo "  ✅ $1 ($SIZE)"
    else
        echo "  ❌ $1 — FALTA"
    fi
}

check_file "checkpoints/gpt2_baseline_best.pt"
check_file "checkpoints/inception_best.pt"
check_file "checkpoints/orchestrator_multidomain_best.pt"
check_file "data/gpt2_baseline_log.json"
check_file "data/inception_training_log.json"
check_file "data/multidomain_training_log.json"

echo ""
echo "============================================================"
echo "  Rebuild completado"
echo "============================================================"
