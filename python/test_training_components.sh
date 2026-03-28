#!/bin/bash
# test_training_components.sh — Quick test script for training components

set -e

echo "=========================================="
echo "LiquidBit Zero-Matrix — Training Test"
echo "=========================================="
echo ""

cd "$(dirname "$0")"

# Test 1: DuplScore Optimizer
echo "[TEST 1] DuplScore Optimizer"
echo "Running: python3 dupl_score_optimizer.py"
python3 dupl_score_optimizer.py \
    --seed 42 \
    --gamma 0.2 \
    --delta 0.001 \
    --tau 0.5 \
    --output test_dupl_wormhole.json

if [ -f test_dupl_wormhole.json ]; then
    echo "✓ DuplScore output generated: test_dupl_wormhole.json"
    wc -l test_dupl_wormhole.json
else
    echo "✗ DuplScore output NOT generated"
    exit 1
fi

echo ""

# Test 2: Fuzzy BSH
echo "[TEST 2] Fuzzy BSH"
echo "Running: python3 fuzzy_bsh.py"
python3 fuzzy_bsh.py \
    --num-epochs 100 \
    --seed 42 \
    --learning-rate 0.01 \
    --harden-every 25 \
    --harden-factor 0.9 \
    --output test_fuzzy_bsh_state.json

if [ -f test_fuzzy_bsh_state.json ]; then
    echo "✓ Fuzzy BSH output generated: test_fuzzy_bsh_state.json"
    wc -l test_fuzzy_bsh_state.json
else
    echo "✗ Fuzzy BSH output NOT generated"
    exit 1
fi

echo ""
echo "=========================================="
echo "[✓] All tests passed!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - test_dupl_wormhole.json (DuplScore graph)"
echo "  - test_fuzzy_bsh_state.json (Fuzzy BSH state)"
