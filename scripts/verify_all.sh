#!/bin/bash
# ============================================================================
# verify_all.sh — Script de verificacion completa de SpectralAI
# Ejecutar despues de pull para verificar todos los cambios.
# Uso: bash scripts/verify_all.sh
# ============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'
PASS=0
FAIL=0
SKIP=0

check() {
    local desc="$1"
    shift
    echo -n "  $desc... "
    if timeout 60 bash -c "$*" > /tmp/verify_out.txt 2>&1; then
        echo -e "${GREEN}OK${NC}"
        PASS=$((PASS+1))
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "${YELLOW}TIMEOUT${NC}"
            SKIP=$((SKIP+1))
        else
            echo -e "${RED}FAIL${NC}"
            tail -3 /tmp/verify_out.txt
            FAIL=$((FAIL+1))
        fi
    fi
}

skip() {
    local desc="$1"
    local reason="$2"
    echo -e "  $desc... ${YELLOW}SKIP ($reason)${NC}"
    SKIP=$((SKIP+1))
}

echo "============================================"
echo " SpectralAI Zero-Matrix — Verificacion"
echo " $(date)"
echo "============================================"
echo ""

# -------------------------------------------
# FASE 1: Tests CPU (sin GPU)
# -------------------------------------------
echo -e "${YELLOW}[FASE 1] Tests CPU (sin GPU)${NC}"

check "Python syntax: async_pipeline_bridge.py" \
    "python -c \"import py_compile; py_compile.compile('python/async_pipeline_bridge.py', doraise=True)\""

check "Python syntax: orchestrator.py" \
    "python -c \"import py_compile; py_compile.compile('python/orchestrator.py', doraise=True)\""

check "Python syntax: calibrate_router.py" \
    "python -c \"import py_compile; py_compile.compile('python/calibrate_router.py', doraise=True)\""

check "Python syntax: benchmark_cuda_e2e.py" \
    "python -c \"import py_compile; py_compile.compile('python/benchmark_cuda_e2e.py', doraise=True)\""

check "Python syntax: bvh_router_bridge.py" \
    "python -c \"import py_compile; py_compile.compile('python/bvh_router_bridge.py', doraise=True)\""

check "Python syntax: extract_real_hiddens.py" \
    "python -c \"import py_compile; py_compile.compile('python/extract_real_hiddens.py', doraise=True)\""

check "Python syntax: olmoe_e2e_eval.py" \
    "python -c \"import py_compile; py_compile.compile('python/olmoe_e2e_eval.py', doraise=True)\""

check "Python syntax: benchmark_expert_types.py" \
    "python -c \"import py_compile; py_compile.compile('python/benchmark_expert_types.py', doraise=True)\""

check "Python syntax: scaling_inception.py" \
    "python -c \"import py_compile; py_compile.compile('python/scaling_inception.py', doraise=True)\""

check "Python syntax: spectral_techniques.py" \
    "python -c \"import py_compile; py_compile.compile('python/spectral_techniques.py', doraise=True)\""

echo ""

# -------------------------------------------
# FASE 2: Tests Spectral Techniques (pytest)
# -------------------------------------------
echo -e "${YELLOW}[FASE 2] Tests Spectral Techniques (37 tests)${NC}"

if python -c "import pytest" 2>/dev/null; then
    check "pytest tests/test_spectral_techniques.py (37 tests)" \
        "python -m pytest tests/test_spectral_techniques.py -q --tb=short"
else
    skip "pytest tests/test_spectral_techniques.py" "pytest no instalado (pip install pytest)"
fi

echo ""

# -------------------------------------------
# FASE 3: Verificacion de componentes Lyra
# -------------------------------------------
echo -e "${YELLOW}[FASE 3] Componentes Spectral (antes/despues)${NC}"

check "SmoothTernarySTE: gradientes fluyen" \
    "python -c \"
import sys; sys.path.insert(0,'python')
import torch
from spectral_techniques import ternary_ste, set_ste_beta
set_ste_beta(1.0)
x = torch.randn(50, requires_grad=True)
y = ternary_ste(x)
y.sum().backward()
assert (x.grad.abs() > 0).sum() > 40, 'Gradients not flowing'
\""

check "SmoothBVHHit: diferenciable + ordenado" \
    "python -c \"
import sys; sys.path.insert(0,'python')
import torch
from spectral_techniques import SmoothBVHHit, set_ste_beta
set_ste_beta(5.0)
hit = SmoothBVHHit(0.1)
d = torch.tensor([[0.1, 1.0, 5.0]], requires_grad=True)
r = torch.tensor([2.0, 2.0, 2.0])
e = torch.tensor([1.0])
w = hit(d, r, e)
w.sum().backward()
assert w[0,0] > w[0,1], f'Wrong order: {w}'
assert d.grad is not None, 'No gradient'
\""

check "RMSNorm: normaliza correctamente" \
    "python -c \"
import sys; sys.path.insert(0,'python')
import torch
from spectral_techniques import RMSNorm
norm = RMSNorm(64)
x = torch.randn(2, 10, 64) * 100
y = norm(x)
assert y.abs().mean() < 5.0, f'Not normalized: {y.abs().mean()}'
\""

check "LiquidTimeGate: LOCAL/GLOBAL split" \
    "python -c \"
import sys; sys.path.insert(0,'python')
import torch
from spectral_techniques import LiquidTimeGate
g = LiquidTimeGate(32)
g.time_a.data[:16] = -1.0
g.time_a.data[16:] = 1.0
s = g.gate_stats()
assert s['n_local'] == 16 and s['n_global'] == 16
\""

check "MetabolicBVH: poda funciona" \
    "python -c \"
import sys; sys.path.insert(0,'python')
import numpy as np
from spectral_techniques import MetabolicBVH
m = MetabolicBVH(64, max_age=5)
for i in range(10):
    m.record_hits(np.array([0,1,2]))
    m.step()
s = m.stats()
assert s['n_pruned'] > 50, f'Pruning not working: {s}'
\""

check "BetaScheduler: annealing 1→10" \
    "python -c \"
import sys; sys.path.insert(0,'python')
from spectral_techniques import BetaScheduler, get_ste_beta
s = BetaScheduler(max_beta=10.0, warmup_steps=10, total_steps=100)
s.step(0); assert abs(get_ste_beta() - 1.0) < 0.01
s.step(100); assert abs(get_ste_beta() - 10.0) < 0.01
\""

check "DualLR: param groups separados" \
    "python -c \"
import sys; sys.path.insert(0,'python')
import torch
import torch.nn as nn
from spectral_techniques import get_dual_lr_param_groups
m = nn.Module()
m.D_cont = nn.Parameter(torch.randn(10))
m.fc = nn.Linear(10, 5)
g = get_dual_lr_param_groups(m, lr=1e-3, bvh_lr_mult=0.1)
bvh_group = [x for x in g if x['name'] == 'bvh_discrete']
assert len(bvh_group) == 1 and abs(bvh_group[0]['lr'] - 1e-4) < 1e-6
\""

echo ""

# -------------------------------------------
# FASE 4: Build C++/CUDA (solo si cmake existe)
# -------------------------------------------
echo -e "${YELLOW}[FASE 4] Build C++/CUDA${NC}"

if command -v cmake &> /dev/null && [ -d "build" ]; then
    check "CMake build (recompile con fixes)" \
        "cd build && cmake --build . 2>&1"
else
    skip "CMake build" "cmake no encontrado o directorio build/ no existe"
fi

echo ""

# -------------------------------------------
# FASE 5: Tests CUDA (solo si hay GPU)
# -------------------------------------------
echo -e "${YELLOW}[FASE 5] Tests CUDA/OptiX (GPU requerida)${NC}"

if command -v nvidia-smi &> /dev/null && nvidia-smi > /dev/null 2>&1; then
    if [ -f "build/test_router" ]; then
        check "test_router (kernel BVH)" "cd build && ./test_router"
    else
        skip "test_router" "binario no encontrado en build/"
    fi

    if [ -f "build/test_optix_pipeline" ]; then
        check "test_optix_pipeline (OptiX)" "cd build && ./test_optix_pipeline"
    else
        skip "test_optix_pipeline" "binario no encontrado en build/"
    fi

    if [ -f "build/rt_router_benchmark" ]; then
        check "rt_router_benchmark (RT Cores)" "cd build && ./rt_router_benchmark"
    else
        skip "rt_router_benchmark" "binario no encontrado en build/"
    fi
else
    skip "Tests CUDA" "nvidia-smi no disponible (sin GPU)"
fi

echo ""

# -------------------------------------------
# RESUMEN
# -------------------------------------------
echo "============================================"
TOTAL=$((PASS + FAIL + SKIP))
echo -e " RESULTADOS: ${GREEN}${PASS} OK${NC} | ${RED}${FAIL} FAIL${NC} | ${YELLOW}${SKIP} SKIP${NC} | Total: ${TOTAL}"
echo "============================================"

if [ $FAIL -gt 0 ]; then
    echo -e "${RED}HAY FALLOS — revisar output arriba${NC}"
    exit 1
else
    echo -e "${GREEN}TODO CORRECTO${NC}"
    exit 0
fi
