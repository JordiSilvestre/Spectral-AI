#!/bin/bash
# ============================================================
# SpectralAI Zero-Matrix — Patent Filing Benchmark Suite
# ============================================================
#
# Runs ALL benchmarks needed to produce fresh, reproducible
# numbers for the three SpectralAI patent applications:
#
#   P1: RT Core Attention Mechanism
#       - BVH routing latency vs PyTorch/CUDA
#       - VRAM savings vs Transformer KV Cache
#       - Orchestrator E2E vs GPT-2 baseline
#
#   P2: Inception Engine (Ternary Experts + Multi-Level BVH)
#       - FP16 / INT8 / Ternary expert comparison
#       - Scaling curves: O(N log N) vs O(N^2)
#       - 16-layer PPL evaluation
#
#   P3: Spectral Routing (Snell's Law for Polysemy)
#       - Spectral ray routing latency (benchmark_e2e_final)
#       - Real model demo (Qwen 1.5B through full pipeline)
#
# Output: results/patent_benchmark_YYYYMMDD.log
#
# Prerequisites:
#   - Python venv activated (with torch, transformers, etc.)
#   - OLMoE model available at MODEL_DIR
#   - CUDA GPU available
#
# Usage (WSL2):
#   cd /mnt/j/Proyectos/SPECTRAL\ AI
#   source .venv_wsl/bin/activate
#   bash scripts/patent_benchmarks.sh
# ============================================================

set -eo pipefail

# ── Auto-detect python command ──────────────────────────────
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: No python3 or python found in PATH"
    exit 1
fi

# ── Configuration ───────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_DIR="${PROJECT_DIR}/python"
SCRIPTS_DIR="${PROJECT_DIR}/scripts"
RESULTS_DIR="${PROJECT_DIR}/results"
TIMESTAMP="$(date +%Y%m%d)"
LOGFILE="${RESULTS_DIR}/patent_benchmark_${TIMESTAMP}.log"
MODEL_DIR="${MODEL_DIR:-/mnt/j/Proyectos/models/olmoe-1b-7b}"
DEVICE="${DEVICE:-cuda}"

# How many seconds to allow per benchmark (default: 30 minutes)
TIMEOUT_SECS="${TIMEOUT_SECS:-1800}"

mkdir -p "${RESULTS_DIR}"

# ── Utilities ───────────────────────────────────────────────

separator() {
    local char="${1:-=}"
    printf '%0.s'"${char}" {1..72}
    echo
}

header() {
    echo
    separator "="
    echo "  $1"
    separator "="
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo
}

vram_snapshot() {
    local label="$1"
    if command -v nvidia-smi &>/dev/null; then
        echo "--- VRAM snapshot [${label}] $(date '+%H:%M:%S') ---"
        nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu \
            --format=csv,noheader,nounits \
        | while IFS=',' read -r idx name used free total util; do
            printf "  GPU %s (%s): %s MB used / %s MB total (%.1f%% util)\n" \
                "$idx" "$(echo "$name" | xargs)" \
                "$(echo "$used" | xargs)" "$(echo "$total" | xargs)" \
                "$(echo "$util" | xargs)"
        done
        echo
    else
        echo "--- VRAM snapshot [${label}]: nvidia-smi not available ---"
        echo
    fi
}

run_benchmark() {
    # Usage: run_benchmark "Label" python_script.py [args...]
    local label="$1"
    shift
    local script="$1"
    shift

    header "${label}"
    vram_snapshot "before: ${label}"

    local rc=0
    timeout "${TIMEOUT_SECS}" "${PY}" "${script}" "$@" || rc=$?

    vram_snapshot "after: ${label}"

    if [ "$rc" -eq 0 ]; then
        echo "  >>> ${label}: PASSED (exit 0)"
    elif [ "$rc" -eq 124 ]; then
        echo "  >>> ${label}: TIMEOUT after ${TIMEOUT_SECS}s"
    else
        echo "  >>> ${label}: FAILED (exit ${rc})"
    fi

    echo
    return 0  # Don't abort the suite on individual failures
}

# ── Pre-flight checks ──────────────────────────────────────

preflight() {
    header "PRE-FLIGHT CHECKS"

    echo "  Python:      ${PY} ($(${PY} --version 2>&1))"
    echo "  Project dir: ${PROJECT_DIR}"
    echo "  Model dir:   ${MODEL_DIR}"
    echo "  Device:      ${DEVICE}"
    echo "  Timestamp:   ${TIMESTAMP}"
    echo "  Log file:    ${LOGFILE}"
    echo

    # Check GPU
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU detected:"
        nvidia-smi --query-gpu=index,name,driver_version,memory.total \
            --format=csv,noheader 2>/dev/null \
        | while IFS=',' read -r idx name driver mem; do
            printf "    GPU %s: %s | Driver %s | %s MB\n" \
                "$idx" "$(echo "$name" | xargs)" \
                "$(echo "$driver" | xargs)" "$(echo "$mem" | xargs)"
        done
        echo
    else
        echo "  WARNING: nvidia-smi not found. GPU benchmarks may fail."
        echo
    fi

    # Check key Python imports
    echo "  Checking Python dependencies..."
    local deps_ok=true
    for pkg in torch numpy; do
        if ${PY} -c "import ${pkg}" 2>/dev/null; then
            local ver
            ver=$(${PY} -c "import ${pkg}; print(${pkg}.__version__)" 2>/dev/null)
            echo "    ${pkg}: ${ver}"
        else
            echo "    ${pkg}: MISSING"
            deps_ok=false
        fi
    done

    # Check CUDA availability
    local cuda_ok
    cuda_ok=$(${PY} -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
    echo "    torch.cuda: ${cuda_ok}"

    if [ "${cuda_ok}" = "True" ]; then
        ${PY} -c "import torch; print(f'    CUDA device: {torch.cuda.get_device_name(0)}')" 2>/dev/null || true
    fi

    # Check OLMoE model directory
    if [ -d "${MODEL_DIR}" ]; then
        echo "    OLMoE model: FOUND at ${MODEL_DIR}"
    else
        echo "    OLMoE model: NOT FOUND at ${MODEL_DIR}"
        echo "    (PPL evaluation will be skipped)"
    fi

    echo
    vram_snapshot "baseline (before any benchmarks)"

    if [ "${deps_ok}" = false ]; then
        echo "ERROR: Missing critical dependencies. Install them and retry."
        exit 1
    fi
}

# ============================================================
# PATENT P1: RT Core Attention Mechanism
# ============================================================

run_p1_benchmarks() {
    separator "#"
    echo "  PATENT P1: RT Core Attention Mechanism"
    echo "  (Routing speedup, VRAM savings, Orchestrator E2E)"
    separator "#"

    # P1-A: Routing latency — PyTorch vs CUDA Extension
    run_benchmark \
        "P1-A: Routing Latency (PyTorch vs CUDA Ext)" \
        "${PYTHON_DIR}/benchmark_e2e_final.py"

    # P1-B: CUDA kernel integrated into Orchestrator
    run_benchmark \
        "P1-B: Orchestrator E2E (CUDA Kernel vs GPT-2)" \
        "${PYTHON_DIR}/benchmark_cuda_e2e.py" \
        --device "${DEVICE}"
}

# ============================================================
# PATENT P2: Inception Engine (Ternary + Multi-Level BVH)
# ============================================================

run_p2_benchmarks() {
    separator "#"
    echo "  PATENT P2: Inception Engine"
    echo "  (Ternary experts, scaling curves, PPL evaluation)"
    separator "#"

    # P2-A: Expert quantization comparison (FP16/INT8/Ternary)
    run_benchmark \
        "P2-A: Expert Types (FP32/FP16/INT8/Ternary)" \
        "${PYTHON_DIR}/benchmark_expert_types.py"

    # P2-B: Scaling curves O(N log N) vs O(N^2)
    run_benchmark \
        "P2-B: Scaling Curves (Inception vs cuBLAS vs Flash)" \
        "${PYTHON_DIR}/scaling_inception.py" \
        --mode analytical \
        --output "${RESULTS_DIR}/scaling_inception_${TIMESTAMP}.json"

    # P2-C: 16-layer PPL evaluation (requires OLMoE model)
    if [ -d "${MODEL_DIR}" ]; then
        run_benchmark \
            "P2-C: PPL Evaluation (16/16 layers)" \
            "${SCRIPTS_DIR}/eval_all_16_layers.py" \
            --model-dir "${MODEL_DIR}" \
            --device "${DEVICE}" \
            --max-tokens 50000
    else
        echo
        echo "  SKIP P2-C: OLMoE model not found at ${MODEL_DIR}"
        echo "  Set MODEL_DIR to run PPL evaluation."
        echo
    fi
}

# ============================================================
# PATENT P3: Spectral Routing (Snell's Law + Polysemy)
# ============================================================

run_p3_benchmarks() {
    separator "#"
    echo "  PATENT P3: Spectral Routing (Snell's Law)"
    echo "  (Spectral ray latency, real model demo)"
    separator "#"

    # P3-A: Full E2E with spectral routing (same as P1-A but
    #        this benchmark specifically exercises Snell refraction
    #        via the spectral_dim=64 codepath in BVHRouter)
    #        Already run in P1-A; reference those numbers.
    echo
    echo "  NOTE: P3-A spectral routing latency was captured in P1-A."
    echo "  The benchmark_e2e_final.py exercises the Snell refraction"
    echo "  codepath (spectral_dim=64, snell_w/snell_b per BVH node)."
    echo

    # P3-B: Real model demo — Qwen 1.5B through full pipeline
    run_benchmark \
        "P3-B: Real Model Demo (Qwen 1.5B)" \
        "${PYTHON_DIR}/real_model_demo.py" \
        --model qwen-1.5b \
        --max-tokens 64 \
        --device "${DEVICE}"
}

# ============================================================
# Summary Table
# ============================================================

print_summary() {
    header "PATENT BENCHMARK SUMMARY"

    echo "  Date:       $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Log file:   ${LOGFILE}"
    echo

    # Extract key numbers from the log if possible
    echo "  Key metrics to extract from the log above:"
    echo
    printf "  %-45s | %-20s | %-8s\n" "Benchmark" "Key Metric" "Patent"
    separator "-"
    printf "  %-45s | %-20s | %-8s\n" \
        "P1-A: BVH Routing (batch=256)" "Latency (us), Speedup" "P1" \
        "P1-B: Orchestrator vs GPT-2" "Throughput (tok/s)" "P1" \
        "P2-A: Ternary vs FP16 experts" "VRAM (MB), tok/s, PPL" "P2" \
        "P2-B: Scaling N=8..128K" "Crossover N, Speedup" "P2" \
        "P2-C: 16-layer PPL (OLMoE)" "PPL (baseline vs BVH)" "P2" \
        "P3-B: Qwen 1.5B real demo" "tok/s, VRAM (MB)" "P3"
    echo

    # Auto-extract specific numbers if they appear in the log
    echo "  Auto-extracted highlights (if available):"
    separator "-"

    # Routing speedup
    if grep -q "Speedup\|speedup\|mas rapido" "${LOGFILE}" 2>/dev/null; then
        echo "  Routing speedups:"
        grep -i "speedup\|mas rapido" "${LOGFILE}" | head -10 | sed 's/^/    /'
        echo
    fi

    # PPL numbers
    if grep -q "PPL\|ppl\|perplexity" "${LOGFILE}" 2>/dev/null; then
        echo "  Perplexity results:"
        grep -i "PPL\|perplexity" "${LOGFILE}" | grep -i "[0-9]" | head -10 | sed 's/^/    /'
        echo
    fi

    # VRAM numbers
    if grep -q "VRAM\|vram\|memory" "${LOGFILE}" 2>/dev/null; then
        echo "  VRAM / Memory:"
        grep -i "VRAM\|active.*MB\|memory.*MB" "${LOGFILE}" | head -10 | sed 's/^/    /'
        echo
    fi

    # Throughput
    if grep -q "tok/s\|tokens/s\|throughput" "${LOGFILE}" 2>/dev/null; then
        echo "  Throughput:"
        grep -i "tok/s\|tokens/s\|throughput" "${LOGFILE}" | head -10 | sed 's/^/    /'
        echo
    fi

    vram_snapshot "final (after all benchmarks)"

    separator "="
    echo "  ALL PATENT BENCHMARKS COMPLETE"
    echo "  Full log: ${LOGFILE}"
    separator "="
}

# ============================================================
# Main
# ============================================================

main() {
    preflight

    local wall_start
    wall_start=$(date +%s)

    run_p1_benchmarks
    run_p2_benchmarks
    run_p3_benchmarks

    local wall_end
    wall_end=$(date +%s)
    local wall_elapsed=$(( wall_end - wall_start ))
    local wall_min=$(( wall_elapsed / 60 ))
    local wall_sec=$(( wall_elapsed % 60 ))
    echo
    echo "  Total wall time: ${wall_min}m ${wall_sec}s"

    print_summary
}

# Tee all output to both console and log file
main 2>&1 | tee "${LOGFILE}"
