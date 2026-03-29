#!/usr/bin/env python3
"""
benchmark_scaling.py — Curva de escalado O(N log N) vs O(N²)

Testa múltiples tamaños de N: 8, 16, 32, 64, 128, 256, 512 tokens.
Objetivo: cruzar el punto donde OptiX destroza a cuBLAS.

Ejecuta:
    python benchmark_scaling.py --output scaling_results.json
"""

import sys
import os
import json
import struct
import subprocess
import tempfile
import argparse
import math
from pathlib import Path
from typing import List, Optional
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BUILD_DIR   = PROJECT_DIR / "build" / "Release"
BUILD_ROOT  = PROJECT_DIR / "build"
PTX_PATH    = BUILD_ROOT / "spectral_kernels.ptx"
BATCH_EXE   = BUILD_DIR / "batch_runner.exe"

from inference import (
    EmbeddingDB, embedding_to_fourier,
    pack_sphere, pack_string, pack_portal_identity,
    RESULT_SIZE,
)
from benchmark import (
    write_batch, read_batch_results, benchmark_cublas_attention,
    scene_from_sentence,
)

BATCH_MAGIC = 0x4C424243

# ─────────────────────────────────────────────────────────────────
# Generar frase de exactamente N tokens (con repetición si hace falta)
# ─────────────────────────────────────────────────────────────────

def generate_sentence_of_n(db: EmbeddingDB, n: int) -> str:
    """Genera una frase de exactamente N tokens del vocab.
    Filtra abreviaturas y tokens con puntuación para evitar OOV."""
    clean_vocab = [w for w in db.vocab[:min(len(db.vocab), 5000)]
                   if w.isalpha() and len(w) > 1]
    words = list(np.random.choice(clean_vocab, size=n, replace=True))
    return " ".join(words)

# ─────────────────────────────────────────────────────────────────
# Benchmark OptiX para una sola frase larga
# ─────────────────────────────────────────────────────────────────

def benchmark_optix_single(db: EmbeddingDB, sentence: str,
                            num_rays: int = 64, repeats: int = 5) -> dict:
    """
    Corre batch_runner con `repeats` copias de la misma escena.
    Retorna estadísticas de launch_ms (media, min, max).
    """
    sd = scene_from_sentence(db, sentence, num_rays)
    if sd is None:
        return {"error": "no tokens"}

    scenes = [sd] * repeats  # misma escena repetida para estadísticas

    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        batch_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        results_path = f.name

    try:
        write_batch(Path(batch_path), scenes, num_rays)

        result = subprocess.run(
            [str(BATCH_EXE), str(PTX_PATH), batch_path, results_path],
            capture_output=True, text=True, timeout=120
        )

        if result.returncode != 0:
            return {"error": result.stderr[:200]}

        batch_results = read_batch_results(Path(results_path))
        if not batch_results:
            return {"error": "no results"}

        # Descartar primera escena (warm-up dentro del batch_runner)
        # El batch_runner ya hace warmup interno, todas son válidas
        launch_times = [r["launch_ms"] for r in batch_results]
        build_times  = [r["build_ms"]  for r in batch_results]
        hits_list    = [r["hits"]       for r in batch_results]

        return {
            "launch_ms_mean": np.mean(launch_times),
            "launch_ms_min":  np.min(launch_times),
            "launch_ms_max":  np.max(launch_times),
            "build_ms_mean":  np.mean(build_times),
            "hits_mean":      np.mean(hits_list),
            "tokens":         len(db.tokenize(sentence)),
        }
    finally:
        try:
            os.unlink(batch_path)
            os.unlink(results_path)
        except OSError:
            pass

# ─────────────────────────────────────────────────────────────────
# Benchmark cuBLAS para N tokens (repetido)
# ─────────────────────────────────────────────────────────────────

def benchmark_cublas_single(db: EmbeddingDB, sentence: str,
                             repeats: int = 10) -> dict:
    """Corre cuBLAS `repeats` veces y retorna estadísticas."""
    try:
        import torch
    except ImportError:
        return {"error": "torch not installed"}

    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    tokens = db.tokenize(sentence)
    if not tokens:
        return {"error": "no tokens"}

    emb_arr = np.array([db.emb[tid] for tid, _ in tokens], dtype=np.float32)
    Q = torch.from_numpy(emb_arr).cuda()
    d = Q.shape[-1]

    # Warmup
    for _ in range(3):
        _ = torch.matmul(Q, Q.t())
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        scores = torch.matmul(Q, Q.t()) / (d ** 0.5)
        attn   = torch.nn.functional.softmax(scores, dim=-1)
        out    = torch.matmul(attn, Q)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    return {
        "time_ms_mean": np.mean(times),
        "time_ms_min":  np.min(times),
        "time_ms_max":  np.max(times),
        "tokens":       len(tokens),
    }

# ─────────────────────────────────────────────────────────────────
# Main: curva de escalado
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scaling benchmark: O(N log N) vs O(N^2)")
    parser.add_argument("--output",   type=str, default="scaling_results.json")
    parser.add_argument("--num-rays", type=int, default=64)
    parser.add_argument("--repeats",  type=int, default=5,
                        help="Repeticiones por punto N para estadisticas")
    parser.add_argument("--n-values", type=str,
                        default="4,8,16,32,64,128,256,512",
                        help="Valores de N (tokens) separados por coma")
    args = parser.parse_args()

    n_values = [int(x) for x in args.n_values.split(",")]

    print("[scaling] Cargando embeddings...")
    db = EmbeddingDB()

    # Warmup cuBLAS
    print("[scaling] Warmup cuBLAS + OptiX...")
    _ = benchmark_cublas_single(db, generate_sentence_of_n(db, 8), repeats=3)

    print(f"\n{'N':>5} | {'OptiX ms':>10} | {'cuBLAS ms':>10} | {'Speedup':>8} | "
          f"{'O(Nlog N)':>10} | {'O(N^2)':>8} | {'Hits':>6}")
    print(f"{'-'*5}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*6}")

    results = []

    for n in n_values:
        sentence = generate_sentence_of_n(db, n)
        actual_tokens = len(db.tokenize(sentence))

        # Medir OptiX
        optix = benchmark_optix_single(db, sentence,
                                        num_rays=args.num_rays,
                                        repeats=args.repeats)
        # Medir cuBLAS
        cublas = benchmark_cublas_single(db, sentence, repeats=args.repeats)

        if "error" in optix or "error" in cublas:
            print(f"{n:>5} | ERROR: {optix.get('error', '')} {cublas.get('error', '')}")
            continue

        optix_ms  = optix["launch_ms_mean"]
        cublas_ms = cublas["time_ms_mean"]
        speedup   = cublas_ms / optix_ms if optix_ms > 0 else 0

        # Valores teóricos normalizados al punto N=8
        n_log_n_ratio = (n * math.log2(n)) / (8 * math.log2(8))
        n2_ratio      = (n * n) / (8 * 8)

        hits = optix.get("hits_mean", 0)

        print(f"{actual_tokens:>5} | {optix_ms:>10.4f} | {cublas_ms:>10.4f} | "
              f"{speedup:>8.2f}x | {n_log_n_ratio:>10.1f}x | {n2_ratio:>8.1f}x | "
              f"{hits:>6.1f}")

        results.append({
            "n_requested":    n,
            "n_actual":       actual_tokens,
            "optix_ms":       optix_ms,
            "optix_ms_min":   optix["launch_ms_min"],
            "optix_ms_max":   optix["launch_ms_max"],
            "cublas_ms":      cublas_ms,
            "cublas_ms_min":  cublas["time_ms_min"],
            "cublas_ms_max":  cublas["time_ms_max"],
            "speedup":        speedup,
            "hits_mean":      hits,
            "num_rays":       args.num_rays,
            "n_log_n_growth": n_log_n_ratio,
            "n2_growth":      n2_ratio,
        })

    # Guardar
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[scaling] Resultados: {args.output}")

    # Resumen
    if results:
        print(f"\n{'='*75}")
        print("CURVA DE ESCALADO")
        print(f"{'='*75}")
        for r in results:
            bar_optix  = "#" * max(1, int(r["optix_ms"] * 200))
            bar_cublas = "#" * max(1, int(r["cublas_ms"] * 200))
            print(f"N={r['n_actual']:>4}  OptiX  [{bar_optix:<40}] {r['optix_ms']:.4f}ms")
            print(f"       cuBLAS [{bar_cublas:<40}] {r['cublas_ms']:.4f}ms  {r['speedup']:.1f}x")
            print()

if __name__ == "__main__":
    main()
