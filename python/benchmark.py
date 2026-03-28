#!/usr/bin/env python3
"""
benchmark.py — Comparativa OptiX (batch, sin reinit) vs cuBLAS attention

Ejecuta:
    python benchmark.py --num-sentences 50 --output results.json

Dos modos:
  1. batch_runner.exe   : OptiX con contexto vivo — mide latencia real de traversal
  2. PyTorch attention  : cuBLAS Q·K^T/sqrt(d) + softmax + V — baseline O(N²)

El overhead de inicializacion OptiX (~190ms) NO se incluye en los tiempos reportados.
"""

import sys
import os
import json
import csv
import time
import subprocess
import struct
import argparse
from pathlib import Path
from typing import List, Optional
import tempfile

import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BUILD_DIR   = PROJECT_DIR / "build" / "Release"
BUILD_ROOT  = PROJECT_DIR / "build"
PTX_PATH    = BUILD_ROOT / "liquidbit_kernels.ptx"
BATCH_EXE   = BUILD_DIR / "batch_runner.exe"
CUDA_BENCH  = BUILD_DIR / "liquidbit_benchmark_full.exe"

from inference import (
    EmbeddingDB, embedding_to_fourier,
    pack_sphere, pack_string, pack_portal_identity, write_scene,
    SPHERE_SIZE, RESONANCE_SIZE, STRING_SIZE, RESULT_SIZE,
    SCENE_MAGIC,
)

BATCH_MAGIC   = 0x4C424243   # 'LBBC'
BRESULT_MAGIC = 0x4C424252   # 'LBBR'

# ─────────────────────────────────────────────────────────────────
# Generación de frases
# ─────────────────────────────────────────────────────────────────

def generate_random_sentences(db: EmbeddingDB, num: int, max_words: int = 8) -> List[str]:
    # Filtra abreviaturas y tokens con puntuación para evitar OOV "ignorado"
    clean_vocab = [w for w in db.vocab[:min(500, len(db.vocab))]
                   if w.isalpha() and len(w) > 1]
    sentences = []
    for _ in range(num):
        n = np.random.randint(3, max_words + 1)
        sentences.append(" ".join(np.random.choice(clean_vocab, size=n, replace=True)))
    return sentences

# ─────────────────────────────────────────────────────────────────
# Construir datos de escena desde una frase
# ─────────────────────────────────────────────────────────────────

def scene_from_sentence(db: EmbeddingDB, sentence: str, num_rays: int = 16):
    """Retorna (spheres_data, strings_data, portals_data, omega, tokens) o None."""
    tokens = db.tokenize(sentence)
    if not tokens:
        return None

    raw_positions = [db.get_3d(tid) for tid, _ in tokens]
    dists = [np.linalg.norm(p) + 1e-6 for p in raw_positions]
    scale = 2.5 / max(np.mean(dists), 0.1)

    spheres_data, strings_data = [], []
    for i, (tid, word) in enumerate(tokens):
        pos3d = raw_positions[i] * scale
        spheres_data.append(pack_sphere(
            float(pos3d[0]), float(pos3d[1]), float(pos3d[2]),
            1.2, i, 3, float(i * 0.15 % (2 * np.pi))
        ))
        a, b = embedding_to_fourier(db.emb[tid])
        strings_data.append(pack_string(
            a, b, 8, 1.0, tid,
            float(pos3d[0]), float(pos3d[1]), float(pos3d[2]), i
        ))

    portals_data = [pack_portal_identity() for _ in range(4)]
    return spheres_data, strings_data, portals_data, 0.787, tokens

# ─────────────────────────────────────────────────────────────────
# Serialización de batch
# ─────────────────────────────────────────────────────────────────

def write_batch(path: Path, scenes: list, num_rays: int = 16):
    """
    Escribe un archivo batch con N escenas.
    scenes: lista de (spheres_data, strings_data, portals_data, omega, tokens)

    Formato batch_runner esperado:
      Header: magic(4) + version(4) + numScenes(4)
      Por escena: numSpheres(4) + numStrings(4) + numPortals(4) + numRays(4) + baseOmega(float4)
                  + SemanticSphere[] + SemanticString[] + AffinePortal[]
    """
    with open(path, "wb") as f:
        f.write(struct.pack("<III", BATCH_MAGIC, 1, len(scenes)))
        for spheres_data, strings_data, portals_data, omega, _tokens in scenes:
            f.write(struct.pack("<IIII", len(spheres_data), len(strings_data),
                                len(portals_data), num_rays))
            f.write(struct.pack("<f", omega))
            for s in spheres_data:
                f.write(s)
            for s in strings_data:
                f.write(s)
            for p in portals_data:
                f.write(p)

def read_batch_results(path: Path) -> List[dict]:
    """Lee batch_results.bin y retorna lista de {launch_ms, build_ms, hits, rays}."""
    results = []
    with open(path, "rb") as f:
        hdr = f.read(12)
        if len(hdr) < 12:
            return []
        magic, num_scenes, total_rays = struct.unpack("<III", hdr)
        if magic != BRESULT_MAGIC:
            print(f"[results] Magic incorrecto: 0x{magic:08X}", file=sys.stderr)
            return []
        for _ in range(num_scenes):
            meta = f.read(16)  # numRays + launch_ms + build_ms + hits
            if len(meta) < 16:
                break
            num_rays, launch_ms_raw, build_ms_raw, hits = struct.unpack("<IffI", meta)
            # Leer SpectralAttentionResult[] (los skipeamos para el benchmark)
            f.read(num_rays * RESULT_SIZE)
            results.append({
                "launch_ms": launch_ms_raw,
                "build_ms":  build_ms_raw,
                "hits":      hits,
                "rays":      num_rays,
            })
    return results

# ─────────────────────────────────────────────────────────────────
# Benchmark OptiX BATCH (latencia real sin overhead de init)
# ─────────────────────────────────────────────────────────────────

def benchmark_optix_batch(db: EmbeddingDB, sentences: List[str],
                          num_rays: int = 16) -> Optional[List[dict]]:
    """
    Procesa todas las frases de una vez con batch_runner.
    Retorna lista de {launch_ms, build_ms, hits} por escena.
    """
    if not BATCH_EXE.exists():
        print(f"[bench] batch_runner.exe no encontrado: {BATCH_EXE}")
        print("        Compile con: cmake --build build --target batch_runner")
        return None
    if not PTX_PATH.exists():
        print(f"[bench] PTX no encontrado: {PTX_PATH}")
        return None

    # Construir escenas
    scenes = []
    valid_indices = []
    for i, s in enumerate(sentences):
        sd = scene_from_sentence(db, s, num_rays)
        if sd is not None:
            scenes.append(sd)
            valid_indices.append(i)

    if not scenes:
        return None

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

        if result.stdout:
            print(result.stdout)
        if result.returncode != 0:
            print(f"[bench] batch_runner error: {result.stderr}", file=sys.stderr)
            return None

        batch_results = read_batch_results(Path(results_path))
        return batch_results, valid_indices

    except subprocess.TimeoutExpired:
        print("[bench] batch_runner timeout", file=sys.stderr)
        return None
    finally:
        try:
            os.unlink(batch_path)
            os.unlink(results_path)
        except OSError:
            pass

# ─────────────────────────────────────────────────────────────────
# Benchmark cuBLAS (PyTorch)
# ─────────────────────────────────────────────────────────────────

def benchmark_cublas_attention(db: EmbeddingDB, sentence: str,
                               warmup: bool = False) -> dict:
    """Mide tiempo de atención O(N²) con cuBLAS via PyTorch."""
    try:
        import torch
    except ImportError:
        return {"time_ms": 0.0, "tokens": 0, "error": "torch not installed"}

    if not torch.cuda.is_available():
        return {"time_ms": 0.0, "tokens": 0, "error": "CUDA not available"}

    tokens = db.tokenize(sentence)
    if not tokens:
        return {"time_ms": 0.0, "tokens": 0, "error": "no tokens"}

    emb_arr = np.array([db.emb[tid] for tid, _ in tokens], dtype=np.float32)
    Q = torch.from_numpy(emb_arr).cuda()
    d = Q.shape[-1]

    # Warmup: asegura que CUDA está caliente antes de medir
    if warmup:
        _ = torch.matmul(Q, Q.t())
        torch.cuda.synchronize()

    # Medir con CUDA events (tiempo GPU real)
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev   = torch.cuda.Event(enable_timing=True)

    start_ev.record()
    scores = torch.matmul(Q, Q.t()) / (d ** 0.5)
    attn   = torch.nn.functional.softmax(scores, dim=-1)
    output = torch.matmul(attn, Q)
    end_ev.record()
    torch.cuda.synchronize()

    return {
        "time_ms": start_ev.elapsed_time(end_ev),
        "tokens":  len(tokens),
    }

# ─────────────────────────────────────────────────────────────────
# Benchmark CUDA kernel (liquidbit_benchmark_full.exe)
# ─────────────────────────────────────────────────────────────────

def run_cuda_benchmark() -> Optional[str]:
    """Corre liquidbit_benchmark_full.exe y retorna la salida."""
    if not CUDA_BENCH.exists():
        return None
    try:
        result = subprocess.run(
            [str(CUDA_BENCH)],
            capture_output=True, text=True, timeout=120
        )
        return result.stdout + result.stderr
    except Exception as e:
        return str(e)

# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LiquidBit Benchmark: OptiX vs cuBLAS")
    parser.add_argument("--num-sentences", type=int, default=20)
    parser.add_argument("--num-rays",      type=int, default=16)
    parser.add_argument("--output",        type=str, default="benchmark_results.json")
    parser.add_argument("--cuda-bench",    action="store_true",
                        help="Tambien correr liquidbit_benchmark_full.exe")
    args = parser.parse_args()

    print("[bench] Cargando embeddings...")
    db = EmbeddingDB()

    print(f"[bench] Generando {args.num_sentences} frases...")
    sentences = generate_random_sentences(db, args.num_sentences)

    # ── cuBLAS warmup ────────────────────────────────────────────
    print("[bench] Warmup cuBLAS...")
    _ = benchmark_cublas_attention(db, sentences[0], warmup=True)

    # ── OptiX batch ──────────────────────────────────────────────
    print(f"[bench] Lanzando batch_runner con {len(sentences)} escenas...")
    batch_out = benchmark_optix_batch(db, sentences, num_rays=args.num_rays)

    optix_results = {}
    if batch_out is not None:
        br, valid_idx = batch_out
        for i, res in zip(valid_idx, br):
            optix_results[i] = res
    else:
        print("[bench] batch_runner no disponible — solo cuBLAS")

    # ── cuBLAS por frase ─────────────────────────────────────────
    print("[bench] Midiendo cuBLAS attention...")
    cublas_results = []
    for s in sentences:
        cublas_results.append(benchmark_cublas_attention(db, s))

    # ── Tabla comparativa ────────────────────────────────────────
    print(f"\n{'#':>3} | {'Tok':>3} | {'OptiX launch ms':>16} | {'cuBLAS ms':>10} | {'Speedup':>8} | {'Hits':>4}")
    print(f"{'-'*3}-+-{'-'*3}-+-{'-'*16}-+-{'-'*10}-+-{'-'*8}-+-{'-'*4}")

    records = []
    for i, sentence in enumerate(sentences):
        tokens      = len(db.tokenize(sentence))
        optix_ms    = optix_results.get(i, {}).get("launch_ms", 0.0)
        cublas_ms   = cublas_results[i].get("time_ms", 0.0)
        hits        = optix_results.get(i, {}).get("hits", 0)
        speedup     = cublas_ms / optix_ms if optix_ms > 0 else 0.0

        records.append({
            "id": i, "sentence": sentence, "tokens": tokens,
            "optix_launch_ms": optix_ms,
            "optix_build_ms":  optix_results.get(i, {}).get("build_ms", 0.0),
            "cublas_ms": cublas_ms,
            "speedup": speedup, "hits": hits, "rays": args.num_rays,
            "cublas_error": cublas_results[i].get("error"),
        })

        print(f"{i+1:>3} | {tokens:>3} | {optix_ms:>16.4f} | {cublas_ms:>10.4f} | {speedup:>8.2f}x | {hits:>4}")

    # ── Resumen ──────────────────────────────────────────────────
    valid = [r for r in records if r["optix_launch_ms"] > 0 and r["cublas_ms"] > 0]
    if valid:
        import statistics
        mean_optix   = statistics.mean(r["optix_launch_ms"] for r in valid)
        mean_cublas  = statistics.mean(r["cublas_ms"]       for r in valid)
        mean_speedup = statistics.mean(r["speedup"]         for r in valid)
        mean_hits    = statistics.mean(r["hits"]            for r in valid)

        print(f"\n{'='*65}")
        print(f"RESUMEN ({len(valid)}/{len(records)} validos)")
        print(f"{'='*65}")
        print(f"OptiX launch ms medio:  {mean_optix:.4f} ms  (GPU time real, sin init)")
        print(f"cuBLAS attention medio: {mean_cublas:.4f} ms")
        print(f"Speedup medio:          {mean_speedup:.2f}x")
        print(f"Hits medio:             {mean_hits:.1f}/{args.num_rays}")
        if mean_speedup >= 1.0:
            print(f"\n=> OptiX es {mean_speedup:.1f}x MAS RAPIDO que cuBLAS")
        else:
            print(f"\n=> cuBLAS es {1/mean_speedup:.1f}x mas rapido (esperado: N pequeno)")
        print(f"   NOTA: ventaja de OptiX crece con N (O(N log N) vs O(N^2))")

    # ── CUDA benchmark completo (opcional) ───────────────────────
    if args.cuda_bench:
        print(f"\n{'='*65}")
        print("liquidbit_benchmark_full.exe:")
        print(f"{'='*65}")
        out = run_cuda_benchmark()
        if out:
            print(out)
        else:
            print(f"[No encontrado: {CUDA_BENCH}]")

    # ── Guardar resultados ───────────────────────────────────────
    output_path = Path(args.output) if Path(args.output).is_absolute() else Path(args.output)
    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n[bench] Resultados guardados: {output_path}")


if __name__ == "__main__":
    main()
