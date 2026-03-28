#!/usr/bin/env python3
"""
ternary_quantize.py — Cuantización post-training de coeficientes Fourier a ternario

OBJETIVO:
=========
Convertir los coeficientes Fourier a[], b[] de las SemanticStrings de FP32
a representación ternaria {-1, 0, +1}, siguiendo el paradigma BitNet b1.58.

MOTIVACIÓN (Idea 5 de LiquidBit v4.0 "Inception Engine"):
===========================================================
  Phase 1 (FP16/FP32):   out = Σ W[i] · x[i]      ← multiplicaciones FP
  Phase 2 (Ternario):     out = Σ { +x[i] si W[i]=+1
                               {  0   si W[i]= 0
                               { -x[i] si W[i]=-1   ← solo suma/resta

  Beneficios:
    - RAM: ~10x menor (1.58 bits vs 32 bits por parámetro)
    - Cómputo: eliminación de FMAs (Fused Multiply-Add)
    - Calor: mínimo — no usa Tensor Cores
    - VRAM: cabe en L1/L2 cache de los RT Cores

ALGORITMO:
==========
1. Calibración: calcular la distribución de los coeficientes en datos reales
2. Thresholding: determinar umbral óptimo τ que minimiza la degradación
3. Cuantización: a[k] → round_to_ternary(a[k], τ)
4. Validación: medir degradación de la función W(ω) vs FP32 original

FORMATO DE SALIDA PARA C++:
============================
  - Array int8_t [K × 2M] donde {-1, 0, +1} caben en 1 byte
  - Header JSON con metadata: num_strings, num_modes, threshold
  - Compatible con ternaryStringResonance() en CUDA

USO:
====
  python ternary_quantize.py                         # usa fourier_coeffs.npy
  python ternary_quantize.py --input my_coeffs.npy   # coeficientes custom
  python ternary_quantize.py --threshold 0.05        # umbral manual
  python ternary_quantize.py --benchmark              # medir degradación

@author LiquidBit Zero-Matrix Team
@date 2026
"""

import sys
import os
import json
import argparse
import numpy as np
from typing import Optional, Tuple, Dict

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Número de modos Fourier (debe coincidir con RESONANCE_NUM_MODES en C++)
RESONANCE_NUM_MODES = 8

# ─────────────────────────────────────────────────────────────────────────────
# Core: cuantización ternaria
# ─────────────────────────────────────────────────────────────────────────────

def quantize_to_ternary(
    params: np.ndarray,
    threshold: Optional[float] = None,
    percentile: float = 20.0,
) -> Tuple[np.ndarray, float]:
    """
    Cuantiza un array de coeficientes FP32 a ternario {-1, 0, +1}.

    Algoritmo:
      τ = threshold o percentile(|params|, percentile%)
      result[i] = +1 si params[i] >  τ
                    0 si |params[i]| <= τ  (zona de silencio)
                   -1 si params[i] < -τ

    El umbral τ se puede elegir automáticamente como el percentil de los
    valores absolutos — valores pequeños (< τ) se redondean a 0.

    Args:
        params:      Array de coeficientes FP32
        threshold:   Umbral manual. Si None, se calcula desde percentil.
        percentile:  Percentil de |params| a usar como umbral automático.

    Returns:
        (ternary_params, threshold_used)
    """
    if threshold is None:
        # Umbral automático: percentil de los valores absolutos
        threshold = float(np.percentile(np.abs(params), percentile))
        # Umbral mínimo para evitar cuantizar todo a 0
        threshold = max(threshold, 1e-6)

    result = np.zeros_like(params, dtype=np.int8)
    result[params >  threshold] = +1
    result[params < -threshold] = -1

    return result, threshold


def ternary_sparsity(ternary: np.ndarray) -> float:
    """Fracción de ceros en el array ternario (sparsidad)."""
    return float(np.mean(ternary == 0))


def compute_scale_factor(fp32: np.ndarray, ternary: np.ndarray) -> float:
    """
    Calcula el factor de escala óptimo para minimizar |fp32 - scale * ternary|².

    Para usar en inferencia: W_ternary_scaled = scale * ternary
    """
    nonzero = ternary != 0
    if not nonzero.any():
        return 1.0
    # Mínimos cuadrados: scale = mean(fp32[nonzero] * ternary[nonzero]) / mean(ternary[nonzero]^2)
    numerator   = np.mean(fp32[nonzero] * ternary[nonzero].astype(np.float32))
    denominator = np.mean(ternary[nonzero].astype(np.float32) ** 2)
    return float(numerator / denominator) if denominator > 0 else 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Resonancia Fourier (para validación de calidad)
# ─────────────────────────────────────────────────────────────────────────────

def resonance_fp32(a: np.ndarray, b: np.ndarray, omega: float, scale: float = 1.0) -> float:
    """
    W(ω) = scale · tanh(Σ a_k·sin(kω) + b_k·cos(kω))  — versión FP32
    """
    M = len(a)
    total = sum(a[k] * np.sin((k + 1) * omega) + b[k] * np.cos((k + 1) * omega)
                for k in range(M))
    return scale * np.tanh(total)


def resonance_ternary(
    a_t: np.ndarray, b_t: np.ndarray,
    omega: float,
    scale_a: float = 1.0, scale_b: float = 1.0,
    output_scale: float = 1.0,
) -> float:
    """
    W(ω) = output_scale · tanh(Σ {±s_a·sin(kω) si a_t[k]!=0} + {±s_b·cos(kω) si b_t[k]!=0})
    Versión ternaria — solo sumas y restas (sin multiplicación FP).
    """
    M = len(a_t)
    total = 0.0
    for k in range(M):
        if a_t[k] == +1:
            total += scale_a * np.sin((k + 1) * omega)
        elif a_t[k] == -1:
            total -= scale_a * np.sin((k + 1) * omega)
        if b_t[k] == +1:
            total += scale_b * np.cos((k + 1) * omega)
        elif b_t[k] == -1:
            total -= scale_b * np.cos((k + 1) * omega)
    return output_scale * np.tanh(total)


def measure_degradation(
    fp32_coeffs: np.ndarray,   # [K, 2M]
    ternary_coeffs: np.ndarray, # [K, 2M] int8
    scales: np.ndarray,         # [K] factores de escala
    n_omega_points: int = 100,
) -> Dict[str, float]:
    """
    Mide la degradación de la función W(ω) entre FP32 y ternario.

    Evalúa en n_omega_points puntos equiespaciados en [0, 2π] para cada
    SemanticString y calcula el error relativo promedio.

    Returns:
        dict con métricas: mean_relative_error, max_relative_error, mean_mse
    """
    K   = fp32_coeffs.shape[0]
    M   = fp32_coeffs.shape[1] // 2
    omegas = np.linspace(0, 2 * np.pi, n_omega_points)

    errors_rel = []
    errors_mse = []

    for k in range(K):
        a_fp  = fp32_coeffs[k, :M]
        b_fp  = fp32_coeffs[k, M:]
        a_ter = ternary_coeffs[k, :M]
        b_ter = ternary_coeffs[k, M:]
        scale_k = float(scales[k]) if scales is not None else 1.0

        fp32_vals = np.array([resonance_fp32(a_fp, b_fp, w) for w in omegas])
        ter_vals  = np.array([resonance_ternary(a_ter, b_ter, w,
                                                scale_a=scale_k, scale_b=scale_k)
                              for w in omegas])

        # MSE
        mse = np.mean((fp32_vals - ter_vals) ** 2)
        errors_mse.append(mse)

        # Error relativo (evitar división por cero)
        range_fp32 = np.max(np.abs(fp32_vals)) + 1e-8
        rel_error  = np.sqrt(mse) / range_fp32
        errors_rel.append(rel_error)

    return {
        "mean_relative_error": float(np.mean(errors_rel)),
        "max_relative_error":  float(np.max(errors_rel)),
        "mean_mse":            float(np.mean(errors_mse)),
        "target":              "<2% degradation (mean_relative_error < 0.02)",
        "pass":                float(np.mean(errors_rel)) < 0.02,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Búsqueda de umbral óptimo
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_threshold(
    fp32_coeffs: np.ndarray,
    max_degradation: float = 0.02,
    n_candidates: int = 20,
) -> Tuple[float, Dict]:
    """
    Busca el umbral τ máximo que mantiene degradación < max_degradation.

    Un τ mayor → más ceros (mayor compresión) → posiblemente más degradación.
    Objetivo: máxima compresión con degradación < 2%.

    Returns:
        (threshold_optimal, metrics_at_optimal)
    """
    # Candidatos: de 0 a percentil 80 de |coeffs|
    max_tau = np.percentile(np.abs(fp32_coeffs), 80)
    thresholds = np.linspace(0.001, max_tau, n_candidates)

    best_tau = 0.001
    best_metrics = None

    for tau in thresholds:
        ternary, _ = quantize_to_ternary(fp32_coeffs, threshold=tau)
        K = fp32_coeffs.shape[0]

        # Calcular escalas por string
        scales = np.array([
            compute_scale_factor(fp32_coeffs[k], ternary[k])
            for k in range(K)
        ])

        metrics = measure_degradation(fp32_coeffs, ternary, scales, n_omega_points=50)
        metrics["threshold"]  = float(tau)
        metrics["sparsity"]   = ternary_sparsity(ternary)

        if metrics["mean_relative_error"] < max_degradation:
            best_tau     = tau
            best_metrics = metrics

    if best_metrics is None:
        # Ningún umbral es aceptable — usar el mínimo
        best_tau = 0.001
        ternary, _ = quantize_to_ternary(fp32_coeffs, threshold=best_tau)
        scales = np.ones(fp32_coeffs.shape[0])
        best_metrics = measure_degradation(fp32_coeffs, ternary, scales)
        best_metrics["threshold"] = best_tau
        best_metrics["sparsity"]  = ternary_sparsity(ternary)
        best_metrics["warning"]   = "No threshold met quality target — using minimum"

    return best_tau, best_metrics


# ─────────────────────────────────────────────────────────────────────────────
# Export para C++
# ─────────────────────────────────────────────────────────────────────────────

def export_for_cpp(
    ternary_coeffs: np.ndarray,  # [K, 2M] int8
    scales: np.ndarray,          # [K] float32
    output_scales: np.ndarray,   # [K] float32 (outputScale del nodo)
    output_dir: str,
    metadata: Dict,
) -> None:
    """
    Exporta los coeficientes ternarios en formato listo para C++/CUDA.

    Archivos creados:
      ternary_coeffs.npy    — int8 [K, 2M]
      ternary_scales.npy    — float32 [K] (scale_a = scale_b por string)
      ternary_output_scales.npy — float32 [K] (outputScale de SemanticString)
      ternary_metadata.json — num_strings, num_modes, threshold, sparsity, etc.

    Formato en C++ (ternary_resonance.cu):
      int8_t ternary_params[NUM_STRINGS][2 * RESONANCE_NUM_MODES]
      float  ternary_scales[NUM_STRINGS]
    """
    np.save(os.path.join(output_dir, "ternary_coeffs.npy"), ternary_coeffs)
    np.save(os.path.join(output_dir, "ternary_scales.npy"), scales.astype(np.float32))
    np.save(os.path.join(output_dir, "ternary_output_scales.npy"), output_scales.astype(np.float32))

    with open(os.path.join(output_dir, "ternary_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[export] ternary_coeffs.npy:        {ternary_coeffs.shape} int8")
    print(f"[export] ternary_scales.npy:        {scales.shape} float32")
    print(f"[export] ternary_output_scales.npy: {output_scales.shape} float32")
    print(f"[export] ternary_metadata.json:     threshold={metadata.get('threshold', '?'):.4f}, "
          f"sparsity={metadata.get('sparsity', 0):.1%}")


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    print("=" * 60)
    print(" LiquidBit v4.0 — Ternary Quantization (Phase 2 upgrade)")
    print("=" * 60)

    # ── 1. Cargar coeficientes FP32 ──────────────────────────────────────
    input_path = args.input or os.path.join(SCRIPT_DIR, "fourier_coeffs.npy")
    if not os.path.exists(input_path):
        print(f"[ERROR] {input_path} no encontrado.")
        print("        Ejecutar primero: python train_liquidbit.py")
        sys.exit(1)

    fp32_coeffs = np.load(input_path).astype(np.float32)
    K, TwoM = fp32_coeffs.shape
    M = TwoM // 2
    print(f"[load] {K} SemanticStrings × {M} modos (={TwoM} coeficientes)")
    print(f"[load] FP32 memory: {fp32_coeffs.nbytes / 1024:.1f} KB")

    # ── 2. Determinar umbral ─────────────────────────────────────────────
    if args.threshold is not None:
        print(f"\n[quant] Usando umbral manual τ={args.threshold}")
        threshold = args.threshold
        ternary_coeffs, _ = quantize_to_ternary(fp32_coeffs, threshold=threshold)
        scales = np.array([
            compute_scale_factor(fp32_coeffs[k], ternary_coeffs[k])
            for k in range(K)
        ])
    else:
        print(f"\n[quant] Buscando umbral óptimo (max_degradation={args.max_degradation:.1%})...")
        threshold, metrics = find_optimal_threshold(
            fp32_coeffs,
            max_degradation=args.max_degradation,
        )
        ternary_coeffs, _ = quantize_to_ternary(fp32_coeffs, threshold=threshold)
        scales = np.array([
            compute_scale_factor(fp32_coeffs[k], ternary_coeffs[k])
            for k in range(K)
        ])
        print(f"[quant] Umbral óptimo: τ={threshold:.4f}")
        print(f"[quant] Sparsidad:     {ternary_sparsity(ternary_coeffs):.1%} de ceros")

    # ── 3. Estadísticas ──────────────────────────────────────────────────
    fp32_bytes    = fp32_coeffs.nbytes
    ternary_bytes = ternary_coeffs.nbytes  # int8 = 1 byte vs 4 bytes FP32
    sparsity      = ternary_sparsity(ternary_coeffs)

    print(f"\n[stats] Distribución ternaria:")
    print(f"  +1: {np.mean(ternary_coeffs == +1):.1%}")
    print(f"   0: {np.mean(ternary_coeffs ==  0):.1%}  (silencio — sin operación)")
    print(f"  -1: {np.mean(ternary_coeffs == -1):.1%}")
    print(f"  Memory: {fp32_bytes/1024:.1f} KB (FP32) → {ternary_bytes/1024:.1f} KB (int8)")
    print(f"  Compression: {fp32_bytes/ternary_bytes:.1f}x raw, "
          f"~{fp32_bytes/(ternary_bytes*(1-sparsity)+1e-6):.1f}x efectivo (sparsidad)")

    # ── 4. Benchmark de calidad ──────────────────────────────────────────
    if args.benchmark or args.threshold is None:
        print(f"\n[bench] Midiendo degradación vs FP32...")
        metrics = measure_degradation(fp32_coeffs, ternary_coeffs, scales)
        print(f"  Mean relative error:  {metrics['mean_relative_error']:.4f} "
              f"({'PASS' if metrics['pass'] else 'WARN'})")
        print(f"  Max relative error:   {metrics['max_relative_error']:.4f}")
        print(f"  Mean MSE:             {metrics['mean_mse']:.6f}")
        print(f"  Target:               {metrics['target']}")
    else:
        metrics = {"threshold": threshold, "sparsity": sparsity}

    # ── 5. Exportar ──────────────────────────────────────────────────────
    output_scales = np.ones(K, dtype=np.float32)  # por defecto 1.0

    metadata = {
        "num_strings":      K,
        "num_modes":        M,
        "threshold":        float(threshold),
        "sparsity":         float(sparsity),
        "fp32_memory_kb":   float(fp32_bytes / 1024),
        "int8_memory_kb":   float(ternary_bytes / 1024),
        "compression_x":    float(fp32_bytes / ternary_bytes),
        **metrics,
        "cpp_dtype":        "int8_t",
        "cpp_array":        f"int8_t ternary_params[{K}][{TwoM}]",
        "cuda_kernel":      "ternaryStringResonance() in cuda/ternary_resonance.cu",
    }

    export_for_cpp(ternary_coeffs, scales, output_scales, SCRIPT_DIR, metadata)

    print("\n[OK] Cuantización ternaria completada.")
    print(f"     Siguiente paso: compilar cuda/ternary_resonance.cu con las nuevas tablas")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="LiquidBit v4.0 — Ternary Quantization")
    p.add_argument("--input",            type=str,   default=None,
                   help="Ruta al .npy de coeficientes FP32 (default: fourier_coeffs.npy)")
    p.add_argument("--threshold",        type=float, default=None,
                   help="Umbral de cuantización manual τ (default: automático)")
    p.add_argument("--max-degradation",  type=float, default=0.02,
                   help="Máxima degradación admisible (default: 0.02 = 2%%)")
    p.add_argument("--benchmark",        action="store_true",
                   help="Medir degradación vs FP32 original")
    p.add_argument("--demo",             action="store_true",
                   help="Usar datos sintéticos para demo rápida")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.demo:
        # Demo con coeficientes sintéticos
        print("[demo] Generando coeficientes sintéticos...")
        np.random.seed(42)
        K, M = 8, 8
        fp32_demo = np.random.randn(K, 2 * M).astype(np.float32) * 0.3

        # Guardar temporalmente para que main() los encuentre
        demo_path = os.path.join(SCRIPT_DIR, "fourier_coeffs.npy")
        np.save(demo_path, fp32_demo)
        args.input = demo_path
        args.benchmark = True

    main(args)
