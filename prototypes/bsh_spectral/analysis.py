#!/usr/bin/env python3
"""
Módulo de análisis y post-procesamiento de resultados del Prototipo A.
Calcula métricas, verifica propiedades teóricas y genera reportes.
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class ComplexityResult:
    """Resultado de análisis de complejidad."""
    vocab_size: int
    nodes_visited: float
    traversal_time_ms: float
    log_n: float
    ratio_to_log_n: float
    is_valid_ologn: bool


def verify_ologn_complexity(vocab_sizes: List[int],
                            nodes_visited_list: List[float]) -> List[ComplexityResult]:
    """
    Verifica que la complejidad de traversal es O(log N).

    Args:
        vocab_sizes: Lista de tamaños de vocabulario probados
        nodes_visited_list: Lista de promedio de nodos visitados

    Returns:
        Lista de ComplexityResult con análisis detallado
    """
    results = []

    for vocab_size, nodes_visited in zip(vocab_sizes, nodes_visited_list):
        log_n = np.log2(vocab_size) if vocab_size > 1 else 0.0
        ratio = nodes_visited / (log_n + 1e-8)

        # O(log N) válido si: 0.5 * log(N) ≤ nodos ≤ 2.0 * log(N)
        is_valid = 0.5 <= ratio <= 2.0

        result = ComplexityResult(
            vocab_size=vocab_size,
            nodes_visited=nodes_visited,
            traversal_time_ms=0.0,  # Se calcula después
            log_n=log_n,
            ratio_to_log_n=ratio,
            is_valid_ologn=is_valid
        )
        results.append(result)

    return results


def compute_speedup_vs_dense(N: int, k: int, D: int = 768) -> float:
    """
    Calcula speedup teórico de MatMul selectivo vs MatMul denso.

    Args:
        N: Tamaño de vocabulario (número de tokens)
        k: Tamaño del bloque selectivo (típicamente k = N^(1/3))
        D: Dimensión de embeddings

    Returns:
        Ratio de speedup (ops_dense / ops_selective)
    """
    # MatMul denso: O(N * D²)
    ops_dense = N * D * D

    # MatMul selectivo: O(k²)
    ops_selective = k * k

    speedup = ops_dense / (ops_selective + 1e-8)
    return speedup


def estimate_vram_savings(N: int, num_layers: int = 96) -> Dict[str, float]:
    """
    Estima ahorro de VRAM para KV Cache usando BSH vs Transformer denso.

    Args:
        N: Número de tokens en secuencia
        num_layers: Número de capas (típicamente 96 para GPT-4)

    Returns:
        Dict con estimaciones: bytes_transformer, bytes_bsh, ratio
    """
    # Transformer: KV Cache = 2 * num_layers * N * D * bytes_per_element
    D = 768  # dimensión de embeddings
    bytes_per_element = 2  # FP16

    kv_cache_transformer = 2 * num_layers * N * D * bytes_per_element

    # BSH: Árbol + embeddings comprimidos ≈ O(N log N) en mejor caso
    # Estimación conservadora: 50 bytes por nodo + overhead
    bytes_per_node = 50
    num_nodes_estimate = N * np.log2(N) if N > 1 else N
    kv_cache_bsh = num_nodes_estimate * bytes_per_node

    ratio = kv_cache_transformer / (kv_cache_bsh + 1e-8)

    return {
        "bytes_transformer": kv_cache_transformer,
        "bytes_bsh": kv_cache_bsh,
        "ratio": ratio,
        "gb_saved": (kv_cache_transformer - kv_cache_bsh) / 1e9
    }


def compute_routing_accuracy(correct_routings: int,
                             total_routings: int) -> float:
    """
    Calcula accuracy de routing de rayos a esferas correctas.

    Args:
        correct_routings: Número de routings correctos
        total_routings: Total de routings intentados

    Returns:
        Accuracy como porcentaje [0, 100]
    """
    if total_routings == 0:
        return 0.0
    return 100.0 * correct_routings / total_routings


def verify_ray_energy_conservation(initial_energy: float,
                                   final_energies: List[float],
                                   tolerance: float = 0.1) -> bool:
    """
    Verifica que la energía se conserva dentro de tolerancia.
    (Validación de corrección de física óptica)

    Args:
        initial_energy: Energía inicial del rayo
        final_energies: Energías finales tras traversal
        tolerance: Tolerancia relativa (%)

    Returns:
        True si se conserva dentro de tolerancia
    """
    for final_energy in final_energies:
        relative_loss = abs(final_energy - initial_energy) / (initial_energy + 1e-8)
        if relative_loss > tolerance:
            return False
    return True


def estimate_latency(vocab_size: int, num_rays: int = 8,
                     time_per_traversal_us: float = 150.0) -> Dict[str, float]:
    """
    Estima latencia de inferencia completa.

    Args:
        vocab_size: Número de tokens en secuencia
        num_rays: Número de rayos lanzados
        time_per_traversal_us: Tiempo por traversal (microsegundos)

    Returns:
        Dict con latencias en milisegundos
    """
    # Tiempo de traversal: O(log N)
    log_n = np.log2(vocab_size) if vocab_size > 1 else 0.0
    time_traversal_all_rays_us = num_rays * log_n * time_per_traversal_us

    # Tiempo de MatMul selectivo: O(k²) donde k ≈ N^(1/3)
    k = int(vocab_size ** (1/3))
    flops_matmul = k * k
    time_matmul_us = (flops_matmul / 100e9) * 1e6  # 100 TFLOPS GPU

    # Overhead: comunicación, compilación, etc
    overhead_us = 100.0

    total_us = time_traversal_all_rays_us + time_matmul_us + overhead_us
    total_ms = total_us / 1000.0

    return {
        "traversal_ms": time_traversal_all_rays_us / 1000.0,
        "matmul_ms": time_matmul_us / 1000.0,
        "overhead_ms": overhead_us / 1000.0,
        "total_ms": total_ms
    }


def print_detailed_report(vocab_sizes: List[int],
                         nodes_visited_list: List[float],
                         correct_routings: int = 1,
                         total_routings: int = 9):
    """
    Imprime un reporte detallado de análisis.
    """
    print("\n" + "=" * 70)
    print("REPORTE DETALLADO DE ANÁLISIS - PROTOTIPO A")
    print("=" * 70)

    # 1. Validación O(log N)
    print("\n[1. VALIDACIÓN COMPLEJIDAD O(log N)]")
    print("-" * 70)
    results = verify_ologn_complexity(vocab_sizes, nodes_visited_list)

    all_valid = True
    for result in results:
        status = "✓" if result.is_valid_ologn else "✗"
        print(f"  N={result.vocab_size:5d} | Nodos={result.nodes_visited:5.1f} | "
              f"log₂(N)={result.log_n:6.2f} | Ratio={result.ratio_to_log_n:5.2f} {status}")
        if not result.is_valid_ologn:
            all_valid = False

    status_text = "PASSED ✓" if all_valid else "FAILED ✗"
    print(f"\n  Estado: {status_text}\n")

    # 2. Speedup MatMul
    print("[2. SPEEDUP MatMul SELECTIVO vs DENSO]")
    print("-" * 70)
    test_sizes = [(1000, 32), (5000, 64), (10000, 128)]
    avg_speedup = []

    for N, k in test_sizes:
        speedup = compute_speedup_vs_dense(N, k)
        avg_speedup.append(speedup)
        print(f"  N={N:5d}, k={k:3d} (N^(1/3)) → Speedup: {speedup:7.1f}x")

    overall_speedup = np.mean(avg_speedup)
    print(f"\n  Speedup Promedio: {overall_speedup:.1f}x\n")

    # 3. Ahorro de VRAM
    print("[3. ESTIMACIÓN AHORRO VRAM (KV Cache)]")
    print("-" * 70)

    for N in [100000, 1000000]:
        vram_analysis = estimate_vram_savings(N)
        gb_transformer = vram_analysis["bytes_transformer"] / 1e9
        gb_bsh = vram_analysis["bytes_bsh"] / 1e9
        ratio = vram_analysis["ratio"]

        print(f"  N={N:7d} tokens:")
        print(f"    Transformer (FP16, 96 layers): {gb_transformer:8.1f} GB")
        print(f"    BSH (optimizado):               {gb_bsh:8.3f} GB")
        print(f"    Ratio: {ratio:7.1f}x menor con BSH\n")

    # 4. Accuracy de Routing (Polisemia)
    print("[4. ACCURACY DE ROUTING (POLISEMIA)]")
    print("-" * 70)
    accuracy = compute_routing_accuracy(correct_routings, total_routings)
    print(f"  Correctos: {correct_routings}/{total_routings}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Estado: {'✓ Aceptable' if accuracy >= 50.0 else '✗ Bajo (requiere ajustes)'}\n")

    # 5. Estimación de Latencia
    print("[5. ESTIMACIÓN LATENCIA DE INFERENCIA]")
    print("-" * 70)

    for N in [1000, 10000, 100000]:
        latency = estimate_latency(N)
        print(f"  N={N:6d} tokens (8 rayos):")
        print(f"    Traversal BSH:   {latency['traversal_ms']:7.3f} ms")
        print(f"    MatMul selectivo: {latency['matmul_ms']:7.3f} ms")
        print(f"    Overhead:         {latency['overhead_ms']:7.3f} ms")
        print(f"    TOTAL:            {latency['total_ms']:7.3f} ms")
        print()

    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Ejemplo de uso con datos simulados
    vocab_sizes = [50, 100, 500, 1000, 2000, 5000]
    nodes_visited = [6.0, 7.0, 7.0, 7.0, 7.0, 7.0]

    print_detailed_report(vocab_sizes, nodes_visited, correct_routings=1, total_routings=9)
