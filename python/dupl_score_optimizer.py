#!/usr/bin/env python3
"""
dupl_score_optimizer.py — DuplScore Optimizer para SpectralAI Zero-Matrix

Implementa la fórmula de decisión de duplicación vs wormhole para conceptos polisémicos:

    DuplScore(C) = (Σ_{s} f(C,s) · R(C,s)) · exp(-γ · D(Sc)) - δ · (|Sc|-1) · size(C)

Donde:
  - f(C,s): frecuencia de acceso al concepto C desde la esfera s
  - R(C,s): relevancia = similitud coseno ponderada por contextos de s
  - D(Sc): distancia jerárquica/geométrica media entre esferas
  - γ: penalización por distancia (favorece dup solo si esferas cercanas)
  - δ: costo de almacenamiento por duplicado
  - size(C): tamaño del tensor local del fragmento
  - τ: umbral calibrado por presupuesto global

Uso:
    python3 dupl_score_optimizer.py --seed 42 --output wormhole_graph.json
"""

import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import math


# ============================================================================
# Estructuras de datos
# ============================================================================

@dataclass
class ConceptInfo:
    """Información sobre un concepto polisémico."""
    concept_id: int
    concept_name: str
    spheres: List[int]  # IDs de esferas donde aparece
    size_bytes: int     # Tamaño del tensor del concepto


@dataclass
class SphereInfo:
    """Información sobre una esfera semántica."""
    sphere_id: int
    sphere_name: str
    position_3d: Tuple[float, float, float]


@dataclass
class DuplScoreResult:
    """Resultado del cálculo de DuplScore."""
    concept_id: int
    concept_name: str
    spheres: List[int]
    dupl_score: float
    decision: str  # 'DUPLICAR' o 'WORMHOLE'
    memory_delta_kb: float  # Cambio de memoria respecto a baseline
    num_spheres: int


# ============================================================================
# Generador de datos sintéticos
# ============================================================================

def create_synthetic_vocabulary() -> Tuple[List[ConceptInfo], List[SphereInfo]]:
    """
    Crea un vocabulario sintético con conceptos polisémicos distribuidos
    en 3 esferas (Programación, Música, Física).

    Returns:
        (lista de conceptos, lista de esferas)
    """
    # Definir esferas
    spheres = [
        SphereInfo(
            sphere_id=0,
            sphere_name="Programación",
            position_3d=(0.0, 0.0, 0.0)
        ),
        SphereInfo(
            sphere_id=1,
            sphere_name="Música",
            position_3d=(5.0, 5.0, 0.0)
        ),
        SphereInfo(
            sphere_id=2,
            sphere_name="Física",
            position_3d=(5.0, 0.0, 5.0)
        ),
    ]

    # Definir conceptos
    # Formato: (nombre, esferas_donde_aparece, tamaño_bytes)
    concept_definitions = [
        # Monopólicos (aparecen en UNA sola esfera)
        ("if", [0], 1024),
        ("while", [0], 1024),
        ("variable", [0], 2048),
        ("función", [0], 2048),
        ("clase", [0], 2048),
        ("array", [0], 1024),

        ("ritmo", [1], 2048),
        ("sample", [1], 1024),
        ("beat", [1], 1536),
        ("tempo", [1], 1536),
        ("acorde", [1], 2048),
        ("melodía", [1], 2048),

        ("orbita", [2], 2048),
        ("campo", [2], 2048),
        ("fuerza", [2], 1536),
        ("masa", [2], 1536),
        ("vector", [2], 1536),

        # Polisémicos (aparecen en MÚLTIPLES esferas)
        ("bucle", [0, 1], 2048),          # Loop en código + ciclo musical
        ("energía", [1, 2], 2048),       # Energía musical + energía física
        ("frecuencia", [0, 1, 2], 2560), # Tasa de muestreo + notas + oscilaciones
        ("amplitud", [1, 2], 2048),      # Amplitud de onda + amplitud de oscilación
    ]

    concepts = []
    for idx, (name, sphere_ids, size) in enumerate(concept_definitions):
        concepts.append(ConceptInfo(
            concept_id=idx,
            concept_name=name,
            spheres=sphere_ids,
            size_bytes=size
        ))

    return concepts, spheres


# ============================================================================
# Cálculo de métricas
# ============================================================================

def compute_sphere_distance(s1: SphereInfo, s2: SphereInfo) -> float:
    """Calcula distancia euclidiana entre dos esferas en espacio 3D."""
    d = 0.0
    for i in range(3):
        d += (s1.position_3d[i] - s2.position_3d[i]) ** 2
    return math.sqrt(d)


def compute_access_frequency(concept: ConceptInfo, sphere_id: int, seed: int) -> float:
    """
    Simula la frecuencia de acceso f(C,s).

    Basada en:
      - Conceptos raros tienen frecuencia baja (~0.1-0.3)
      - Conceptos comunes tienen frecuencia alta (~0.7-0.9)
      - Variación pseudo-aleatoria por esfera
    """
    # Usar hash determinista del concepto+esfera para reproducibilidad
    hash_val = (concept.concept_id * 1000 + sphere_id) * 37 + seed
    np.random.seed(hash_val % (2**31))

    # Frequencia base según complejidad del concepto
    base_freq = 0.3 + 0.6 * (concept.size_bytes / 2560.0)

    # Variación por esfera
    variation = np.random.uniform(-0.1, 0.2)

    return max(0.05, min(0.95, base_freq + variation))


def compute_relevance(concept: ConceptInfo, sphere_id: int, seed: int) -> float:
    """
    Simula la relevancia R(C,s) = similitud coseno ponderada.

    Más alto si el concepto aparece en esferas cercanas.
    """
    hash_val = (concept.concept_id * 2000 + sphere_id) * 41 + seed
    np.random.seed(hash_val % (2**31))

    # Relevancia base (inversamente proporcional a número de esferas)
    # Concepto muy raro (pocas esferas) → alta relevancia local
    # Concepto muy común (muchas esferas) → baja relevancia local
    n_spheres = len(concept.spheres)
    base_relevance = 0.7 + 0.3 * (1.0 / n_spheres)

    # Variación aleatoria
    variation = np.random.uniform(-0.1, 0.2)

    return max(0.1, min(0.95, base_relevance + variation))


def compute_sphere_distance_mean(concept: ConceptInfo, spheres: List[SphereInfo]) -> float:
    """
    Calcula D(Sc) = distancia media entre las esferas donde aparece el concepto.
    """
    if len(concept.spheres) < 2:
        return 0.0

    # Calcular distancias pairwise
    distances = []
    for i in range(len(concept.spheres)):
        for j in range(i + 1, len(concept.spheres)):
            s1 = spheres[concept.spheres[i]]
            s2 = spheres[concept.spheres[j]]
            distances.append(compute_sphere_distance(s1, s2))

    return np.mean(distances) if distances else 0.0


def compute_dupl_score(
    concept: ConceptInfo,
    spheres: List[SphereInfo],
    gamma: float = 0.2,
    delta: float = 0.001,
    seed: int = 42
) -> float:
    """
    Calcula DuplScore(C) según la fórmula del documento.

    DuplScore(C) = (Σ_{s} f(C,s) · R(C,s)) · exp(-γ · D(Sc)) - δ · (|Sc|-1) · size(C)

    Args:
        concept: Concepto a evaluar
        spheres: Lista de todas las esferas
        gamma: Penalización por distancia
        delta: Costo de almacenamiento
        seed: Semilla para reproducibilidad

    Returns:
        DuplScore (valor flotante)
    """
    # Término 1: Σ_{s} f(C,s) · R(C,s)
    access_relevance_product = 0.0
    for sphere_id in concept.spheres:
        f_cs = compute_access_frequency(concept, sphere_id, seed)
        r_cs = compute_relevance(concept, sphere_id, seed)
        access_relevance_product += f_cs * r_cs

    # Término 2: exp(-γ · D(Sc))
    d_mean = compute_sphere_distance_mean(concept, spheres)
    exp_factor = math.exp(-gamma * d_mean)

    # Término 3: -δ · (|Sc|-1) · size(C)
    num_spheres = len(concept.spheres)
    storage_cost = -delta * (num_spheres - 1) * concept.size_bytes

    # Fórmula completa
    dupl_score = access_relevance_product * exp_factor + storage_cost

    return dupl_score


# ============================================================================
# Lógica de decisión
# ============================================================================

def decide_duplication(dupl_score: float, tau: float = 0.5) -> str:
    """
    Decide si duplicar o usar wormhole basado en DuplScore.

    Si DuplScore > τ: DUPLICAR en todas las esferas relevantes
    Sino: WORMHOLE — puntero O(1) entre esferas

    Args:
        dupl_score: DuplScore calculado
        tau: Umbral de decisión

    Returns:
        'DUPLICAR' o 'WORMHOLE'
    """
    return 'DUPLICAR' if dupl_score > tau else 'WORMHOLE'


def calculate_memory_delta(concept: ConceptInfo, decision: str) -> float:
    """
    Calcula el cambio de memoria (en KB) según la decisión.

    DUPLICAR: Incremento de (num_spheres - 1) * size / 1024 KB
    WORMHOLE: Decremento de (num_spheres - 1) * size / 1024 KB
    """
    num_spheres = len(concept.spheres)
    if num_spheres < 2:
        return 0.0

    base_bytes = concept.size_bytes

    if decision == 'DUPLICAR':
        # Guardamos el concepto en todas las esferas
        # Costo: (num_spheres - 1) * tamaño adicional
        delta_bytes = (num_spheres - 1) * base_bytes
    else:  # WORMHOLE
        # Guardamos solo una vez + punteros
        # Ahorro: (num_spheres - 1) * tamaño - overhead de punteros
        pointer_overhead = 16  # 8 bytes de puntero + metadata
        delta_bytes = -((num_spheres - 1) * (base_bytes - pointer_overhead))

    return delta_bytes / 1024.0


# ============================================================================
# Análisis y reportes
# ============================================================================

def analyze_vocabulary(
    concepts: List[ConceptInfo],
    spheres: List[SphereInfo],
    gamma: float = 0.2,
    delta: float = 0.001,
    tau: float = 0.5,
    seed: int = 42
) -> List[DuplScoreResult]:
    """
    Analiza todo el vocabulario y calcula decisiones de duplicación.

    Returns:
        Lista de resultados (solo conceptos polisémicos)
    """
    results = []

    for concept in concepts:
        # Solo analizar conceptos polisémicos (>1 esfera)
        if len(concept.spheres) < 2:
            continue

        dupl_score = compute_dupl_score(concept, spheres, gamma, delta, seed)
        decision = decide_duplication(dupl_score, tau)
        memory_delta = calculate_memory_delta(concept, decision)

        result = DuplScoreResult(
            concept_id=concept.concept_id,
            concept_name=concept.concept_name,
            spheres=concept.spheres,
            dupl_score=dupl_score,
            decision=decision,
            memory_delta_kb=memory_delta,
            num_spheres=len(concept.spheres)
        )

        results.append(result)

    return results


def print_analysis_table(
    results: List[DuplScoreResult],
    spheres: List[SphereInfo]
) -> Tuple[int, int, float]:
    """
    Imprime tabla de análisis.

    Returns:
        (num_wormholes, num_duplicados, total_memory_delta_kb)
    """
    print("\n" + "="*100)
    print("[DUPL SCORE OPTIMIZER]")
    print("="*100)

    # Tabla de decisiones
    print(f"\n{'Concepto':<15} {'Esferas':<20} {'DuplScore':<12} {'Decisión':<12} {'Ahorro/Costo':<15}")
    print("-" * 100)

    wormhole_count = 0
    duplicado_count = 0
    total_memory_delta = 0.0

    for result in sorted(results, key=lambda r: r.dupl_score, reverse=True):
        sphere_names = ", ".join([spheres[s].sphere_name for s in result.spheres])

        # Formatear memoria delta con sign
        if result.memory_delta_kb > 0:
            delta_str = f"+{result.memory_delta_kb:.1f}KB"
        else:
            delta_str = f"{result.memory_delta_kb:.1f}KB"

        print(
            f"{result.concept_name:<15} {sphere_names:<20} "
            f"{result.dupl_score:>10.3f}  {result.decision:<12} {delta_str:>15}"
        )

        if result.decision == 'WORMHOLE':
            wormhole_count += 1
        else:
            duplicado_count += 1

        total_memory_delta += result.memory_delta_kb

    # Resumen
    print("-" * 100)
    print(f"\nTotal concepts analyzed: {len(results)}")
    print(f"  - Wormholes: {wormhole_count}")
    print(f"  - Duplicados: {duplicado_count}")
    print(f"\nMemory impact: {total_memory_delta:+.1f} KB")
    if total_memory_delta < 0:
        print(f"  → Ahorro de memoria: {abs(total_memory_delta):.1f} KB")
    else:
        print(f"  → Costo adicional: {total_memory_delta:.1f} KB")

    print("="*100 + "\n")

    return wormhole_count, duplicado_count, total_memory_delta


# ============================================================================
# Generación de grafo de wormholes
# ============================================================================

def build_wormhole_graph(
    results: List[DuplScoreResult],
    spheres: List[SphereInfo]
) -> Dict:
    """
    Construye un grafo JSON de wormholes y relaciones de duplicación.

    Estructura:
    {
        "spheres": [...],
        "concepts": [
            {
                "name": "bucle",
                "spheres": [0, 1],
                "decision": "WORMHOLE",
                "dupl_score": 0.23,
                "wormhole_edges": [
                    {"from_sphere": 0, "to_sphere": 1, "cost": 0}
                ]
            },
            ...
        ]
    }
    """
    graph = {
        "spheres": [
            {
                "id": s.sphere_id,
                "name": s.sphere_name,
                "position": s.position_3d
            }
            for s in spheres
        ],
        "concepts": []
    }

    for result in results:
        concept_entry = {
            "name": result.concept_name,
            "concept_id": result.concept_id,
            "spheres": result.spheres,
            "decision": result.decision,
            "dupl_score": float(result.dupl_score),
            "memory_delta_kb": float(result.memory_delta_kb)
        }

        if result.decision == 'WORMHOLE':
            # Crear edges entre esferas
            wormhole_edges = []
            for i in range(len(result.spheres)):
                for j in range(i + 1, len(result.spheres)):
                    wormhole_edges.append({
                        "from_sphere": result.spheres[i],
                        "to_sphere": result.spheres[j],
                        "cost_bytes": 16  # Costo de puntero
                    })
            concept_entry["wormhole_edges"] = wormhole_edges

        graph["concepts"].append(concept_entry)

    return graph


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DuplScore Optimizer para SpectralAI Zero-Matrix"
    )

    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla para RNG (default: 42)')
    parser.add_argument('--gamma', type=float, default=0.2,
                        help='Penalización por distancia (default: 0.2)')
    parser.add_argument('--delta', type=float, default=0.001,
                        help='Costo de almacenamiento (default: 0.001)')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Umbral de decisión (default: 0.5)')
    parser.add_argument('--output', type=str, default='wormhole_graph.json',
                        help='Archivo de salida JSON (default: wormhole_graph.json)')

    args = parser.parse_args()

    # ========================================================================
    # PASO 1: Generar vocabulario sintético
    # ========================================================================
    print("[PASO 1] Generando vocabulario sintético...")
    concepts, spheres = create_synthetic_vocabulary()
    print(f"  ✓ {len(concepts)} conceptos en {len(spheres)} esferas")

    # ========================================================================
    # PASO 2: Calcular DuplScore para conceptos polisémicos
    # ========================================================================
    print("\n[PASO 2] Calculando DuplScore...")
    results = analyze_vocabulary(
        concepts=concepts,
        spheres=spheres,
        gamma=args.gamma,
        delta=args.delta,
        tau=args.tau,
        seed=args.seed
    )
    print(f"  ✓ {len(results)} conceptos polisémicos analizados")

    # ========================================================================
    # PASO 3: Imprimir tabla de análisis
    # ========================================================================
    print("\n[PASO 3] Análisis de decisiones...")
    wormhole_count, duplicado_count, total_delta = print_analysis_table(results, spheres)

    # ========================================================================
    # PASO 4: Generar grafo de wormholes
    # ========================================================================
    print("[PASO 4] Generando grafo de wormholes...")
    wormhole_graph = build_wormhole_graph(results, spheres)

    # Guardar a archivo JSON
    try:
        with open(args.output, 'w') as f:
            json.dump(wormhole_graph, f, indent=2)
        print(f"  ✓ Grafo guardado en {args.output}")
    except IOError as e:
        print(f"  ✗ Error al guardar {args.output}: {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================================
    # PASO 5: Resumen final
    # ========================================================================
    print("\n" + "="*100)
    print("[RESUMEN FINAL]")
    print("="*100)
    print(f"Parámetros:")
    print(f"  - γ (gamma, penalización distancia): {args.gamma}")
    print(f"  - δ (delta, costo almacenamiento): {args.delta}")
    print(f"  - τ (tau, umbral decisión): {args.tau}")
    print(f"\nResultados:")
    print(f"  - Wormholes creados: {wormhole_count}")
    print(f"  - Duplicaciones: {duplicado_count}")
    print(f"  - Ahorro/costo de memoria: {total_delta:+.1f} KB")
    print(f"  - Salida JSON: {args.output}")
    print("="*100)

    print("\n[✓] DuplScore optimizer completado exitosamente.\n")


if __name__ == '__main__':
    main()
