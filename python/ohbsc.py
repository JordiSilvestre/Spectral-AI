#!/usr/bin/env python3
"""
ohbsc.py — Overlapping Hierarchical Bounding Sphere Clustering

ALGORITMO:
==========
OHBSC construye un árbol BSH (Bounding Sphere Hierarchy) donde las esferas
pueden solaparse. Los conceptos polisémicos viven en la INTERSECCIÓN de dos
o más esferas, resolviendo la polisemia sin duplicar nodos.

FLUJO:
  1. Farthest Point Sampling (FPS) → centros iniciales de esferas
  2. Fuzzy assignment (softmax de distancias inversas) → membresías difusas
  3. Detectar polisemia: entropía de membresía > H_umbral → nodo polisémico
  4. Construir IntersectionNode para nodos polisémicos
  5. Recursive build en cada cluster (depth+1)
  6. Mínimo bounding sphere para nodos padre

DIFERENCIAS VS K-MEANS CLÁSICO:
  - Solapamiento permitido: P(token ∈ esfera_k) puede ser > 0 para múltiples k
  - Nodos intersección: dual-parent (DAG en lugar de árbol puro)
  - Temperature annealing: T → 0 → asignación dura al final
  - FPS en lugar de centroides aleatorios: mejor cobertura inicial

USO:
  from ohbsc import OHBSCBuilder, OHBSCNode
  builder = OHBSCBuilder(num_clusters=3, overlap_alpha=0.3)
  root = builder.build(embeddings_3d, token_ids)
  tree_dict = root.to_dict()

COMPATIBILIDAD:
  - numpy >= 1.21
  - scipy (opcional, para bounding sphere exacta)
  - GloVe-300d embeddings proyectados a 3D via PCA

@author SpectralAI Zero-Matrix Team
@date 2026
"""

import sys
import os
import json
import numpy as np
from typing import List, Optional, Dict, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# Constantes
# ─────────────────────────────────────────────────────────────────────────────

MIN_CLUSTER_SIZE   = 2       # Mínimo de tokens para crear un nodo hoja
MAX_DEPTH          = 4       # Máximo de niveles en la jerarquía
TEMPERATURE_INIT   = 1.0     # Temperatura inicial para membresía difusa
TEMPERATURE_DECAY  = 0.9     # Factor de decay por nivel
ENTROPY_THRESHOLD  = 0.6     # H(membership) > umbral → polisémico
POLYSEMY_THRESHOLD = 0.3     # P(k) > umbral → token pertenece a esfera k

# ─────────────────────────────────────────────────────────────────────────────
# Estructuras de datos
# ─────────────────────────────────────────────────────────────────────────────

class NodeType(Enum):
    DOMAIN     = 0   # Nivel 0 — dominio temático grande
    SUBDOMAIN  = 1   # Nivel 1 — subdominio
    CONCEPT    = 2   # Nivel 2 — concepto específico
    LEAF       = 3   # Nivel 3 — SemanticString


@dataclass
class OHBSCNode:
    """Nodo en el árbol OHBSC."""
    node_type:     NodeType
    depth:         int
    center:        np.ndarray          # Posición 3D del centroide de la esfera
    radius:        float               # Radio de la esfera
    token_ids:     List[int]           # IDs de tokens en este nodo
    memberships:   np.ndarray          # Vector de membresías de cada token [0,1]
    children:      List['OHBSCNode']   = field(default_factory=list)
    polysemic_ids: List[int]           = field(default_factory=list)  # tokens polisémicos
    intersection_with: List[int]       = field(default_factory=list)  # IDs de nodos gemelos
    label:         str                 = ""
    node_id:       int                 = 0

    def is_leaf(self) -> bool:
        return self.node_type == NodeType.LEAF or len(self.children) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serializa el nodo a dict (para JSON export)."""
        return {
            "node_id":         self.node_id,
            "node_type":       self.node_type.name,
            "depth":           self.depth,
            "center":          self.center.tolist(),
            "radius":          float(self.radius),
            "token_ids":       self.token_ids,
            "polysemic_ids":   self.polysemic_ids,
            "intersection_with": self.intersection_with,
            "label":           self.label,
            "num_children":    len(self.children),
            "children":        [c.to_dict() for c in self.children],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Algoritmos auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def farthest_point_sampling(positions: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Farthest Point Sampling (FPS) para inicializar centros de clusters.

    Garantiza que los centros están lo más separados posible — mejor cobertura
    inicial que k-means++ y mucho mejor que aleatorio.

    Args:
        positions: Array [N, 3] de posiciones 3D
        n_samples: Número de centros a seleccionar

    Returns:
        Array [n_samples, 3] de centros seleccionados
    """
    N = len(positions)
    n_samples = min(n_samples, N)
    if n_samples <= 0:
        return np.empty((0, 3))

    selected = np.zeros(n_samples, dtype=int)
    # Primer punto: el más cercano al centroide (robusto contra outliers)
    centroid = positions.mean(axis=0)
    dists_to_centroid = np.linalg.norm(positions - centroid, axis=1)
    selected[0] = np.argmin(dists_to_centroid)

    if n_samples == 1:
        return positions[selected]

    # Distancias mínimas al conjunto ya seleccionado
    min_dists = np.full(N, np.inf)
    min_dists -= np.linalg.norm(positions - positions[selected[0]], axis=1)
    min_dists = np.linalg.norm(positions - positions[selected[0]], axis=1)

    for i in range(1, n_samples):
        selected[i] = np.argmax(min_dists)
        new_dists = np.linalg.norm(positions - positions[selected[i]], axis=1)
        min_dists = np.minimum(min_dists, new_dists)

    return positions[selected]


def fuzzy_assignment(
    positions: np.ndarray,
    centers: np.ndarray,
    temperature: float = 1.0
) -> np.ndarray:
    """
    Asignación difusa de tokens a clusters vía softmax de distancias inversas.

    P(token_i ∈ cluster_k) = softmax(-d²(i,k) / T²)

    Args:
        positions:   Array [N, D] de posiciones
        centers:     Array [K, D] de centros
        temperature: T para softmax — T→0 → hard assignment

    Returns:
        memberships: Array [N, K] con P(token_i ∈ cluster_k) ∈ [0,1], suma = 1 por fila
    """
    N, K = len(positions), len(centers)
    # Distancias cuadráticas [N, K]
    d_sq = np.sum(
        (positions[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2,
        axis=2
    )
    # Softmax (-d² / T²)
    logits = -d_sq / max(temperature ** 2, 1e-8)
    # Estabilización numérica (restar max por fila)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    memberships = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    return memberships


def shannon_entropy(probs: np.ndarray) -> float:
    """Entropía de Shannon H(p) = -Σ p_i log(p_i), normalizada a [0,1]."""
    K = len(probs)
    if K <= 1:
        return 0.0
    eps = 1e-12
    h = -np.sum(probs * np.log(probs + eps))
    h_max = np.log(K)  # entropía máxima = uniforme
    return h / h_max if h_max > 0 else 0.0


def minimum_bounding_sphere(positions: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Esfera de bounding mínima (aproximación: centroide + percentil 90).

    La solución exacta (Miniball) es O(N·D) iterativa — aquí usamos
    una aproximación O(N) más simple pero suficientemente cercana.

    Returns:
        (center, radius) de la esfera
    """
    if len(positions) == 0:
        return np.zeros(3), 0.0
    center = positions.mean(axis=0)
    dists  = np.linalg.norm(positions - center, axis=1)
    radius = np.percentile(dists, 90) * 1.1  # 10% de margen
    return center, float(radius)


# ─────────────────────────────────────────────────────────────────────────────
# Constructor principal OHBSC
# ─────────────────────────────────────────────────────────────────────────────

class OHBSCBuilder:
    """
    Constructor del árbol OHBSC (Overlapping Hierarchical Bounding Sphere Clustering).

    Atributos configurables:
        branching:      Número de hijos por nodo (factor de ramificación)
        overlap_alpha:  Umbral de membresía para solapamiento (0.3 = 30%)
        entropy_thresh: Entropía normalizada para detectar polisemia
        max_depth:      Máximo de niveles en el árbol
        min_size:       Tamaño mínimo de cluster para continuar dividiendo
        temp_init:      Temperatura inicial (T=1.0 → completamente difuso)
        temp_decay:     Factor de decaimiento de T por nivel
    """

    def __init__(
        self,
        branching:      int   = 3,
        overlap_alpha:  float = POLYSEMY_THRESHOLD,
        entropy_thresh: float = ENTROPY_THRESHOLD,
        max_depth:      int   = MAX_DEPTH,
        min_size:       int   = MIN_CLUSTER_SIZE,
        temp_init:      float = TEMPERATURE_INIT,
        temp_decay:     float = TEMPERATURE_DECAY,
    ):
        self.branching      = branching
        self.overlap_alpha  = overlap_alpha
        self.entropy_thresh = entropy_thresh
        self.max_depth      = max_depth
        self.min_size       = min_size
        self.temp_init      = temp_init
        self.temp_decay     = temp_decay
        self._node_counter  = 0

    def _new_id(self) -> int:
        self._node_counter += 1
        return self._node_counter

    def build(
        self,
        positions: np.ndarray,
        token_ids: Optional[List[int]] = None,
        labels: Optional[List[str]] = None,
        depth: int = 0,
    ) -> OHBSCNode:
        """
        Construye el árbol OHBSC recursivamente.

        Args:
            positions:  Array [N, 3] de posiciones 3D de los tokens
            token_ids:  Lista de IDs de tokens (si None, usa índices 0..N-1)
            labels:     Etiquetas legibles por token (opcional, para debug)
            depth:      Profundidad actual en la jerarquía (0 = raíz)

        Returns:
            OHBSCNode raíz del árbol construido
        """
        N = len(positions)

        if token_ids is None:
            token_ids = list(range(N))
        if labels is None:
            labels = [str(tid) for tid in token_ids]

        # Temperatura decrece con la profundidad
        temperature = self.temp_init * (self.temp_decay ** depth)

        # Tipo de nodo según profundidad
        node_types = [NodeType.DOMAIN, NodeType.SUBDOMAIN, NodeType.CONCEPT, NodeType.LEAF]
        node_type = node_types[min(depth, len(node_types) - 1)]

        # ── Caso base: hoja ──────────────────────────────────────────────
        if N <= self.min_size or depth >= self.max_depth - 1:
            center, radius = minimum_bounding_sphere(positions)
            return OHBSCNode(
                node_type  = NodeType.LEAF,
                depth      = depth,
                center     = center,
                radius     = radius,
                token_ids  = list(token_ids),
                memberships = np.ones(N) / N,
                label      = f"leaf_{depth}_{self._new_id()}",
                node_id    = self._new_id(),
            )

        # ── Inicializar centros via FPS ──────────────────────────────────
        k = min(self.branching, N)
        centers = farthest_point_sampling(positions, k)

        # ── Membresías difusas ───────────────────────────────────────────
        memberships = fuzzy_assignment(positions, centers, temperature)

        # ── Detectar polisemia ───────────────────────────────────────────
        polysemic_mask = np.array([
            shannon_entropy(memberships[i]) > self.entropy_thresh
            for i in range(N)
        ])
        polysemic_ids_in_cluster = [
            token_ids[i] for i in range(N) if polysemic_mask[i]
        ]

        # ── Asignar tokens a clusters ────────────────────────────────────
        # Hard assignment para la construcción del árbol:
        # cada token va al cluster con membresía máxima
        hard_assignment = np.argmax(memberships, axis=1)

        # Tokens polisémicos: añadir también a clusters secundarios
        # donde P(token ∈ cluster) > overlap_alpha
        cluster_token_ids   = [[] for _ in range(k)]
        cluster_positions   = [[] for _ in range(k)]
        cluster_memberships = [[] for _ in range(k)]

        for i in range(N):
            for c in range(k):
                if memberships[i, c] > self.overlap_alpha:
                    cluster_token_ids[c].append(token_ids[i])
                    cluster_positions[c].append(positions[i])
                    cluster_memberships[c].append(memberships[i, c])

        # ── Construir nodos hijo recursivamente ──────────────────────────
        children = []
        for c in range(k):
            if len(cluster_token_ids[c]) == 0:
                continue
            child_pos = np.array(cluster_positions[c])
            child = self.build(
                positions = child_pos,
                token_ids = cluster_token_ids[c],
                labels    = [str(tid) for tid in cluster_token_ids[c]],
                depth     = depth + 1,
            )
            children.append(child)

        # ── Esfera padre = mínima esfera que contiene todos los hijos ────
        child_centers = np.array([c.center for c in children]) if children else positions
        parent_center, parent_radius = minimum_bounding_sphere(positions)

        # Asegurar que la esfera padre contiene todas las esferas hijo
        for child in children:
            dist_to_child = np.linalg.norm(parent_center - child.center)
            parent_radius = max(parent_radius, dist_to_child + child.radius)

        node = OHBSCNode(
            node_type    = node_type,
            depth        = depth,
            center       = parent_center,
            radius       = parent_radius,
            token_ids    = list(token_ids),
            memberships  = memberships.max(axis=1),  # membresía máxima por token
            children     = children,
            polysemic_ids = polysemic_ids_in_cluster,
            label        = f"{node_type.name.lower()}_{depth}_{self._new_id()}",
            node_id      = self._new_id(),
        )

        return node

    def tree_stats(self, root: OHBSCNode) -> Dict[str, Any]:
        """Estadísticas del árbol construido."""
        def _count(node: OHBSCNode, stats: Dict):
            stats["total_nodes"]  += 1
            stats["total_tokens"] += len(node.token_ids)
            stats["polysemic"]    += len(node.polysemic_ids)
            if node.node_type == NodeType.LEAF:
                stats["leaves"]   += 1
                stats["max_depth"] = max(stats["max_depth"], node.depth)
            for child in node.children:
                _count(child, stats)

        stats = {
            "total_nodes": 0, "total_tokens": 0, "leaves": 0,
            "polysemic": 0, "max_depth": 0
        }
        _count(root, stats)
        return stats


# ─────────────────────────────────────────────────────────────────────────────
# Conversión a InceptionScene (para C++)
# ─────────────────────────────────────────────────────────────────────────────

def tree_to_inception_scene(
    root: OHBSCNode,
    embeddings: np.ndarray,
    vocab: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convierte el árbol OHBSC a un InceptionScene serializable.

    El resultado puede leerse en Python (inference.py) para serializar
    en el formato binario esperado por inception_runner.exe.

    Args:
        root:        Raíz del árbol OHBSC
        embeddings:  Array [vocab_size, 3] de embeddings 3D
        vocab:       Lista de palabras del vocabulario (opcional)

    Returns:
        dict compatible con json.dump() y con el formato InceptionScene de C++
    """
    domains    = []
    subdomains = []
    concepts   = []
    leaves     = []

    def _traverse(node: OHBSCNode):
        entry = {
            "node_id":    node.node_id,
            "node_type":  node.node_type.name,
            "depth":      node.depth,
            "center":     node.center.tolist(),
            "radius":     float(node.radius),
            "token_ids":  node.token_ids,
            "label":      node.label,
            "polysemic":  node.polysemic_ids,
        }
        if node.node_type == NodeType.DOMAIN:
            domains.append(entry)
        elif node.node_type == NodeType.SUBDOMAIN:
            subdomains.append(entry)
        elif node.node_type == NodeType.CONCEPT:
            concepts.append(entry)
        else:  # LEAF
            # Generar coeficientes Fourier desde embedding 3D
            if node.token_ids and node.token_ids[0] < len(embeddings):
                pos3d = embeddings[node.token_ids[0]].tolist()
            else:
                pos3d = node.center.tolist()
            entry["position_3d"] = pos3d
            entry["label_word"]  = vocab[node.token_ids[0]] if vocab and node.token_ids else ""
            leaves.append(entry)

        for child in node.children:
            _traverse(child)

    _traverse(root)

    return {
        "domains":    domains,
        "subdomains": subdomains,
        "concepts":   concepts,
        "leaves":     leaves,
        "base_omega": 0.785398,  # π/4
        "num_levels": 4,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Demo / main
# ─────────────────────────────────────────────────────────────────────────────

def run_demo():
    """Demo con embeddings GloVe-300d proyectados a 3D."""
    print("=" * 60)
    print(" OHBSC Demo — Overlapping Hierarchical Bounding Sphere")
    print("=" * 60)

    # Cargar embeddings 3D (ya proyectados por inference.py con PCA)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    emb_3d_path = os.path.join(script_dir, "embeddings_3d.npy")
    vocab_path  = os.path.join(script_dir, "vocab.txt")

    if not os.path.exists(emb_3d_path):
        print(f"[ERROR] {emb_3d_path} no encontrado.")
        print("        Ejecutar primero: python download_embeddings_v2.py")
        return

    print(f"[info] Cargando embeddings 3D desde {emb_3d_path}...")
    emb_3d = np.load(emb_3d_path)[:500]   # primeras 500 palabras
    N = len(emb_3d)
    print(f"[info] {N} tokens cargados, shape {emb_3d.shape}")

    vocab = []
    if os.path.exists(vocab_path):
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = [l.strip() for l in f.readlines()][:N]

    # Construir árbol OHBSC
    print(f"\n[build] Construyendo OHBSC (branching=4, depth=3, overlap_alpha=0.3)...")
    builder = OHBSCBuilder(branching=4, overlap_alpha=0.3, max_depth=3, min_size=3)
    root = builder.build(emb_3d, list(range(N)), vocab)

    stats = builder.tree_stats(root)
    print(f"\n[stats] Árbol construido:")
    print(f"  Total nodos:    {stats['total_nodes']}")
    print(f"  Nodos hoja:     {stats['leaves']}")
    print(f"  Profundidad max:{stats['max_depth']}")
    print(f"  Polisémicos:    {stats['polysemic']}")
    print(f"  Tokens totales: {stats['total_tokens']} (con solapamiento)")

    # Exportar estructura
    scene = tree_to_inception_scene(root, emb_3d, vocab)
    output_path = os.path.join(script_dir, "ohbsc_tree.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scene, f, indent=2, ensure_ascii=False)
    print(f"\n[save] Árbol guardado en {output_path}")
    print(f"       {len(scene['domains'])} dominios, {len(scene['leaves'])} hojas")

    # Mostrar primeras ramas
    print("\n[tree] Primeras 3 ramas:")
    for i, child in enumerate(root.children[:3]):
        labels = [vocab[tid] for tid in child.token_ids[:5] if tid < len(vocab)]
        print(f"  Nodo {i}: depth={child.depth}, r={child.radius:.3f}, "
              f"tokens={len(child.token_ids)}, "
              f"polisémicos={len(child.polysemic_ids)}")
        print(f"    Muestra: {labels}")

    print("\n[OK] OHBSC completado.")


if __name__ == "__main__":
    run_demo()
