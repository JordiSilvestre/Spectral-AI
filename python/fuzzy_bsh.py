#!/usr/bin/env python3
"""
fuzzy_bsh.py — Fuzzy Bounding Sphere Hierarchy para Backpropagation

Implementa un árbol BSH diferenciable usando membresía fuzzy:
  - Durante training: cada token tiene distribución de probabilidad sobre esferas
  - P(token ∈ esfera_k) = softmax(-d²(token, center_k) / (2*T²))
  - Gradiente respecto a center_k vía chain rule
  - Simulated annealing: T → 0 para endurecimiento progresivo del árbol

Uso:
    python3 fuzzy_bsh.py --num-epochs 200 --seed 42 --output clusters.json
"""

import sys
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import math


# ============================================================================
# Estructura de datos
# ============================================================================

@dataclass
class FuzzyBSHState:
    """Estado del árbol Fuzzy BSH."""
    epoch: int
    temperature: float
    loss_spatial: float
    cluster_accuracy: float


# ============================================================================
# Clase FuzzyBSH
# ============================================================================

class FuzzyBSH:
    """
    Árbol BSH diferenciable con membresía fuzzy para training.

    Durante training: la asignación de tokens a esferas es suave (probabilística).
    Durante inferencia: puede discretizarse (T → 0).
    """

    def __init__(
        self,
        n_spheres: int,
        embed_dim: int = 64,
        temperature: float = 1.0,
        learning_rate: float = 0.01,
        seed: int = 42,
        init_from_data: Optional[Tuple[np.ndarray, Dict[int, List[int]]]] = None
    ):
        """
        Inicializa el árbol Fuzzy BSH.

        Args:
            n_spheres: Número de esferas
            embed_dim: Dimensión del embedding (reducido)
            temperature: Temperatura inicial (controla fuzziness)
            learning_rate: Tasa de aprendizaje para gradient descent
            seed: Semilla para reproducibilidad
            init_from_data: Tupla (embeddings, clusters) para inicializar desde datos
        """
        self.n_spheres = n_spheres
        self.embed_dim = embed_dim
        self.T = temperature
        self.lr = learning_rate
        self.seed = seed

        np.random.seed(seed)

        # Parámetros aprendibles
        # centers: (n_spheres, embed_dim)
        if init_from_data is not None:
            # Inicializar desde promedios de datos
            token_embeddings, clusters = init_from_data
            self.centers = np.zeros((n_spheres, embed_dim), dtype=np.float32)

            for sphere_id in range(n_spheres):
                if sphere_id in clusters and clusters[sphere_id]:
                    token_ids = clusters[sphere_id]
                    cluster_embedding = token_embeddings[token_ids].mean(axis=0)
                    self.centers[sphere_id] = cluster_embedding[:embed_dim]
                else:
                    self.centers[sphere_id] = np.random.randn(embed_dim).astype(np.float32) * 0.3
        else:
            # Inicialización aleatoria
            self.centers = np.random.randn(n_spheres, embed_dim).astype(np.float32) * 0.3

        # radii: (n_spheres,) - radio de cada esfera
        self.radii = np.ones(n_spheres, dtype=np.float32) * 1.0

        # Histórico de training
        self.history = {
            'epoch': [],
            'temperature': [],
            'loss_spatial': [],
            'cluster_accuracy': [],
            'loss_prox': [],
            'loss_cover': []
        }

    def membership_probs(self, token_embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula P(token ∈ esfera_k) para todos los tokens.

        Args:
            token_embeddings: (n_tokens, embed_dim)

        Returns:
            membership_matrix: (n_tokens, n_spheres)
            membership_matrix[i, k] = P(token_i ∈ esfera_k)
        """
        n_tokens = token_embeddings.shape[0]

        # Distancias cuadradas: ||token_i - center_k||²
        # Shape: (n_tokens, n_spheres)
        dists_sq = np.zeros((n_tokens, self.n_spheres), dtype=np.float32)

        for i in range(n_tokens):
            for k in range(self.n_spheres):
                diff = token_embeddings[i] - self.centers[k]
                dists_sq[i, k] = np.sum(diff ** 2)

        # Fórmula: P(k) = softmax(-d² / (2*T²))
        # Versión numérica estable: softmax(x) = exp(x - max(x)) / Σ exp(x - max(x))
        logits = -dists_sq / (2 * self.T ** 2)

        # Versión estable
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        return probs

    def forward(
        self,
        token_embeddings: np.ndarray,
        query_colors: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass: calcula distribución de atención usando Fuzzy BSH.

        Args:
            token_embeddings: (n_tokens, embed_dim)
            query_colors: (n_tokens, 64) - contexto espectral (opcional)

        Returns:
            (attention_weights, membership_probs)
            - attention_weights: (n_tokens, n_tokens) - matriz de atención difusa
            - membership_probs: (n_tokens, n_spheres) - distribución de asignación
        """
        # Calcular membresía fuzzy
        membership = self.membership_probs(token_embeddings)  # (n_tokens, n_spheres)

        # Calcular ataques entre esferas
        # Simplificación: dos tokens en la MISMA esfera son más relevantes
        # attention_weight[i, j] = Σ_k P(i ∈ k) · P(j ∈ k)
        n_tokens = token_embeddings.shape[0]
        attention = np.zeros((n_tokens, n_tokens), dtype=np.float32)

        for i in range(n_tokens):
            for j in range(n_tokens):
                # Suma ponderada de co-membresía
                attention[i, j] = np.sum(membership[i] * membership[j])

        # Normalizar por filas (softmax-like)
        for i in range(n_tokens):
            row_sum = np.sum(attention[i])
            if row_sum > 0:
                attention[i] /= row_sum

        return attention, membership

    def compute_loss(
        self,
        token_embeddings: np.ndarray,
        ground_truth_clusters: Dict[int, List[int]],
        alpha_prox: float = 1.0,
        alpha_cover: float = 1.0,
        alpha_inter: float = 0.5
    ) -> Tuple[float, float, float, float]:
        """
        Calcula pérdida espacial total.

        L_spatial = L_prox + L_cover + L_inter

        Donde:
        - L_prox: tokens del mismo cluster deben estar cerca en el espacio
        - L_cover: esferas deben cubrir sus tokens asignados
        - L_inter: tokens polisémicos deben estar en intersecciones de esferas

        Args:
            token_embeddings: (n_tokens, embed_dim)
            ground_truth_clusters: Dict[sphere_id] → List[token_ids]
            alpha_prox, alpha_cover, alpha_inter: pesos de pérdidas

        Returns:
            (L_spatial_total, L_prox, L_cover, L_inter)
        """
        membership = self.membership_probs(token_embeddings)  # (n_tokens, n_spheres)
        n_tokens = token_embeddings.shape[0]

        # ====================================================================
        # L_PROX: Tokens del mismo cluster cercanos
        # ====================================================================
        l_prox = 0.0
        token_to_sphere = {}  # Mapeo inverso: token_id → assigned_sphere (ground truth)

        for sphere_id, token_ids in ground_truth_clusters.items():
            for tid in token_ids:
                token_to_sphere[tid] = sphere_id

        # Para cada par de tokens, si pertenecen al mismo cluster, deben estar cerca
        count_pairs = 0
        for i in range(n_tokens):
            for j in range(i + 1, n_tokens):
                if i in token_to_sphere and j in token_to_sphere:
                    if token_to_sphere[i] == token_to_sphere[j]:
                        # Mismo cluster → distancia debe ser pequeña
                        diff = token_embeddings[i] - token_embeddings[j]
                        dist_sq = np.sum(diff ** 2)
                        l_prox += dist_sq
                        count_pairs += 1

        if count_pairs > 0:
            l_prox /= count_pairs
        else:
            l_prox = 0.0

        # ====================================================================
        # L_COVER: Esferas cubren sus tokens asignados
        # ====================================================================
        l_cover = 0.0
        count_coverage = 0

        for sphere_id, token_ids in ground_truth_clusters.items():
            for tid in token_ids:
                # Distancia del token al centro de su esfera
                diff = token_embeddings[tid] - self.centers[sphere_id]
                dist = np.sqrt(np.sum(diff ** 2) + 1e-8)
                radius = self.radii[sphere_id]

                # Penalizar si token está fuera del radio
                if dist > radius:
                    l_cover += (dist - radius) ** 2
                count_coverage += 1

        if count_coverage > 0:
            l_cover /= count_coverage
        else:
            l_cover = 0.0

        # ====================================================================
        # L_INTER: Tokens polisémicos en intersecciones
        # ====================================================================
        # Simplificación: token es polisémico si pertenece a múltiples clusters
        l_inter = 0.0
        count_poly = 0

        sphere_membership_count = np.zeros(n_tokens, dtype=np.int32)
        for sphere_id, token_ids in ground_truth_clusters.items():
            for tid in token_ids:
                sphere_membership_count[tid] += 1

        for tid in range(n_tokens):
            if sphere_membership_count[tid] > 1:
                # Token polisémico → debe estar en intersección
                # Heurística: debería tener P(k) > 0 para múltiples esferas
                probs = membership[tid]
                # Maximizar entropía de asignación (no estar concentrado en una esfera)
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                # Penalizar baja entropía (asignación concentrada)
                l_inter += (1.0 - entropy) ** 2
                count_poly += 1

        if count_poly > 0:
            l_inter /= count_poly
        else:
            l_inter = 0.0

        # ====================================================================
        # Pérdida total
        # ====================================================================
        l_spatial = alpha_prox * l_prox + alpha_cover * l_cover + alpha_inter * l_inter

        return l_spatial, l_prox, l_cover, l_inter

    def compute_cluster_accuracy(
        self,
        token_embeddings: np.ndarray,
        ground_truth_clusters: Dict[int, List[int]]
    ) -> float:
        """
        Calcula la precisión de clustering.

        Un token se considera bien clasificado si se asigna a su esfera
        ground truth con la máxima probabilidad fuzzy.

        Args:
            token_embeddings: (n_tokens, embed_dim)
            ground_truth_clusters: Dict[sphere_id] → List[token_ids]

        Returns:
            Accuracy (0.0 a 1.0)
        """
        membership = self.membership_probs(token_embeddings)
        n_tokens = token_embeddings.shape[0]

        # Mapeo inverso
        token_to_sphere = {}
        for sphere_id, token_ids in ground_truth_clusters.items():
            for tid in token_ids:
                token_to_sphere[tid] = sphere_id

        correct = 0
        for tid in range(n_tokens):
            if tid in token_to_sphere:
                # Esfera asignada: máxima probabilidad
                assigned_sphere = np.argmax(membership[tid])
                gt_sphere = token_to_sphere[tid]

                # Es correcto si la esfera con máx prob es la ground truth
                if assigned_sphere == gt_sphere:
                    correct += 1

        accuracy = correct / max(len(token_to_sphere), 1)
        return accuracy

    def update_gradient_descent(
        self,
        token_embeddings: np.ndarray,
        ground_truth_clusters: Dict[int, List[int]],
        alpha_prox: float = 1.0,
        alpha_cover: float = 1.0,
        alpha_inter: float = 0.5
    ) -> None:
        """
        Actualiza centros y radios con gradient descent analítico.

        Usamos cálculo directo de gradientes respecto a membership.

        Args:
            token_embeddings: (n_tokens, embed_dim)
            ground_truth_clusters: Dict[sphere_id] → List[token_ids]
            alpha_prox, alpha_cover, alpha_inter: pesos de pérdidas
        """
        # ====================================================================
        # Actualizar centros: mover hacia los tokens que debería cubrir
        # ====================================================================
        for k in range(self.n_spheres):
            if k in ground_truth_clusters:
                token_ids = ground_truth_clusters[k]

                # Centro = promedio ponderado de los tokens asignados
                # Pesos: probabilidad de asignación (membership)
                membership = self.membership_probs(token_embeddings)

                # Gradiente: suma ponderada hacia los tokens de esta esfera
                gradient = np.zeros(self.embed_dim, dtype=np.float32)

                for tid in token_ids:
                    # Peso: qué tan probable es que este token pertenezca a esta esfera
                    weight = membership[tid, k]
                    # Dirección: hacia el token
                    direction = token_embeddings[tid] - self.centers[k]
                    gradient += weight * direction

                # Normalizar
                gradient /= max(len(token_ids), 1)

                # Actualizar centro (gradient ascent hacia los tokens)
                self.centers[k] += self.lr * gradient

        # ====================================================================
        # Actualizar radios: cubrir los tokens asignados
        # ====================================================================
        for k in range(self.n_spheres):
            if k in ground_truth_clusters:
                token_ids = ground_truth_clusters[k]
                dists = []

                for tid in token_ids:
                    diff = token_embeddings[tid] - self.centers[k]
                    dist = np.sqrt(np.sum(diff ** 2) + 1e-8)
                    dists.append(dist)

                # Radio = promedio + un margen (para suavidad)
                if dists:
                    mean_dist = np.mean(dists)
                    max_dist = np.max(dists)
                    # Radio conservador: media + 30% del rango
                    self.radii[k] = mean_dist + 0.3 * (max_dist - mean_dist)

    def harden(self, factor: float = 0.9) -> None:
        """
        Reduce la temperatura para endurecimiento progresivo del árbol.

        T → 0 hace que la distribución sea más concentrada (eventual discretización).

        Args:
            factor: Multiplicador de temperatura (típicamente 0.9-0.95)
        """
        self.T *= factor
        self.T = max(self.T, 0.01)  # Mínimo para estabilidad numérica

    def train(
        self,
        token_embeddings: np.ndarray,
        ground_truth_clusters: Dict[int, List[int]],
        num_epochs: int = 200,
        harden_every: int = 50,
        harden_factor: float = 0.9
    ) -> List[FuzzyBSHState]:
        """
        Entrena el árbol Fuzzy BSH con simulated annealing.

        Args:
            token_embeddings: (n_tokens, embed_dim)
            ground_truth_clusters: Dict[sphere_id] → List[token_ids]
            num_epochs: Número de épocas
            harden_every: Endurecimiento cada N épocas
            harden_factor: Factor de endurecimiento por iteración

        Returns:
            Histórico de training
        """
        print(f"\n{'='*100}")
        print("[FUZZY BSH TRAINING]")
        print(f"{'='*100}\n")

        print(f"{'Epoch':<8} {'T':<8} {'L_spatial':<14} {'L_prox':<14} {'L_cover':<14} {'L_inter':<14} {'Accuracy':<10}")
        print("-" * 100)

        states = []

        for epoch in range(num_epochs):
            # ================================================================
            # Forward pass + Loss computation
            # ================================================================
            l_spatial, l_prox, l_cover, l_inter = self.compute_loss(
                token_embeddings, ground_truth_clusters
            )
            accuracy = self.compute_cluster_accuracy(token_embeddings, ground_truth_clusters)

            # ================================================================
            # Backward pass + Update
            # ================================================================
            self.update_gradient_descent(token_embeddings, ground_truth_clusters)

            # ================================================================
            # Simulated annealing (endurecimiento)
            # ================================================================
            if epoch > 0 and epoch % harden_every == 0:
                self.harden(harden_factor)

            # ================================================================
            # Logging
            # ================================================================
            if epoch % max(1, num_epochs // 20) == 0 or epoch == num_epochs - 1:
                acc_pct = accuracy * 100
                print(
                    f"{epoch:<8} {self.T:<8.3f} {l_spatial:<14.4f} "
                    f"{l_prox:<14.4f} {l_cover:<14.4f} {l_inter:<14.4f} {acc_pct:<9.1f}%"
                )

            # ================================================================
            # Registrar en histórico
            # ================================================================
            self.history['epoch'].append(epoch)
            self.history['temperature'].append(self.T)
            self.history['loss_spatial'].append(float(l_spatial))
            self.history['cluster_accuracy'].append(float(accuracy))
            self.history['loss_prox'].append(float(l_prox))
            self.history['loss_cover'].append(float(l_cover))

            state = FuzzyBSHState(
                epoch=epoch,
                temperature=self.T,
                loss_spatial=l_spatial,
                cluster_accuracy=accuracy
            )
            states.append(state)

        print("-" * 100)
        print(f"{'[TRAINING COMPLETE]':<8}\n")

        return states

    def get_hard_assignments(self, token_embeddings: np.ndarray) -> np.ndarray:
        """
        Obtiene asignaciones discretas (hard) de tokens a esferas.

        Returns:
            assignment: (n_tokens,) - cada token asignado a su esfera más probable
        """
        membership = self.membership_probs(token_embeddings)
        assignments = np.argmax(membership, axis=1)
        return assignments

    def to_dict(self) -> Dict:
        """Serializa el estado del árbol."""
        return {
            'n_spheres': self.n_spheres,
            'embed_dim': self.embed_dim,
            'temperature': float(self.T),
            'centers': self.centers.tolist(),
            'radii': self.radii.tolist(),
            'history': self.history
        }


# ============================================================================
# Funciones de utilidad
# ============================================================================

def create_synthetic_dataset() -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Crea un dataset sintético para training del Fuzzy BSH.

    Crea 22 tokens distribuidos en 3 clusters:
    - Esfera 0 (Programación): 8 tokens
    - Esfera 1 (Música): 7 tokens
    - Esfera 2 (Física): 7 tokens
    - Tokens polisémicos en las intersecciones

    Returns:
        (token_embeddings, ground_truth_clusters)
    """
    np.random.seed(42)

    # Generar centroides de clusters en espacio 3D
    cluster_centers = np.array([
        [0.0, 0.0, 0.0],    # Programación
        [3.0, 3.0, 0.0],    # Música
        [3.0, 0.0, 3.0]     # Física
    ], dtype=np.float32)

    # Tokens por cluster (monosémica)
    tokens_per_cluster = {
        0: ['python', 'for', 'while', 'variable', 'función', 'clase', 'array', 'import'],
        1: ['ritmo', 'sample', 'beat', 'tempo', 'acorde', 'melodía', 'notas'],
        2: ['orbita', 'campo', 'fuerza', 'masa', 'vector', 'energía', 'aceleración']
    }

    # Generar embeddings
    embeddings = []
    ground_truth = {0: [], 1: [], 2: []}
    token_names = []
    token_id = 0

    # Generar tokens monosémicos
    for sphere_id, token_names_list in tokens_per_cluster.items():
        center = cluster_centers[sphere_id]
        for name in token_names_list:
            # Embedding = centroide + ruido pequeño
            embedding = center + np.random.randn(3) * 0.3
            embeddings.append(embedding)
            ground_truth[sphere_id].append(token_id)
            token_names.append(name)
            token_id += 1

    # Generar tokens polisémicos (en intersecciones)
    # "bucle" → entre Prog y Música
    bucle_embedding = (cluster_centers[0] + cluster_centers[1]) / 2 + np.random.randn(3) * 0.2
    embeddings.append(bucle_embedding)
    ground_truth[0].append(token_id)
    token_names.append('bucle')
    token_id += 1

    # "frecuencia" → entre Música y Física
    freq_embedding = (cluster_centers[1] + cluster_centers[2]) / 2 + np.random.randn(3) * 0.2
    embeddings.append(freq_embedding)
    ground_truth[1].append(token_id)
    token_names.append('frecuencia')
    token_id += 1

    embeddings = np.array(embeddings, dtype=np.float32)

    print(f"[INFO] Dataset sintético creado:")
    print(f"  - {len(embeddings)} tokens totales")
    print(f"  - 3 esferas (Programación, Música, Física)")
    for sphere_id, token_ids in ground_truth.items():
        print(f"  - Esfera {sphere_id}: {len(token_ids)} tokens")

    return embeddings, ground_truth, token_names


def print_final_clustering(
    fuzzy_bsh: FuzzyBSH,
    token_embeddings: np.ndarray,
    token_names: List[str],
    ground_truth_clusters: Dict[int, List[int]],
    sphere_names: List[str]
) -> None:
    """
    Imprime el clustering final después del entrenamiento.
    """
    assignments = fuzzy_bsh.get_hard_assignments(token_embeddings)

    print(f"\n{'='*100}")
    print("[HARDENING COMPLETO — Árbol Discreto Final]")
    print(f"{'='*100}\n")

    for sphere_id, sphere_name in enumerate(sphere_names):
        tokens_in_sphere = [token_names[tid] for tid in range(len(token_names))
                            if assignments[tid] == sphere_id]

        print(f"{sphere_name}_Sphere:  {', '.join(tokens_in_sphere)}")

    print()

    # Evaluación
    correct = 0
    for tid, name in enumerate(token_names):
        # Encontrar esfera ground truth
        gt_sphere = None
        for sphere_id, token_ids in ground_truth_clusters.items():
            if tid in token_ids:
                gt_sphere = sphere_id
                break

        assigned_sphere = assignments[tid]

        if gt_sphere == assigned_sphere:
            correct += 1

    accuracy = correct / len(token_names) if token_names else 0.0
    print(f"Final Cluster Accuracy: {accuracy*100:.1f}% ({correct}/{len(token_names)})")
    print(f"{'='*100}\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fuzzy BSH para Backpropagation en LiquidBit"
    )

    parser.add_argument('--num-epochs', type=int, default=200,
                        help='Número de épocas (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Semilla RNG (default: 42)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Tasa de aprendizaje (default: 0.01)')
    parser.add_argument('--harden-every', type=int, default=50,
                        help='Endurecimiento cada N épocas (default: 50)')
    parser.add_argument('--harden-factor', type=float, default=0.9,
                        help='Factor de endurecimiento (default: 0.9)')
    parser.add_argument('--output', type=str, default='fuzzy_bsh_state.json',
                        help='Archivo de salida JSON (default: fuzzy_bsh_state.json)')

    args = parser.parse_args()

    # ========================================================================
    # PASO 1: Crear dataset sintético
    # ========================================================================
    print("\n[PASO 1] Creando dataset sintético...")
    token_embeddings, ground_truth_clusters, token_names = create_synthetic_dataset()

    # ========================================================================
    # PASO 2: Inicializar Fuzzy BSH
    # ========================================================================
    print("\n[PASO 2] Inicializando Fuzzy BSH...")
    fuzzy_bsh = FuzzyBSH(
        n_spheres=3,
        embed_dim=3,
        temperature=1.0,
        learning_rate=args.learning_rate,
        seed=args.seed,
        init_from_data=(token_embeddings, ground_truth_clusters)
    )
    print(f"  ✓ {fuzzy_bsh.n_spheres} esferas, dim={fuzzy_bsh.embed_dim}, T={fuzzy_bsh.T:.3f}")
    print(f"  ✓ Inicialización desde datos (centroides de clusters)")

    # ========================================================================
    # PASO 3: Entrenar con simulated annealing
    # ========================================================================
    print("\n[PASO 3] Entrenando Fuzzy BSH...")
    states = fuzzy_bsh.train(
        token_embeddings=token_embeddings,
        ground_truth_clusters=ground_truth_clusters,
        num_epochs=args.num_epochs,
        harden_every=args.harden_every,
        harden_factor=args.harden_factor
    )

    # ========================================================================
    # PASO 4: Imprimir clustering final
    # ========================================================================
    print("\n[PASO 4] Mostrando clustering final...")
    sphere_names = ["Programación", "Música", "Física"]
    print_final_clustering(
        fuzzy_bsh,
        token_embeddings,
        token_names,
        ground_truth_clusters,
        sphere_names
    )

    # ========================================================================
    # PASO 5: Guardar estado
    # ========================================================================
    print("[PASO 5] Guardando estado...")
    try:
        state_dict = fuzzy_bsh.to_dict()
        with open(args.output, 'w') as f:
            json.dump(state_dict, f, indent=2)
        print(f"  ✓ Estado guardado en {args.output}")
    except IOError as e:
        print(f"  ✗ Error al guardar: {e}", file=sys.stderr)
        sys.exit(1)

    # ========================================================================
    # Resumen final
    # ========================================================================
    print(f"\n{'='*100}")
    print("[RESUMEN FINAL]")
    print(f"{'='*100}")
    print(f"Configuración:")
    print(f"  - Épocas: {args.num_epochs}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Endurecimiento cada: {args.harden_every} épocas")
    print(f"  - Factor de endurecimiento: {args.harden_factor}")
    print(f"\nResultados finales:")
    final_state = states[-1]
    print(f"  - Pérdida espacial: {final_state.loss_spatial:.4f}")
    print(f"  - Precisión de clustering: {final_state.cluster_accuracy*100:.1f}%")
    print(f"  - Temperatura final: {final_state.temperature:.4f}")
    print(f"  - Output: {args.output}")
    print(f"{'='*100}\n")

    print("[✓] Fuzzy BSH training completado exitosamente.\n")


if __name__ == '__main__':
    main()
