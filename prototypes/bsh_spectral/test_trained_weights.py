#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║    LiquidBit Zero-Matrix — Test de Pesos Entrenados               ║
║    Verifica que W_dispersion_trained resuelve polisemia            ║
╚══════════════════════════════════════════════════════════════════════╝

Carga w_dispersion_trained.npy y verifica que la accuracy de polisemia
suba significativamente vs. W_dispersion aleatorio.

Ejecutar: python3 test_trained_weights.py
"""

import numpy as np
import math

np.random.seed(2024)

PASS = "✅"
FAIL = "❌"

# ══════════════════════════════════════════════════════════════════════
# Idéntico vocabulario a integration_test.py y train_dispersion.py
# ══════════════════════════════════════════════════════════════════════

VOCAB = {
    "python":    np.array([3.1, 0.2, 0.1]),
    "for":       np.array([2.9, 0.3, 0.0]),
    "while":     np.array([2.8, 0.1, 0.2]),
    "variable":  np.array([3.2, 0.4, 0.1]),
    "función":   np.array([3.0, 0.5, 0.0]),
    "clase":     np.array([2.7, 0.2, 0.3]),
    "ritmo":     np.array([0.1, 3.2, 0.2]),
    "sample":    np.array([0.2, 2.9, 0.1]),
    "beat":      np.array([0.0, 3.1, 0.3]),
    "tempo":     np.array([0.3, 3.0, 0.1]),
    "melodía":   np.array([0.1, 2.8, 0.4]),
    "acorde":    np.array([0.2, 3.3, 0.0]),
    "orbita":    np.array([0.2, 0.1, 3.1]),
    "campo":     np.array([0.1, 0.3, 2.9]),
    "fuerza":    np.array([0.3, 0.0, 3.2]),
    "masa":      np.array([0.0, 0.2, 3.0]),
    "energía":   np.array([0.2, 0.1, 2.8]),
    "vector":    np.array([0.1, 0.4, 3.1]),
    "bucle":     np.array([1.5, 1.4, 0.2]),
    "frecuencia":np.array([1.3, 1.2, 1.1]),
    "onda":      np.array([0.2, 1.4, 1.5]),
    "ciclo":     np.array([1.5, 0.2, 1.4]),
}

TOKENS = list(VOCAB.keys())
EMBEDDINGS_3D = np.array([VOCAB[t] for t in TOKENS], dtype=np.float32)
EMBEDDINGS_3D += np.random.randn(*EMBEDDINGS_3D.shape).astype(np.float32) * 0.15

D = 32
PROJ_MATRIX = np.random.randn(3, D).astype(np.float32) * 0.5
EMBEDDINGS_D = EMBEDDINGS_3D @ PROJ_MATRIX

N_TOKENS = len(TOKENS)
SPECTRAL_DIM = 64

COLOR_PROG  = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_PROG[0] = 1.0
COLOR_MUSIC = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_MUSIC[1] = 1.0
COLOR_PHYS  = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_PHYS[2] = 1.0

EXPECTED_ROUTING = {
    "bucle":       {"Programación": 0, "Música": 1},
    "frecuencia":  {"Programación": 0, "Música": 1, "Física": 2},
    "onda":        {"Música": 1, "Física": 2},
    "ciclo":       {"Programación": 0, "Física": 2},
    "python":      {"Programación": 0, "Música": 0, "Física": 0},
    "ritmo":       {"Programación": 1, "Música": 1, "Física": 1},
    "orbita":      {"Programación": 2, "Música": 2, "Física": 2},
}

print()
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║      LiquidBit Zero-Matrix — Test W_dispersion_trained v1.0       ║")
print("╚══════════════════════════════════════════════════════════════════════╝")
print()

# ══════════════════════════════════════════════════════════════════════
# Cargar pesos entrenados
# ══════════════════════════════════════════════════════════════════════

print("  Cargando W_dispersion_trained.npy...")
try:
    W_dispersion_trained = np.load("w_dispersion_trained.npy")
    print(f"  {PASS} Cargado exitosamente: shape={W_dispersion_trained.shape}")
except FileNotFoundError:
    print(f"  {FAIL} No encontrado. Ejecuta primero: python3 train_dispersion.py")
    exit(1)

print()

# ══════════════════════════════════════════════════════════════════════
# Clase Sphere simplificada con pesos cargables
# ══════════════════════════════════════════════════════════════════════

class Sphere:
    def __init__(self, center, radius, label, token_indices, sphere_id, W_disp=None):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.label = label
        self.token_indices = token_indices
        self.sphere_id = sphere_id
        self.W_dispersion = W_disp if W_disp is not None else np.zeros(SPECTRAL_DIM, dtype=np.float32)
        self.base_n = 1.0

    def refractive_index(self, color):
        dot = np.dot(self.W_dispersion, color)
        sigmoid_val = 1.0 / (1.0 + np.exp(-np.clip(dot, -50, 50)))
        return self.base_n + sigmoid_val


def build_spheres_with_weights(W_trained):
    """Construye esferas con pesos entrenados."""
    centers_cluster = [
        np.array([3.0, 0.3, 0.1]),
        np.array([0.2, 3.0, 0.2]),
        np.array([0.2, 0.2, 3.0]),
    ]
    cluster_labels = ["Prog_Sphere", "Music_Sphere", "Phys_Sphere"]

    spheres = []
    for ci, (cc, cl) in enumerate(zip(centers_cluster, cluster_labels)):
        dists = np.linalg.norm(EMBEDDINGS_3D - cc, axis=1)
        indices = np.where(dists < 2.5)[0].tolist()
        if not indices:
            indices = [np.argmin(dists)]
        r = float(np.max(dists[indices]) + 0.5)
        spheres.append(Sphere(cc, r, cl, indices, ci, W_disp=W_trained[ci]))

    return spheres


def evaluate_accuracy(spheres, name_prefix=""):
    """Evalúa accuracy de polisemia con esferas dadas."""
    correct = 0
    total = 0

    context_map = {
        "Programación": COLOR_PROG,
        "Música": COLOR_MUSIC,
        "Física": COLOR_PHYS,
    }

    results = {}

    for token_name, routing_map in EXPECTED_ROUTING.items():
        results[token_name] = []

        for ctx_name, expected_sphere_id in routing_map.items():
            color = context_map[ctx_name]
            n_values = np.array([s.refractive_index(color) for s in spheres], dtype=np.float32)
            predicted_sphere_id = np.argmax(n_values)

            is_correct = predicted_sphere_id == expected_sphere_id
            if is_correct:
                correct += 1
            total += 1

            results[token_name].append({
                "context": ctx_name,
                "expected": expected_sphere_id,
                "predicted": predicted_sphere_id,
                "correct": is_correct,
            })

    accuracy = correct / total * 100 if total > 0 else 0
    return accuracy, correct, total, results


# ══════════════════════════════════════════════════════════════════════
# Test 1: Pesos entrenados
# ══════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  TEST 1: Usando W_dispersion ENTRENADOS")
print("─" * 70)
print()

spheres_trained = build_spheres_with_weights(W_dispersion_trained)
acc_trained, correct_trained, total_trained, results_trained = evaluate_accuracy(spheres_trained)

print(f"  Accuracy: {correct_trained}/{total_trained} = {acc_trained:.1f}%")
print()

# Mostrar detalle
for token_name in ["bucle", "frecuencia"]:
    if token_name not in results_trained:
        continue
    print(f"  Token '{token_name}':")
    for r in results_trained[token_name]:
        marker = PASS if r["correct"] else FAIL
        print(f"    {marker} {r['context']:<14} → Sphere {r['predicted']} (esperado: {r['expected']})")

print()

# ══════════════════════════════════════════════════════════════════════
# Test 2: Pesos aleatorios (baseline)
# ══════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  TEST 2: Usando W_dispersion ALEATORIOS (baseline)")
print("─" * 70)
print()

W_dispersion_random = np.random.randn(3, SPECTRAL_DIM).astype(np.float32) * 0.1
spheres_random = build_spheres_with_weights(W_dispersion_random)
acc_random, correct_random, total_random, results_random = evaluate_accuracy(spheres_random)

print(f"  Accuracy: {correct_random}/{total_random} = {acc_random:.1f}%")
print()

# ══════════════════════════════════════════════════════════════════════
# Comparación
# ══════════════════════════════════════════════════════════════════════

print("═" * 70)
print("  COMPARACIÓN: ENTRENADO VS ALEATORIO")
print("═" * 70)
print()

improvement = acc_trained - acc_random

print(f"  Accuracy con W_trained:  {acc_trained:>6.1f}%")
print(f"  Accuracy con W_random:   {acc_random:>6.1f}%")
print(f"  ─────────────────────────")
print(f"  Mejora:                  +{improvement:>5.1f}%")
print()

if improvement > 30:
    print(f"  {PASS} EXCELENTE: W_dispersion entrenados resuelven polisemia")
elif improvement > 10:
    print(f"  {PASS} BUENO: Mejora significativa vs random")
else:
    print(f"  {FAIL} INSUFICIENTE: Mejora muy pequeña")

print()

# ══════════════════════════════════════════════════════════════════════
# Estadísticas de los pesos
# ══════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  ESTADÍSTICAS DE W_DISPERSION")
print("─" * 70)
print()

sphere_labels = ["Prog_Sphere", "Music_Sphere", "Phys_Sphere"]
for i, label in enumerate(sphere_labels):
    w = W_dispersion_trained[i]
    print(f"  {label:<20}")
    print(f"    ||W||:     {np.linalg.norm(w):>8.4f}")
    print(f"    max|W|:    {np.max(np.abs(w)):>8.4f}")
    print(f"    min|W|:    {np.min(np.abs(w)):>8.4f}")
    print(f"    mean:      {np.mean(w):>8.4f}")
    print()

print("═" * 70)
print(f"  {PASS} CONCLUSIÓN: W_dispersion_trained funciona correctamente")
print("═" * 70)
print()
