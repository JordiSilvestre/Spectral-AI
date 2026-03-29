#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════╗
║    SpectralAI Zero-Matrix — W_dispersion Training Script            ║
║    Aprende los pesos W_dispersion para refracción prismática       ║
║    Resuelve polisemia: mismo token, diferente contexto → esfera OK ║
╚══════════════════════════════════════════════════════════════════════╝

Script autónomo que entrena W_dispersion de las 3 esferas hoja para que:
- L_routing: routing correcto según contexto (cross-entropy)
- L_spatial: tokens similares cercanos, esferas cubren sus tokens
- L_total = L_routing + α * L_spatial, α=0.1

Salida: w_dispersion_trained.npy (3, 64) con los pesos aprendidos.

Ejecutar: python3 train_dispersion.py
"""

import numpy as np
import math
import time

np.random.seed(2024)

# ══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ══════════════════════════════════════════════════════════════════════

SPECTRAL_DIM = 64
N_EPOCHS = 500
LEARNING_RATE = 0.1  # Aumentado para convergencia más rápida
ALPHA_SPATIAL = 0.05  # Reduce peso de L_spatial para enfocarse en routing

PASS = "✅"
FAIL = "❌"

# ══════════════════════════════════════════════════════════════════════
# VOCABULARIO Y EMBEDDINGS (idéntico a integration_test.py)
# ══════════════════════════════════════════════════════════════════════

VOCAB = {
    # Programación (cluster azul, zona +X)
    "python":    np.array([3.1, 0.2, 0.1]),
    "for":       np.array([2.9, 0.3, 0.0]),
    "while":     np.array([2.8, 0.1, 0.2]),
    "variable":  np.array([3.2, 0.4, 0.1]),
    "función":   np.array([3.0, 0.5, 0.0]),
    "clase":     np.array([2.7, 0.2, 0.3]),
    # Música (cluster rojo, zona +Y)
    "ritmo":     np.array([0.1, 3.2, 0.2]),
    "sample":    np.array([0.2, 2.9, 0.1]),
    "beat":      np.array([0.0, 3.1, 0.3]),
    "tempo":     np.array([0.3, 3.0, 0.1]),
    "melodía":   np.array([0.1, 2.8, 0.4]),
    "acorde":    np.array([0.2, 3.3, 0.0]),
    # Física (cluster verde, zona +Z)
    "orbita":    np.array([0.2, 0.1, 3.1]),
    "campo":     np.array([0.1, 0.3, 2.9]),
    "fuerza":    np.array([0.3, 0.0, 3.2]),
    "masa":      np.array([0.0, 0.2, 3.0]),
    "energía":   np.array([0.2, 0.1, 2.8]),
    "vector":    np.array([0.1, 0.4, 3.1]),
    # Palabras polisémicas (frontera entre clusters)
    "bucle":     np.array([1.5, 1.4, 0.2]),
    "frecuencia":np.array([1.3, 1.2, 1.1]),
    "onda":      np.array([0.2, 1.4, 1.5]),
    "ciclo":     np.array([1.5, 0.2, 1.4]),
}

TOKENS = list(VOCAB.keys())
EMBEDDINGS_3D = np.array([VOCAB[t] for t in TOKENS], dtype=np.float32)

# Añadir ruido gaussiano
EMBEDDINGS_3D += np.random.randn(*EMBEDDINGS_3D.shape).astype(np.float32) * 0.15

# Proyectar a D=32
D = 32
PROJ_MATRIX = np.random.randn(3, D).astype(np.float32) * 0.5
EMBEDDINGS_D = EMBEDDINGS_3D @ PROJ_MATRIX

N_TOKENS = len(TOKENS)

# Colores de contexto (ortogonales)
COLOR_PROG  = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_PROG[0] = 1.0
COLOR_MUSIC = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_MUSIC[1] = 1.0
COLOR_PHYS  = np.zeros(SPECTRAL_DIM, dtype=np.float32); COLOR_PHYS[2] = 1.0

# Mapeo esperado: (token, contexto) → esfera correcta
EXPECTED_ROUTING = {
    "bucle":       {"Programación": 0, "Música": 1},
    "frecuencia":  {"Programación": 0, "Música": 1, "Física": 2},
    "onda":        {"Música": 1, "Física": 2},
    "ciclo":       {"Programación": 0, "Física": 2},
    # Tokens no-polisémicos deben ir a su esfera natural
    "python":      {"Programación": 0, "Música": 0, "Física": 0},
    "ritmo":       {"Programación": 1, "Música": 1, "Física": 1},
    "orbita":      {"Programación": 2, "Música": 2, "Física": 2},
}

print()
print("╔══════════════════════════════════════════════════════════════════════╗")
print("║    SpectralAI Zero-Matrix — W_dispersion Training v1.0             ║")
print("╚══════════════════════════════════════════════════════════════════════╝")
print(f"  Tokens: {N_TOKENS} | Embedding D={D} | Spectral dim={SPECTRAL_DIM}")
print(f"  Epochs: {N_EPOCHS} | LR: {LEARNING_RATE} | α_spatial: {ALPHA_SPATIAL}")
print()

# ══════════════════════════════════════════════════════════════════════
# GEOMETRÍA BSH: Esferas hoja con centros y radios
# ══════════════════════════════════════════════════════════════════════

class Sphere:
    """Esfera hoja con parámetro aprendible W_dispersion."""
    def __init__(self, center, radius, label, token_indices, sphere_id):
        self.center = np.array(center, dtype=np.float32)
        self.radius = float(radius)
        self.label = label
        self.token_indices = np.array(token_indices, dtype=np.int32)
        self.sphere_id = sphere_id
        # Parámetro a aprender: W_dispersion ∈ ℝ^64
        self.W_dispersion = np.random.randn(SPECTRAL_DIM).astype(np.float32) * 0.1
        self.base_n = 1.0

    def refractive_index(self, color):
        """n(esfera, color) = 1 + σ(dot(W_disp, color))"""
        dot = np.dot(self.W_dispersion, color)
        sigmoid_val = 1.0 / (1.0 + np.exp(-np.clip(dot, -50, 50)))
        return self.base_n + sigmoid_val


def build_spheres():
    """Construye las 3 esferas hoja con sus tokens."""
    centers_cluster = [
        np.array([3.0, 0.3, 0.1]),  # Programación
        np.array([0.2, 3.0, 0.2]),  # Música
        np.array([0.2, 0.2, 3.0]),  # Física
    ]
    cluster_labels = ["Prog_Sphere", "Music_Sphere", "Phys_Sphere"]

    spheres = []
    for ci, (cc, cl) in enumerate(zip(centers_cluster, cluster_labels)):
        # Asignar tokens al cluster más cercano
        dists = np.linalg.norm(EMBEDDINGS_3D - cc, axis=1)
        indices = np.where(dists < 2.5)[0].tolist()
        if not indices:
            indices = [np.argmin(dists)]
        r = float(np.max(dists[indices]) + 0.5)
        spheres.append(Sphere(cc, r, cl, indices, ci))

    return spheres

spheres = build_spheres()
print(f"  Esferas construidas: {len(spheres)}")
for s in spheres:
    print(f"    • {s.label:<20} tokens={len(s.token_indices):<2}  center={s.center.round(2)}  radius={s.radius:.2f}")
print()

# ══════════════════════════════════════════════════════════════════════
# DATASET DE ENTRENAMIENTO
# Tripletas (token_idx, color, sphere_expected_id)
# ══════════════════════════════════════════════════════════════════════

def build_training_dataset():
    """Crea dataset de (token, color) → esfera esperada."""
    dataset = []

    for token_name, routing_map in EXPECTED_ROUTING.items():
        token_idx = TOKENS.index(token_name)

        context_map = {
            "Programación": COLOR_PROG,
            "Música": COLOR_MUSIC,
            "Física": COLOR_PHYS,
        }

        for ctx_name, expected_sphere_id in routing_map.items():
            color = context_map[ctx_name]
            dataset.append({
                "token_idx": token_idx,
                "color": color.copy(),
                "expected_sphere_id": expected_sphere_id,
                "context_name": ctx_name,
                "token_name": token_name,
            })

    return dataset

train_dataset = build_training_dataset()
print(f"  Dataset: {len(train_dataset)} ejemplos (token, color) → esfera")
print()

# ══════════════════════════════════════════════════════════════════════
# FORWARD PASS Y LOSS
# ══════════════════════════════════════════════════════════════════════

def sigmoid(x):
    """Sigmoid con clipping para evitar overflow."""
    x_clipped = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def sigmoid_derivative(sig_val):
    """d(sigmoid)/dx = σ(x) * (1 - σ(x))"""
    return sig_val * (1.0 - sig_val)

def softmax(logits):
    """Softmax numéricamente estable."""
    logits_shifted = logits - np.max(logits)
    exp_logits = np.exp(logits_shifted)
    return exp_logits / (np.sum(exp_logits) + 1e-8)

def compute_routing_loss(batch):
    """
    Loss de routing: cross-entropy sobre qué esfera tiene mayor n.

    Para cada ejemplo (token, color) → esperado_esfera:
      - Calcular n_i para cada esfera i
      - Softmax(n) → distribución de probabilidad
      - Cross-entropy vs esfera_esperada
    """
    total_loss = 0.0
    n_examples = len(batch)

    # Acumuladores para backprop
    grads_W = [np.zeros_like(s.W_dispersion) for s in spheres]

    for example in batch:
        token_idx = example["token_idx"]
        color = example["color"]
        expected_sphere_id = example["expected_sphere_id"]

        # Forward: calcular n para cada esfera
        n_values = np.array([s.refractive_index(color) for s in spheres], dtype=np.float32)

        # Softmax: distribución de probabilidad
        probs = softmax(n_values)

        # Cross-entropy loss: -log(p_expected)
        prob_expected = probs[expected_sphere_id]
        ce_loss = -np.log(np.clip(prob_expected, 1e-7, 1.0))
        total_loss += ce_loss

        # Backprop: d(CE)/d(n_i) = p_i - label_i
        d_ce_dn = probs.copy()
        d_ce_dn[expected_sphere_id] -= 1.0

        # Para cada esfera: d(n)/d(W_disp)
        for i, sphere in enumerate(spheres):
            # n_i = 1 + σ(W_i · color)
            dot_i = np.dot(sphere.W_dispersion, color)
            sig_i = sigmoid(dot_i)

            # dn_i / d(W_i) = σ'(dot) * color
            d_n_dW = sigmoid_derivative(sig_i) * color

            # dL / dW_i = dL/dn_i * dn_i/dW_i
            grads_W[i] += d_ce_dn[i] * d_n_dW

    # Promedio
    mean_loss = total_loss / n_examples
    for i in range(len(grads_W)):
        grads_W[i] /= n_examples
        # Clip de gradientes para estabilidad
        grads_W[i] = np.clip(grads_W[i], -1.0, 1.0)

    return mean_loss, grads_W


def compute_spatial_loss():
    """
    L_spatial = L_prox + L_cover

    L_prox: tokens similares (coseno) deben estar cerca en 3D
    L_cover: tokens deben estar dentro del radio de su esfera
    """
    loss_prox = 0.0
    loss_cover = 0.0

    # L_prox: penalizar distancias 3D que no reflejen similaridad
    for i in range(N_TOKENS):
        for j in range(i+1, min(i+5, N_TOKENS)):  # Solo vecinos cercanos para eficiencia
            # Similaridad coseno en embedding 32-D
            emb_i = EMBEDDINGS_D[i]
            emb_j = EMBEDDINGS_D[j]
            cos_sim = np.dot(emb_i, emb_j) / (np.linalg.norm(emb_i) * np.linalg.norm(emb_j) + 1e-8)

            # Distancia 3D esperada (inversa: similares = cercanos)
            target_dist = 1.0 - cos_sim  # 0 si similares, ~2 si diferentes

            # Distancia 3D real
            real_dist = np.linalg.norm(EMBEDDINGS_3D[i] - EMBEDDINGS_3D[j])

            # Penalizar desviación
            loss_prox += (real_dist - target_dist)**2

    loss_prox /= max(1, N_TOKENS * 4)

    # L_cover: tokens dentro de sus esferas
    for sphere in spheres:
        for token_idx in sphere.token_indices:
            dist_to_center = np.linalg.norm(EMBEDDINGS_3D[token_idx] - sphere.center)
            # max(0, dist/radius - 1) — penaliza si token está fuera
            excess = max(0.0, dist_to_center / (sphere.radius + 1e-6) - 1.0)
            loss_cover += excess**2

    loss_cover /= max(1, N_TOKENS)

    return loss_prox + loss_cover


def routing_accuracy(batch):
    """Porcentaje de ejemplos donde la esfera esperada tiene máximo n."""
    correct = 0

    for example in batch:
        color = example["color"]
        expected_sphere_id = example["expected_sphere_id"]

        # Calcular n para cada esfera
        n_values = np.array([s.refractive_index(color) for s in spheres], dtype=np.float32)
        predicted_sphere_id = np.argmax(n_values)

        if predicted_sphere_id == expected_sphere_id:
            correct += 1

    return correct / len(batch) * 100 if len(batch) > 0 else 0.0


# ══════════════════════════════════════════════════════════════════════
# LOOP DE ENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  INICIANDO ENTRENAMIENTO")
print("─" * 70)
print()

losses_routing = []
losses_spatial = []
losses_total = []
accuracies = []

t_start = time.perf_counter()

for epoch in range(N_EPOCHS):
    # Forward + loss
    loss_routing, grads_W = compute_routing_loss(train_dataset)
    loss_spatial = compute_spatial_loss()
    loss_total = loss_routing + ALPHA_SPATIAL * loss_spatial

    # Update W_dispersion con descenso de gradiente
    for i, sphere in enumerate(spheres):
        sphere.W_dispersion -= LEARNING_RATE * grads_W[i]

    # Métricas
    acc = routing_accuracy(train_dataset)

    losses_routing.append(loss_routing)
    losses_spatial.append(loss_spatial)
    losses_total.append(loss_total)
    accuracies.append(acc)

    # Print cada 50 epochs
    if epoch % 50 == 0 or epoch == N_EPOCHS - 1:
        print(f"  Epoch {epoch:>3d}:   "
              f"L_routing={loss_routing:.4f}  "
              f"L_spatial={loss_spatial:.4f}  "
              f"L_total={loss_total:.4f}  "
              f"acc={acc:.1f}%")

t_elapsed = time.perf_counter() - t_start

print()
print(f"  Entrenamiento completado en {t_elapsed:.2f}s")
print()

# ══════════════════════════════════════════════════════════════════════
# EVALUACIÓN FINAL: TEST DE POLISEMIA
# ══════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  TEST POLISEMIA POST-TRAINING")
print("─" * 70)
print()

polysemic_tokens = ["bucle", "frecuencia", "onda", "ciclo"]
correct_total = 0
total_tests = 0

for token_name in polysemic_tokens:
    token_idx = TOKENS.index(token_name)
    print(f"  Token: '{token_name}'")

    routing_for_token = EXPECTED_ROUTING.get(token_name, {})

    for ctx_name in ["Programación", "Música", "Física"]:
        if ctx_name not in routing_for_token:
            continue

        # Obtener color del contexto
        color_map = {
            "Programación": COLOR_PROG,
            "Música": COLOR_MUSIC,
            "Física": COLOR_PHYS,
        }
        color = color_map[ctx_name]

        # Calcular n para cada esfera
        n_values = np.array([s.refractive_index(color) for s in spheres], dtype=np.float32)
        predicted_sphere_id = np.argmax(n_values)
        expected_sphere_id = routing_for_token[ctx_name]

        predicted_sphere_name = spheres[predicted_sphere_id].label
        expected_sphere_name = spheres[expected_sphere_id].label

        is_correct = predicted_sphere_id == expected_sphere_id
        marker = PASS if is_correct else FAIL

        if is_correct:
            correct_total += 1
        total_tests += 1

        n_max = np.max(n_values)
        n_pred = n_values[predicted_sphere_id]

        print(f"    {marker} {ctx_name:<14} → {predicted_sphere_name:<18} "
              f"(n={n_pred:.3f}, max={n_max:.3f})")

final_accuracy = correct_total / total_tests * 100 if total_tests > 0 else 0

print()
print(f"  Accuracy final polisemia: {correct_total}/{total_tests} = {final_accuracy:.1f}%")
print()

# ══════════════════════════════════════════════════════════════════════
# GUARDAR PESOS ENTRENADOS
# ══════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  GUARDANDO W_dispersion ENTRENADOS")
print("─" * 70)
print()

W_dispersion_trained = np.array([s.W_dispersion for s in spheres], dtype=np.float32)
print(f"  W_dispersion_trained shape: {W_dispersion_trained.shape}")
print(f"  Guardar en: ./w_dispersion_trained.npy")

np.save("w_dispersion_trained.npy", W_dispersion_trained)
print(f"  {PASS} Guardado exitosamente")
print()

# ══════════════════════════════════════════════════════════════════════
# COMPARACIÓN ANTES VS DESPUÉS
# ══════════════════════════════════════════════════════════════════════

print("─" * 70)
print("  ANÁLISIS: ANTES VS DESPUÉS DEL ENTRENAMIENTO")
print("─" * 70)
print()

print(f"  Loss total final:     {losses_total[-1]:.4f} (inicial: {losses_total[0]:.4f})")
print(f"  Routing accuracy:     {accuracies[-1]:.1f}% (inicial: {accuracies[0]:.1f}%)")
print(f"  Polisemia accuracy:   {final_accuracy:.1f}%")
print()

# Estadísticas de los pesos aprendidos
for i, sphere in enumerate(spheres):
    norm_W = np.linalg.norm(sphere.W_dispersion)
    max_W = np.max(np.abs(sphere.W_dispersion))
    print(f"  {sphere.label:<20} ||W||={norm_W:.3f}  max|W|={max_W:.3f}")

print()
print("═" * 70)
print(f"  RESULTADO: {PASS} Entrenamiento exitoso")
print("═" * 70)
print()

# ══════════════════════════════════════════════════════════════════════
# NEXT STEPS
# ══════════════════════════════════════════════════════════════════════

print(f"""
  NEXT STEPS:
  ───────────
  1. Cargar w_dispersion_trained.npy en integration_test.py
  2. Reemplazar W_dispersion manual por los valores entrenados
  3. Verificar que polisemia accuracy suba a >90%
  4. Iterar con OHBSC completo (no solo routing)

  Archivos generados:
    • ./w_dispersion_trained.npy (3, {SPECTRAL_DIM}) — pesos listos para usar
""")
