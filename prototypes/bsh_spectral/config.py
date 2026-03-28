#!/usr/bin/env python3
"""
Configuración y parámetros del Prototipo A: BSH Espectral.
Centraliza todos los hiperparámetros ajustables.
"""

# ============================================================================
# PARÁMETROS DE CONSTRUCCIÓN DEL ÁRBOL
# ============================================================================

# Dimensionalidad de embeddings de entrada
EMBEDDING_DIM = 256

# Dimensión de proyección PCA (espacio 3D para geometría)
TARGET_SPATIAL_DIM = 3

# Profundidad máxima del árbol BSH
MAX_TREE_DEPTH = 6

# Número mínimo de tokens por hoja (caso base de recursión)
MIN_TOKENS_PER_LEAF = 2

# Semilla para reproducibilidad
RANDOM_SEED = 42


# ============================================================================
# PARÁMETROS DE RAYOS ESPECTRALES
# ============================================================================

# Dimensión del vector de contexto (color) del rayo
SPECTRAL_COLOR_DIM = 64

# Energía inicial de cada rayo
INITIAL_RAY_ENERGY = 1.0

# Máxima profundidad de traversal (prevención de bucles infinitos)
MAX_TRAVERSAL_DEPTH = 20

# Número de rayos a lanzar por query (análogo a num_heads en Transformer)
NUM_RAYS_PER_QUERY = 8


# ============================================================================
# PARÁMETROS ÓPTICOS (PRISMAS)
# ============================================================================

# Índice de refracción base de todas las esferas
BASE_REFRACTIVE_INDEX = 1.0

# Rango de índices: [BASE, BASE + REFRACTIVE_INDEX_RANGE]
REFRACTIVE_INDEX_RANGE = 1.0  # → n ∈ [1.0, 2.0)

# Escala de los pesos W_dispersion (aprendidos en training)
W_DISPERSION_SCALE = 0.1

# Escala del coeficiente de absorción (atención decay)
ABSORPTION_COEFFICIENT = 0.1


# ============================================================================
# PARÁMETROS DE MatMul SELECTIVO
# ============================================================================

# Tamaño de los bloques de matriz por esfera
MATRIX_BLOCK_SIZE = 16

# Dimensión del espacio de salida de MatMul
OUTPUT_DIM = 768

# Simulación de TFLOPS en GPU (para estimación de tiempo)
GPU_TFLOPS_FP32 = 100.0  # RTX 4090 ≈ 100 TFLOPS FP32

# Factor de escala para simulación de tiempo
TIME_SCALE_FACTOR = 10.0


# ============================================================================
# PARÁMETROS DE DEMO POLISEMIA
# ============================================================================

# Número de contextos semánticos distintos en demo
NUM_SEMANTIC_CONTEXTS = 3

# Número de palabras polisémicas (aparecen en 2+ contextos)
NUM_POLYSEMIC_WORDS = 1  # "bucle"

# Tokens por contexto en demo de polisemia
TOKENS_PER_CONTEXT_DEMO = 6


# ============================================================================
# PARÁMETROS DE BENCHMARK
# ============================================================================

# Tamaños de vocabulario a probar
BENCHMARK_VOCAB_SIZES = [50, 100, 500, 1000, 2000, 5000]

# Número de traversals por tamaño
TRAVERSALS_PER_SIZE = 5

# Tamaños para benchmark de MatMul selectivo
BENCHMARK_MATMUL_SIZES = [(1000, 32), (5000, 64), (10000, 128)]


# ============================================================================
# PARÁMETROS DE VALIDACIÓN
# ============================================================================

# Tolerancia para verificación de O(log N)
# Si ratio de (nodos / log N) está en [0.5, 2.0], se considera válido
COMPLEXITY_VALIDATION_BOUNDS = (0.5, 2.0)

# Tolerancia numérica para comparaciones vectoriales
NUMERICAL_TOLERANCE = 1e-6


# ============================================================================
# FUNCIÓN AUXILIAR
# ============================================================================

def print_config():
    """Imprime la configuración actual."""
    import sys
    print("=" * 60)
    print("CONFIGURACIÓN: PROTOTIPO A - BSH ESPECTRAL")
    print("=" * 60)

    config_dict = {k: v for k, v in globals().items()
                   if not k.startswith('_') and k.isupper()}

    for key, value in sorted(config_dict.items()):
        if isinstance(value, (int, float, str)):
            print(f"  {key:.<40} {value}")
        elif isinstance(value, (list, tuple)):
            print(f"  {key:.<40} {value}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    print_config()
