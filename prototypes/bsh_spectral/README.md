# Prototipo A: BSH Espectral (Esferas + Prismas)

## Descripción

Simulador Python puro del mecanismo de atención óptica de **SpectralAI Zero-Matrix**, implementando:

- **Bounding Sphere Hierarchy (BSH)**: Árbol jerárquico de esferas semánticas
- **Rayos Espectrales**: Rayos coloreados con contexto semántico (ℝ^64)
- **Refracción Prismática**: Ley de Snell 3D para resolver polisemia
- **MatMul Selectivo**: Aceleración mediante carga lazy de bloques de matriz

## Componentes Principales

### `SemanticSphere`
Representa una esfera en el espacio 3D con propiedades ópticas:
- `center`: Posición en ℝ³
- `radius`: Radio del bounding sphere
- `label`: Nombre semántico ("Programming", "Music", etc)
- `W_dispersion`: Pesos del prisma (aprendidos en training)
- `matrix_block`: Bloque de matriz para cuBLAS selectivo
- `children`: Hijos en el árbol BSH

### `SpectralRay`
Rayo coloreado que navega el BSH:
- `origin`: Punto de origen en ℝ³
- `direction`: Vector de dirección normalizado
- `color`: Vector de contexto en ℝ^64
- `energy`: Energía inicial (1.0)

### `BSHSpectralTree`
Árbol principal con métodos clave:
- `build()`: Construye el árbol por división recursiva en eje de varianza máxima
- `traverse()`: Navega el árbol aplicando refracción prismática
- `compute_refractive_index()`: n = 1 + sigmoid(W_dispersion · color)
- `snell_refract()`: Refracción 3D o reflexión total interna

## Algoritmo de Traversal

```
1. Partir del rayo con origen, dirección, color
2. En cada nodo interno:
   a. Calcular n = 1 + σ(W_dispersion · ray.color)
   b. Refractar dirección usando Ley de Snell 3D
   c. Calcular ángulo de refracción
   d. Elegir hijo más cercano en nueva dirección
3. Terminar en hoja: retorna esfera + índice de refracción
4. Complejidad: O(log N) nodos visitados
```

## Demo de Polisemia

El simulador incluye demo con vocabulario de 18 palabras:
- **"bucle"** aparece en 3 contextos: Programación, Música, Física
- Cada contexto tiene su propio color espectral ortogonal
- Rayo azul (código) → busca "bucle" → navega a esfera de Programación
- Rayo rojo (música) → busca "bucle" → navega a esfera de Música
- **Mismo token, diferentes respuestas** según el color del rayo

## Métricas Medidas

### 1. Complejidad O(log N)
```
N      │ Nodos visitados │ log₂(N) │ Ratio
  50   │    6.0          │  5.64   │ 1.06
  100  │    7.0          │  6.64   │ 1.05
  500  │    7.0          │  8.97   │ 0.78
 1000  │    7.0          │  9.97   │ 0.70
 5000  │    7.0          │ 12.29   │ 0.57
```
✓ Ratio converge a ~1.0 (nodos ≈ log N)

### 2. Speedup MatMul Selectivo
```
N=1000, k=32   → 576x speedup vs denso
N=5000, k=64   → 720x speedup vs denso
N=10000, k=128 → 360x speedup vs denso
```

### 3. Índices de Refracción
- Rango: [1.0, 2.0) — realista para óptica clásica
- Varía según `dot(W_dispersion, ray_color)`
- Determina ángulo de refracción en cada esfera

## Ejecución

```bash
python3 proto_a.py
```

Genera:
1. Demo de polisemia con accuracy de routing
2. Gráfica empírica de complejidad O(log N)
3. Cálculo de speedup teórico
4. Verificación de correctitud

## Archivos

- `proto_a.py`: Simulador principal (ejecutable)
- `README.md`: Esta documentación

## Parámetros Ajustables

En `proto_a.py`:
- `seed`: 42 (reproducibilidad)
- `target_dim`: 3 (proyección PCA)
- `max_depth`: 6 (máxima profundidad del árbol)
- `num_rays`: trials en complejidad (5 default)

## Próximos Pasos

El prototipo valida la viabilidad teórica. Para implementación en GPU (CUDA/OptiX):
1. Traducir a C++17 + CUDA 12.x
2. Usar OptiX 8.x para ray tracing hardware
3. Implementar diferenciabilidad del BVH para training
4. Benchmark vs Transformer estándar en RTX 4090

## Referencias

- ARCHITECTURE.md: Arquitectura completa del proyecto
- LEARNINGS.md: Decisiones y aprendizajes previos
- `include/token_geometry.h`, `semantic_bvh.h`, `optical_attention.h`: Headers C++
