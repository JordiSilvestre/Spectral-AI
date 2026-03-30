# SpectralAI Zero-Matrix — CLAUDE.md
> Guía de arquitectura y contexto para agentes de IA trabajando en este proyecto.

## 🎯 Visión del Proyecto

**SpectralAI Zero-Matrix** es un prototipo de modelo de lenguaje de nueva generación que **elimina completamente la multiplicación de matrices (MatMul)** del mecanismo de atención, sustituyéndolo por **geometría espacial acelerada por hardware** usando los RT Cores de NVIDIA (OptiX / Vulkan RT).

### El Problema que Resuelve

Los Transformers tradicionales (GPT-4, Gemini) tienen complejidad **O(N²)** en el mecanismo de atención:
- Para 100.000 tokens → 10.000 millones de celdas en la matriz de atención
- ~80 billones de FLOPs solo para la capa de atención
- Requiere racks de H100 (~30.000€/unidad) en paralelo
- El KV Cache con 96 capas consume ~307 GB de VRAM

### La Solución: Atención Óptica O(N log N)

En lugar de matrices, mapeamos tokens como **polígonos en un espacio 3D** estructurado en un árbol **BVH (Bounding Volume Hierarchy)**:
- Complejidad: **O(N log N)** — para 100K tokens, solo log₂(100.000) ≈ 17 pasos de traversal
- Operaciones: ~6.900 millones de intersecciones de rayos (vs 80 billones de FLOPs)
- Hardware target: **NVIDIA RTX 5070 Ti** (o RTX 4090) con RT Cores dedicados
- VRAM: Un BVH de 100K polígonos pesa ~10-50 MB (vs 307 GB de KV Cache)

---

## 🏗️ Arquitectura del Sistema

```
spectral-ai/
├── CLAUDE.md               ← Este archivo (contexto para agentes)
├── LEARNINGS.md            ← Registro de decisiones, fallos y aprendizajes
├── STATUS.md               ← Estado real: inventario, accuracy, roadmap
├── README.md               ← Descripción del proyecto
│
├── include/                ← Headers C++ públicos
│   ├── token_geometry.h    ← Struct TokenNode: token → objeto geométrico BVH
│   ├── semantic_bvh.h      ← Gestión del árbol BVH semántico
│   ├── optical_attention.h ← Interfaz del mecanismo de atención óptica
│   ├── spectral_ray.h      ← Codificación espectral: PrismaticRay, Snell
│   ├── alpha_bsh.h          ← Esferas semánticas + MatrixBlocks
│   └── spectral_resonance.h ← Resonancia Fourier (v4 research)
│
├── src/                    ← Implementaciones C++
│   ├── token_geometry.cpp  ← Proyección embedding → espacio 3D
│   ├── semantic_bvh.cpp    ← Construcción y actualización del BVH
│   └── alpha_bsh.cpp        ← Orchestración AlphaBSH Phase A/B
│
├── cuda/                   ← Kernels CUDA/OptiX
│   ├── ray_attention.cu    ← Kernel principal: ray_traced_attention_kernel
│   ├── ray_generation.cu   ← Generación de rayos + spectral color
│   ├── closest_hit.cu      ← Programa OptiX: ClosestHit + Ley de Snell
│   ├── miss.cu             ← Programa OptiX: Miss (token irrelevante)
│   ├── spectral_kernels.cu  ← Kernels OptiX raygen 4-level IAS
│   └── v5/                  ← Kernels CUDA producción
│       ├── bvh_router_kernel.cu   ← Router fusionado 3-nivel (105x speedup)
│       ├── bvh_torch_ext.cu       ← PyTorch extension zero-copy
│       └── ternary_torch_ext.cu   ← Expert ternario POPCOUNT
│
├── python/                 ← Pipeline Python (training + inference)
│   ├── olmoe_bvh_distill.py       ← Training BVH Router (EnhancedBVHRouter)
│   ├── olmoe_e2e_eval.py          ← Evaluación PPL end-to-end
│   ├── calibrate_router.py        ← Calibración linear/affine
│   ├── spectral_techniques.py     ← 6 técnicas: SmoothBVHHit, RMSNorm, DualLR...
│   ├── bvh_router.py              ← BVHRouter diferenciable (Gumbel-Softmax)
│   ├── bvh_router_bridge.py       ← Hybrid: PyTorch ↔ CUDA kernel auto-select
│   ├── orchestrator.py            ← Pipeline v5.0: Tokens → Router → Experts
│   └── spectral_bridge.py         ← Python ↔ C++ bridge (tokenize → 3D → binary)
│
├── tests/                  ← Tests unitarios y de integración
│   ├── test_spectral_techniques.py ← 37 tests CPU (todas pasando)
│   └── benchmark.cu        ← Benchmarks vs Transformer estándar
│
├── scripts/                ← Scripts de automatización
│   ├── train_remaining_layers.sh  ← Retrain capas débiles con --lyra
│   └── verify_all.sh              ← Verificación one-command
│
├── CMakeLists.txt          ← Sistema de build (CMake + CUDA + OptiX)
└── .gitattributes          ← Force LF para .sh (WSL compat)
```

---

## 📐 Conceptos Matemáticos Clave

### 1. Proyección Token → Geometría 3D

Cada token tiene un embedding de D dimensiones (ej. D=768 para BERT-base, D=4096 para GPT-4).
Lo proyectamos a un espacio 3D mediante **PCA esférica**:

```
embedding ∈ R^D  →  posición ∈ R^3  (centroide del polígono)
```

La posición 3D preserva la métrica coseno del espacio de embeddings:
- Tokens semánticamente similares → AABBs cercanos en el espacio 3D
- La agrupación geométrica refleja clústeres semánticos

### 2. Estructura del TokenNode (Objeto BVH)

```cpp
struct TokenNode {
    // Identidad
    uint32_t token_id;           // ID del token en el vocabulario
    uint32_t position_in_seq;    // Posición en la secuencia (0..N-1)

    // Geometría (para RT Cores)
    float3   centroid;           // Posición 3D del token en el espacio semántico
    float3   aabb_min;           // Bounding box mínimo (semántica)
    float3   aabb_max;           // Bounding box máximo (semántica)
    float    semantic_radius;    // Radio semántico (diversidad de contextos)

    // Embedding comprimido
    half     embedding[256];     // Embedding reducido FP16 (proyección de D→256)

    // Atención acumulada
    float    attention_weight;   // Peso de atención calculado por el rayo
    float    energy_remaining;   // Energía restante del rayo tras la colisión
};
```

### 3. Mecanismo de Atención Óptica

**Ray Generation (Prompt → Rayos):**
- El prompt actual emite `num_rays` rayos desde su posición semántica
- Cada rayo representa una "dimensión de pensamiento" (análogo a los query heads)
- Dirección inicial: calculada desde el embedding del token de query

**Closest Hit (Colisión semántica):**
- Cuando un rayo golpea un TokenNode, calcula la relevancia
- La "Pérdida de Energía" actúa como Attention Decay

**Fórmula de Attention Decay:**
```
attention_weight = E₀ · exp(-λ · d_semantic)
```
Donde:
- `E₀` = energía inicial del rayo (1.0)
- `λ` = coeficiente de absorción semántica (hyperparámetro, ~0.1)
- `d_semantic` = distancia semántica en el espacio 3D (proxy de irrelevancia)

**Complejidad resultante:** O(N log N) — el BVH descarta mitades del espacio en cada nivel

### 4. Ventaja Computacional

| Métrica | GPT-4 (MatMul) | SpectralAI (Ray Tracing) | Diferencia |
|---|---|---|---|
| Complejidad | O(N²) | O(N log₂ N) | ~5.882x para N=100K |
| Operaciones (N=100K) | ~80T FLOPs | ~6.9B intersecciones | ~11.500x menos |
| Hardware | Tensor Cores saturados | RT Cores dedicados | Silicio previamente inactivo |
| VRAM (KV Cache) | ~307 GB (96 capas) | ~10-50 MB (BVH) | ~6.000x menos |
| Hardware mínimo | Rack de H100 | RTX 4090 / 5070 Ti | Democratización total |

---

## 🔧 Stack Tecnológico

| Componente | Tecnología |
|---|---|
| Lenguaje principal | C++17 + CUDA 12.x |
| API de Ray Tracing | NVIDIA OptiX 8.x |
| API alternativa | Vulkan + VK_KHR_ray_tracing |
| Build system | CMake 3.28+ |
| Python bindings | pybind11 |
| Embeddings base | Pre-trained (Word2Vec / GloVe / BERT proyectados) |
| Hardware target | NVIDIA RTX 4090 / RTX 5070 Ti (Ada Lovelace / Blackwell) |
| CUDA Compute | sm_89 (Ada) / sm_100 (Blackwell) |

---

---

## 🌈 Arquitectura Ultra: SpectralAI Espectral (3 Ideas Combinadas)

### Idea 3: Codificación Espectral + Refracción Prismática

**El concepto:** El rayo lleva un "color" — un vector de contexto `f ∈ ℝ^64` que codifica el estado conversacional actual. Las esferas actúan como prismas ópticos: el índice de refracción `n(esfera, f) = σ(W_dispersion · f)` determina el ángulo de refracción, que selecciona qué sub-bloque de matrices cargar.

**Resuelve la polisemia sin duplicar matrices:**
- Rayo AZUL (contexto=Código) llega a esfera "Bucle" → refracta 45° → matrices de programación
- Rayo ROJO (contexto=Música) llega a la MISMA esfera → refracta 90° → matrices de ritmo
- Un único punto en el espacio. Dos respuestas distintas. Overhead: 0.03% del cómputo total.

**Ley de Snell semántica:**
```
n(esfera, f) = σ( W_dispersion · f )     ← W aprendida en training
sin(θ_salida) = sin(θ_entrada) / n(esfera, f)
d_salida = n_ratio * d_entrada + (n_ratio * cos_i - cos_t) * normal
```

### Pipeline de Inferencia Completo (3 Fases)

```
Fase 0: Spectral Encoding
  context_history → W_spectral (64×D) → color f ∈ ℝ^64
  [encodeContext() en SpectralBSH]

Fase A: OHBSC Traversal O(log N)
  PrismaticRay(origin=query_pos, color=f) navega el árbol BSH
  En cada nodo: n = σ(W_disp · f) → Snell → nueva dirección
  Wormholes: O(1) para conceptos en esferas lejanas
  Salida: leaf_sphere_id + matrix_block_id + ángulo_final

Fase B: cuBLAS Selectivo O(k²), k = N^(1/3)
  Carga lazy del MatrixBlock seleccionado por refracción
  MatMul FP16 completo → calidad GPT-4 en el contexto exacto
```

### Mecanismo de Wormholes (De los Documentos de Training)

Para conceptos polisémicos, en lugar de duplicar matrices:
- **Duplicar** si: `f_ij·s_k/C_d > exp(-λ·d_ij)/C_w + Φ(i,j)` (alta frecuencia + esferas cercanas)
- **Wormhole** en caso contrario: puntero O(1) entre esferas lejanas

```
DuplScore(C) = (Σ f(C,s)·R(C,s)) · exp(-γ·D(Sc)) - δ·(|Sc|-1)·size(C)
Si DuplScore > τ: duplicar. Sino: wormhole.
```

### Función de Pérdida Total (OHBSC Training)

```
L_total = L_task + α·L_spatial

L_spatial = L_prox + L_cover + L_inter

L_prox = Σ w_ij · (d_g(c_i,c_j) - δ_ij)²           ← similares deben estar cerca
L_cover = Σ [avg(d_g(c,center_n)/r_n) - 1]+         ← esferas deben cubrir sus tokens
L_inter = Σ ||pos(c) - proj_{Si∩Sj}(pos(c))||²      ← polisémicos en intersecciones
```

### Structs de la Arquitectura Ultra

| Struct | Archivo | Función |
|---|---|---|
| `TokenNode` | `token_geometry.h` | Token → objeto BVH (Idea 1) |
| `SemanticSphereAlpha` | `alpha_bsh.h` | Esfera + MatrixBlock (Idea 2) |
| `SpectralContext` | `spectral_ray.h` | Color del rayo (Idea 3) |
| `PrismaticSphere` | `spectral_ray.h` | Esfera-prisma con W_dispersion |
| `PrismaticRay` | `spectral_ray.h` | Rayo con color + Snell |
| `SpectralAttentionResult` | `spectral_ray.h` | Resultado completo 3 fases |

---

## 🚦 Estado Actual del Prototipo

| Fase | Estado | Descripción |
|---|---|---|
| Arquitectura conceptual | ✅ Completa | Diseño matemático validado |
| CUDA kernels v5 | ✅ Operativos | 105x speedup routing, POPCOUNT ternary |
| BVH Router distillation | ✅ 16/16 capas | top-8: 80-93% por capa |
| Spectral Techniques | ✅ 6 técnicas | SmoothBVHHit, RMSNorm, DualLR, etc. |
| PPL actual (16/16 BVH) | 🔄 ~8.27 | Baseline OLMoE: 6.11. Objetivo: <7.0 |
| Lyra en capas débiles | 🔄 En progreso | L1 con Lyra OK, L3/L5/L11 pendientes |
| Pipeline OptiX RT Cores | ⏳ Pendiente | PTX no funcional en sm_89 |
| Demo end-to-end | ⏳ Pendiente | Qwen 51.9 tok/s validado, falta integrar |
| Patentes | ⏳ 3 redactadas | Filing USPTO pendiente ($1,050) |

---

## ⚠️ Decisiones de Diseño Críticas

1. **Proyección D→3D:** Usamos PCA con preservación de métrica coseno. La pérdida de información es aceptable porque el BVH solo necesita la topología relativa, no la semántica exacta. Los 256 floats del embedding comprimido preservan el 95%+ de la varianza.

2. **Diferenciabilidad:** El mayor desafío para el entrenamiento. Los RT Cores no son diferenciables por defecto. Solución potencial: usar **Soft BVH** con gradientes aproximados, o entrenar los embeddings con un Transformer y solo sustituir la atención en inferencia.

3. **Equivalencia de operaciones:** Una intersección rayo-triángulo ≈ 20-30 FLOPs elementales. Esto reduce el factor de ventaja real a ~380x (vs los teóricos 11.500x), pero sigue siendo demoledor.

4. **Construcción del BVH:** El árbol se construye una vez por secuencia y se reutiliza para todos los layers. La construcción tiene coste O(N log N) amortizado.

---

## 📋 Instrucciones para Agentes

Cuando trabajes en este proyecto:

1. **Leer siempre LEARNINGS.md** antes de implementar algo — puede haber decisiones ya tomadas o errores ya cometidos.
2. **Actualizar LEARNINGS.md** cuando descubras un fallo, tomes una decisión importante, o encuentres una alternativa mejor.
3. **Los headers en `include/` son la fuente de verdad** — si hay conflicto entre un .cpp y un .h, el .h manda.
4. **No uses std::vector en código CUDA hot path** — usa arrays planos o thrust::device_vector.
5. **Toda memoria de GPU debe tener su counterpart de liberación** — no hay GC en CUDA.
6. **El prototipo no necesita ser production-ready** — necesita ser correcto y demostrar la viabilidad del O(N log N).
