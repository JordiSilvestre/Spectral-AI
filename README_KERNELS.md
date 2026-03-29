# SpectralAI Zero-Matrix — Kernels CUDA/OptiX

## Descripción General

Este directorio contiene los kernels CUDA/OptiX que implementan el mecanismo de atención óptica usando ray tracing acelerado por hardware (RT Cores de NVIDIA).

## Estructura de Archivos

### Headers Compartidos

- **`include/optical_attention.h`**
  - Definiciones de tipos: `RayPayload`, `TokenNode`, `AttentionResult`
  - Constantes de configuración: `SPECTRAL_LAMBDA`, `SPECTRAL_ENERGY_THRESHOLD`
  - Funciones auxiliares device: `compute_attention_weight()`, `insert_top_token()`
  - **Uso:** Incluir en todos los kernels CUDA/OptiX

- **`include/token_geometry.h`**
  - Función de distancia semántica: `semantic_distance_3d()`
  - Utilidades de AABB: `point_in_aabb()`, `aabb_center()`, `aabb_half_extent()`
  - Estructura de configuración de proyección: `ProjectionConfig`
  - **Uso:** Utilidades para cálculos geométricos en kernels

### Kernels CUDA

- **`cuda/ray_attention.cu`** — Kernel Principal
  - **Función:** `ray_traced_attention_kernel()` — Ejecuta la traversal del BVH para cada query token
  - **Entrada:** Array de query tokens, posiciones 3D, direcciones de rayos
  - **Salida:** Array de `AttentionResult` con top-K tokens y pesos
  - **Complejidad:** O(N log N) por query (traversal del BVH)
  - **Características:**
    - Genera `rays_per_query` rayos por query token
    - Lanza `optixTrace()` para cada rayo
    - Acumula resultados en memoria compartida
    - Normaliza pesos de atención

### Programas OptiX

- **`cuda/closest_hit.cu`** — Shader OptiX ClosestHit
  - **Programas:**
    - `__closesthit__ch_optical_attention()` — Versión estándar
    - `__closesthit__ch_optical_attention_topk()` — Versión con top-K optimizado
  - **Ejecutado cuando:** Un rayo intersecta un TokenNode en el BVH
  - **Responsabilidades:**
    1. Recuperar payload del rayo
    2. Calcular peso de atención: `w = E₀ · exp(-λ · d_semantic)`
    3. Actualizar payload (accumulated_attention, energy_remaining, hit_count)
    4. Opcionalmente terminar el rayo si energía es muy baja
  - **Fórmula Matemática:** Beer-Lambert Law análogo
    ```
    attention_weight = energy_remaining * exp(-SPECTRAL_LAMBDA * semantic_distance)
    energy_remaining *= exp(-SPECTRAL_LAMBDA * semantic_distance)  // Absorción
    ```

- **`cuda/miss.cu`** — Shader OptiX Miss
  - **Programas:**
    - `__miss__ms_optical_attention()` — Sin contribución al miss
    - `__miss__ms_optical_attention_with_background()` — Con pequeño weight de fondo
  - **Ejecutado cuando:** Un rayo NO golpea ningún TokenNode
  - **Responsabilidades:**
    - Mantener payload sin cambios (miss = sin atención)
    - Terminar la traversal

- **`cuda/ray_generation.cu`** — Shader OptiX RayGen
  - **Programas:**
    - `__raygen__rg_optical_attention()` — Distribución uniforme en hemisferio
    - `__raygen__rg_optical_attention_gaussian()` — Distribución gaussiana
  - **Ejecutado una vez por rayo** (distribuido en grid de OptiX)
  - **Responsabilidades:**
    1. Determinar query token y rayo índice
    2. Generar dirección de rayo (hemisférica)
    3. Inicializar payload (energy=1.0)
    4. Lanzar `optixTrace()`
    5. Procesar y almacenar resultados

## Fórmula de Atención Óptica

La atención se calcula usando analogía con la absorción de luz en un medio (Beer-Lambert Law):

```
attention_weight = E₀ · exp(-λ · d_semantic)
```

Donde:
- **E₀ ∈ [0, 1]**: Energía inicial del rayo (decrece con cada hit)
- **λ ≈ 0.1**: Coeficiente de absorción semántica (hiperparámetro)
- **d_semantic**: Distancia euclídea 3D entre centros de tokens

### Interpretación:
- Tokens **cercanos** (baja distancia) → mayor peso de atención
- Tokens **lejanos** (alta distancia) → menor peso de atención
- Similar a softmax pero O(log N) en lugar de O(N²)

## Constantes Configurables

Definidas en `include/optical_attention.h`:

| Constante | Valor | Descripción |
|---|---|---|
| `SPECTRAL_LAMBDA` | 0.1f | Coeficiente de absorción |
| `SPECTRAL_ENERGY_THRESHOLD` | 0.001f | Energía mínima para continuar |
| `SPECTRAL_MAX_TOP_TOKENS` | 64 | Tokens top-K almacenados |
| `SPECTRAL_RAYS_PER_QUERY` | 8 | Rayos por query (multi-head) |
| `SPECTRAL_MAX_SEQUENCE_LENGTH` | 100000 | Máximo tokens en secuencia |

## Estructura de Datos: RayPayload

Datos que viaján con el rayo durante la traversal del BVH (OptiX payload):

```cpp
struct RayPayload {
    float accumulated_attention;     // Suma de pesos acumulados
    float energy_remaining;          // Energía del rayo (0 a 1.0)
    uint32_t hit_count;              // Número de tokens golpeados
    uint32_t top_tokens[64];         // IDs de tokens más relevantes
    float top_weights[64];           // Pesos correspondientes
    uint32_t ray_origin_{x,y,z};     // Posición 3D del rayo
};
```

**Nota:** En OptiX, los payloads se pasan como 32-bit words. El payload completo requiere ~132 words.

## Estructura de Datos: TokenNode

Representa un token en el espacio 3D semántico:

```cpp
struct TokenNode {
    uint32_t token_id;           // ID en vocabulario
    uint32_t position_in_seq;    // Índice en secuencia
    float3 centroid;             // Centro en espacio 3D
    float3 aabb_min, aabb_max;   // Bounding box
    float semantic_radius;       // Radio semántico
    __half embedding[256];       // Embedding comprimido FP16
    float attention_weight;      // Peso calculado
    float energy_remaining;      // Energía tras impacto
};
```

## Flujo de Ejecución

```
Host Code
    ↓
[ray_attention.cu] ray_traced_attention_kernel
    ↓ (para cada rayo)
optixTrace(bvh_handle, origin, direction)
    ↓
[ray_generation.cu] __raygen__
    ↓ (BVH traversal, hardware-accelerated)
    ├→ ClosestHit en token N
    │      ↓
    │  [closest_hit.cu] __closesthit__
    │      ↓ (calcular weight)
    │      ├→ optixIgnoreIntersection() [terminar rayo]
    │      └→ continuar traversal
    │
    └→ Miss (sin token golpeado)
           ↓
       [miss.cu] __miss__
           ↓ (sin acción)
           └→ fin de rayo
    ↓
Resultados: AttentionResult[] con top-K tokens
```

## Compilación

Los kernels requieren:
1. **CUDA Toolkit 12.x** con soporte OptiX 8.x
2. **OptiX SDK 8.x** instalado
3. **NVIDIA RTX 40xx / 50xx** (Ada/Blackwell architecture)

Ejemplo de CMake:

```cmake
find_package(OptiX 8.0 REQUIRED)
find_package(CUDA 12.0 REQUIRED)

add_library(spectral_kernels OBJECT
    cuda/ray_attention.cu
    cuda/closest_hit.cu
    cuda/miss.cu
    cuda/ray_generation.cu
)

target_compile_options(spectral_kernels PRIVATE
    --use_fast_math
    --forward-unknown-to-host-compiler
)

target_include_directories(spectral_kernels PRIVATE
    ${OPTIX_INCLUDE_DIRS}
    include/
)
```

## Uso desde Host Code

```cpp
#include "include/optical_attention.h"
#include "include/semantic_bvh.h"

// 1. Preparar datos en GPU
uint32_t* d_query_tokens;
float3* d_query_positions;
float3* d_ray_directions;
AttentionResult* d_results;

// 2. Compilar y configurar pipeline OptiX
OptixDeviceContext context;
OptixPipeline pipeline;
// ... (configuración de modules, programas, SBT)

// 3. Configurar constantes device
cudaMemcpyToSymbol(c_bvh_handle, &bvh.getOptixAccelStructure(), sizeof(OptixTraversableHandle));
cudaMemcpyToSymbol(c_token_nodes, &d_tokens, sizeof(TokenNode*));
cudaMemcpyToSymbol(c_rays_per_query, &rays_per_query, sizeof(uint32_t));

// 4. Lanzar kernel
dim3 grid((num_queries + 255) / 256);
dim3 block(256);
ray_traced_attention_kernel<<<grid, block>>>(
    d_query_tokens,
    d_query_positions,
    num_queries,
    d_ray_directions,
    d_results
);

// 5. Procesar resultados
AttentionResult results_host[num_queries];
cudaMemcpy(results_host, d_results, num_queries * sizeof(AttentionResult), cudaMemcpyDeviceToHost);
```

## Optimizaciones y Pendientes

1. **Top-K Management**: Actualmente lineal O(k) por hit. Considerar heap para k > 16.
2. **Shared Memory Reduction**: Usar warp-level primitivos para mejor escalabilidad.
3. **Occupancy**: Estudiar occupancy del kernel con diferentes configuraciones.
4. **Compaction**: Implementar compaction de rayos terminados para mejor utilización de SMs.
5. **Persistent Threads**: Considerar modelo persistent para mejor load balancing.

## Referencias Matemáticas

- **Beer-Lambert Law (Óptica):** I = I₀ · e^(-μx)
  - Adaptado a: weight = E₀ · e^(-λ·d)
  
- **Attention Mechanism (NLP):** softmax(QK^T/√d)V
  - Análogo aproximado en ray tracing: top-K tokens ponderados por decay exponencial

- **Complejidad BVH:** O(log N) por ray-tree intersection
  - Total: O(N log N) para N queries × 1 rayo mínimo

## Testing

Ver `tests/test_attention.cu` para:
- Validación de pesos contra softmax estándar
- Correctitud de top-K tokens
- Benchmarks de throughput

---

**Última actualización:** 2026-03-24
**Autor:** SpectralAI Zero-Matrix Team
