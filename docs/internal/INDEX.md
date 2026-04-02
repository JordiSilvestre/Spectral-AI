# SpectralAI Zero-Matrix — Índice de Archivos (2026-03-24)

## 📍 Navegación Rápida

### Documentación Principal
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Guía de contexto y arquitectura general del proyecto
- **[LEARNINGS.md](LEARNINGS.md)** — Registro vivo de decisiones arquitectónicas
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** — Resumen de implementación de kernels
- **[README_KERNELS.md](README_KERNELS.md)** — Documentación técnica detallada de kernels

### Headers C++ (Tipos y Interfaces)
- **[include/optical_attention.h](include/optical_attention.h)** (219 líneas)
  - `struct RayPayload` — Datos que viajan en los rayos
  - `struct TokenNode` — Representación geométrica de tokens
  - `struct AttentionResult` — Resultados de salida
  - Constantes: LAMBDA, THRESHOLD, MAX_TOP_TOKENS, etc.
  - Funciones device: compute_attention_weight(), insert_top_token()

- **[include/token_geometry.h](include/token_geometry.h)** (353 líneas)
  - Utilidades de geometría 3D
  - Función: semantic_distance_3d()
  - Función: point_in_aabb(), aabb_center(), etc.

- **[include/semantic_bvh.h](include/semantic_bvh.h)** (439 líneas) [Pre-existente]
  - `class SemanticBVH` — Gestor del árbol BVH
  - `struct BVHNode` — Nodo del árbol
  - API para build(), getOptixAccelStructure(), etc.

### Kernels CUDA/OptiX (Programas de Shaders)

#### 1. Ray Attention Kernel (Orquestador Principal)
- **[cuda/ray_attention.cu](cuda/ray_attention.cu)** (288 líneas)
  - `__global__ ray_traced_attention_kernel()` — Kernel CUDA principal
  - Genera rayos desde queries
  - Lanza optixTrace() y acumula resultados
  - Normaliza pesos finales

#### 2. Shader ClosestHit (Cuando un rayo golpea un token)
- **[cuda/closest_hit.cu](cuda/closest_hit.cu)** (250 líneas)
  - `__closesthit__ch_optical_attention()` — Shader principal
  - `__closesthit__ch_optical_attention_topk()` — Variante optimizada
  - Fórmula: weight = E₀ · exp(-λ · d_semantic)
  - Verifica threshold de energía
  - Actualiza payload y opcionalmente termina rayo

#### 3. Shader Miss (Cuando un rayo no golpea nada)
- **[cuda/miss.cu](cuda/miss.cu)** (118 líneas)
  - `__miss__ms_optical_attention()` — Shader principal
  - `__miss__ms_optical_attention_with_background()` — Variante con fondo
  - Finaliza traversal sin contribución

#### 4. Shader RayGen (Generación de rayos)
- **[cuda/ray_generation.cu](cuda/ray_generation.cu)** (369 líneas)
  - `__raygen__rg_optical_attention()` — Distribución uniforme
  - `__raygen__rg_optical_attention_gaussian()` — Distribución gaussiana
  - Genera direcciones hemisféricas (Fibonacci sphere)
  - Inicializa payload y lanza optixTrace()

### Implementaciones C++ (Pendiente)
- **[src/token_geometry.cpp](src/token_geometry.cpp)** — Proyección D→3D [Pendiente]
- **[src/semantic_bvh.cpp](src/semantic_bvh.cpp)** — Construcción del BVH [Pendiente]

### Python (Pendiente)
- **[python/embedding_bridge.py](python/embedding_bridge.py)** — Carga embeddings [Pendiente]
- **[python/inference.py](python/inference.py)** — Script de inferencia [Pendiente]

### Tests (Pendiente)
- **[tests/benchmark.cu](tests/benchmark.cu)** — Benchmarks vs attention estándar [Pendiente]

### Build
- **[CMakeLists.txt](CMakeLists.txt)** — Sistema de build
- **[BUILD.md](BUILD.md)** — Instrucciones de compilación

---

## 🎯 Guía de Lectura por Rol

### Si eres un Investigador
1. Lee [ARCHITECTURE.md](ARCHITECTURE.md) para entender la visión del proyecto
2. Lee [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) para ver lo completado
3. Lee [LEARNINGS.md](LEARNINGS.md) para entender las decisiones

### Si eres un Ingeniero CUDA/OptiX
1. Lee [README_KERNELS.md](README_KERNELS.md) primero
2. Revisa [include/optical_attention.h](include/optical_attention.h) para tipos
3. Estudia [cuda/ray_attention.cu](cuda/ray_attention.cu) (orquestador)
4. Entiende [cuda/closest_hit.cu](cuda/closest_hit.cu) (lógica de atención)
5. Revisa [cuda/ray_generation.cu](cuda/ray_generation.cu) (distribución)
6. Consulta [cuda/miss.cu](cuda/miss.cu) (terminación)

### Si eres un Integrador/DevOps
1. Revisa [CMakeLists.txt](CMakeLists.txt)
2. Lee [BUILD.md](BUILD.md)
3. Consulta [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) para status

### Si eres un Contribuidor
1. Actualiza [LEARNINGS.md](LEARNINGS.md) cuando tomes decisiones
2. Documenta cambios en [README_KERNELS.md](README_KERNELS.md)
3. Ejecuta tests en [tests/](tests/) antes de push

---

## 📊 Estadísticas del Proyecto

| Métrica | Valor |
|---|---|
| Headers implementados | 3 (2 nuevos) |
| Kernels CUDA/OptiX | 4 |
| Programas OptiX | 6 (ClosestHit×2, Miss×2, RayGen×2) |
| Líneas de código kernel | 1.025 |
| Líneas de tipos compartidos | 572 |
| Documentación (líneas) | 700+ |
| Tamaño total | ~240 KB |

---

## 🔄 Flujo de Datos (Ejecución)

```
Host (C++)
    ↓ cudaMemcpy(query_tokens, d_tokens)
    ↓ cudaMemcpyToSymbol(c_token_nodes, ...)
    ↓
ray_traced_attention_kernel <<<grid, block>>>
    ↓ (para cada thread = query token)
    ├─ genera rays_per_query rayos
    ├─ para cada rayo:
    │   ↓
    │   optixTrace(bvh, origin, direction)
    │       ↓ (hardware BVH traversal)
    │       ├─ si hit token → __closesthit__
    │       │   ├─ calcula weight = E₀·exp(-λ·d)
    │       │   ├─ actualiza payload
    │       │   └─ continúa o termina
    │       │
    │       └─ si no hit → __miss__
    │           └─ finaliza rayo
    │   ↓
    │   ClosestHit/Miss actualiza payload
    ↓
Acumula resultados en shared memory
    ↓
Normaliza pesos (sum-to-1)
    ↓
Escribe AttentionResult[] en GPU
    ↓ cudaMemcpy(results_host, d_results)
Host recibe resultados
```

---

## ✅ Checklist de Implementación Completada

- [x] RayPayload structure (132 words, OptiX compatible)
- [x] TokenNode structure (3D geometry + FP16 embedding)
- [x] AttentionResult structure (output with top-K)
- [x] Attention decay formula (exponential, Beer-Lambert analog)
- [x] Ray generation (hemispheric, Fibonacci distribution)
- [x] ClosestHit handler (weight computation + energy absorption)
- [x] Miss handler (ray termination)
- [x] Top-K management (ordered insertion)
- [x] Shared memory reduction (local accumulation)
- [x] Weight normalization (sum-to-1)
- [x] Configurable constants (6 defines)
- [x] Detailed comments (inline math formulas)
- [x] Host API wrapper (launch_ray_traced_attention_kernel)
- [x] Documentation (README_KERNELS.md)
- [x] LEARNINGS.md updated

---

## ⏳ Próximos Hitos

### Fase 2: Host Integration (2-3 días)
- [ ] OptiX context setup code
- [ ] Module compilation from CUDA sources
- [ ] SBT (Shader Binding Table) configuration
- [ ] Device constant setup
- [ ] Host-to-device data transfer

### Fase 3: BVH + Tests (2-3 días)
- [ ] SemanticBVH::build() implementation
- [ ] Correctness tests vs CPU brute force
- [ ] Weight distribution validation (KL divergence)
- [ ] Performance benchmarks

### Fase 4: Python Bindings (1-2 días)
- [ ] pybind11 wrapper
- [ ] Python inference script
- [ ] Pre-trained embedding loading

---

## 📝 Nota Importante

Este índice se actualiza con cada cambio significativo. Consulta [LEARNINGS.md](LEARNINGS.md) para histórico de decisiones y [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) para estado de completitud.

**Última actualización:** 2026-03-24 00:00 UTC  
**Estado:** ✅ Kernels completados y documentados

