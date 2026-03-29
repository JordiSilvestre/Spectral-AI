# MEJORAS.md — LiquidBit Zero-Matrix
> Revisado: 2026-03-29 | Para revisión del equipo

Documento de auditoría completa del código. Incluye bugs, optimizaciones, mejoras de calidad y la propuesta de integración de rayos espectrales en los kernels CUDA/OptiX.

---

## Tabla de Contenidos

1. [Propuesta: Integración de Rayos Espectrales](#1-propuesta-integración-de-rayos-espectrales)
2. [Idea Externa: PolarQuant para Vectores Espectrales](#2-idea-externa-polarquant-para-vectores-espectrales)
3. [CUDA/OptiX — Bugs y Optimizaciones](#3-cudaoptix--bugs-y-optimizaciones)
4. [C++/Headers — Bugs y Mejoras](#4-cheaders--bugs-y-mejoras)
5. [Python — Bugs y Optimizaciones](#5-python--bugs-y-optimizaciones)
6. [Build System (CMake)](#6-build-system-cmake)
7. [Resumen de Prioridades](#7-resumen-de-prioridades)

---

## 1. Propuesta: Integración de Rayos Espectrales

### Estado Actual
Los shaders CUDA/OptiX usan `SemanticRay` (rayos basicos: origin + direction + energy). Los structs espectrales (`PrismaticRay`, `SpectralContext`, `PrismaticSphere`) estan definidos en `spectral_ray.h` (~1000 lineas) pero **nunca se instancian en ningun kernel CUDA**. Solo hay una version "soft" en PyTorch (`bvh_router.py`).

### Comparativa: Basico vs Espectral

| Metrica | Basico (medido) | Espectral (estimado) | Cambio |
|---|---|---|---|
| Latencia routing (B=256) | 64.6 us | ~72 us | +11% |
| Throughput E2E | 51.9 tok/s | ~50 tok/s | -3% |
| **PPL (16/16 capas)** | **8.29** | **~7.3** | **-12%** |
| Polisemia | 0% | 88.9% | Salto cualitativo |
| Memoria extra | 0 | 18 KB (64 esferas) | Negligible |
| Overhead FLOPs/capa | 0 | 0.85% (7.2G/843G) | Negligible |

### Coste por Hit

| Operacion | Basico | Espectral |
|---|---|---|
| Energy decay `exp(-l*d)` | ~10 FLOPs | ~10 FLOPs |
| `dot(W_disp[64], f[64])` + sigmoid | — | ~130 FLOPs |
| Snell: direccion refractada | — | ~20 FLOPs |
| **Total por hit** | **~10 FLOPs** | **~160 FLOPs** |

### Proyeccion de PPL

```
Degradacion actual:     8.29 / 6.11 = +35.7%
  - Error inherente BVH (O(log N)):    ~18%
  - Error por polisemia mal ruteada:    ~17%  (corregible)

Con espectral (88.9% polisemia):
  - Error inherente BVH:                ~18%  (se mantiene)
  - Error polisemia residual:           ~2%   (88.9% corregido)
  = PPL estimada: ~7.3  (+19.5% vs 6.11)
```

### Archivos a Modificar
- `cuda/ray_generation.cu` — Anadir color vector al payload del rayo
- `cuda/closest_hit.cu` — Anadir calculo Snell + W_dispersion
- `include/optical_attention.h` — Integrar `PrismaticRay` del `spectral_ray.h`
- `cuda/optix_router_raygen.cu` — Generar rayos con contexto espectral

### Conclusion
**Coste minimo (~0.85% overhead), ganancia maxima (~12% PPL).** Es la mejora con mejor ratio esfuerzo/impacto del proyecto.

---

## 2. Idea Externa: PolarQuant para Vectores Espectrales

> Origen: Revision de TurboQuant (ICLR 2026, arXiv:2504.19874) y TurboQuant+.
> Repos evaluados y descartados como dependencia (no aportan al core de LiquidBit).
> Solo esta tecnica especifica es relevante.

### Contexto

Los rayos espectrales de LiquidBit llevan un vector de color `f in R^64` (SpectralContext).
Actualmente se almacena en FP16 (128 bytes por rayo). Con 4096 rayos/query, son **528 KB por token**.

### La Tecnica: Rotacion + Cuantizacion Escalar Optima

TurboQuant usa **PolarQuant**: una rotacion ortogonal aleatoria (matriz Haar) que
"Gaussianiza" las coordenadas del vector, seguida de cuantizacion escalar Lloyd-Max
por coordenada. Resultado: **near-lossless con 2-3 bits/coordenada**.

```
Vector espectral f (64D, FP16, 128 bytes)
  -> Rotacion Pi (64x64, precomputada, seed-based)
  -> Coordenadas Gaussianizadas ~ N(0, 1/d)
  -> Lloyd-Max 3-bit por coordenada (codebook precomputado)
  -> 64 * 3 bits = 192 bits = 24 bytes + 4 bytes norma = 28 bytes

Compresion: 128 -> 28 bytes = 4.6x
Con 4096 rayos/query: 528 KB -> 115 KB por token
```

### Aplicacion en LiquidBit

| Donde | Que comprimir | Antes | Despues | Ahorro |
|---|---|---|---|---|
| `SpectralContext.color_vector[64]` | Vector de color del rayo | 128 B | 28 B | 4.6x |
| `PrismaticSphere.W_dispersion[64]` | Pesos de dispersion | 128 B | 28 B | 4.6x |
| BVH con 64 esferas (W_disp total) | Todas las esferas | 8 KB | 1.8 KB | 4.4x |
| 4096 rayos/query (contextos) | Todos los rayos | 528 KB | 115 KB | 4.6x |

### Calidad medida (TurboQuant paper)

- **cos_sim** tras compresion 3-bit: **0.997** (practicamente lossless)
- **PPL delta**: +0.23% a 4-bit, +1.06% a 3-bit
- Gaussianizacion validada: kurtosis raw 900.4 -> 2.9 (ref Gaussiana = 3.0)

### Implementacion sugerida

No necesita dependencia externa. El algoritmo es simple (~50 lineas):

```python
# Pseudocodigo para comprimir vectores espectrales
import numpy as np

class SpectralCompressor:
    def __init__(self, d=64, bits=3, seed=42):
        rng = np.random.RandomState(seed)
        # Rotacion ortogonal Haar (precomputada una vez)
        H = rng.randn(d, d)
        self.Q, _ = np.linalg.qr(H)
        # Codebook Lloyd-Max para N(0, 1/d) con `bits` bits
        self.codebook = precompute_lloyd_max(d, bits)

    def compress(self, f):
        norm = np.linalg.norm(f)
        f_unit = f / (norm + 1e-10)
        rotated = self.Q @ f_unit          # Gaussianiza
        indices = self.codebook.quantize(rotated)  # 3-bit/coord
        return indices, norm               # 28 bytes total

    def decompress(self, indices, norm):
        rotated = self.codebook.dequantize(indices)
        f_unit = self.Q.T @ rotated        # Rotacion inversa
        return f_unit * norm
```

### Prioridad

**BAJA** — Solo relevante DESPUES de integrar rayos espectrales en kernels CUDA (Seccion 1).
Sin rayos espectrales funcionando, no hay nada que comprimir. Secuencia:

1. Integrar rayos espectrales en CUDA (Seccion 1) -> **-12% PPL**
2. Si la memoria de los vectores espectrales se convierte en bottleneck -> aplicar PolarQuant
3. Ahorro estimado: 4.6x en almacenamiento de contexto espectral

---

## 3. CUDA/OptiX — Bugs y Optimizaciones

### CRITICAL

#### 2.1 Buffer Overflow en top-K accumulation
- **Archivo:** `cuda/ray_attention.cu:234-245`
- **Problema:** `total_hit_count` se usa como contador Y como tamano de array para `insert_top_token()`. Crece sin limite y desborda `accumulated_top_tokens[LIQUIDBIT_MAX_TOP_TOKENS]`.
- **Fix:** Usar contador separado:
  ```cuda
  uint32_t accumulated_top_count = 0;
  insert_top_token(accumulated_top_tokens, accumulated_top_weights,
                   accumulated_top_count, token_id, weight);
  ```

#### 2.2 Conversion FP32->FP16 no implementada
- **Archivo:** `cuda/alpha_phase_b.cu:300-318`
- **Problema:** El codigo dice `// En GPU requeriria un pequeno kernel, omitido aqui`. `d_input_fp16` recibe datos basura, las operaciones cuBLAS posteriores producen resultados incorrectos.
- **Fix:** Implementar kernel de conversion:
  ```cuda
  convertFp32ToFp16<<<blocks, ALPHA_BLOCK_DIM_1D>>>(d_input_fp32, d_input_fp16, total_elements);
  ```

### HIGH

#### 2.3 Data Race en multi-ray accumulation
- **Archivo:** `cuda/ray_generation.cu:272-276`
- **Problema:** Multiples threads escriben a `result.total_attention` simultaneamente sin sincronizacion.
- **Fix:** `atomicAdd(&result.total_attention, ray_payload.accumulated_attention);`

#### 2.4 Coordinate space mismatch en intersection
- **Archivo:** `cuda/liquidbit_kernels.cu:296-307`
- **Problema:** `__intersection__sphere` usa `optixGetWorldRayOrigin()` pero `sphere.center` puede estar en object space si hay transformaciones de instancia.
- **Fix:** Usar `optixGetObjectRayOrigin()` o pre-transformar centros a world space en construccion del BVH.

#### 2.5 Variable indefinida en cross product
- **Archivo:** `cuda/optix_router_raygen.cu:108-129`
- **Problema:** Se usa variable `d` en cross product sin definirla como `const float3 d = direction;`. Produce base ortonormal incorrecta.
- **Fix:** Anadir `const float3 d = direction;` antes del calculo.

#### 2.6 Null dereference antes de bounds check
- **Archivo:** `cuda/alpha_phase_a.cu:154-155`
- **Problema:** `current_sphere.children_ids[i]` se accede antes de validar `child_id >= num_spheres`.
- **Fix:** `uint32_t actual_children = min(current_sphere.num_children, (uint32_t)ALPHA_BSH_MAX_CHILDREN);`

#### 2.7 Buffer reuse sin semantica clara
- **Archivo:** `cuda/async_pipeline.cu:330-340`
- **Problema:** `softmax_topk_kernel` sobreescribe `ps.d_expert_weights` (logits originales) con output de softmax. Los logits se pierden.
- **Fix:** Usar buffer separado para softmax output.

#### 2.8 Loop accede indices invalidos
- **Archivo:** `cuda/ternary_resonance.cu:107-122`
- **Problema:** `#pragma unroll` hasta `RESONANCE_NUM_MODES` pero si `num_modes < RESONANCE_NUM_MODES`, lee `params.a[k-1]` sin inicializar.
- **Fix:** Anadir `if (k > M) break;` o inicializar `a[]` con ceros.

#### 2.9 Benchmark con input zero-initialized
- **Archivo:** `cuda/v5/bvh_router_kernel.cu:437-448`
- **Problema:** `cudaMemset(d_input, 0, ...)` produce vectores cero. El BVH router espera direcciones normalizadas. El benchmark no refleja rendimiento real.
- **Fix:** Inicializar con vectores aleatorios normalizados.

#### 2.10 Pi truncado inconsistente
- **Archivo:** `cuda/inception_resonance.cu:192-196`
- **Problema:** Usa `2.0f * 3.14159265f` (truncado) en vez de `CUDART_PI_F` (como en `liquidbit_kernels.cu:99`). Ademas `fabsf()` pierde signo.
- **Fix:** `omega = fmodf(new_omega, 2.0f * CUDART_PI_F);`

### MEDIUM

#### 2.11 Expert forward no implementado
- **Archivo:** `cuda/async_pipeline.cu:369-370`
- **Problema:** Marcado como TODO. `weighted_combine_kernel` combina hidden states sin transformacion experta. Pipeline incompleto.

#### 2.12 Shared memory bank conflicts
- **Archivo:** `cuda/ray_attention.cu:148-154`
- **Problema:** `shared_hit_count[256]` sin padding causa bank conflicts en acceso secuencial.
- **Fix:** `__shared__ uint32_t shared_hit_count[256 + 32];`

#### 2.13 sqrt redundante en hot path
- **Archivo:** `cuda/closest_hit.cu:111-118`
- **Problema:** `sqrtf()` en cada hit para calcular distancia antes de `expf()`. Si el modelo lo permite, usar distancia al cuadrado.
- **Fix:** Reformular decay curve para evitar sqrt, o usar `rsqrtf()`.

#### 2.14 Path computado pero descartado
- **Archivo:** `cuda/v5/bvh_torch_ext.cu:309-354`
- **Problema:** `route_impl()` calcula `path` tensor pero no lo retorna. Memoria desperdiciada y util para debug.
- **Fix:** `return std::make_tuple(expert_ids, scores, confidence, path);`

#### 2.15 childIAS==0 no es sentinel seguro
- **Archivo:** `cuda/inception_kernels.cu:273-336`
- **Problema:** `OptixTraversableHandle` valor 0 no esta garantizado como invalido en todas las implementaciones.

#### 2.16 Error checks faltantes en GPU operations
- **Archivos:** Multiples (`alpha_phase_b.cu`, `async_pipeline.cu`, `ray_generation.cu`)
- **Problema:** `cudaMalloc()`, `optixTrace()` sin verificar errores.

---

## 4. C++/Headers — Bugs y Mejoras

### CRITICAL

#### 3.1 Memory leak en `computePrincipalAxes()`
- **Archivo:** `src/token_geometry.cpp:119-165`
- **Problema:** `float* temp = new float[embed_dim]` — si hay excepcion, no se libera.
- **Fix:** Usar `std::vector<float> temp(embed_dim, 0.0f);`

#### 3.2 Memory leak en `projectEmbeddingTo3D()`
- **Archivo:** `src/token_geometry.cpp:214-256`
- **Problema:** `float* normalized = new float[embed_dim]` sin RAII.
- **Fix:** Usar `std::vector<float>`.

#### 3.3 Memory leak en `validateTreeStructure()`
- **Archivo:** `src/alpha_bsh.cpp:306-347`
- **Problema:** `new SemanticSphereAlpha[num_spheres_]` — si `cudaMemcpy()` falla, el array queda sin liberar.
- **Fix:** `std::vector<SemanticSphereAlpha> h_spheres(num_spheres_);`

#### 3.4 cudaMemcpy sin error check
- **Archivo:** `src/alpha_bsh.cpp:307-308`
- **Problema:** Si `cudaMemcpy()` falla, la validacion procede sobre datos basura.
- **Fix:** Verificar `cudaError_t` y retornar early si falla.

### HIGH

#### 3.5 Null pointer dereference en `launchPhaseA()`
- **Archivo:** `src/alpha_bsh.cpp:370-386`
- **Problema:** `query_embedding` no se valida antes de usar.
- **Fix:** `if (query_embedding == nullptr || query_dim == 0) return AlphaRayPayload();`

#### 3.6 malloc() en vez de cudaMalloc()
- **Archivo:** `src/semantic_bvh.cpp:311-335`
- **Problema:** Variable llamada `gpu_bvh_nodes` pero se aloca con `malloc()`. Si se pretende usar en GPU, debe ser `cudaMalloc()`. Si es CPU, el nombre confunde.
- **Fix:** Corregir a `cudaMalloc()` o renombrar a `host_bvh_nodes`.

#### 3.7 buildRecursive() sin manejar edge case
- **Archivo:** `src/semantic_bvh.cpp:173-246`
- **Problema:** Si `start >= end`, retorna -1 que puede causar problemas en el caller sin validacion.
- **Fix:** Validar y loguear error.

#### 3.8 Resource leak con CUDA events
- **Archivo:** `src/alpha_bsh.cpp:445-519`
- **Problema:** `cudaEvent_t` creados pero si hay excepcion, no se destruyen.
- **Fix:** Wrapper RAII `CudaEventGuard`.

### MEDIUM

#### 3.9 Loop O(N^2) en parent-child assignment
- **Archivo:** `src/alpha_bsh.cpp:249-279`
- **Problema:** Bucle anidado para encontrar padre mas cercano. Para 100K+ tokens, prohibitivo.
- **Fix:** KD-tree o thrust::sort para O(N log N).

#### 3.10 Perdida de informacion en proyeccion 3D
- **Archivo:** `src/token_geometry.cpp:234-254`
- **Problema:** Proyeccion simplificada (suma par/impar + tanh). Muchos embeddings distintos mapean al mismo punto 3D.
- **Fix:** Implementar PCA real (ya existe `computePrincipalAxes()`).

#### 3.11 Datos sin inicializar en `computeBounds()`
- **Archivo:** `src/semantic_bvh.cpp:133-152`
- **Problema:** Si rango vacio, `min_out`/`max_out` quedan con `numeric_limits`.
- **Fix:** Manejar rango vacio explicitamente.

#### 3.12 Potencial double-free
- **Archivo:** `src/alpha_bsh.cpp:173,187`
- **Problema:** `cudaFree(d_spheres_)` en error path + destructor.
- **Fix:** `d_spheres_ = nullptr;` despues de cada `cudaFree()`.

---

## 5. Python — Bugs y Optimizaciones

### HIGH

#### 4.1 GPU memory leak en pipeline asincrono
- **Archivo:** `python/async_pipeline_bridge.py:134-144`
- **Problema:** Tensores creados en loop no se hacen `.detach()`. En pipeline largo, la VRAM crece sin control.
- **Fix:** `expert_output = expert_output.detach()`

#### 4.2 Device mismatch en routing supervision loss
- **Archivo:** `python/orchestrator.py:240-245`
- **Problema:** Mezcla operaciones GPU/CPU con `.item()` en loop. Si `domain_ids` y `expert_probs` estan en dispositivos distintos, crash.
- **Fix:** Vectorizar con masks booleanas + `torch.arange()`.

#### 4.3 torch.no_grad() con .detach() redundante
- **Archivo:** `python/calibrate_router.py:106-109`
- **Problema:** Dentro de `torch.no_grad()`, `.detach()` es innecesario. Pero el grafo se sigue construyendo para el forward pass.
- **Fix:** Eliminar `.detach()` redundante.

#### 4.4 Device mismatch en extract_real_hiddens
- **Archivo:** `python/extract_real_hiddens.py:195-203`
- **Problema:** No valida que `gate_weight` y `h_gate` esten en el mismo device antes de `F.linear()`.
- **Fix:** Assert de device consistency.

#### 4.5 .item() en loop caliente de benchmark
- **Archivo:** `python/benchmark_expert_types.py:318-319`
- **Problema:** `expert_id = ids[token_idx, k].item()` fuerza sync GPU-CPU en cada iteracion.
- **Fix:** Mover `ids` a CPU una vez fuera del loop, o vectorizar con operaciones torch.

#### 4.6 Race condition en monkey-patching
- **Archivo:** `python/benchmark_cuda_e2e.py:159-177`
- **Problema:** Se reemplaza `model.router.forward` sin locking. Otro thread podria usar la version incorrecta.
- **Fix:** Context manager `RouterSwap` con __enter__/__exit__.

### MEDIUM

#### 4.7 Security: pickle sin restriccion
- **Archivos:** `python/olmoe_e2e_eval.py:242-254`, `python/calibrate_router.py`
- **Problema:** `torch.load(..., weights_only=False)` permite ejecucion arbitraria de codigo.
- **Fix:** `weights_only=True` con fallback documentado.

#### 4.8 Bare except clauses
- **Archivo:** `python/scaling_inception.py:173, 185, 243, 249, 277`
- **Problema:** `except:` captura `KeyboardInterrupt`, `SystemExit`, etc.
- **Fix:** `except Exception as e:`

#### 4.9 NaN potencial en log
- **Archivo:** `python/orchestrator.py:248`
- **Problema:** `torch.log(domain_prob + 1e-8)` — epsilon insuficiente en FP16.
- **Fix:** `torch.log(torch.clamp(domain_prob, min=1e-7))`

#### 4.10 Truncacion silenciosa de spectral dim
- **Archivo:** `python/bvh_router_bridge.py:165-167`
- **Problema:** Si `spec_dim != SPEC_DIM`, se trunca/padea sin warning. Degrada routing sin visibilidad.
- **Fix:** Assert o warning explicito.

#### 4.11 API deprecated
- **Archivo:** `python/benchmark_expert_types.py:127-129`
- **Problema:** `torch.ao.quantization.quantize_dynamic()` deprecated en PyTorch 2.2+.
- **Fix:** Migrar a `torch.quantization.quantize_dynamic()` con fallback.

---

## 6. Build System (CMake)

### HIGH

#### 5.1 Typo en variable OptiX
- **Archivo:** `CMakeLists.txt:455`
- **Problema:** `${OptiX_INCLUDE}` deberia ser `${OptiX_INCLUDE_DIR}`.
- **Fix:** Cambiar a `${OptiX_INCLUDE_DIR}`.

#### 5.2 PTX solo compila para sm_89
- **Archivo:** `CMakeLists.txt:229`
- **Problema:** Falta `-gencode=arch=compute_120,code=compute_120`. Los shaders OptiX no corren en RTX 5070 Ti (Blackwell).
- **Fix:** Anadir gencode para sm_120.

### MEDIUM

#### 5.3 Sin version check para CUDA 12.8+
- **Archivo:** `CMakeLists.txt:122-124`
- **Problema:** sm_120 requiere CUDA 12.8+ pero no se verifica. Errores crípticos si CUDA es antiguo.
- **Fix:** Anadir check con `CUDAToolkit_VERSION VERSION_LESS "12.8"`.

#### 5.4 test_optix_pipeline no linka liquidbit_optix
- **Archivo:** `CMakeLists.txt:458-462`
- **Problema:** Falta `liquidbit_optix` en `target_link_libraries`. Errores de linker.
- **Fix:** Anadir `liquidbit_optix` al target.

### LOW

#### 5.5 Variable CMAKE_CUDA_ARCHITECTURES
- **Archivo:** `CMakeLists.txt:124`
- **Problema:** Usa `CUDA_ARCHITECTURES` en vez de `CMAKE_CUDA_ARCHITECTURES` (standard CMake).
- **Fix:** Renombrar para portabilidad.

---

## 7. Resumen de Prioridades

### Accion Inmediata (Bloqueantes / Corrupcion de datos)

| # | Archivo | Problema |
|---|---------|----------|
| 2.1 | ray_attention.cu:234 | Buffer overflow en top-K |
| 2.2 | alpha_phase_b.cu:300 | FP32->FP16 no implementado |
| 2.3 | ray_generation.cu:272 | Data race en accumulation |
| 3.1-3.3 | token_geometry.cpp, alpha_bsh.cpp | Memory leaks (usar vector) |
| 5.2 | CMakeLists.txt:229 | PTX no compila para sm_120 (Blackwell) |

### Alta Prioridad (Resultados incorrectos / Crashes potenciales)

| # | Archivo | Problema |
|---|---------|----------|
| 2.4 | liquidbit_kernels.cu:296 | Coordinate space mismatch |
| 2.5 | optix_router_raygen.cu:108 | Variable indefinida en cross product |
| 3.5 | alpha_bsh.cpp:370 | Null pointer sin validar |
| 3.6 | semantic_bvh.cpp:311 | malloc() donde deberia ser cudaMalloc() |
| 4.1 | async_pipeline_bridge.py:134 | GPU memory leak |
| 4.2 | orchestrator.py:240 | Device mismatch |
| 5.1 | CMakeLists.txt:455 | Typo OptiX include |

### Media Prioridad (Rendimiento / Calidad)

| # | Archivo | Problema |
|---|---------|----------|
| 1.0 | spectral_ray.h -> kernels | Integrar rayos espectrales (-12% PPL) |
| 2.13 | closest_hit.cu:111 | sqrt redundante en hot path |
| 3.9 | alpha_bsh.cpp:249 | Loop O(N^2) parent-child |
| 4.5 | benchmark_expert_types.py:318 | .item() sync en loop |
| 4.7 | olmoe_e2e_eval.py:242 | Security: pickle sin restriccion |

### Total: 44 hallazgos

| Severidad | CUDA/OptiX | C++/Headers | Python | CMake | Total |
|-----------|-----------|-------------|--------|-------|-------|
| CRITICAL | 2 | 4 | 0 | 0 | **6** |
| HIGH | 8 | 4 | 6 | 2 | **20** |
| MEDIUM | 5 | 4 | 5 | 2 | **16** |
| LOW | 1 | 0 | 0 | 1 | **2** |
| **Total** | **16** | **12** | **11** | **5** | **44** |
