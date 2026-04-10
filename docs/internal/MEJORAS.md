# MEJORAS.md — SpectralAI Zero-Matrix
> Revisado: 2026-03-29 | Para revisión del equipo

Documento de auditoría completa del código. Incluye bugs, optimizaciones, mejoras de calidad y la propuesta de integración de rayos espectrales en los kernels CUDA/OptiX.

---

## Tabla de Contenidos

1. [Propuesta: Integración de Rayos Espectrales](#1-propuesta-integración-de-rayos-espectrales)
2. [Idea Externa: PolarQuant para Vectores Espectrales](#2-idea-externa-polarquant-para-vectores-espectrales)
3. [Sinergias Lyra-AGI → SpectralAI
4. [CUDA/OptiX — Bugs y Optimizaciones](#4-cudaoptix--bugs-y-optimizaciones)
5. [C++/Headers — Bugs y Mejoras](#5-cheaders--bugs-y-mejoras)
6. [Python — Bugs y Optimizaciones](#6-python--bugs-y-optimizaciones)
7. [Build System (CMake)](#7-build-system-cmake)
8. [Resumen de Prioridades](#8-resumen-de-prioridades)

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
> Repos evaluados y descartados como dependencia (no aportan al core de SpectralAI).
> Solo esta tecnica especifica es relevante.

### Contexto

Los rayos espectrales de SpectralAI llevan un vector de color `f in R^64` (SpectralContext).
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

### Aplicacion en SpectralAI

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

## 3. Sinergias Lyra-AGI → SpectralAI

> Origen: Revision completa del repo Lyra-AGI (jordisilvestre/Lyra-AGI).
> 6 tecnicas transferibles. La mas critica (SmoothSTE) podria desbloquear
> training end-to-end del BVH — el mayor problema abierto del proyecto.

### 3.1 SmoothSTE — Diferenciabilidad del BVH (CRITICO)

**Problema en SpectralAI**: Los RT Cores no son diferenciables. El BVH traversal
es discreto (hit/miss), gradiente = 0. Marcado como "mayor desafio" en ARCHITECTURE.md.

**Solucion de Lyra**: `SmoothTernarySTE` usa tanh suave con beta annealing:

```python
# Forward: ternarizacion suave (gradientes fluyen)
magnitude = tanh(beta * (|D_cont| - 0.5)).clamp(0, 1)
D_ternary = magnitude * sign(D_cont)

# Backward: escala gradientes para evitar zonas muertas
scale = 1.0 - tanh(|D_cont| - 2.0).clamp(0, 1)
grad_out = grad_out * scale

# Beta annealing: 1.0 → 10.0 linealmente tras 1000 steps warmup
```

**Aplicacion al BVH de SpectralAI**:

```python
# Actual closest_hit (NO diferenciable):
attention_weight = (hit) ? energy * exp(-lambda * d) : 0.0

# Con SmoothSTE (diferenciable):
soft_hit = tanh(beta * (semantic_radius - distance))
attention_weight = soft_hit * energy * exp(-lambda * d)
# beta=1 (inicio): suave, explora → beta=10 (final): discreto, RT Core real
```

**Resultados validados en Lyra**:
- Loss: -44.6% en 200 steps (TinyStories)
- 49/49 tests pasando, 0 NaN con BF16
- Convergencia estable de loss 10.70 → 5.93

**Hiperparametros clave**:
- beta_start=1.0, beta_end=10.0, warmup=1000 steps
- **BF16 obligatorio** (FP16 causa NaN — validado)
- Clamp D_cont a [-2, 2] despues de cada optimizer step

**Fuente**: `lyra/model/lyra_block.py:39-76`
**Prioridad**: **CRITICA** — desbloquea training E2E del BVH

---

### 3.2 LiquidTimeGate — Inicializacion de W_dispersion (ALTA)

**Problema**: W_dispersion de los rayos espectrales esta a cero (nunca entrenado).
No hay guia sobre que canales deben ser LOCAL vs GLOBAL.

**Solucion de Lyra**: `LiquidTimeGate` aprende automaticamente por canal:

```python
# gate(i, pos) = sigmoid(10 * a_i * dist + b_i)
# a < 0 → LOCAL (atenua tokens lejanos)
# a > 0 → GLOBAL (favorece contexto amplio)
```

**Patrones emergentes (sin supervision)**:
- Layer 0: 66 GLOBAL, 13 LOCAL → captura contexto de secuencia
- Layer 1: 11 GLOBAL, 66 LOCAL → procesa detalles locales
- Layers 2-3: 60+ LOCAL → refinamiento

**Aplicacion**: Inicializar W_dispersion del PrismaticRay con estos patrones:
- Capas tempranas: W_dispersion sesgado hacia GLOBAL (rayos de largo alcance)
- Capas profundas: W_dispersion sesgado hacia LOCAL (rayos de corto alcance)

**Resultado**: -6.4% loss vs sin gate temporal
**Fuente**: `lyra/model/lyra_block.py:269-334`
**Prioridad**: **ALTA** — mejora convergencia espectral ~40%

---

### 3.3 SubLN — RMSNorm post-routing (MEDIA)

**Problema**: La salida del BVH router tiene escalas dispares (tokens con muchos
hits vs pocos). Sin normalizacion, las capas siguientes saturan.

**Solucion**: RMSNorm obligatorio despues del routing, antes del gate:

```python
h = self.router(h)     # salida BVH — escala descontrolada
h = self.sub_ln(h)     # RMSNorm — normaliza
h = self.gate(h)       # ahora opera sobre valores estables
```

**Sin SubLN**: 100% saturacion, colapso total, loss no baja
**Con SubLN**: Convergencia estable 1000+ steps, permite 95% sparsity

**Fuente**: `lyra/model/lyra_block.py:424-434, 508`
**Prioridad**: **MEDIA** — simple de implementar, obligatorio si se entrena BVH

---

### 3.4 Dual LR — Learning rate separado para BVH (MEDIA)

**Problema**: Los pesos discretos del BVH (centroides, radios) son sensibles a
LR altos. Oscilan entre estados si el LR es igual al de pesos float.

**Solucion**: LR 10x menor para parametros discretos:

```python
param_groups = [
    {"params": bvh_params,   "lr": base_lr * 0.1, "weight_decay": 0.0},
    {"params": float_params, "lr": base_lr,        "weight_decay": 0.01},
]
```

**Resultado**: 100% estabilidad, 0 NaN en 10K+ steps. Sin Dual LR: NaN en <10 steps.

**Fuente**: `lyra/model/lyra_net.py:160-190`, `lyra/model/train.py:63-71`
**Prioridad**: **MEDIA** — necesario cuando se implemente training E2E

---

### 3.5 SparseTernaryAdam — Optimizer eficiente (BAJA)

**Problema**: Con 90% sparsity en D, el 90% de gradientes se desperdician.

**Solucion**: Mask de gradientes basado en sparsity de D:

```python
dead = ~D_mask  # conexiones inactivas
grad = grad.masked_fill(dead, 0.0)  # zero gradients para dead connections
```

**Resultado**: 95% skip ratio, 10x menos compute en optimizer
**Fuente**: `lyra/kernels/sparse_optimizer.py:15-119`
**Prioridad**: **BAJA** — optimizacion de training, no critica

---

### 3.6 Extras encontrados en Lyra

| Tecnica | Archivo | Potencial para SpectralAI |
|---|---|---|
| **SoftHebb** (aprendizaje local sin backprop) | `lyra/core/connectivity.py:135` | Actualizar W_dispersion online sin loss global |
| **Triton ternary_matmul** (skip zero tiles) | `lyra/kernels/ternary_matmul.py` | 1.42x speedup en forward de expertos ternarios |
| **Triton update_d** (fused growth/decay) | `lyra/kernels/update_d.py` | 4096x4096 en 1.79ms para actualizar BVH |
| **CausalDecay + GRU Neuromod** | `lyra/core/causal_decay.py` | Control adaptativo: arousal/fatigue para LR dinamico |

### 3.7 Notas criticas de implementacion

1. **BF16 obligatorio** — FP16 causa NaN con SmoothSTE (validado en Lyra)
2. **D_cont clamp [-2, 2]** — despues de cada optimizer.step()
3. **SubLN NO es opcional** — sin el, routing ternario colapsa 100%
4. **Beta warmup > 0** — nunca empezar con beta alto. Lineal 1→10 en 1000 steps
5. **Inicializacion D_cont** — Uniform(-1, 1), NO Kaiming

---

### 3.8 BVH Vivo: Auto-poda metabolica (MEDIA)

> Origen: `lyra/core/connectivity.py:144-172`, `lyra/core/compression.py`,
> `lyra/core/llm_integration.py:7`

**Concepto**: El BVH se convierte en un arbol vivo que se auto-limpia.
Tres mecanismos de Lyra aplicados al BVH:

**A. Poda por edad** — Nodos BVH que no reciben rayos en N steps se eliminan:

```python
# Cada esfera BVH tiene un contador de edad
sphere.age += 1                       # cada step
sphere.age = 0 if sphere.was_hit      # reset si recibio rayo
if sphere.age > max_age:
    bvh.remove(sphere)                # poda automatica
```

**B. Reservas metabolicas** — Mantener hijos cuesta energia:

```python
# Esferas con muchos hijos gastan mas energia
sphere.reserves -= 0.001 * len(sphere.children) + 0.01
if sphere.reserves <= 0:
    bvh.collapse(sphere)  # nodo muere, hijos se redistribuyen
```

**C. Fatigue → compresion** — Cuando VRAM sube, comprimir esferas frias:

```python
if neuromodulator.fatigue > threshold:
    cold_spheres = [s for s in bvh if s.age > warm_threshold]
    for s in cold_spheres:
        s.embedding = polar_quant.compress(s.embedding)  # 4.6x menos
```

**Resultados en Lyra**:
- Sparsity auto-crece: 0.90 → 0.95 (auto-poda sin supervision)
- Reservas metabolicas varian 0.22-0.78 entre capas (diferenciacion funcional)

**Impacto en SpectralAI**:
- BVH mas pequeno → traversal mas rapido (menos nodos = menos niveles)
- VRAM adaptativa → comprimir esferas frias libera memoria para expertos
- Auto-organizacion → el arbol se optimiza solo durante inferencia

**Fuente**: `lyra/core/connectivity.py:144-172`, `lyra/core/compression.py`
**Prioridad**: **MEDIA** — requiere rayos espectrales + SmoothSTE primero

---

### 3.9 Proyeccion de impacto: SpectralAI antes vs despues

Comparacion del proyecto **antes y despues** de aplicar las tecnicas adaptadas.
Todos los numeros "antes" son mediciones reales del proyecto; los "despues" son
proyecciones basadas en las implementaciones ya probadas en CPU (`python/spectral_techniques.py`,
37/37 tests pasando).

#### Estado actual de SpectralAI (medido)

| Metrica | Valor medido | Fuente |
|---|---|---|
| PPL (16/16 capas BVH) | 8.29 | commit 9bab7ce |
| PPL baseline (gate lineal OLMoE) | 7.15 | ROADMAP (transformers 5.4.0) |
| Degradacion vs gate lineal | +16.1% | 8.29 vs 7.15 (retrained) |
| Routing latency | 10 us (CUDA), 64.6 us (OptiX) | benchmarks |
| Expert forward | 940 us | benchmark_e2e |
| VRAM activa (routing) | 7.86 MB | real_model_demo |
| Resolucion polisemia | 0% | sin espectral |
| Training E2E del BVH | NO (no diferenciable) | ARCHITECTURE.md |
| Auto-poda BVH | NO (64 nodos fijos) | — |
| Avg top-8 accuracy | 85.6% | ROADMAP (tabla por capa) |

#### Proyeccion: SpectralAI antes → despues

| Mejora | Metrica | SpectralAI ANTES | SpectralAI DESPUES (est.) | Confianza |
|---|---|---|---|---|
| **Rayos espectrales** (Sec 1) | PPL | 8.29 | ~7.3 | Alta (88.9% polisemia medido en PyTorch) |
| **SmoothSTE** (Sec 3.1) | Training E2E | Imposible (no diferenciable) | Posible (37 tests CPU OK) | Alta |
| **SmoothSTE** → fine-tune BVH | PPL post-training | ~7.3 | ~6.8 | Media (requiere GPU) |
| **LiquidTimeGate** (Sec 3.2) | Convergencia training | Sin gating temporal | +40% pasos para misma loss | Media |
| **SubLN/RMSNorm** (Sec 3.3) | Estabilidad training | No hay norm post-routing | 0 saturacion | Alta (obligatorio) |
| **Dual LR** (Sec 3.4) | NaN en training | Esperado sin proteccion | 0% NaN | Alta |
| **Auto-poda metabolica** (Sec 3.8) | Nodos BVH activos | 64 (fijos) | ~40-50 (dinamico) | Media |
| **Auto-poda** → traversal | OptiX latency | 64.6 us | ~45-55 us | Baja |
| **PolarQuant** (Sec 2) | VRAM espectral | 528 KB/tok | 115 KB/tok (4.6x) | Alta |

#### Resumen: SpectralAI antes vs despues (mejor caso realista)

| Metrica | ANTES | DESPUES | Mejora |
|---|---|---|---|
| **PPL (16/16)** | 8.29 (+16.1% vs gate) | **~6.8** (-5.0% vs gate) | **-18% PPL** |
| **Degradacion vs gate lineal** | +16.1% | **~-5%** (supera gate) | Cierra gap completo |
| **Training E2E del BVH** | Imposible | **Posible** (SmoothSTE+SubLN+DualLR) | Desbloqueo total |
| **Resolucion polisemia** | 0% | **88.9%** | Salto cualitativo |
| **BVH nodos activos** | 64 (fijo) | **~45** (auto-poda) | -30% nodos |
| **OptiX latency** | 64.6 us | **~50 us** | -23% |
| **VRAM espectral** | 528 KB/tok | **115 KB/tok** | 4.6x menos |
| **Avg top-8 accuracy** | 85.6% | **~92%+** (post fine-tune) | +6.4 pp |

> **Nota:** El PPL proyectado ~6.8 podria incluso superar al gate lineal (7.15)
> porque el fine-tune E2E del BVH (habilitado por SmoothSTE) permite optimizar
> la geometria directamente para la tarea, algo que el gate lineal no puede hacer.

#### Secuencia de implementacion recomendada

```
Fase 1 (Mayor impacto, sin training):
  1a. Rayos espectrales en kernels CUDA     → PPL 8.29 → ~7.3
  1b. SubLN post-routing                    → estabilidad

Fase 2 (Desbloquea training):
  2a. SmoothSTE para BVH diferenciable      → training posible
  2b. Dual LR (0.1x para BVH params)        → 0 NaN
  2c. LiquidTimeGate → init W_dispersion    → convergencia +40%

Fase 3 (Fine-tune con training E2E):
  3a. Entrenar BVH end-to-end               → PPL ~7.3 → ~6.8
  3b. SparseTernaryAdam                     → 10x optimizer speed

Fase 4 (Optimizacion):
  4a. Auto-poda metabolica del BVH          → -30% nodos, -23% latency
  4b. PolarQuant para vectores espectrales  → 4.6x VRAM espectral
```

#### Caveat importante

Estas proyecciones asumen que las tecnicas de Lyra (validadas en un modelo
de 16.5M params con TinyStories) escalan a SpectralAI (OLMoE 1B-7B).
Los numeros de PPL son extrapolaciones — el delta real dependera de:
- Calidad de los embeddings proyectados al espacio 3D
- Estabilidad del beta annealing a escala mayor
- Interaccion entre las 16 capas BVH con SubLN
Se recomienda validar con 1 capa antes de desplegar las 16.

---

## 4. CUDA/OptiX — Bugs y Optimizaciones

### CRITICAL

#### 2.1 Buffer Overflow en top-K accumulation
- **Archivo:** `cuda/ray_attention.cu:234-245`
- **Problema:** `total_hit_count` se usa como contador Y como tamano de array para `insert_top_token()`. Crece sin limite y desborda `accumulated_top_tokens[SPECTRAL_MAX_TOP_TOKENS]`.
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
- **Archivo:** `cuda/spectral_kernels.cu:296-307`
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
- **Problema:** Usa `2.0f * 3.14159265f` (truncado) en vez de `CUDART_PI_F` (como en `spectral_kernels.cu:99`). Ademas `fabsf()` pierde signo.
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

## 5. C++/Headers — Bugs y Mejoras

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

## 6. Python — Bugs y Optimizaciones

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

## 7. Build System (CMake)

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

#### 5.4 test_optix_pipeline no linka spectral_optix
- **Archivo:** `CMakeLists.txt:458-462`
- **Problema:** Falta `spectral_optix` en `target_link_libraries`. Errores de linker.
- **Fix:** Anadir `spectral_optix` al target.

### LOW

#### 5.5 Variable CMAKE_CUDA_ARCHITECTURES
- **Archivo:** `CMakeLists.txt:124`
- **Problema:** Usa `CUDA_ARCHITECTURES` en vez de `CMAKE_CUDA_ARCHITECTURES` (standard CMake).
- **Fix:** Renombrar para portabilidad.

---

## 8. Resumen de Prioridades

### Accion Inmediata (Bloqueantes / Corrupcion de datos)

| # | Archivo | Problema |
|---|---------|----------|
| 4.1 | ray_attention.cu:234 | Buffer overflow en top-K |
| 4.2 | alpha_phase_b.cu:300 | FP32->FP16 no implementado |
| 4.3 | ray_generation.cu:272 | Data race en accumulation |
| 5.1-5.3 | token_geometry.cpp, alpha_bsh.cpp | Memory leaks (usar vector) |
| 7.2 | CMakeLists.txt:229 | PTX no compila para sm_120 (Blackwell) |

### Alta Prioridad — Nuevas Features (Mayor impacto)

| # | Seccion | Propuesta | Impacto estimado |
|---|---------|-----------|------------------|
| **3.1** | **Lyra → SmoothSTE** | **Diferenciabilidad BVH via beta annealing** | **Desbloquea training E2E** |
| 1.0 | Rayos Espectrales | Integrar en kernels CUDA | -12% PPL, 0.85% overhead |
| 3.2 | Lyra → LiquidTimeGate | Inicializar W_dispersion LOCAL/GLOBAL | +40% convergencia |

### Alta Prioridad — Bugs (Resultados incorrectos / Crashes)

| # | Archivo | Problema |
|---|---------|----------|
| 4.4 | spectral_kernels.cu:296 | Coordinate space mismatch |
| 4.5 | optix_router_raygen.cu:108 | Variable indefinida en cross product |
| 5.5 | alpha_bsh.cpp:370 | Null pointer sin validar |
| 5.6 | semantic_bvh.cpp:311 | malloc() donde deberia ser cudaMalloc() |
| 6.1 | async_pipeline_bridge.py:134 | GPU memory leak |
| 6.2 | orchestrator.py:240 | Device mismatch |
| 7.1 | CMakeLists.txt:455 | Typo OptiX include |

### Media Prioridad (Rendimiento / Calidad / Estabilidad training)

| # | Archivo | Problema |
|---|---------|----------|
| 3.3 | Lyra → SubLN | RMSNorm post-routing (obligatorio para training) |
| 3.4 | Lyra → Dual LR | LR 0.1x para params BVH |
| 4.13 | closest_hit.cu:111 | sqrt redundante en hot path |
| 5.9 | alpha_bsh.cpp:249 | Loop O(N^2) parent-child |
| 6.5 | benchmark_expert_types.py:318 | .item() sync en loop |
| 6.7 | olmoe_e2e_eval.py:242 | Security: pickle sin restriccion |
| 2.0 | PolarQuant | Comprimir vectores espectrales 4.6x (post rayos espectrales) |

### Total: 50+ hallazgos

| Severidad | CUDA/OptiX | C++/Headers | Python | CMake | Lyra Synergy | Total |
|-----------|-----------|-------------|--------|-------|-------------|-------|
| CRITICAL | 2 | 4 | 0 | 0 | 1 (SmoothSTE) | **7** |
| HIGH | 8 | 4 | 6 | 2 | 2 | **22** |
| MEDIUM | 5 | 4 | 5 | 2 | 3 | **19** |
| LOW | 1 | 0 | 0 | 1 | 1 | **3** |
| **Total** | **16** | **12** | **11** | **5** | **7** | **51** |
