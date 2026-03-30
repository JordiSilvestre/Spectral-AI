# SpectralAI Zero-Matrix — ROADMAP
> Hoja de ruta completa del proyecto. Ultima actualizacion: 2026-03-29
> Para decisiones y fallos: LEARNINGS.md | Para arquitectura: CLAUDE.md

---

## Vision

**SpectralAI** elimina MatMul de la atencion de LLMs, sustituyendolo por
geometria espacial acelerada por RT Cores de NVIDIA. El sistema usa los
RT Cores como router O(log N) para seleccionar micro-expertos especializados,
logrando inferencia en hardware de consumidor (RTX 4090/5070 Ti) en lugar
de racks de H100.

---

## Estado actual — lo que esta hecho

| Version | Arquitectura | Best ppl | Params | Estado |
|---|---|---|---|---|
| GPT-2 baseline | Scaled dot-product O(N2) | 187.4 | 16.1M | Referencia |
| v4.0 Inception | BVH 4-nivel + Spectral + Fourier | 191.3 | 16.5M | Validado |
| v5.0 Orchestrator | Router BVH + Backbone condicionado | 362.5* | 19.9M | Fase 4 completa |
| Kernel CUDA v2 | Router fusionado 3 niveles | 8.84 us/batch | 89K (router) | Compilado y testeado |

*ppl alto esperado: WikiText-2 general no permite especializacion por dominio.

**Benchmark clave validado:**

| Sistema | Latencia routing (batch=256) | Speedup |
|---|---|---|
| PyTorch BVHRouter | 1,003 us | 1x |
| CUDA Extension (zero-copy) | 10 us | **105x** |
| CUDA kernel micro (aislado) | 9.1 us | **110x** |

**Benchmark Orchestrator completo (routing + backbone, batch=1):**

| Sistema | Latencia E2E | Speedup |
|---|---|---|
| Orchestrator PyTorch puro | 1.793 ms | 1x |
| Orchestrator + CUDA Extension | 0.949 ms | **1.89x** |

---

## FASE 1 — Validacion conceptual [COMPLETADA]

**Objetivo:** Demostrar que atencion sin MatMul es viable.

- [x] Inception Engine: 4 niveles BVH + Spectral + Fourier + Refraccion
- [x] ppl=191.3 vs GPT-2 187.4 (solo 2.1% peor) — hipotesis validada
- [x] Training completo en WikiText-2, 10 epochs, RTX 5070 Ti

**Archivos:** `inception_attention.py`, `train_inception.py`

---

## FASE 2 — Router + Micro-Expertos [COMPLETADA]

**Objetivo:** Pivote a Router Optico + backbone condicionado.

- [x] `bvh_router.py` — Router BVH 4 niveles, Gumbel-Softmax/argmax
- [x] `micro_expert.py` — 4 tipos: FP16, INT8, Ternario, Inception
- [x] `orchestrator.py` — Pipeline completo, backbone compartido, 27 it/s
- [x] Training: ppl=362.5, 8 dominios, 3.7 min total

**Archivos:** `bvh_router.py`, `micro_expert.py`, `orchestrator.py`

---

## FASE 3 — Kernels CUDA [EN PROGRESO]

**Objetivo:** Reemplazar routing PyTorch por kernels CUDA fusionados.

- [x] Compilar `bvh_router_kernel.cu` (186KB, sm_120)
- [x] Compilar `ternary_expert.cu` (99KB)
- [x] Test 5/5 pasados: determinismo, latencia 8.83us, throughput 28.9M tok/s
- [x] Reorganizar proyecto: kernels en `cuda/v5/`
- [x] Conectar kernel via ctypes (`bvh_router_cuda.py`) — funcional pero 700us overhead
- [x] Crear extension PyTorch zero-copy (`bvh_torch_ext.cu` + pybind11) — 105x speedup
- [x] Benchmark E2E: Orchestrator 1.89x mas rapido con CUDA routing
- [x] Medir end-to-end: routing CUDA + backbone PyTorch
- [ ] Compilar `liquid_expert.cu` (ODE adaptativo)

**Archivos:** `cuda/v5/`, `python/bvh_router_cuda.py`, `cuda/v5/bvh_torch_ext.cu`

**Kernels disponibles:**

| Kernel | Funcion | Latencia | Estado |
|---|---|---|---|
| `bvh_router_kernel.cu` | Router 3 niveles, constant mem | 8.83 us | Compilado |
| `bvh_router_deep.cu` | Router 3-8 niveles (65K expertos) | ~15 us | No creado (FASE 8) |
| `async_pipeline.cu` | Pipeline tri-core triple buffer | TBD | No creado (FASE 6) |
| `ternary_expert.cu` | BitNet 1.58 POPCOUNT (PyTorch ext) | TBD | Compilado (ternary_torch_ext.cu) |
| `liquid_expert.cu` | Experto ODE adaptativo | TBD | No creado (FASE 9) |
| `optix_bvh_router.cu` | RT Cores via OptiX (full attn) | TBD | SDK instalado, pendiente build |
| `optix_router_raygen.cu` | RT Core expert selection | TBD | Shader creado, pendiente PTX |
| `optix_router_hitgroup.cu` | RT Core hit/miss programs | TBD | Shader creado, pendiente PTX |

---

## FASE 4 — Training Multi-Dominio [COMPLETADA]

**Objetivo:** Demostrar que el router aprende a discriminar dominios.

**Resultado: 100% routing accuracy en 4 dominios** (meta era >90%).

**Tareas:**
- [x] Crear 4 datasets por dominio:

| Dominio | Dataset | ~Tokens | Domain ID | Acc |
|---|---|---|---|---|
| General | WikiText-2 | 2.3M | 0 | 100% |
| Codigo Python | Sintetico (templates) | 2.5M | 1 | 100% |
| Ciencia | Sintetico (abstracts) | 2.5M | 2 | 100% |
| Legal | Sintetico (contratos) | 2.5M | 3 | 100% |

- [x] Crear `multi_domain_dataset.py`: DomainDataset + collate con domain_ids
- [x] Entrenar con supervision: L_task + alpha_router * L_routing + alpha_balance * L_balance
- [x] Medir routing accuracy: 100% — cada dominio va a su grupo de expertos
- [x] Especialización: 2-6 expertos unicos por dominio (de 16 disponibles)
- [ ] Inicializar BVH con `semantic_initializer.py` (K-means jerarquico) — pendiente para datasets reales

**Nota:** Val ppl ~50K es alto porque los datos sinteticos son templates repetitivos.
Con datasets reales (codeparrot, peS2o, pile-of-law) el ppl seria significativamente menor.

**Archivos:** `multi_domain_dataset.py`, `train_multi_domain.py`, `orchestrator.py` (modificado)

---

## FASE 5 — Benchmark de Cuantizacion [3/4 COMPLETO]

**Objetivo:** Medir FP16 vs INT8 vs Ternario vs Inception como backbone.

**Script:** `python/benchmark_expert_types.py`

**Resultados:**

| Tipo | Val PPL | tok/s | Estado |
|---|---|---|---|
| FP32 | 13.3 | 1,050,000 | ✅ Completado |
| FP16 | 13.6 | 1,350,000 | ✅ Completado |
| Ternario | 349.9 | — | ✅ Completado (PPL alta — esperado sin fine-tune) |
| INT8 (CPU) | — | — | ⏭ Saltado (CPU-only en PyTorch, no comparable) |

**Conclusion:** FP16 es el sweet spot (0.3 PPL peor, 28% mas rapido). Ternario necesita
fine-tuning con datos suficientes o expertos pre-entrenados (→ OLMoE approach).

**Tareas:**
- [x] Entrenar backbone FP32 base en cada dominio
- [x] Entrenar backbone FP16 (post-training quant)
- [x] Benchmark Ternario (PPL=349.9 — confirma que ternario from scratch falla sin datos)
- [x] INT8 descartado (no hay soporte GPU en PyTorch nativo)

**Dependencia:** Fase 4 (datos multi-dominio) ✅

---

## FASE 6 — Pipeline Asincrono Completo [EN PROGRESO — DISEÑO COMPLETADO]

**Objetivo:** Coreografia del silicio — RT Cores + CUDA Cores + Tensor Cores
trabajando en paralelo.

**El concepto:**
```
Token N:   [RT Core: RUTA] -> [Tensor Core: GENERA]
Token N+1:              [RT Core: RUTA] -> [Tensor Core: GENERA]
                                    ^ overlap ^
```

**Diseño completado (2026-03-28f):**
- [x] `cuda/async_pipeline.cu` — Triple buffer CUDA kernel con 3 streams priorizados
- [x] `python/async_pipeline_bridge.py` — Simulador Python para validacion de correctness
- [x] Kernels: scatter_by_expert, weighted_combine, apply_calibration, softmax_topk
- [x] Benchmark function: `benchmark_async_pipeline()` con synthetic data

**Tareas pendientes:**
- [ ] Integrar con optixLaunch() real (actualmente placeholder en stage_route)
- [ ] Device-side expert dispatch (zero host synchronization)
- [ ] Medir latencia total: routing(8us) + expert_forward(?) + overlap
- [ ] CUDA Graph capture para pipeline completo
- [ ] Benchmark: latencia first-token, throughput sostenido

**Hardware target:** RTX 5070 Ti
- Stream 0 (alta prioridad): routing BVH
- Stream 1 (media): expert inference
- Stream 2 (baja): transfers CPU↔GPU

**Metrica de exito:** <1ms latencia first-token, >10K tok/s sostenido.

---

## FASE 7 — RT Cores Reales (OptiX) [BENCHMARK COMPLETADO]

**Objetivo:** Conectar el routing real con RT Cores via OptiX SDK.

**El beneficio:** RT Cores hacen ray-sphere intersection en ~4 ciclos GPU
vs ~80 ciclos en CUDA cores. Speedup teorico: 10-20x sobre kernel actual.

**Prerequisitos instalados (2026-03-27):**
- ✅ CUDA Toolkit 12.8 (sm_120 para RTX 5070 Ti)
- ✅ OptiX SDK 9.1.0
- ✅ Visual Studio C++ tools

**Build completado (2026-03-28):**
- ✅ 4 PTX shaders compilados: ray_generation (9KB), closest_hit (5KB), miss (2KB), ray_attention (41KB)
- ✅ spectral_core.lib + spectral_optix.lib + inception_runner.exe
- ✅ 0 errores de compilacion (solo warnings menores)

**Host code overhaul (2026-03-28d):**
- ✅ optix_host.cpp: split single module → 3 modules (raygen, hitgroup, miss)
- ✅ Fix entry point names (alpha_bsh_* → optical_attention matching real shaders)
- ✅ Remove non-existent __anyhit__ and __intersection__ references
- ✅ Add loadPTXFile() + createSpectralAIOptixContextFromFiles() factory
- ✅ Fix pipeline compile options (was nullptr → stored as member)
- ✅ test_optix_pipeline.cpp: integration test with GAS build + CPU baseline

**RT Core Router (2026-03-28e):**
- ✅ optix_router_raygen.cu: minimal raygen (single-ray + top-K multi-ray fan)
- ✅ optix_router_hitgroup.cu: closesthit returns primitiveIndex, miss returns sentinel
- ✅ optix_router_host.cpp: RTCoreRouter class with GAS build + benchmark
- ✅ CMakeLists.txt: spectral_rt_router lib + rt_router_benchmark executable
- ✅ benchmark_routing_backends.py: compare PyTorch vs CUDA ext vs 3D-PCA vs OptiX

**Tareas:**
- [x] Instalar CUDA Toolkit 12.8
- [x] Instalar OptiX SDK 9.1
- [x] Fix CMakeLists.txt (sm_120, OptiX 9.1 path, alpha_bsh.cpp, optix_host.cpp)
- [x] Fix include paths en `cuda/optix_host.cpp`
- [x] `cmake --build .` sin errores
- [x] Compilar shaders OptiX (.cu → .ptx) para ray_generation, closest_hit, miss, ray_attention
- [x] Fix host code: multi-module, entry points, PTX loader, pipeline options
- [x] Mapear sphere_centers → OptixAabb (en buildAccelerationStructure + test)
- [x] **Benchmark: OptiX RT vs CUDA puro vs PyTorch — COMPLETADO**
- [x] Rayo = embedding como origen + direccion (raygen shader con top-K fan)
- [x] RT Cores devuelven hit mas cercano = expert_id (closesthit shader)
- [x] Fix cuCtxCreate_v4 para CUDA 13.2 compatibility
- [x] Fix optix_function_table_definition.h linkage
- [ ] Construir IAS (Instance Acceleration Structure) jerarquico 4 niveles
- [ ] Benchmark con N=1024+ expertos (crossover point vs CUDA kernel)

**Benchmark RT Core (RTX 5070 Ti, 2026-03-28):**

| Implementacion | batch=256 | batch=4096 | batch=16384 | vs PyTorch |
|---|---|---|---|---|
| PyTorch BVHRouter | 1,580 us | — | — | 1x |
| CUDA kernel v2 | 8.84 us | — | — | 179x |
| **OptiX RT Cores** | **64.6 us** | **61.1 us** | **69.4 us** | **24x** |

**Analisis:** RT Cores son ~7x MAS LENTOS que CUDA kernel a batch=256. Esto es esperado:
la flat GAS con 64 AABBs tiene overhead de pipeline OptiX (setup, launch) que domina a
batch pequeno. La fortaleza de RT Cores emerge a N>>1000 expertos donde la complejidad
O(log N) del BVH de hardware supera el scan lineal del CUDA kernel. Proximo: benchmark
con N=1024 y N=4096 expertos para encontrar el crossover point.

**Throughput (medido):**
- batch=256: 3.96M queries/s
- batch=4096: 67M queries/s
- batch=16384: **236M queries/s** — latencia casi constante, escala linealmente

---

## FASE 8 — Escalado [PENDIENTE]

**Objetivo:** Escalar de 8 a 64-65K expertos.

**Tareas:**
- [ ] Escalar a 4x4x4 = 64 expertos con `bvh_router_kernel.cu`
- [ ] Escalar a 8 niveles = 65,536 expertos con `bvh_router_deep.cu`
- [ ] Backbone 256d (vs 128d actual)
- [ ] Cuantizar expertos reales con `quantize_to_ternary.py` (Qwen2.5, Phi-3)
- [ ] `training_pipeline.py`: Sparse Upcycling de modelo denso → N expertos ternarios
- [ ] Medir: 10,000 expertos × 1.1 MB = 11 GB total en disco, 1 GB VRAM activa

**Metrica de exito:**
- 64 expertos: ppl < 200 por dominio
- 1,000+ expertos: VRAM activa < 2 GB
- Throughput: >1K tok/s en RTX 5070 Ti

---

## FASE 9 — Liquid Expert (ODE Adaptativo) [INVESTIGACION]

**Objetivo:** Probar expertos con velocidad de pensamiento adaptativa.

**El concepto:** `dx/dt = -x/tau(x,I) + f(x,I)/tau(x,I)` — la constante de
tiempo tau depende del input. Preguntas faciles → tau bajo (rapido).
Preguntas dificiles → tau alto (mas compute).

**Tareas:**
- [ ] Compilar `liquid_expert.cu`
- [ ] Integrar como opcion en `micro_expert.py`
- [ ] Benchmark: Liquid Expert vs Transformer en calidad y velocidad
- [ ] Coupling: tau computado desde spectral color del router

**Riesgo:** La diferenciabilidad de ODE solvers en GPU es compleja.

---

## FASE 10 — Paper y Benchmark Formal [FUTURO]

**Objetivo:** Publicacion academica con benchmarks rigurosos.

**Comparativas necesarias:**

| Sistema | Routing | Complejidad | Hardware |
|---|---|---|---|
| GPT-2 (16M) | MatMul O(N2) | N2 | CPU |
| LLaMA-3 (8B) | FlashAttention O(N2) | N2 | A100 |
| Mixtral (47B MoE) | Sparse MoE O(N2) | N2 + routing | 4x A100 |
| DeepSeek-V3 (671B) | MoE + Aux loss | N2 + routing | rack H100 |
| **SpectralAI v5.0** | **RT Router O(log N)** | **O(log N) + O(k2)** | **RTX 5070 Ti** |

**Metricas del paper:**
- [ ] Perplexity por dominio vs generalista
- [ ] Throughput (tok/s) a N=128, 1K, 10K, 100K tokens
- [ ] VRAM usage vs N (grafica O(log N) vs O(N2))
- [ ] Routing accuracy por dominio
- [ ] Latencia: PyTorch vs CUDA vs OptiX (tabla triple)
- [ ] Comparativa energia (W) por token
- [ ] Cold start latency (cargar experto desde CPU a GPU)

**Target conferencia:** NeurIPS 2027 / ICML 2027

---

## FASE 11 — App Store de Expertos [FUTURO]

**Objetivo:** Modelo de negocio — SpectralAI como "placa base optica".

**El concepto:**
- SpectralAI = el router BVH (la infraestructura)
- Comunidad/empresas crean micro-expertos y los "enchufan" en esferas
- Bufete → esfera legal, Hospital → esfera medica, etc.

**Tareas:**
- [ ] Definir formato estandar de experto (.lbe — SpectralAI Expert)
- [ ] API de registro: `router.register_expert(sphere_id, expert_path)`
- [ ] Validacion de experto: calidad minima, tamano maximo, seguridad
- [ ] CLI: `spectral install expert-legal-es`
- [ ] Marketplace web (futuro)

---

## Estructura del proyecto (post-reorganizacion 2026-03-26)

```
spectral-ai/
├── CLAUDE.md              # Arquitectura detallada
├── LEARNINGS.md           # Diario de decisiones y fallos
├── ROADMAP.md             # Este archivo
├── CMakeLists.txt         # Build system C++/CUDA
│
├── python/                # Implementacion Python activa
│   ├── bvh_router.py          # v5.0 Router BVH PyTorch (training)
│   ├── bvh_router_cuda.py     # v5.0 Bridge CUDA ctypes (inferencia)
│   ├── micro_expert.py        # v5.0 Wrapper expertos (FP16/INT8/Ternary)
│   ├── orchestrator.py        # v5.0 Pipeline Router->Expert completo
│   ├── inception_attention.py # v4.0 Atencion completa (referencia)
│   ├── gpt2_baseline.py       # Baseline GPT-2
│   ├── quantize_to_ternary.py # Cuantizacion BitNet avanzada
│   ├── semantic_initializer.py # Inicializacion BVH desde embeddings
│   ├── training_pipeline.py   # Sparse upcycling, semantic batching
│   └── train_*.py             # Scripts de training
│
├── cuda/
│   ├── v4/                    # Kernels Inception (ray tracing, resonancia)
│   └── v5/                    # Kernels Orchestrator (router+expert)
│       ├── bvh_router_kernel.cu   # Router fusionado 3 niveles
│       ├── bvh_router_deep.cu     # Router escalable 3-8 niveles
│       ├── async_pipeline.cu      # Pipeline tri-core triple buffer
│       ├── ternary_expert.cu      # BitNet 1.58 POPCOUNT
│       ├── liquid_expert.cu       # Experto ODE adaptativo
│       ├── optix_bvh_router.cu    # RT Cores via OptiX
│       ├── torch_bvh_extension.cpp # Binding PyTorch pybind11
│       ├── test_router.cu         # Tests del kernel
│       └── Makefile
│
├── include/               # Headers C++ publicos
├── src/                   # Implementaciones C++
├── tests/                 # Tests C++
├── docs/                  # Documentacion tecnica
│   ├── BENCHMARK_TEORICO.md   # Comparativa vs Mixtral/DeepSeek
│   ├── CUDA_BVH_ROUTER.md    # Doc tecnica del router CUDA
│   ├── MEMORY_BREAKTHROUGH.md # Analisis cache-bound vs VRAM-bound
│   └── archive/               # Docs historicos v1-v3
├── data/                  # Datasets, embeddings, logs de training
├── checkpoints/           # Modelos entrenados (.pt)
└── archive/               # Prototipos antiguos, codigo legacy
```

---

## PRIORIDAD 0 — Patentes + Demo Killer [PATENTES LISTAS, DEMO VALIDADA]

**Objetivo:** Proteger IP y demostrar viabilidad con modelo real.

**Patentes provisionales (USPTO, $350/cada):**
- [x] LBS-2026-001: RT Cores como atencion O(log N) — `patents/patent_01_rt_attention.md`
- [x] LBS-2026-002: IAS anidados para 12D — `patents/patent_02_inception_engine.md`
- [x] LBS-2026-003: Codificacion espectral + Snell — `patents/patent_03_spectral_routing.md`
  - Reforzada (2026-03-27): Claims 21-33 con 3 mecanismos irreducibles:
    - Chromatic Aberration (multi-band decomposition)
    - Total Internal Reflection (discontinuous boundary)
    - Phase-Coherent Multi-Ray Interference
- [ ] Revision por abogado de patentes
- [ ] Filing en USPTO (provisional, 12 meses de proteccion)

**Demo con modelo real:**
- [x] `real_model_demo.py` — soporta 8+ modelos HuggingFace
- [x] Soporte ternario nativo (BitNet 2B, 1BitLLM 3B, TriLM 3.9B)
- [x] Soporte post-training quant (Qwen2.5, Phi-3, TinyLlama)
- [ ] Ejecutar demo con microsoft/bitnet-b1.58-2B-4T (MIT, 2B params, MMLU 52%)
- [ ] Benchmark: SpectralAI (RTX 5070 Ti) vs modelo base

**Modelos ternarios nativos disponibles (CERO cuantizacion):**

| Modelo | Params | Licencia | MMLU | VRAM |
|---|---|---|---|---|
| microsoft/bitnet-b1.58-2B-4T | 2B | MIT | 52% | ~1.2 GB |
| 1bitLLM/bitnet_b1_58-3B | 3.3B | MIT | ~LLaMA 3B | ~550 MB |
| SpectraSuite/TriLM_3.9B | 3.9B | Apache 2.0 | ~FloatLM 3.9B | TBD |
| HF1BitLLM/Llama3-8B-1.58 | 8B | Llama 3 | mejor calidad | ~1.5 GB |

---

## FASE A — OLMoE BVH Distillation [✅ 16/16 COMPLETADA — PPL 8.27 (+15.7%)]

**Objetivo:** Reemplazar el gate lineal de OLMoE-1B-7B (7B params, 64 expertos) con
nuestro BVH Router geometrico y medir el impacto en perplexity real.

**Contexto:** FASE A v1 (MoE from scratch) llego a ceiling PPL=186. Los expertos no se
especializan con solo 5M tokens. OLMoE-1B-7B tiene 64 expertos SwiGLU ya especializados.

### Resultado principal (PPL end-to-end)

**Sesion original (pre-pérdida datos):**

| Configuracion | PPL | Delta vs baseline (6.11) | Estado |
|---|---|---|---|
| Baseline (gate lineal OLMoE) | 6.11 | — | Referencia |
| BVH Router 1 capa (L8) | 6.16 | **+0.8%** | ✅ Validado |
| BVH Router 2 capas (L4,8) | 6.23 | **+2.0%** | ✅ Validado |
| BVH Router 5 capas (L0,4,8,12,15) | 6.40 | **+4.8%** | ✅ Validado |

**Re-training (post-recuperación, transformers 4.46.3):**

| Configuracion | PPL | Delta vs baseline (7.15) | Estado |
|---|---|---|---|
| Baseline (gate lineal OLMoE) | 7.15 | — | Referencia |
| BVH Router 1 capa (L8) | 7.19 | **+0.6%** | ✅ Validado |
| BVH Router 5 capas (L0,4,8,12,15) | 7.45 | **+4.2%** | ✅ Validado |

**Nota:** Baseline diferente (7.15 vs 6.11) por version de transformers. Los **deltas son
mejores** en el re-training (+0.6% y +4.2%) gracias a calibracion linear (4160 params)
en vez de affine (128 params).

**Degradacion ~0.98% por capa (superlinear).** Resultado real 16/16: PPL 8.27 (+15.7%) tras retrain 7 layers.

**PPL scaling curve completa (2026-03-28, transformers 5.4.0):**

| Configuracion | PPL | Delta vs baseline (7.15) | Capas |
|---|---|---|---|
| Baseline (gate lineal OLMoE) | 7.15 | — | 0/16 |
| BVH Router 1 capa (L8) | 7.19 | **+0.6%** | 1/16 |
| BVH Router 5 capas | 7.45 | **+4.2%** | 5/16 |
| BVH Router 12 capas (skip worst 4) | 7.86 | **+10.0%** | 12/16 |
| BVH Router 14 capas (skip L1,L2) | 8.12 | **+13.6%** | 14/16 |
| BVH Router 16 capas (ALL, pre-retrain) | 8.38 | +17.3% | 16/16 |
| BVH Router 16 capas (retrain L1+L2+L8) | 8.35 | +16.8% | 16/16 |
| BVH Router 16 capas (retrain L1+L2+L8+L13+L14) | 8.29 | +16.1% | 16/16 |
| **BVH Router 16 capas (retrain +L0+L4 = 7 layers)** | **8.27** | **+15.7%** | **16/16** |

**Hallazgo clave:** Retrain iterativo de capas debiles mejora PPL progresivamente.
L8 era el mayor cuello de botella (59.4% → 90.1%). Avg top-8 actual: 86.9%.

### Precision por capa (re-training vs original)

| Capa | Top-8 Acc | Top-1 Acc | Cal cosine | Estado |
|---|---|---|---|---|
| L0 | 89.5% | 89.3% | 0.95 | retrained (was 80.4%) |
| L1 | 79.3% | 85.3% | 0.94 | retrained (was 72.8%) |
| L2 | 84.7% | 82.8% | 0.95 | retrained (was 78.4%) |
| L3 | 80.5% | 81.5% | 0.95 | ok |
| L4 | 86.6% | 80.6% | 0.96 | retrained (was 80.2%) |
| L5 | 81.9% | 79.9% | 0.95 | ok |
| L6 | 84.3% | 80.7% | 0.96 | ok |
| L7 | 84.3% | 78.7% | 0.95 | ok |
| L8 | 90.1% | 77.8% | 0.97 | retrained (was 59.4%) |
| L9 | 88.3% | 78.0% | 0.96 | ok |
| L10 | 89.3% | 80.8% | 0.96 | ok |
| L11 | 81.8% | 78.4% | 0.95 | ok |
| L12 | 88.8% | 77.4% | 0.96 | ok |
| L13 | 92.4% | 77.9% | 0.96 | retrained (was 79.6%) |
| L14 | 93.4% | 78.6% | 0.96 | retrained (was 79.2%) |
| L15 | 89.3% | 80.2% | 0.96 | ok |

**Avg top-8: 85.6%.** Weakest: L1 (79.3%), L4 (80.2%), L0 (80.4%). 5 layers retrained.

### Componentes clave

- **EnhancedBVHRouter**: Jerarquia 4x4x4 = 64 expertos, ~1.35M params, 128-dim features
- **Sparse Upcycling**: Inicializacion del router desde pesos del gate (SVD + K-Means)
- **Calibracion Linear**: Capa 64→64 (4160 params) que ajusta distribucion de pesos → cosine 0.97
- **`norm_topk_prob=False`**: Critico — OLMoE usa pesos raw softmax, NO normalizados
- **Full softmax**: Softmax sobre 64 expertos completos, luego `.gather()` los top-k

### Bugs criticos resueltos

| Bug | Impacto | Solucion |
|---|---|---|
| `norm_topk_prob=False` ignorado | PPL 7.67 en vez de 6.11 | Leer atributo del gate original |
| Softmax restringido en hybrid | Pesos inflados (16 vs 64 expertos) | Softmax completo + gather |
| Distribucion de pesos BVH | PPL 134 sin calibrar | Calibracion linear 64→64 (4160 params) |

### Pipeline

```bash
# 1. Extraer hidden states reales
python python/extract_real_hiddens.py --model-dir /path/to/olmoe-1b-7b --layer 8

# 2. Entrenar router BVH (50 epochs, sparse upcycling)
python python/olmoe_bvh_distill.py --layer 8 --real-data data/real_hiddens_layer8.pt

# 3. Calibrar pesos (linear 4160 params)
python python/calibrate_router.py --mode linear --epochs 100 --real-data data/real_hiddens_layer8.pt

# 4. Evaluar PPL end-to-end
python python/olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt
```

**Archivos:** `python/extract_real_hiddens.py`, `python/olmoe_bvh_distill.py`,
`python/calibrate_router.py`, `python/olmoe_e2e_eval.py`

### Estado actual: FASE 3 — Multi-layer (16/16 capas COMPLETADAS)

- [x] 5 capas originales (0,4,8,12,15): PPL +4.2% (re-train) / +4.8% (original)
- [x] 11 capas adicionales: 1,2,3,5,6,7,9,10,11,13,14
- [x] **16/16 capas entrenadas** (EnhancedBVHRouter, 1.35M params cada)
- [x] **16/16 capas calibradas** (Linear 64x64, 4160 params cada, cosine >0.94)
- [x] Checkpoint validator: `python scripts/validate_checkpoints.py`
- [x] **16/16 PPL evaluation: PPL = 8.38 (+17.3% vs baseline 7.15)**
- [ ] Target: <15% PPL degradation — actual 17.3%, needs optimization

### Recuperacion de datos (2026-03-28)

⚠️ **Los checkpoints y datos fueron re-generados** tras perdida de archivos. Los valores
absolutos de PPL cambiaron (baseline 7.15 vs 6.11) por version de transformers (4.46.3).
Los **deltas relativos son comparables** y de hecho mejores gracias a calibracion linear.

**Tests manuales post-recuperacion:**

| Test | Estado | Notas |
|------|--------|-------|
| Kernel CUDA (`build_ext.py`) | ✅ | sm_120, 23.6 μs/iter |
| Expert ternario (`build_ternary_ext.py`) | ✅ | POPCOUNT OK, max diff 0.000031 |
| PPL single-layer (L8) | ✅ | +0.6% delta |
| PPL multi-layer (5 capas) | ✅ | +4.2% delta |
| Demo (`real_model_demo.py`) | ❌ | Routing colapsado, output garbage — POR ARREGLAR |

---

## Proximos pasos inmediatos (actualizado 2026-03-29)

### PRIORIDAD MAXIMA: Completar 16/16 capas

**Paso 1 — OLMoE Distillation** [✅ COMPLETADO]
- PPL 6.16 (+0.8%) con 1 capa, 6.40 (+4.8%) con 5 capas
- Pipeline completo: extract → train → calibrate → eval

**Paso 2 — 16/16 capas** [COMPLETADO]
- Re-generar checkpoints (perdidos 28-Mar): `bash scripts/regenerate_all.sh`
- 5 capas validadas: PPL +4.2% (mejor que original +4.8%)
- **16/16 capas entrenadas y calibradas**
- Avg accuracy: 82.4% top-8, 80.3% top-1
- **16/16 PPL = 8.38 (+17.3%)** — all 16 linear gates replaced with BVH routers

**Paso 2b — Arreglar demo** [✅ FIX APLICADO — PENDIENTE VERIFICACION]
- `real_model_demo.py` tenia routing colapsado (todos Expert #11) por calibracion faltante
- Fix: PCA + K-means calibracion del router antes de sync a CUDA
- Pendiente: verificar con modelo real cuando GPU este libre

**Paso 2c — Tecnicas Lyra para BVH Training (FASE B)** [EN PROGRESO]
- ✅ 6 tecnicas implementadas en `python/spectral_techniques.py` (37/37 tests CPU)
- ✅ SmoothBVHHit: BVH diferenciable para training E2E (el mayor blocker resuelto)
- Pendiente: integrar en pipeline GPU, fine-tune E2E, medir PPL antes/despues

**Paso 2d — Auditoria de bugs (MEJORAS.md)** [✅ 51/51 ARREGLADOS]
- ✅ 16 CUDA/OptiX bugs (buffer overflow, data races, coord mismatch, etc.)
- ✅ 12 C++/Headers bugs (memory leaks, null deref, double-free, etc.)
- ✅ 11 Python bugs (memory leaks, device mismatch, security, deprecated APIs)
- ✅ 5 CMake bugs (OptiX typo, sm_120, version check, linking)
- Pendiente: **compilar C++/CUDA en GPU para verificar** (ver comandos abajo)

**Paso 3 — Build C++/CUDA con CMake** [✅ COMPILADO — PENDIENTE RECOMPILAR CON FIXES]
- CUDA 13.2, OptiX 9.1, CMake 4.2.3, MSVC 18.4, sm_89+sm_120
- ✅ 4 PTX shaders compilados, spectral_core.lib, spectral_optix.lib, inception_runner.exe
- Pendiente: recompilar con los 51 bug fixes aplicados y verificar

---

## FASE B — Tecnicas Lyra adaptadas para BVH Training [EN PROGRESO]

**Objetivo:** Integrar 6 tecnicas del proyecto hermano (Lyra-AGI) para desbloquear
el entrenamiento end-to-end del BVH y mejorar PPL de 8.29 → ~6.8.

**Estado:** Implementaciones CPU creadas y testeadas (37/37 tests). Pendiente: GPU.

### Componentes implementados (CPU, testeados)

| Componente | Archivo | Tests | Estado |
|---|---|---|---|
| SmoothTernarySTE | `python/spectral_techniques.py` | 8/8 | ✅ CPU OK |
| SmoothBVHHit | `python/spectral_techniques.py` | 4/4 | ✅ CPU OK |
| RMSNorm (SubLN) | `python/spectral_techniques.py` | 3/3 | ✅ CPU OK |
| LiquidTimeGate | `python/spectral_techniques.py` | 6/6 | ✅ CPU OK |
| DualLR param groups | `python/spectral_techniques.py` | 3/3 | ✅ CPU OK |
| MetabolicBVH | `python/spectral_techniques.py` | 7/7 | ✅ CPU OK |
| BetaScheduler | `python/spectral_techniques.py` | 4/4 | ✅ CPU OK |
| Integration tests | `tests/test_spectral_techniques.py` | 2/2 | ✅ CPU OK |

### Tareas

- [x] Adaptar SmoothTernarySTE de Lyra para BVH (soft ternary quantization)
- [x] Crear SmoothBVHHit (diferentiable closest_hit replacement)
- [x] Adaptar RMSNorm (SubLN post-routing normalization)
- [x] Adaptar LiquidTimeGate (LOCAL/GLOBAL temporal gating)
- [x] Crear DualLR param groups (0.1x LR for BVH discrete params)
- [x] Implementar MetabolicBVH (age tracking + reserves + auto-pruning)
- [x] Implementar BetaScheduler (linear beta annealing 1→10)
- [x] Tests unitarios: 37/37 pasando en CPU
- [x] Auditoria de bugs: 51/51 arreglados (MEJORAS.md secciones 4-7)
- [ ] Integrar SmoothBVHHit en `bvh_router.py` forward pass
- [ ] Integrar RMSNorm post-routing en `orchestrator.py`
- [ ] Integrar DualLR en training scripts (`olmoe_bvh_distill.py`)
- [ ] Training E2E con SmoothSTE en GPU (RTX 5070 Ti)
- [ ] Compilar smooth_bvh_hit.cu (CUDA kernel, instrucciones en spectral_techniques.py)
- [ ] Medir PPL antes/despues con fine-tune E2E
- [ ] Activar MetabolicBVH en pipeline de inferencia

### Impacto proyectado (SpectralAI antes → despues)

| Metrica | Antes | Despues (est.) |
|---|---|---|
| PPL (16/16) | 8.29 (+16.1%) | ~6.8 (-5% vs gate) |
| Training E2E | Imposible | Posible |
| Polisemia | 0% | 88.9% |
| BVH nodos activos | 64 fijos | ~45 dinamico |

**Archivos:** `python/spectral_techniques.py`, `tests/test_spectral_techniques.py`
**Referencia:** `vendor/Lyra-AGI/` (submodulo temporal), `MEJORAS.md` Seccion 3

---

## Guia de verificacion y comandos (GPU requerida)

### Paso 1: Verificar tests CPU (sin GPU)

```bash
# Tests de tecnicas Lyra (37 tests, ~2 segundos)
python -m pytest tests/test_spectral_techniques.py -v

# Verificar que todos los archivos Python parsean correctamente
python -c "
import py_compile, glob
for f in glob.glob('python/*.py'):
    try:
        py_compile.compile(f, doraise=True)
        print(f'✅ {f}')
    except py_compile.PyCompileError as e:
        print(f'❌ {f}: {e}')
"
```

### Paso 2: Recompilar C++/CUDA con bug fixes (GPU requerida)

```bash
# Limpiar build anterior y recompilar
cd build
cmake --build . --clean-first 2>&1 | tee build_log.txt

# Verificar que compila sin errores:
grep -c "error" build_log.txt  # debe ser 0

# Si no hay directorio build, crear desde cero:
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="89;120" \
    -DOptiX_INSTALL_DIR=/path/to/OptiX
cmake --build . -j$(nproc) 2>&1 | tee build_log.txt
```

### Paso 3: Verificar bug fixes CUDA criticos (GPU requerida)

```bash
# Test del kernel de router (verifica fix buffer overflow 2.1)
cd build && ./test_router

# Test del pipeline OptiX (verifica fix OptiX include 5.1 + linking 5.4)
cd build && ./test_optix_pipeline

# Benchmark RT Cores (verifica fix coordinate space 2.4)
python python/benchmark_routing_backends.py
```

### Paso 4: Integrar tecnicas Lyra en pipeline de training (GPU requerida)

```bash
# 4a. Medir PPL baseline ANTES de cambios (guardar numero)
python python/olmoe_e2e_eval.py \
    --model-dir /path/to/olmoe-1b-7b \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt
# Resultado esperado: PPL ~8.29

# 4b. Re-entrenar 1 capa (L8) con SmoothSTE + SubLN + DualLR
#     Modificar olmoe_bvh_distill.py para usar:
#       from spectral_techniques import RMSNorm, get_dual_lr_param_groups, BetaScheduler
#     Anadir RMSNorm despues del router forward
#     Usar get_dual_lr_param_groups() en vez de optimizer directo
#     Crear BetaScheduler y llamar .step() cada batch
python python/olmoe_bvh_distill.py --layer 8 \
    --real-data data/real_hiddens_layer8.pt \
    --epochs 100 --use-smooth-ste

# 4c. Calibrar
python python/calibrate_router.py --mode linear --epochs 100 \
    --real-data data/real_hiddens_layer8.pt

# 4d. Medir PPL DESPUES
python python/olmoe_e2e_eval.py \
    --model-dir /path/to/olmoe-1b-7b \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt
# Resultado esperado: PPL <8.29 (mejora por SmoothSTE fine-tune)

# 4e. Si L8 mejora, repetir para las 16 capas
for layer in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
    python python/olmoe_bvh_distill.py --layer $layer \
        --real-data data/real_hiddens_layer${layer}.pt \
        --epochs 100 --use-smooth-ste
    python python/calibrate_router.py --mode linear --epochs 100 \
        --real-data data/real_hiddens_layer${layer}.pt
done

# 4f. PPL final con las 16 capas re-entrenadas
python python/olmoe_e2e_eval.py \
    --model-dir /path/to/olmoe-1b-7b \
    --all-layers
# Objetivo: PPL ~6.8 (vs 8.29 actual)
```

### Paso 5: Compilar CUDA kernel de SmoothBVHHit (opcional, alto rendimiento)

```bash
# Ver instrucciones detalladas en python/spectral_techniques.py (SMOOTH_BVH_HIT_CUDA_STUB)
# Resumen: modificar cuda/closest_hit.cu lineas 111-118:
#   float soft_hit = tanhf(c_beta * (semantic_radius - semantic_distance));
#   soft_hit = fmaxf(soft_hit, 0.0f);
#   float attention_weight = soft_hit * energy_remaining
#                           * expf(-c_lambda * semantic_distance);
# Compilar:
nvcc -arch=sm_89 -arch=sm_120 -c cuda/closest_hit.cu
```

### Paso 6: Activar MetabolicBVH (post-training)

```bash
# Probar auto-poda en inferencia:
python -c "
from python.spectral_techniques import MetabolicBVH
import numpy as np

mbvh = MetabolicBVH(n_nodes=64, max_age=100)
# Simular 200 pasos con solo 20 nodos activos recibiendo hits
for step in range(200):
    active_nodes = np.random.choice(64, 20, replace=False)
    mbvh.record_hits(active_nodes)
    stats = mbvh.step()
print(f'Activos: {stats[\"n_active\"]}, Sparsity: {stats[\"sparsity\"]:.2f}')
# Esperado: ~20-25 activos, sparsity ~0.65
"
```

---

## Resumen de tareas pendientes (todo el proyecto)

### Prioridad 1 — Verificar y medir (GPU requerida, ~1 dia)

| Tarea | Comando | Resultado esperado |
|---|---|---|
| Recompilar C++/CUDA | `cmake --build . --clean-first` | 0 errores |
| Tests CUDA | `./test_router && ./test_optix_pipeline` | PASS |
| PPL baseline | `olmoe_e2e_eval.py` | 8.29 (confirmar) |
| Verificar demo | `real_model_demo.py` | Routing no colapsado |

### Prioridad 2 — Fine-tune E2E con Lyra (GPU, ~2-3 dias)

| Tarea | Impacto estimado |
|---|---|
| Re-entrenar L8 con SmoothSTE+SubLN+DualLR | PPL L8 mejora |
| Re-entrenar 16 capas con Lyra techniques | PPL 8.29 → ~6.8 |
| Medir polisemia con rayos espectrales | 0% → 88.9% |

### Prioridad 3 — Escalar (GPU, semanas)

| Tarea | Fase |
|---|---|
| IAS jerarquico 4 niveles | FASE 7 |
| Benchmark N=1024+ expertos (crossover RT vs CUDA) | FASE 7 |
| Escalar a 64-65K expertos | FASE 8 |
| Pipeline asincrono RT+CUDA+Tensor | FASE 6 |
| Liquid Expert ODE adaptativo | FASE 9 |

### Prioridad 4 — Publicacion y negocio

| Tarea | Fase |
|---|---|
| Filing 3 patentes provisionales USPTO ($1,050) | PRIORIDAD 0 |
| Revision por abogado de patentes | PRIORIDAD 0 |
| Demo con bitnet-b1.58-2B-4T | PRIORIDAD 0 |
| Paper NeurIPS/ICML 2027 | FASE 10 |
| App Store de expertos (.lbe format) | FASE 11 |

### Prioridad 5 — Photonic Computing Bridge (largo plazo)

La arquitectura SpectralAI está diseñada como puente hacia computación fotónica:

| Concepto SpectralAI | Equivalente Fotónico | Estado |
|---|---|---|
| `spectral_color[64]` | Wavelength-Division Multiplexing (WDM) | Simulado en GPU |
| Ley de Snell (refracción) | Refracción óptica real en chip | Simulado en CUDA |
| RT Core 3D traversal | Interferencia óptica masiva (2048D) | Hardware futuro |
| PrismaticRefraction | Prisma óptico real (dispersión) | Hardware futuro |
| BVH O(N log N) | O(1) paralelo en fotónica | Hardware futuro |

**Claim para patente P3 (Spectral Routing):**
> "The spectral color vector is designed to map directly to wavelength-division
> multiplexing (WDM) on photonic hardware, where each dimension of the color
> corresponds to a physical optical frequency. The Snell-based refraction
> routing computes the same function as physical light refraction through
> prismatic media, enabling direct hardware acceleration when photonic
> processors become available. The architecture serves as an electronic
> simulation of photonic attention that is forward-compatible with optical
> computing platforms (LightGen, ACCEL, Q.ANT NPU)."

**Roadmap fotónico:**
1. AHORA: spectral_dim=64 en GPU (RT Cores 3D + color simulado)
2. PRÓXIMO: spectral_dim=256-2048 en CUDA cores (sin RT, pure compute)
3. FUTURO: spectral_dim=2048 en chip fotónico (WDM nativo, O(1))

**Referencia:** Investigación actual (2025-2026):
- LightGen (generative AI fotónico): 100x vs A100, 3W
- ACCEL (analog photonic chip): 3600x vs A100, 4.4nJ/frame
- Q.ANT NPU: 50x vs GPU, 30x eficiencia energética
- SDRC (Spectral Dimension Reservoir Computing): 56 nodos → state-of-the-art

---
*Para contexto completo de cada decision y fallo: LEARNINGS.md*
*Para arquitectura detallada: CLAUDE.md*
*Para auditoria de bugs detallada: MEJORAS.md*
