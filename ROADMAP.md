# LiquidBit Zero-Matrix — ROADMAP
> Hoja de ruta completa del proyecto. Ultima actualizacion: 2026-03-28
> Para decisiones y fallos: LEARNINGS.md | Para arquitectura: CLAUDE.md

---

## Vision

**LiquidBit** elimina MatMul de la atencion de LLMs, sustituyendolo por
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
| `bvh_router_deep.cu` | Router 3-8 niveles (65K expertos) | ~15 us | Compilado |
| `async_pipeline.cu` | Pipeline tri-core triple buffer | TBD | Compilado |
| `ternary_expert.cu` | BitNet 1.58 POPCOUNT | TBD | Compilado |
| `liquid_expert.cu` | Experto ODE adaptativo | TBD | Por compilar |
| `optix_bvh_router.cu` | RT Cores via OptiX | TBD | SDK instalado, pendiente build |

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

## FASE 6 — Pipeline Asincrono Completo [PENDIENTE]

**Objetivo:** Coreografia del silicio — RT Cores + CUDA Cores + Tensor Cores
trabajando en paralelo.

**El concepto:**
```
Token N:   [RT Core: RUTA] → [Tensor Core: GENERA]
Token N+1:              [RT Core: RUTA] → [Tensor Core: GENERA]
                                    ↑ overlap ↑
```

**Tareas:**
- [ ] Integrar `async_pipeline.cu` (triple buffer, 3 streams con prioridades)
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

## FASE 7 — RT Cores Reales (OptiX) [LISTO PARA COMPILAR]

**Objetivo:** Conectar el routing real con RT Cores via OptiX SDK.

**El beneficio:** RT Cores hacen ray-sphere intersection en ~4 ciclos GPU
vs ~80 ciclos en CUDA cores. Speedup teorico: 10-20x sobre kernel actual.

**Prerequisitos instalados (2026-03-27):**
- ✅ CUDA Toolkit 12.8 (sm_120 para RTX 5070 Ti)
- ✅ OptiX SDK 9.1.0
- ✅ Visual Studio C++ tools

**Tareas:**
- [x] Instalar CUDA Toolkit 12.8
- [x] Instalar OptiX SDK 9.1
- [ ] Fix CMakeLists.txt (sm_120, OptiX 9.1 path, alpha_bsh.cpp, optix_host.cpp)
- [ ] Fix include paths en `cuda/optix_host.cpp`
- [ ] `cmake --build .` sin errores
- [ ] Compilar shaders OptiX (.cu → .ptx) para ray_generation, closest_hit, miss, any_hit
- [ ] Mapear sphere_centers → OptixAabb
- [ ] Construir IAS (Instance Acceleration Structure) jerarquico 4 niveles
- [ ] Rayo = embedding como origen + direccion
- [ ] RT Cores devuelven hit mas cercano = expert_id
- [ ] Benchmark: OptiX RT vs CUDA puro vs PyTorch

**Speedup esperado:**

| Implementacion | Latencia batch=256 | vs PyTorch |
|---|---|---|
| PyTorch | 1,580 us | 1x |
| CUDA kernel v2 | 8.84 us | 179x |
| OptiX RT Cores | ~0.5-1 us (estimado) | ~1,500-3,000x |

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
| **LiquidBit v5.0** | **RT Router O(log N)** | **O(log N) + O(k2)** | **RTX 5070 Ti** |

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

**Objetivo:** Modelo de negocio — LiquidBit como "placa base optica".

**El concepto:**
- LiquidBit = el router BVH (la infraestructura)
- Comunidad/empresas crean micro-expertos y los "enchufan" en esferas
- Bufete → esfera legal, Hospital → esfera medica, etc.

**Tareas:**
- [ ] Definir formato estandar de experto (.lbe — LiquidBit Expert)
- [ ] API de registro: `router.register_expert(sphere_id, expert_path)`
- [ ] Validacion de experto: calidad minima, tamano maximo, seguridad
- [ ] CLI: `liquidbit install expert-legal-es`
- [ ] Marketplace web (futuro)

---

## Estructura del proyecto (post-reorganizacion 2026-03-26)

```
liquidbit-zero-matrix/
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
- [ ] Benchmark: LiquidBit (RTX 5070 Ti) vs modelo base

**Modelos ternarios nativos disponibles (CERO cuantizacion):**

| Modelo | Params | Licencia | MMLU | VRAM |
|---|---|---|---|---|
| microsoft/bitnet-b1.58-2B-4T | 2B | MIT | 52% | ~1.2 GB |
| 1bitLLM/bitnet_b1_58-3B | 3.3B | MIT | ~LLaMA 3B | ~550 MB |
| SpectraSuite/TriLM_3.9B | 3.9B | Apache 2.0 | ~FloatLM 3.9B | TBD |
| HF1BitLLM/Llama3-8B-1.58 | 8B | Llama 3 | mejor calidad | ~1.5 GB |

---

## FASE A — OLMoE BVH Distillation [✅ VALIDADA — PPL +0.8% single, +4.8% multi]

**Objetivo:** Reemplazar el gate lineal de OLMoE-1B-7B (7B params, 64 expertos) con
nuestro BVH Router geometrico y medir el impacto en perplexity real.

**Contexto:** FASE A v1 (MoE from scratch) llego a ceiling PPL=186. Los expertos no se
especializan con solo 5M tokens. OLMoE-1B-7B tiene 64 expertos SwiGLU ya especializados.

### Resultado principal (PPL end-to-end)

| Configuracion | PPL | Delta vs baseline (6.11) | Estado |
|---|---|---|---|
| Baseline (gate lineal OLMoE) | 6.11 | — | Referencia |
| BVH Router 1 capa (L8) | 6.16 | **+0.8%** | ✅ Validado |
| BVH Router 2 capas (L4,8) | 6.23 | **+2.0%** | ✅ Validado |
| BVH Router 5 capas (L0,4,8,12,15) | 6.40 | **+4.8%** | ✅ Validado |

**Degradacion lineal ~1% por capa reemplazada.** Extrapolacion: 16/16 capas → ~15% PPL.

### Precision por capa

| Capa | Top-8 | Top-1 | Calibracion cosine |
|---|---|---|---|
| L0 | 87.8% | 89.0% | 0.97 |
| L4 | 86.4% | 73.0% | 0.97 |
| L8 | 91.7% | 71.1% | 0.97 |
| L12 | 92.2% | 74.5% | 0.97 |
| L15 | 93.2% | 74.7% | 0.97 |

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

### Estado actual: FASE 3 — Multi-layer (5/16 capas)

- [x] 5 capas reemplazadas (0,4,8,12,15): PPL 6.40 (+4.8%)
- [ ] Entrenar capas restantes: 1,2,3,5,6,7,9,10,11,13,14
- [ ] 16/16 capas: target <15% PPL degradation

---

## Proximos pasos inmediatos (actualizado 2026-03-28)

### PRIORIDAD MAXIMA: Completar 16/16 capas

**Paso 1 — OLMoE Distillation** [✅ COMPLETADO]
- PPL 6.16 (+0.8%) con 1 capa, 6.40 (+4.8%) con 5 capas
- Pipeline completo: extract → train → calibrate → eval

**Paso 2 — 16/16 capas** [🔄 EN PROGRESO — 5/16]
- Re-generar checkpoints (perdidos 28-Mar): `bash scripts/regenerate_all.sh`
- Entrenar 11 capas restantes: 1,2,3,5,6,7,9,10,11,13,14
- Target: PPL <7.0 (≈+15% degradacion)

**Paso 3 — Build C++/CUDA con CMake** [✅ COMPILADO — 11 targets]
- CUDA 13.2, OptiX 9.1, CMake 4.2.3, MSVC 18.4, sm_89+sm_120
- Shaders OptiX → PTX: pendiente integrar con optixModuleCreate()

### DESPUES DEL PROTOTIPO

4. **Patentes:** Filing 3 provisionales USPTO ($1,050 total)
5. **Pipeline async (Fase 6):** RT + CUDA + Tensor Cores en paralelo
6. **Escalado (Fase 8):** 64 → 65K expertos con bvh_router_deep.cu
7. **Paper (Fase 10):** NeurIPS/ICML 2027

---
*Para contexto completo de cada decision y fallo: LEARNINGS.md*
*Para arquitectura detallada: CLAUDE.md*
