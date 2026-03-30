# STATUS.md — SpectralAI Zero-Matrix
> Estado real del proyecto, inventario de archivos, y roadmap por fases.
> Ultima actualizacion: 2026-03-30

---

## Estado General

| Aspecto | Estado |
|---|---|
| Concepto matematico | Validado (O(N log N) vs O(N^2)) |
| CUDA kernels v5 | Operativos (105x speedup routing, POPCOUNT ternary) |
| Demo killer (Qwen 1.5B) | 51.9 tok/s, 375x menos VRAM, ambos kernels CUDA activos |
| BVH Router distillation | ✅ 91.7% top-8 (L8), datos reales, calibracion linear |
| E2E PPL (1 capa) | ✅ PPL 6.16 (+0.8%) — BVH Router L8 con calibracion linear |
| E2E PPL (5 capas) | ✅ PPL 6.40 (+4.8%) — Capas 0,4,8,12,15 reemplazadas |
| Bugs criticos resueltos | norm_topk_prob=False, restricted softmax, calibracion |
| Pipeline OptiX v5 | ✅ Compilado: 6 PTX, benchmark 39µs/batch, 100% accuracy (triángulos) |
| RT Training Bridge | ✅ StraightThroughRT (STE): RT hard forward + soft backward |
| Patentes | 3 provisionales redactadas, Patent 3 reforzada con Claims 21-33 |

---

## INVENTARIO COMPLETO DE ARCHIVOS

### Python — CORE ACTIVO (usados en el pipeline actual)

| Archivo | Funcion | Usado por |
|---|---|---|
| `python/bvh_router.py` | Router BVH diferenciable (PyTorch). Gumbel-Softmax, 3D jerarquico | train_router, orchestrator |
| `python/bvh_router_bridge.py` | HybridBVHRouter: auto-selecciona PyTorch vs CUDA kernel | real_model_demo, train_moe |
| `python/ternary_expert_ext_bridge.py` | Expert ternario POPCOUNT (PyTorch ext, dims flexibles) | real_model_demo |
| `python/micro_expert.py` | 4 tipos de expert: FP16, INT8, Ternary, Inception | orchestrator, real_model_demo |
| `python/orchestrator.py` | Pipeline v5.0 completo: Tokens -> Router -> Experts -> Logits | demos, benchmarks |
| `python/real_model_demo.py` | Demo killer: Qwen + BVH + Ternary en RTX 5070 Ti | standalone |
| `python/expert_lru_cache.py` | GPU memory manager: top-k experts en GPU, LRU eviction | orchestrator, train_router |
| `python/semantic_initializer.py` | K-Means inicializacion del BVH desde embeddings reales | train_router |

### Python — TRAINING (scripts de entrenamiento)

| Archivo | Funcion | Estado |
|---|---|---|
| `python/train_router.py` | FASE 6: Entrena solo router+blend_gate (~500K params) sobre backbone frozen | Activo |
| `python/train_moe.py` | FASE A: MoE from scratch (embeddings + router + experts) | Techo PPL=186 |
| `python/train_multi_domain.py` | Entrenamiento supervisado 4 dominios (Wiki/Code/Science/Legal) | Validado 100% |
| `python/train_inception.py` | v4.0 Inception con L_task + L_spatial | Prototipo (no usado en v5) |
| `python/train_spectral_lm.py` | Baseline training SpectralAIForCausalLM | Prototipo |

### Python — DISTILLATION OLMoE (pipeline actual de desarrollo)

| Archivo | Funcion | Estado |
|---|---|---|
| `python/olmoe_extract.py` | Carga 64 experts SwiGLU + gate de OLMoE safetensors | Validado |
| `python/olmoe_bvh_distill.py` | EnhancedBVHRouter v2.1 + KD loss + RealHiddensDataset | Activo (necesita datos reales) |
| `python/extract_real_hiddens.py` | Extrae hidden states reales de OLMoE en WikiText-2 | NUEVO - pendiente ejecutar |
| `python/olmoe_e2e_eval.py` | Evaluacion PPL: BVH Router vs gate lineal | Activo (bugs corregidos) |
| `python/rt_training_bridge.py` | StraightThroughRT: RT Core forward + SmoothBVHHit backward | NUEVO |
| `python/distill_gate_labels.py` | Pre-computa gate labels de OLMoE (KL div) | Activo |
| `python/inspect_olmoe.py` | Inspeccion de arquitectura OLMoE (sin cargar pesos) | Utilidad |

### Python — MODELOS Y ATENCION

| Archivo | Funcion | Relacion |
|---|---|---|
| `python/inception_attention.py` | v4.0: 4 niveles nested 3D + spectral + Fourier | Arquitectura v4 completa |
| `python/optix_attention.py` | RT Core attention (modos: REAL/APPROX/MATMUL) | Requiere batch_runner.exe |
| `python/spectral_lm.py` | LM standalone: Embedding -> OptiXBlocks -> LMHead | Prototipo v4 |
| `python/gpt2_baseline.py` | Baseline GPT-2 para comparacion justa | Referencia |
| `python/spatial_loss.py` | Loss diferenciable BSH: L_prox + L_cover + L_inter | Usado en train_inception |
| `python/trainable_experts.py` | SwiGLU experts entrenables (no frozen) | FASE A |

### Python — DATASETS Y EMBEDDINGS

| Archivo | Funcion | Estado |
|---|---|---|
| `python/multi_domain_dataset.py` | 4 dominios (Wiki/Code/Science/Legal) con labels | Activo |
| `python/download_embeddings.py` | GloVe fallback (sintetico si falla descarga) | Obsoleto |
| `python/download_embeddings_v2.py` | Gensim loader (moderno) | Activo |
| `python/embedding_bridge.py` | Serializer: embeddings -> 3D -> binary TokenNode | Incompleto |
| `python/spectral_bridge.py` | Bridge completo: tokenize -> 3D -> binary + validacion | Activo |
| `python/tokenizer.py` | Tokenizador simple (BPE o word-level) | Utilidad |

### Python — BENCHMARKS

| Archivo | Funcion | Estado |
|---|---|---|
| `python/benchmark_e2e_final.py` | DEFINITIVO: PyTorch vs CUDA extension (170us vs 957us) | Activo |
| `python/benchmark_cuda_e2e.py` | Orchestrator completo con kernel CUDA | Activo |
| `python/benchmark_expert_types.py` | FASE 5: FP32/FP16/Ternary/INT8 comparativa | Activo |
| `python/benchmark_comparativo.py` | Modelo de coste: FLOPs + memoria + energia | Activo |
| `python/benchmark.py` | OptiX vs cuBLAS (requiere batch_runner.exe) | Requiere C++ |
| `python/benchmark_e2e.py` | Router + Expert types (pre-extension) | Antiguo |
| `python/benchmark_scaling.py` | Curva de escalado N=8->512 | Requiere C++ |
| `python/scaling_inception.py` | v4.0 benchmark OptiX vs cuBLAS vs FlashAttention | Analitico |

### Python — UTILIDADES Y EXPERIMENTAL

| Archivo | Funcion | Estado |
|---|---|---|
| `python/bvh_router_cuda.py` | ctypes binding a libbvh_router.so (957us) | Supersedido por ext (170us) |
| `python/bvh_router_hybrid.py` | PyTorch + CUDA kernel simple | Supersedido por bridge |
| `python/ternary_expert_cuda.py` | ctypes binding dims fijas (64->1024->4096) | ROTO (dims incompatibles) |
| `python/quantize_to_ternary.py` | Cuantizacion post-training a {-1,0,+1} | Activo |
| `python/train_dispersion.py` | W_dispersion para routing polisemico | Experimental |
| `python/dupl_score_optimizer.py` | DuplScore: decidir duplicar vs wormhole | Experimental |
| `python/fuzzy_bsh.py` | Fuzzy BSH: memberships diferenciables (numpy) | Experimental |
| `python/fuzzy_bsh_autograd.py` | BVH traversal como autograd.Function | Experimental |
| `python/simulator.py` | Simulacion O(N log N) vs O(N^2) (numpy puro) | Validacion teorica |
| `python/inference.py` | Pipeline alto nivel: embed -> 3D -> binary -> exe | Prototipo |
| `python/optix_router_bridge.py` | Export esferas -> OptiX + run inception_runner | Prototipo |
| `python/training_pipeline.py` | Sparse Upcycling + Semantic Batching (avanzado) | Prototipo |

### CUDA v5 — PRODUCCION

| Archivo | Funcion | Estado |
|---|---|---|
| `cuda/v5/bvh_router_kernel.cu` | Router kernel principal (constant mem, warp-level) | ACTIVO - 170us, 105x |
| `cuda/v5/bvh_torch_ext.cu` | PyTorch extension zero-copy | ACTIVO - auto-seleccionado |
| `cuda/v5/ternary_torch_ext.cu` | Expert ternario POPCOUNT (dims flexibles) | ACTIVO - auto-seleccionado |
| `cuda/v5/bvh_router_deep.cu` | BVH 65K+ experts jerarquico | Compilado, sin training |
| `cuda/v5/ternary_expert.cu` | Expert ternario dims fijas (64->1024->4096) | ROTO (dims incompatibles) |
| `cuda/v5/build_ext.py` | JIT compiler bvh_router_ext | Activo |
| `cuda/v5/build_ternary_ext.py` | JIT compiler ternary_expert_ext | Activo |
| `cuda/v5/liquid_expert.cu` | Expert experimental | No usado |
| `cuda/v5/optix_bvh_router.cu` | Router via OptiX (alternativa) | No usado |
| `cuda/v5/optix_bvh_router.h` | Header del anterior | No usado |
| `cuda/v5/async_pipeline.cu` | Pipeline 3-core asincrono (RT+CUDA+Tensor) | Futuro |
| `cuda/v5/torch_bvh_extension.cpp` | Binding legacy (supersedido) | Obsoleto |

### CUDA v4 — INVESTIGACION (compilacion condicional, OFF por defecto)

| Archivo | Funcion | Estado |
|---|---|---|
| `cuda/v4/spectral_kernels.cu` | OptiX raygen/closesthit/miss 4-level IAS | Activo (v4 research) |
| `cuda/v4/inception_kernels.cu` | Salto dimensional explicito | Activo (v4 research) |
| `cuda/v4/spectral_resonance.cu` | Fourier resonance (device code) | Activo (v4 research) |
| `cuda/v4/inception_resonance.cu` | Gradientes + AdamW (training v4) | Activo (v4 research) |
| `cuda/v4/optix_host.cpp` | Pipeline OptiX completo (host) | Activo (v4 research) |
| `cuda/v4/alpha_phase_a.cu` | Phase A: ray traversal | Reemplazado por v5 |
| `cuda/v4/alpha_phase_b.cu` | Phase B: cuBLAS selection | Reemplazado por v5 |
| `cuda/v4/ray_attention.cu` | Prototipo v1 original | Historico |
| `cuda/v4/ray_generation.cu` | Prototipo v1 | Historico |
| `cuda/v4/closest_hit.cu` | Prototipo v1 | Historico |
| `cuda/v4/miss.cu` | Prototipo v1 | Historico |
| `cuda/v4/ternary_resonance.cu` | Fourier ternario (experimental) | No integrado |

### C++ Headers y Sources

| Archivo | Funcion | Estado |
|---|---|---|
| `include/token_geometry.h` | Struct TokenNode, SemanticRay, operadores float3 | Activo |
| `include/semantic_bvh.h` | BVHNode, BVHBuildConfig, clase SemanticBVH | Activo |
| `include/optical_attention.h` | AttentionConfig, OpticalAttention | Activo |
| `include/inception_engine.h` | InceptionScene, InceptionEngine (v4 4-level IAS) | Activo (v4) |
| `include/alpha_bsh.h` | SemanticSphereAlpha, AlphaBSH Phase A/B | Activo |
| `include/spectral_resonance.h` | ResonanceParams, SemanticString, SemanticSphere | Activo (v4) |
| `include/spectral_ray.h` | Spectral encoding (Snell) | Incompleto |
| `src/token_geometry.cpp` | PCA projection helpers | Activo |
| `src/semantic_bvh.cpp` | BVH construction | Activo |
| `src/alpha_bsh.cpp` | AlphaBSH orchestration | Activo |

### Documentacion

| Archivo | Funcion | Estado |
|---|---|---|
| `README.md` | Descripcion proyecto para humanos | Actual |
| `CLAUDE.md` | Guia arquitectura para agentes IA | Actual |
| `LEARNINGS.md` | Registro de decisiones y fallos | Actual (~2800 lineas) |
| `ROADMAP.md` | Roadmap por fases | Necesita reescritura |
| `STATUS.md` | ESTE ARCHIVO | Nuevo |
| `docs/ARCHITECTURE.md` | Diseno del sistema | Actual |
| `docs/CUDA_BVH_ROUTER.md` | Doc tecnica kernel CUDA | Actual |
| `docs/MEMORY_BREAKTHROUGH.md` | Analisis VRAM y cache | Actual |
| `docs/BENCHMARK_TEORICO.md` | Benchmarks vs estado del arte | Actual |
| `python/README_TRAINING.md` | Guia de training | Actual |
| `patents/patent_01_rt_attention.md` | Patente 1: RT Core Attention | Lista |
| `patents/patent_02_inception_engine.md` | Patente 2: Inception Engine | Lista |
| `patents/patent_03_spectral_routing.md` | Patente 3: Spectral/Snell (reforzada) | Lista |

---

## DUPLICACIONES Y PROBLEMAS DETECTADOS

### Archivos con funcionalidad duplicada

| Grupo | Archivos | Cual usar | Que sobra |
|---|---|---|---|
| Router CUDA binding | `bvh_router_cuda.py`, `bvh_router_hybrid.py`, `bvh_router_bridge.py` | `bvh_router_bridge.py` (auto-selecciona el mejor backend) | Los otros 2 son fallbacks lentos |
| Expert ternario | `ternary_expert_cuda.py` (dims fijas ROTAS), `ternary_expert_ext_bridge.py` (flexible) | `ternary_expert_ext_bridge.py` | `ternary_expert_cuda.py` esta ROTO |
| Embedding bridge | `embedding_bridge.py` (incompleto), `spectral_bridge.py` (completo) | `spectral_bridge.py` | `embedding_bridge.py` incompleto |
| Embeddings download | `download_embeddings.py` (antiguo), `download_embeddings_v2.py` (gensim) | `download_embeddings_v2.py` | v1 obsoleto |
| v5 CUDA obsoletos | `ternary_expert.cu` (dims fijas), `torch_bvh_extension.cpp` (legacy) | Los .cu activos (ver tabla CUDA v5) | Dims fijas + legacy binding |

### Bugs conocidos pendientes

| Bug | Archivo | Severidad | Estado |
|---|---|---|---|
| CMake: alpha_phase_a/b.cu no se compilan pero alpha_bsh.cpp los referencia con extern | CMakeLists.txt + src/alpha_bsh.cpp | CRITICO para build | No afecta Python pipeline |
| CMake: projectEmbeddingTo3D 2 firmas incompatibles | token_geometry.cpp vs alpha_phase_a.cu | CRITICO para build | No afecta Python pipeline |
| CMake: SemanticBVH redefinida en .cpp vs .h | semantic_bvh.cpp vs semantic_bvh.h | CRITICO para build | No afecta Python pipeline |
| PTX hardcoded sm_120 (no funciona en RTX 4090 sm_89) | CMakeLists.txt lineas 308-309, 418-419 | MEDIO | Solo afecta v4 OptiX |
| BVHGateWrapper output dtype | olmoe_e2e_eval.py | CORREGIDO | Ya arreglado |
| Synthetic data != real distribution | olmoe_bvh_distill.py | CRITICO para PPL | Fix: extract_real_hiddens.py |

---

## ROADMAP POR FASES

### FASE 0: Limpieza y estructura (AHORA)
- [ ] Revisar archivos candidatos a archivo (NO borrar, mover a archive/)
- [ ] Consolidar duplicados
- [ ] Verificar que el pipeline Python funciona limpio
- [ ] Este STATUS.md como fuente de verdad

### FASE 1: Distillation con datos reales — ✅ COMPLETADA
- [x] Ejecutar `extract_real_hiddens.py` → 199,680 samples (856 MB)
- [x] Re-entrenar BVH Router con `--real-data` → 91.7% top-8, 71.1% top-1
- [x] Corregir bug `norm_topk_prob=False` → PPL 7.67→6.11
- [x] Corregir hybrid restricted softmax → PPL 6.11 exacto
- [x] Evaluar PPL con `olmoe_e2e_eval.py` → BVH puro PPL 134 (sin calibrar)
- [x] TARGET: PPL delta < 5% → **LOGRADO con calibracion: +0.8%**

### FASE 2: Calibracion de pesos — ✅ COMPLETADA
- [x] Affine calibracion (128 params) → PPL 6.27 (+2.5%), cosine 0.88
- [x] **Linear 64→64 calibracion (4160 params) → PPL 6.16 (+0.8%), cosine 0.97**
- [x] `calibrate_router.py` soporta ambos modos

### FASE 3: Multi-layer replacement — 🔄 EN PROGRESO (Spectral Techniques)
- [x] Extraer hidden states TODAS las 16 capas (data/real_hiddens_layer*.pt)
- [x] Entrenar BVH router todas las 16 capas
- [x] L1 reentrenada con --spectral: 79.3% → 81.9% top-8 (+2.6pp), beta=10.0
- [x] L11 reentrenada con --spectral: 81.8% → 93.3% top-8 (+11.5pp!)
- [x] PPL actual (16/16 sin Spectral completo): ~8.27
- [ ] Retrain TODAS las capas con --spectral (script: scripts/train_remaining_layers.sh)
- [ ] Calibrar todas las 16 capas post-Spectral
- [ ] Eval PPL 16/16 post-Spectral (objetivo: 8.27 → ~7.0)

**Per-layer accuracy (estado 2026-03-30):**

| Capa | Top-8 | Top-1 | Epochs | Spectral | Estado |
|------|-------|-------|--------|----------|--------|
| L0  | 89.5% | 89.3% | 198 | No  | Retrain con --spectral |
| L1  | 81.9% | 86.3% | 50  | YES | Retrain con --spectral |
| L2  | 84.7% | 82.8% | 100 | No  | WEAK — retrain con --spectral |
| L3  | **94.6%** | **82.2%** | 100 | **YES (dim=64)** | ✅ DONE — era 80.5% (+14.1pp!) |
| L4  | 86.6% | 80.6% | 197 | No  | Retrain con --spectral |
| L5  | **86.9%** | — | 51  | **YES (dim=64)** | ✅ DONE — era 81.9% sin Spectral |
| L6  | 84.3% | 80.7% | 47  | No  | WEAK — retrain con --spectral |
| L7  | 84.3% | 78.7% | 49  | No  | WEAK — retrain con --spectral |
| L8  | 90.1% | 77.8% | 191 | No  | Retrain con --spectral |
| L9  | 88.3% | 77.9% | 49  | No  | Retrain con --spectral |
| L10 | 89.3% | 80.8% | 50  | No  | Retrain con --spectral |
| L11 | **93.3%** | **79.5%** | 100 | **YES** | **DONE** (+11.5pp!) |
| L12 | 88.8% | 77.4% | 47  | No  | Retrain con --spectral |
| L13 | 92.4% | 77.9% | 200 | No  | Retrain con --spectral |
| L14 | 93.4% | 78.6% | 188 | No  | Retrain con --spectral |
| L15 | 89.3% | 80.2% | 50  | No  | Retrain con --spectral |

**Plan:** Retrain TODAS las 16 capas con --spectral (débiles primero, luego fuertes)

**Test A/B en curso: spectral_dim 64 vs 256 (misma capa L3)**

| Test | dim | save_dir | Estado |
|------|-----|----------|--------|
| A | 64  | `checkpoints/olmoe_distill_layer3/` | ✅ **94.6% top-8, 82.2% top-1** |
| B | 256 | `checkpoints/olmoe_distill_layer3_dim256/` | 🔄 Lanzado |

Coste dim=256: +61 MB total (16 capas), +0.2% latencia RT Core. Seguimos 3,750x más eficientes que Transformer.

### FASE 4: C++ / OptiX build — ✅ COMPILADO (2026-03-30)
- [x] Fix CMakeLists.txt: 3 fixes (projectEmbeddingTo3D, alpha_bsh, SPECTRAL_MAX_TOP_TOKENS)
- [x] Fix PTX: single -arch=compute_89 (--ptx no soporta multi-gencode)
- [x] Compilar PTX para compute_89 (6 shaders)
- [x] RT Core benchmark: 39.24 µs/batch, 6.52M queries/s
- [ ] Conectar optix_host.cpp con el BVH Router entrenado

### FASE 5: Demo end-to-end
- [ ] Pipeline completo: texto -> tokenize -> BVH routing -> expert inference -> texto
- [ ] Benchmark: latencia, throughput, VRAM
- [ ] Comparativa con OLMoE original

### FASE 6: Patentes
- [ ] Review final de las 3 provisionales
- [ ] Buscar abogado de patentes
- [ ] Filing ($1,050 total para las 3)

### FASE 7: Escalado (futuro)
- [ ] 65K experts (`bvh_router_deep.cu`)
- [ ] NVMe-backed expert cache
- [ ] Training E2E con Soft BVH diferenciable
- [ ] LLaMA 8B / 70B

---

## RESUMEN DE CONTEO

| Categoria | Total | Activos | Prototipo/Exp | Obsoletos |
|---|---|---|---|---|
| Python (.py) | ~50 | 36 (72%) | 10 (20%) | 4 (8%) |
| CUDA (.cu) | ~22 | 8 (36%) | 6 (27%) | 8 (36%) |
| C++ (.cpp/.h) | ~17 | 10 (59%) | 3 (18%) | 4 (24%) |
| Docs (.md) | ~25 | 14 (56%) | 0 | 11 (44% historico) |
| **Total** | **~114** | **68** | **19** | **27** |
