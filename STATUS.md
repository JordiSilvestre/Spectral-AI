# STATUS.md — SpectralAI Zero-Matrix
> Estado real del proyecto, inventario de archivos, y roadmap por fases.
> Ultima actualizacion: 2026-04-01

---

## Estado General

| Aspecto | Estado |
|---|---|
| Concepto matematico | Validado (O(N log N) vs O(N^2)) |
| CUDA kernels v5 | Operativos (105x speedup routing, POPCOUNT ternary) |
| Demo killer (Qwen 0.5B) | ✅ 33 tok/s, 6/6 prompts, ternary experts |
| BVH Router distillation | ✅ 16/16 capas, 80-97.2% top-8 (L11 best: 97.2%) |
| E2E PPL (1 capa) | ✅ PPL 6.16 (+0.8%) — BVH Router L8 con calibracion linear |
| E2E PPL (5 capas) | ✅ PPL 6.40 (+4.8%) — Capas 0,4,8,12,15 reemplazadas |
| E2E PPL (16 capas hybrid) | ✅ PPL 7.15 (0.0%) — BVH selecciona, gate original pesa |
| E2E PPL (16 capas pure) | ✅ PPL 8.42 (+17.8%) — MicroPredictor 16 params (sin gate original) |
| E2E PPL (3 capas pure) | ✅ PPL 7.42 (+3.9%) — L3,L8,L15 BVH puro (sin gate) |
| E2E PPL (3 capas mixto) | ✅ PPL 7.17 (+0.4%) — BVH selecciona + gate pesa |
| E2E PPL (16 capas mixto) | ✅ PPL 7.30 (+2.1%) — 16/16 capas hybrid_residual |
| E2E PPL (1 capa pure) | ✅ PPL 7.19 (+0.6%) — L8 MicroPredictor |
| Bugs criticos resueltos | norm_topk_prob=False, restricted softmax, calibracion |
| Pipeline OptiX v5 | ✅ Compilado: 6 PTX, benchmark 39µs/batch, 100% accuracy (triángulos) |
| RT Training Bridge | ✅ StraightThroughRT (STE): RT hard forward + soft backward |
| OptiX Training Ext | ✅ pybind11 extension + JIT build + test script + integration wrapper |
| Inception v4.0 opt | ✅ PPL 185.4 — gap 1.75% vs GPT-2 (objetivo <=2.1% CUMPLIDO) |
| Patent claims certificados | ✅ 9/10 cumplidos, 3 superados (C4, C5, C9) |
| Patentes | 3 provisionales redactadas, Patent 3 reforzada con Claims 21-33 |
| FASE D: Retrain con topk_loss | ✅ COMPLETADA — 16/16 capas, L11=97.2% top-8 |
| FASE E: Pure mode PPL | ✅ CERRADA — Best: 3 capas PPL 7.42 (+3.9%), 16 capas PPL 8.42 (+17.8%) |
| Multi-ray ensemble | ✅ Implementado, NO mejora (7.43 vs 7.42 sin multi-ray) |
| FASE G: Demo generacion texto | ✅ COMPLETADA — 3 capas 15 tok/s, 16 capas 4.7 tok/s, texto coherente |
| FASE G+: Benchmarks CUDA | ✅ COMPLETADA — BVH 85-170x speedup (confirmado), Ternary 7.9x compresion |
| Ternary decision | ✅ DESCARTADO para modelos actuales — usar FP16 (Tensor Cores 10x mas rapido) |
| hybrid_residual mode | ✅ PPL 7.17 (+0.4%) — BVH selecciona, gate pesa. Brecha cerrada. |
| benchmark_scaling.py | ✅ CREADO — O(log N) vs O(N) curva para N=64..4096 expertos |
| retrofit_bvh.py | ✅ CREADO — Universal: MoE (replace gate) + Dense (sparsity dinamica) |
| FASE F: BVH accuracy | 🔄 PARCIAL — Solo L1 (93.4%) necesita retrain. L4,L8 rozando 96%. L9-L15 ya 96.8-97.6% |
| Cross-disciplinary weights | ✅ 11 modos probados. render_eq PPL 7.33 (+2.5%) NUEVO RECORD puro |
| Expert Analysis (deep) | ✅ 30 categorias + token-level + co-activacion 16 capas. Expertos = sintacticos, no semanticos |
| Expert Permutation (L3) | ✅ 86.2% top-8, PPL 7.55 (+5.6%) — permutacion no mejora PPL, mejora traversal RT |
| Expert Permutation (16 capas) | 🔄 EN PROGRESO — Entrenando 16 capas con --expert-perm + real data |
| Bugs corregidos (distill) | ✅ distillation_loss NaN, benchmark inv_perm, olmoe_layer=None |
| E2E PPL (3 capas render_eq) | ✅ PPL 7.33 (+2.5%) — logit × geometry, puro sin gate |
| E2E PPL (6 capas render_eq) | ✅ PPL 7.51 (+5.0%) — 6 capas FASE F (96%+), ~0.03 PPL/capa |
| E2E PPL (16 capas render_eq)| PPL 9.17 (+28%) — degradado por L1 (93.4%) y capas FASE D |
| FASE H: Patentes | ⏳ Pendiente — Filing USPTO, tests completos para claims |
| FASE I: Paper | ⏳ Pendiente — Resultados publicables (PPL 7.33 puro, 7.17 hybrid) |

---

## RESUMEN EJECUTIVO (2026-04-01)

### Que tenemos HOY

SpectralAI reemplaza el gate lineal de modelos MoE con un router BVH geometrico O(log N).
Probado en OLMoE-1B-7B (64 expertos, 16 capas).

**Resultados clave:**
```
ROUTING SPEED (CUDA kernel, RTX 5070 Ti):
  BVH Router:   10.4 us/batch (batch=256) → 24.7M tok/s
  Gate lineal: ~927 us/batch (PyTorch)    → 94x mas lento
  Speedup:      85-170x segun batch size

PPL (Perplexity — menor = mejor):
  Baseline OLMoE (gate original):     7.15
  Modo PURO 3 capas (render_eq):      7.33  (+2.5%)  ← SIN gate original
  Modo PURO 6 capas (render_eq):      7.51  (+5.0%)
  Modo MIXTO 3 capas (hybrid):        7.17  (+0.4%)  ← USA gate original
  Modo MIXTO 16 capas (hybrid):       7.30  (+2.1%)
  Modo PURO 16 capas (render_eq):     9.17  (+28%)   ← degradado por L1

ACCURACY por capa (top-8 overlap con gate original):
  Promedio FASE F (6 capas, 200ep):   96.1%
  Promedio FASE D (10 capas, 100ep):  96.5%
  Peor capa: L1 = 93.4%  ← cuello de botella
  Mejor capa: L15 = 97.6%
```

### Descubrimiento del dia: Cross-disciplinary weight modes

Se probaron 11 modos de asignar pesos inspirados en otros campos:
- **render_eq** (Ec. renderizado 3D): `weight = sqrt(logit) * 1/sqrt(dist_BVH)` → **7.33**
- **gravity** (Economia/ALiBi): `logit - alpha*log(dist)` → **7.33**
- **ray_march** (Volumetric rendering): `logit * exp(-dist)` → **7.33**
- Tres metodos convergen en 7.33 → posible suelo para 3 capas a 96% accuracy

**Insight principal:** La geometria del BVH contiene informacion util para pesos.
Combinar logits + distancia geometrica baja PPL de 7.42 a 7.33 en modo puro.

### Mapa de capas (accuracy top-8)

```
L0  [##########----] 95.4%  FASE F     L8  [##########----] 95.9%  FASE D
L1  [########------] 93.4%  DEBIL !!   L9  [###########---] 96.8%  FASE D OK
L2  [##########----] 96.1%  FASE F     L10 [###########---] 97.2%  FASE D OK
L3  [##########----] 96.2%  FASE F     L11 [###########---] 97.2%  FASE D OK
L4  [##########----] 95.1%  FASE D     L12 [###########---] 97.4%  FASE D OK
L5  [##########----] 96.1%  FASE F     L13 [###########---] 97.0%  FASE D OK
L6  [##########----] 96.4%  FASE F     L14 [###########---] 97.5%  FASE D OK
L7  [##########----] 96.6%  FASE F     L15 [############--] 97.6%  FASE D OK
```

### Descubrimiento: Expert Analysis Deep (16 capas)

Analisis exhaustivo de los 64 expertos de OLMoE revela:

```
1. Expertos NO especializan por tema (math, code, etc.)
   - Mejor especialista tematico: solo 6.8% (vs 3.3% uniforme)
   - Especializan por TIPO DE TOKEN: content_word, function_word, punctuation

2. Co-activacion forma 4 clusters de 16 (perfecto para BVH 4x4x4)
   - C0: content puro (63%)
   - C1: content + function words (62% + 21%)
   - C2: mixto/transicion (42% content + 19% punct)
   - C3: estructura (30% function + 27% punct)

3. Clusters CAMBIAN entre capas (estabilidad 4.7-6.2%)
   - Cada capa reorganiza los 64 expertos de forma diferente
   - Un BVH unico no servira para todas las capas

4. Selectividad tiene forma de U
   - L0-L3: alta (0.52-0.61) -- capas tempranas discriminan fuerte
   - L5-L7: baja (0.38-0.42) -- capas medias procesan uniforme
   - L12-L15: alta (0.51-0.59) -- capas tardias discriminan fuerte

5. Evolucion por capa: L0=55 content experts, L9=19 function experts
```

**Implicacion:** El BVH deberia organizarse per-layer por co-activacion, no por tema.

### Que falta — PROXIMO PASO

**1. Reentrenar L1 (PRIORIDAD ALTA)**
- L1 es 93.4% — la unica capa < 95%. Cuello de botella para 16 capas.
- Comando: `python3 olmoe_bvh_distill.py --layer 1 --epochs 200 --spectral --topk-weight 0.3`
- Estimado: ~35 min. Objetivo: subir a 96%+
- Opcional: L4 (95.1%) y L8 (95.9%)

**2. BVH semantico per-layer (basado en expert analysis)**
- Usar clusters de co-activacion de cada capa para organizar el arbol BVH
- Cada capa tiene su propio arbol optimizado
- Integrar en retrofit_bvh.py

**3. Re-evaluar 16 capas con render_eq (tras retrain L1 + BVH semantico)**
- Prediccion: si L1 sube a 96%, PPL 16 capas deberia bajar de 9.17 a ~8.0-8.5
- Con BVH semantico: potencialmente mejor

**4. FASE H: Patentes**
- 3 provisionales ya redactadas
- Falta: tests completos para todos los claims, filing USPTO ($1,050)
- Los 11 weight modes son patentables (cross-disciplinary routing)
- NUEVO: "Expert specialization is syntactic, not semantic" -- publicable

**5. FASE I: Paper**
- Resultados ya publicables:
  - Modo puro: PPL +2.5% (3 capas render_eq)
  - Modo mixto: PPL +0.4% (3 capas hybrid)
  - Routing: 85-170x speedup
  - Scaling: O(log N) demostrado
  - Cross-disciplinary: 11 modos, 3 convergen
  - NUEVO: Deep expert analysis en 16 capas
- Falta: completar 16 capas para tabla completa

### Decisiones tomadas

- **Ternary POPCOUNT**: DESCARTADO para datacenter (0.1x vs FP16). Future work para edge.
- **FP16 expertos**: Estandar. La ventaja es el ROUTING BVH, no la cuantizacion.
- **Weight mode recomendado**: render_eq (puro) o hybrid_residual (mixto)
- **FASE F 200 epochs**: NO necesaria para L9-L15 (ya estaban bien con 100ep FASE D)
- **Expert analysis**: Organizacion BVH por co-activacion per-layer, no por tema

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
| `python/train_inception.py` | v4.0 Inception con L_task + L_spatial (optimizado: warmup, spatial every step) | ✅ gap 1.75% — objetivo cumplido |
| `python/finetune_ternary_experts.py` | QAT ternario: STE + KD + learnable scale (recrear 14h training) | NUEVO |
| `python/train_spectral_lm.py` | Baseline training SpectralAIForCausalLM | Prototipo |

### Python — DISTILLATION OLMoE (pipeline actual de desarrollo)

| Archivo | Funcion | Estado |
|---|---|---|
| `python/olmoe_extract.py` | Carga 64 experts SwiGLU + gate de OLMoE safetensors | Validado |
| `python/olmoe_bvh_distill.py` | EnhancedBVHRouter v2.1 + KD loss + RealHiddensDataset | Activo (necesita datos reales) |
| `python/extract_real_hiddens.py` | Extrae hidden states reales de OLMoE en WikiText-2 | NUEVO - pendiente ejecutar |
| `python/olmoe_e2e_eval.py` | Evaluacion PPL + demo generacion texto (--generate) | Activo |
| `python/benchmark_cuda_pipeline.py` | Benchmarks CUDA: BVH kernel + Ternary POPCOUNT + pipeline | Activo |
| `python/rt_training_bridge.py` | StraightThroughRT: RT Core forward + SmoothBVHHit backward | NUEVO |
| `python/optix_training_bridge.py` | OptiXTrainingBridge: STE con pybind11 ext (zero-copy GPU) | NUEVO |
| `python/optix_router_integration.py` | Drop-in OptiX wrapper para EnhancedBVHRouter | NUEVO |
| `python/test_optix_training.py` | Test: Gumbel-Softmax vs SmoothBVHHit vs OptiX+STE | NUEVO |
| `cuda/v5/optix_training_ext.cu` | pybind11 extension wrapping RTCoreRouter | NUEVO |
| `cuda/v5/build_optix_ext.py` | JIT compilation script para optix_training_ext | NUEVO |
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
| `python/benchmark_scaling.py` | BVH O(log N) vs Linear Gate O(N) scaling curve | Activo |
| `python/retrofit_bvh.py` | Universal BVH retrofit: any HF model (MoE + Dense) | NUEVO |
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

### FASE 3: Multi-layer replacement — 🔄 EN PROGRESO (Spectral + topk_matching_loss)
- [x] Extraer hidden states TODAS las 16 capas (data/real_hiddens_layer*.pt)
- [x] Entrenar BVH router todas las 16 capas (legacy, sin topk_loss)
- [x] L1 reentrenada con --spectral: 79.3% → 81.9% top-8 (+2.6pp), beta=10.0
- [x] L11 reentrenada con --spectral: 81.8% → 93.3% top-8 (+11.5pp!)
- [x] PPL actual (16/16 sin topk_loss): ~8.38
- [x] Patent claims certificados: 9/10, 3 superados (2026-03-30)
- [x] `topk_matching_loss` integrada en training loop (weight=0.3)
- [x] `FORCE_RETRAIN=true` para re-entrenar capas ya spectral
- [🔄] FASE D: Retrain TODAS 16 capas con spectral + topk_loss (EN CURSO)
- [ ] Calibrar todas las 16 capas post-FASE D
- [ ] Eval PPL 16/16 post-FASE D (objetivo: 8.38 → <7.0)

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

**Test A/B completado: spectral_dim 64 vs 256 (misma capa L3)**

| Test | dim | save_dir | Resultado |
|------|-----|----------|-----------|
| A | 64  | `checkpoints/olmoe_distill_layer3/` | 94.6% top-8, 82.2% top-1 |
| B | 256 | `checkpoints/olmoe_distill_layer3_dim256/` | **95.1% top-8, 82.7% top-1** ← WINNER |

**RESULTADO: dim=256 gana. Retrain masivo usa `--spectral-dim 256`.**

---

## 🚀 PASOS A EJECUTAR (en orden, actualizado 2026-03-30)

### ✅ PASO 1: Inception v4.0 optimizado — COMPLETADO
**Resultado:** PPL 185.4 — gap 1.75% vs GPT-2 (182.2). Objetivo <=2.1% CUMPLIDO.

### ✅ PASO 2: Demo real_model_demo — COHERENTE. Optimizaciones aplicadas.
**Fix 1:** RoPE position_embeddings (HF transformers 5.x API break) → rotary_emb extraído, pos_emb=(cos,sin)
**Fix 2:** Ternary MLP 58% sparsity → usar FP16 MLP original para generacion coherente
**Fix 3 (NUEVO):** KV Cache dos fases (quitar el bucle):
  - Fase 1: Prompt forward una vez → rellena DynamicCache (28 layers × S tokens)
  - Fase 2: Cada token nuevo → solo 1 posicion × 28 layers (KV reutilizado)
  - Speedup esperado: 1.4 tok/s → 15-30 tok/s
**Fix 4 (NUEVO):** SpectralKV Pruner (el laser que recorta el KV cache):
  - Proyecta hidden states a 3D espectral (dims [0, H/2, H-1])
  - Para cada token nuevo: selecciona K=64 tokens del prompt mas cercanos
  - Mask -inf al resto → atencion O(K) en vez de O(S)
  - En prompts 256 tokens: 4x reduccion atencion. En 2048: 32x.
**Resultado actual (antes de KV cache):** 1.4 tok/s, 30x VRAM, coherente
**Ejecutar ahora:**
```bash
cd /mnt/j/Proyectos/SPECTRAL\ AI
source .venv_wsl/bin/activate
python3 python/real_model_demo.py --model qwen-1.5b --max-tokens 64
```
**Esperado:** ~20-40 tok/s (KV cache), texto coherente, routing diverso

### PASO 3: Compilar y testear OptiX Training Extension
**Fixes aplicados:**
  1. Symlinks sin espacios para OptiX SDK, include/, cuda/:
     `/tmp/optix_sdk_inc`, `/tmp/spectral_include`, `/tmp/spectral_cuda`
  2. **Fix linker (NUEVO):** Symlink del paquete torch COMPLETO a `/tmp/_torch_pkg_optix`
     + redirige `torch.__file__` y `torch.__path__[0]` a traves del symlink.
     Esto corrige la segunda inyeccion `-L` que PyTorch hace internamente en `_prepare_ldflags`.
     Mismo fix aplicado a `build_ext.py` (`/tmp/_torch_pkg_bvh`).
  3. `optix_router_host.cpp` copiado a `parent_of_build_dir` para que la ruta
     relativa `#include "../optix_router_host.cpp"` funcione.
```bash
# Paso 3a: Compilar extension pybind11
python3 cuda/v5/build_optix_ext.py

# Paso 3b: Test de validacion (Gumbel vs SmoothBVH vs OptiX+STE)
python3 python/test_optix_training.py --device cuda --steps 200

# Paso 3c: Test integracion con routing wrapper
python3 python/optix_router_integration.py --mode test --device cuda
```
**Esperado:** Extension compila + linka OK, SmoothBVHHit gradients fluyen, OptiX+STE accuracy ~= Gumbel
**Nota:** Si OptiX PTX no encontrado, los tests igual pasan con fallback soft

### PASO 4: Fine-tuning ternario QAT (recrear los 14h de entrenamiento perdidos)
**Script NUEVO:** `python/finetune_ternary_experts.py`
Quantization-Aware Training con Straight-Through Estimator:
  - Forward: w_q = sign(w_latent) * (|w_latent| > threshold) → {-1, 0, +1}
  - Backward: STE con atenuacion gaussiana cerca del umbral
  - Loss: KD (MSE normalizado) + cosine similarity + sparsity regularization
  - Per-row learnable scale (LearnableScale con softplus)
  - Online hidden state extraction (no requiere datos pre-extraidos)
```bash
# Test rapido: 1 capa, 20 epochs (~30 min)
python3 python/finetune_ternary_experts.py --model qwen-0.5b --layer 8 --epochs 20

# Pipeline completo: todas las capas, 50 epochs (~14h RTX 4090)
python3 python/finetune_ternary_experts.py --model qwen-0.5b --epochs 50

# Exporta: checkpoints/ternary/ternary_experts/{layer_N}/*.npy
```
**Target patente:** 375x VRAM (7.86 MB active), cosine >0.97, sparsity ~50%

### PASO 5: Retrain TODAS las 16 capas con Spectral + topk_matching_loss — 🔄 EN CURSO
```bash
bash scripts/train_remaining_layers.sh   # FORCE_RETRAIN=true, weight_topk=0.3
```
**Cambio clave:** `topk_matching_loss` integrada (weight=0.3). Optimiza directamente el top-8 set.
**Esperado:** Cada capa sube 5-15pp. PPL 16/16: 8.38 → ~6.5-7.0
**Duracion:** ~50-80 minutos (100 epochs x 16 capas)

### PASO 6: Demo final end-to-end
- Pipeline completo: texto -> tokenize -> BVH routing -> ternary expert inference -> texto
- Benchmark: latencia (target 51.9 tok/s), throughput, VRAM (target 7.86 MB active)
- Comparativa con OLMoE original (PPL target: 6.16, +0.8%)

### PASO 7: Patentes
- Review final de las 3 provisionales contra codigo real
- Buscar abogado de patentes
- Filing ($1,050 total para las 3)

---

### FASE 4: C++ / OptiX build — ✅ COMPILADO (2026-03-30)
- [x] Fix CMakeLists.txt: 3 fixes (projectEmbeddingTo3D, alpha_bsh, SPECTRAL_MAX_TOP_TOKENS)
- [x] Fix PTX: single -arch=compute_89 (--ptx no soporta multi-gencode)
- [x] Compilar PTX para compute_89 (6 shaders)
- [x] RT Core benchmark: 39.24 µs/batch, 6.52M queries/s
- [x] Conectar optix_host.cpp con el BVH Router entrenado (optix_training_bridge.py + optix_training_ext.cu)

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
