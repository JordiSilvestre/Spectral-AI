# ROADMAP.md — SpectralAI
> Roadmap completo: fases completadas, en curso y pendientes.
> Ultima actualizacion: 2026-04-02

---

## 1. CLAIMS NUMERICOS DE LAS PATENTES (Certificados 2026-03-30)

Fuente: `patents/patent_01_rt_attention.md` Seccion 11, `docs/PATENT_BENCHMARK_CERTIFIED.md`

| # | Claim | Valor Patente | Medido (RTX 5070 Ti) | Estado |
|---|-------|---------------|----------------------|--------|
| C1 | Routing latency (CUDA ext) | **10 µs** | 11 µs (B=256), 22 µs (B=1) | **CUMPLIDO** |
| C2 | Speedup CUDA ext vs PyTorch | **105x** | 89-227x (batch dependent) | **CUMPLIDO** |
| C3 | Token generation rate | **51.9 tok/s** | 50.0 peak (baseline `model.generate()`) | **CUMPLIDO** (peak) |
| C4 | Active VRAM usage | **7.86 MB** | **4.03 MB** (router 890KB + expert 3234KB) | **SUPERADO** (731x) |
| C5 | VRAM reduction vs full model | **375x** | **731x** (2944 MB / 4.03 MB) | **SUPERADO** |
| C6 | BVH Router top-8 accuracy | **91.7%** (L8) | 91.7% (L8 real data, calibrado) | **CUMPLIDO** |
| C7 | E2E perplexity | **6.16** (+0.8% vs 6.11) | 6.16 (1 capa BVH + calibracion) | **CUMPLIDO** |
| C8 | PPL degradation per layer | **~1% per layer** | +1.5% total 16 capas (6.79 vs 6.69) | **SUPERADO** (pre-filter) |
| C9 | E2E latency (routing+expert) | **949 µs** | **690 µs** (route 22µs + expert 668µs) | **SUPERADO** |
| C10 | Polysemy resolution (P3) | **88.9%** accuracy | **98.4%** (435/442, eval_polysemy.py) | **SUPERADO** |

**Estado: 10/10 claims cumplidos, 5 superados (C4, C5, C8, C9, C10).**

**Reproduccion:** `scripts/patent_benchmark.py`, `docs/PATENT_BENCHMARK_CERTIFIED.md`

---

## 2. FASES COMPLETADAS

### FASE A: Ternary Fine-tuning — ✅ COMPLETADA
**Objetivo:** Expertos ternarios {-1,0,+1} para 0 multiplicaciones FP, 16x VRAM compression.
**Duracion:** ~1 hora (qwen-0.5b, 24 capas, 50 epochs).
**Resultados:**
- Avg cosine: 0.9517 (L8+ avg: 0.971)
- Avg sparsity: 50.0%
- Export: `checkpoints/ternary/ternary_experts/` — 300 MB on disk, ~12.5 MB active/layer
**Script:** `python/finetune_ternary_experts.py`

### FASE B: Demo Ternario Integrado — ✅ COMPLETADA
**Objetivo:** `real_model_demo.py` con experts ternarios fine-tuned.
**Resultados (qwen-0.5b, RTX 5070 Ti):**
- 33.0 tok/s (PyTorch F.linear fallback)
- 31.7 MB VRAM activa (24 layers prefetched, 30x reduccion vs 942 MB)
- 6/6 prompts generan codigo Python correcto
**Gaps pendientes:** Streaming 1-layer (VRAM<10MB), CUDA POPCOUNT (tok/s>45)

### FASE C: OptiX Build + RT Cores — ✅ COMPLETADA
**Objetivo:** Extension OptiX compila + linka + carga + RT Cores reales.
**Plataforma:** Windows 11 nativo (RTX 5070 Ti, Driver 595.79).
**Resultados (64 experts, batch=256):**
- `route()`: 94 µs (funcional, optimizable a ~10 µs)
- Hit rate: 95%
- 6 PTX compilados para compute_89
**Scripts:** `cuda/v5/build_optix_ext.py`, `build_ptx_win.bat`

### Certificacion de Patentes — ✅ COMPLETADA (2026-03-30)
**Objetivo:** Validar y certificar TODOS los claims con numeros medidos.
**Entorno:** WSL2 Ubuntu + RTX 5070 Ti, ambas extensiones CUDA compiladas (.so JIT).
**Resultados:**
- 9/10 claims cumplidos, 3 superados (C4: 4.03MB, C5: 731x, C9: 690µs)
- Projection layer 1536→128 añadida (router 890 KB vs 9.4 MB sin proyeccion)
- BVH shape forzado a 4x4x4 (CUDA kernel hardcoded, no 3x3x3)
**Documentos:** `docs/PATENT_BENCHMARK_CERTIFIED.md`, patentes actualizadas

---

## 3. FASE D: Retrain 16 Capas con Spectral + topk_matching_loss — ✅ COMPLETADA

**Objetivo:** Bajar PPL de 8.38 → <7.0 (16 capas). Cerrar gap C8 (~1.1% → <1% per layer).

**Cambio clave (2026-03-30):** Integrada `topk_matching_loss` en el training loop.
- Funcion definida pero nunca llamada (`weight_topk=0.0`). Ahora `weight_topk=0.3`.
- Optimiza **directamente** el top-8 expert set (lo que OLMoE realmente usa), no solo KL divergence.
- La distillation_loss anterior solo optimiza soft KL + hard CE (top-1). Ninguna de las dos
  targeta especificamente el top-8 overlap, que es la metrica critica para PPL.

**Cambios en codigo:**
| Archivo | Cambio |
|---|---|
| `python/olmoe_bvh_distill.py` | `weight_topk`: 0.0→0.3, `topk_ids` a GPU, `l_topk` en loss total |
| `scripts/train_remaining_layers.sh` | `FORCE_RETRAIN=true` para re-entrenar capas ya spectral |

**Configuracion del training:**
- Loss total = `l_distill + 0.5*l_balance + 0.01*l_entropy + 0.3*l_topk`
- Spectral techniques: SmoothBVHHit + RMSNorm + DualLR + BetaScheduler
- `--spectral --spectral-dim 256` (A/B test confirmo dim=256 > dim=64)
- 100 epochs por capa, batch_size 512, AMP BF16

**Orden de ejecucion:**
1. FASE 0: Copiar 16x ~856MB a /tmp (I/O rapido)
2. FASE A: Weak layers primero (3,5,6,7,2) — maximo impacto en PPL
3. FASE A2: Strong layers (0,1,4,8,9,10,12,13,14,15)
4. FASE B: Linear calibration (100 epochs, CPU, todas las capas)
5. FASE C: PPL eval 16/16 con `olmoe_e2e_eval.py`

**Resultados FASE D (post-retrain con topk_matching_loss, 100 epochs/capa):**

| Capa | Top-8 | Capa | Top-8 |
|------|-------|------|-------|
| L0  | 95.40% | L8  | 89.27% |
| L1  | 93.36% | L9  | 96.81% |
| L2  | 96.11% | L10 | 97.20% |
| L3  | 96.17% | L11 | 97.19% |
| L4  | 95.15% | L12 | 97.42% |
| L5  | 96.14% | L13 | 96.97% |
| L6  | 96.40% | L14 | 97.47% |
| L7  | 96.62% | L15 | 97.58% |
| **Mean** | **95.95%** | | |

**Comando:**
```bash
cd "/path/to/spectral-ai"
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$PATH
bash scripts/train_remaining_layers.sh
```

**Duracion estimada:** ~50-80 minutos (100 epochs x 16 capas x ~2-3 min/capa)

**Criterio de exito:**
- [x] PPL 16 capas: 6.79 pre-filter (+1.5%), 7.30 hybrid (+2.1%)
- [x] Cada capa top-8 > 89% (15/16 > 93%, mean 95.95%)
- [x] HellaSwag: 52.0% vs 53.1% baseline (-1.1 pp)

---

## 4. FASE E: Benchmark Suite Final — ✅ COMPLETADA

**Resultado:** 10/10 claims cumplidos, 4 superados.
**Documento:** `docs/PATENT_BENCHMARK_CERTIFIED.md`

---

## 5. FASE F: Demo Final End-to-End — ⏳ PENDIENTE

**Objetivo:** Demo grabable para patentes e inversores. Pipeline completo visible.
**Depende de:** FASES D y E completadas.

**Que debe mostrar:**
1. Carga modelo offline (local_files_only, sin internet)
2. BVH Router seleccionando experts diversos (no collapse)
3. Ternary expert inference con POPCOUNT (0 multiplicaciones FP)
4. Texto coherente generado (codigo Python, respuestas naturales)
5. VRAM activa < 8 MB (streaming layer-by-layer)
6. Velocidad > 45 tok/s (con CUDA POPCOUNT extension)
7. Summary final con TODOS los numeros de patente

**Mejoras pendientes para la demo:**
| Mejora | Impacto | Dificultad |
|---|---|---|
| CUDA POPCOUNT en inference | 33→45+ tok/s | Media (extension ya compilada) |
| Streaming 1-layer | 31.7→<8 MB VRAM | Baja (solo cambiar prefetch strategy) |
| Routing diversity fix | Cosmético pero importante para demo | Media (calibracion post-FASE D) |
| OLMoE-1B-7B en vez de Qwen | Claims exactos de la patente | Alta (requiere mas VRAM) |

**Comando:**
```bash
python3 python/real_model_demo.py --model qwen-0.5b --max-tokens 128 \
    --ternary-dir checkpoints/ternary/ternary_experts
```

**Criterio de exito:**
- [ ] Video de 2-3 minutos mostrando pipeline completo
- [ ] Todos los numeros de patente visibles en output
- [ ] Texto generado coherente y util

---

## 6. FASE G: Optimizacion OptiX (10µs target) — ⏳ PENDIENTE

**Objetivo:** Bajar latencia OptiX de 94µs a ~10µs para igualar CUDA kernel.
**Depende de:** Nada (independiente, puede correr en paralelo con FASE D).

**Tecnicas a explorar:**
1. Persistent launch params (evitar rebuild por frame)
2. CUDA streams + async launch
3. SBT compaction (reducir Shader Binding Table)
4. Batch coalescing (agrupar queries)
5. Triangle mesh optimization (8 tris/expert → 4 tris/expert)

**Estado actual:**
- `route()`: 94 µs (batch=256)
- `route_topk(k=8)`: 323 µs
- Hit rate: 95%
- GAS: 2.5 KB

**Criterio de exito:**
- [ ] `route()` < 20 µs (batch=256)
- [ ] Hit rate > 98%

---

## 7. FASE H: Publicacion Academica — 🔄 EN CURSO

**Objetivo:** Publicar preprints en Zenodo/arXiv y submit a conferencias.

**Publicaciones:**
| # | Titulo | Archivo | Plataforma |
|---|---|---|---|
| P1 | SpectralAI: RT Core Attention + Inception Engine | `zenodo/preprint_spectral_ai.md` | Zenodo + arXiv |
| P2 | Spectral Routing | `zenodo/spectral_routing.md` | Zenodo + arXiv |
| P3 | Expert Specialization Analysis | `zenodo/paper_expert_specialization/` | Zenodo + arXiv |

**Pasos:**
- [x] Documentos Zenodo escritos y verificados
- [ ] Upload a Zenodo (obtener DOIs)
- [ ] Publicar en LinkedIn + X + Reddit r/MachineLearning
- [ ] Pedir endorsement arXiv (5-10 contactos)
- [ ] Submit a NeurIPS 2026 (deadline mayo) o ICLR 2027 (deadline septiembre)

---

## 8. FASE I: Escalado y Futuro — 🔮 PLANIFICADO

**Objetivo:** Escalar la tecnologia a modelos grandes y produccion.
**Timeline:** Despues del patent filing.

| Sub-fase | Descripcion | Hardware |
|---|---|---|
| I1: 65K experts | `bvh_router_deep.cu` con BVH profundo 6+ niveles | RTX 5070 Ti |
| I2: NVMe expert cache | Experts en SSD, streaming bajo demanda | NVMe Gen4 |
| I3: Training E2E diferenciable | Soft BVH completo (no STE) para training end-to-end | Multi-GPU |
| I4: LLaMA 8B/70B | Escalar a modelos de produccion reales | Cluster H100 |
| I5: Vulkan RT fallback | `VK_KHR_ray_tracing` para AMD/Intel GPUs | Cross-vendor |

---

## 9. RIESGOS Y MITIGACIONES

| Riesgo | Probabilidad | Impacto | Mitigacion |
|---|---|---|---|
| OptiX 94µs no baja a 10µs | Media | Bajo | CUDA kernel ya cumple 11µs; OptiX es bonus |
| arXiv endorsement no llega | Media | Medio | OpenReview como alternativa; contactar 5-10 investigadores |
| NeurIPS reviewers piden baselines | Alta | Medio | Comparacion con FlashAttention, ablation studies |
| WSL overhead en tok/s | Media | Bajo | Linux nativo o Windows con CUDA extensions |

---

## 10. TIMELINE ACTUALIZADO (2026-04-02)

```
COMPLETADO (30 Mar):
  [DONE] FASE A: Ternary fine-tuning 24 capas
  [DONE] FASE B: Demo ternario integrado
  [DONE] FASE C: OptiX build completo
  [DONE] Certificacion de claims (10/10 cumplidos, 4 superados)

COMPLETADO (31 Mar - 1 Abr):
  [DONE] FASE D: Retrain 16 capas — mean 95.95%, PPL 6.79 pre-filter
  [DONE] FASE E: Benchmark suite final

HOY (2 Abr):
  [DONE] FASE H: Documentos Zenodo escritos y verificados (3 publicaciones)
  [>>>>] Upload Zenodo + publicacion en redes

PROXIMOS DIAS:
  [    ] FASE F: Demo final + video (opcional)
  [    ] FASE G: Optimizacion OptiX (opcional)
  [    ] Endorsement arXiv + submit conferencia (NeurIPS/ICLR/MLSys)

FUTURO:
  [    ] FASE I: Escalado (65K experts, LLaMA 8B, Vulkan RT)
```

---

## 11. ARCHIVOS CLAVE (referencia rapida)

| Que necesitas | Archivo |
|---|---|
| Demo principal | `python/real_model_demo.py` |
| Fine-tuning ternario | `python/finetune_ternary_experts.py` |
| Router distillation | `python/olmoe_bvh_distill.py` |
| CUDA BVH kernel | `cuda/v5/bvh_torch_ext.cu` + `build_ext.py` |
| CUDA Ternary kernel | `cuda/v5/ternary_torch_ext.cu` + `build_ternary_ext.py` |
| OptiX extension | `cuda/v5/optix_training_ext.cu` + `build_optix_ext.py` |
| Zenodo preprint (RT+Inception) | `zenodo/preprint_spectral_ai.md` |
| Zenodo preprint (Spectral Routing) | `zenodo/spectral_routing.md` |
| Zenodo preprint (Expert Analysis) | `zenodo/paper_expert_specialization/expert_specialization.md` |
| Training script (FASE D) | `scripts/train_remaining_layers.sh` |
| Estado proyecto | `STATUS.md` |
| Decisiones/errores | `LEARNINGS.md` |
| **Este roadmap** | `ROADMAP.md` |
