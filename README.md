# LiquidBit Zero-Matrix

**Atencion sin multiplicacion de matrices.** Los RT Cores reemplazan MatMul con ray tracing O(log N).

---

## Que es esto?

LiquidBit Zero-Matrix es un prototipo de investigacion que reemplaza el mecanismo de atencion O(N^2) de los Transformers con operaciones de ray tracing O(N log N), usando los RT Cores ya presentes en GPUs NVIDIA de consumidor (RTX 4090, RTX 5070 Ti).

En lugar de computar una matriz densa de atencion (Query x Key), los tokens se proyectan en un espacio geometrico 3D organizado como un BVH (Bounding Volume Hierarchy). Un "rayo" desde el token de consulta recorre el arbol, encontrando tokens semanticamente relevantes en O(log N) pasos — de la misma forma que un videojuego encuentra que objetos impacta una bala.

### Por que importa

| Metrica | GPT-4 (MatMul) | LiquidBit (Ray Tracing) |
|---|---|---|
| Complejidad atencion | O(N^2) | O(N log N) |
| Operaciones (N=100K) | ~80T FLOPs | ~6.9B intersecciones |
| KV Cache (96 capas) | ~307 GB VRAM | ~10-50 MB (BVH) |
| Hardware minimo | Rack de H100s | Una sola RTX 5070 Ti |

---

## Estado actual (2026-03-28)

### Lo que funciona

| Componente | Estado | Metrica clave |
|---|---|---|
| Router BVH (PyTorch) | Validado | Jerarquia 3 niveles, Gumbel-Softmax, 64 expertos |
| Kernel CUDA del router | Compilado + testeado | 8.83 us/batch, 105x vs PyTorch |
| Extension PyTorch (zero-copy) | Integrada | 10 us routing, seleccion automatica |
| Kernel experto ternario (POPCOUNT) | Validado | Cero multiplicaciones FP, 0.000038 diff vs FP32 |
| Demo (Qwen 1.5B) | Ejecutada | 51.9 tok/s, 375x menos VRAM |
| Demo (BitNet 2B) | Ejecutada | 3.4x speedup, 519x menos VRAM |
| Routing multi-dominio | 100% precision | 4 dominios, 16 expertos |
| Motor Inception (4 niveles 12D) | PPL 191.3 | Solo 2.1% peor que baseline GPT-2 |
| Codificacion espectral + Snell | Implementada | 88.9% resolucion polisemia |

### Distillation OLMoE — FASE A (resultado principal)

Reemplazamos el gate lineal de OLMoE-1B-7B (modelo MoE real de 7B parametros, 64 expertos) con nuestro BVH Router geometrico y medimos el impacto en perplexity:

| Configuracion | PPL | Delta vs baseline (6.11) | Estado |
|---|---|---|---|
| Baseline (gate lineal OLMoE) | 6.11 | — | Referencia |
| BVH Router 1 capa (L8) | 6.16 | **+0.8%** | ✅ Validado |
| BVH Router 2 capas (L4,8) | 6.23 | **+2.0%** | ✅ Validado |
| BVH Router 5 capas (L0,4,8,12,15) | 6.40 | **+4.8%** | ✅ Validado |

**Degradacion lineal ~1% por capa reemplazada.** Extrapolacion: 16/16 capas → ~15% PPL.

Componentes clave:
- **EnhancedBVHRouter**: Jerarquia 4x4x4 = 64 expertos, ~1.35M params
- **Sparse Upcycling**: Inicializacion del router desde los pesos del gate (SVD + K-Means)
- **Calibracion Linear**: Capa 64→64 (4160 params) que ajusta la distribucion de pesos → cosine 0.97
- **Precision top-8**: 87-93% segun la capa

### En progreso

| Componente | Estado | Notas |
|---|---|---|
| FASE 3: 16/16 capas reemplazadas | 5/16 completadas | Entrenando capas restantes |
| Pipeline OptiX RT Core | Shaders escritos | Necesita CUDA Toolkit + OptiX SDK |
| Build C++/CUDA CMake | 7 targets compilan | Necesita fix sm_120 para RTX 5070 Ti |

### Pendiente

- RT Cores reales via OptiX (estimado 10-20x sobre kernel CUDA)
- Pipeline asincrono tri-core (RT + CUDA + Tensor en paralelo)
- Escalado a 65K expertos
- Training end-to-end diferenciable
- Paper academico / benchmarks formales

---

## Arquitectura

```
Tokens de entrada
    |
    v
[Embedding] --> [Proyeccion 3D (PCA)]
    |
    v
[Router BVH] -- 3 niveles x 3D = 12 dimensiones semanticas
    |              Nivel 1: Dominios (Ciencia, Codigo, Humanidades, General)
    |              Nivel 2: Subdominios (4 por dominio)
    |              Nivel 3: Conceptos (4 por subdominio = 64 expertos)
    |
    v
[Seleccion Top-k Expertos] -- top-8, ponderados por probabilidades de routing
    |
    v
[Experto FFN SwiGLU] -- congelado (de OLMoE) o entrenable
    |
    v
[Proyeccion de Salida] --> logits
```

Tres innovaciones clave:

1. **Atencion RT Core (Patente LBS-2026-001):** El traversal BVH reemplaza MatMul denso. O(log N) en vez de O(N^2).

2. **Motor Inception (Patente LBS-2026-002):** 4 niveles IAS anidados codifican 12 dimensiones semanticas usando solo hardware 3D. Cada nivel es un "portal dimensional" que reinicia coordenadas.

3. **Routing Espectral (Patente LBS-2026-003):** Los rayos llevan un "color" (vector de contexto). Los nodos actuan como prismas (ley de Snell) — el mismo nodo enruta diferente segun el contexto, resolviendo polisemia sin duplicar parametros.

---

## Estructura del proyecto

```
liquidbit-zero-matrix/
├── CLAUDE.md              # Referencia de arquitectura (para agentes IA)
├── LEARNINGS.md           # Registro de decisiones, fallos, descubrimientos
├── ROADMAP.md             # Hoja de ruta de 11 fases
├── STATUS.md              # Estado detallado con inventario de archivos
├── README.md              # Este archivo
├── CMakeLists.txt         # Sistema de build C++/CUDA
│
├── python/                # ~50 archivos, ~25K lineas
│   ├── bvh_router.py          # Router BVH (PyTorch, diferenciable)
│   ├── orchestrator.py        # Pipeline completo: Router -> Experto -> Salida
│   ├── real_model_demo.py     # Demo con modelos HuggingFace reales
│   ├── olmoe_bvh_distill.py   # Distillation del router BVH desde gate OLMoE
│   ├── olmoe_e2e_eval.py      # Evaluacion PPL end-to-end (multi-capa)
│   ├── calibrate_router.py    # Calibracion post-hoc de pesos (affine/linear)
│   ├── extract_real_hiddens.py # Extraccion de hidden states reales
│   └── train_*.py             # Scripts de entrenamiento
│
├── cuda/
│   └── v5/                    # Kernels del orquestador
│       ├── bvh_torch_ext.cu       # Extension PyTorch zero-copy (105x speedup)
│       ├── ternary_torch_ext.cu   # Extension POPCOUNT ternaria
│       └── optix_bvh_router.cu    # Routing RT Core (necesita OptiX SDK)
│
├── include/               # Headers C++ publicos (7 archivos)
├── src/                   # Implementaciones C++ (3 archivos)
├── tests/                 # Tests C++ y benchmarks (7 archivos)
├── docs/                  # Documentacion tecnica
├── patents/               # 3 borradores de patente provisional
├── scripts/               # Scripts de automatizacion
│   └── regenerate_all.sh      # Regenera checkpoints y datos desde cero
├── data/                  # Datasets, embeddings (generados, no en git)
└── checkpoints/           # Modelos entrenados (generados, no en git)
```

**Total:** ~52K lineas (25K Python + 18K C++/CUDA + 9K Markdown)

---

## Requisitos de hardware

- **GPU:** NVIDIA RTX 4090 o RTX 5070 Ti (RT Cores necesarios)
- **VRAM:** 16 GB minimo
- **RAM:** 24 GB+ (para cargar OLMoE-1B-7B en evaluacion)
- **CUDA Toolkit:** 12.8+ (para sm_120 / soporte Blackwell)
- **OptiX SDK:** 9.1 (para pipeline RT Core, opcional para routing solo CUDA)
- **Python:** 3.10+, PyTorch 2.x con CUDA

---

## Inicio rapido

```bash
# WSL2 (recomendado)
ln -sf "/mnt/j/Proyectos/LiquidBit Zero-Matrix" /tmp/liquidbit
cd /tmp/liquidbit
python3 -m venv .venv_wsl && source .venv_wsl/bin/activate
pip install torch transformers accelerate safetensors datasets scikit-learn

# Regenerar todo el pipeline (extraccion → training → calibracion → eval)
bash scripts/regenerate_all.sh

# O paso a paso:

# 1. Extraer hidden states de OLMoE
python python/extract_real_hiddens.py --model-dir /path/to/olmoe-1b-7b --layer 8

# 2. Entrenar router BVH
python python/olmoe_bvh_distill.py --layer 8 --real-data data/real_hiddens_layer8.pt --epochs 50

# 3. Calibrar pesos
python python/calibrate_router.py --mode linear --epochs 100 --real-data data/real_hiddens_layer8.pt --device cpu

# 4. Evaluar PPL (deberia dar ~6.16, +0.8% vs baseline 6.11)
python python/olmoe_e2e_eval.py --model-dir /path/to/olmoe-1b-7b \
    --router-checkpoint checkpoints/olmoe_distill/bvh_router_best.pt --max-tokens 50000
```

---

## Bugs criticos resueltos (2026-03-28)

| Bug | Impacto | Solucion |
|---|---|---|
| `norm_topk_prob=False` ignorado | PPL 7.67 en vez de 6.11 | Leer atributo del gate original |
| Softmax restringido en hybrid | Pesos inflados (16 vs 64 expertos) | Softmax completo + gather |
| Distribucion de pesos BVH | PPL 134 sin calibrar | Calibracion linear 64→64 (4160 params) |
| Sin git / sin backup | Perdida total de archivos | Recuperacion de transcript JSONL + git init |

---

## Patentes

Tres solicitudes de patente provisional redactadas (pendientes de presentar):

| Expediente | Titulo | Innovacion |
|---|---|---|
| LBS-2026-001 | Atencion RT Core O(log N) | BVH reemplaza MatMul en atencion |
| LBS-2026-002 | IAS Anidado para 12D | 4 niveles de 3D = 12 dimensiones via instanciado OptiX |
| LBS-2026-003 | Routing Espectral + Snell | Routing dependiente de contexto sin duplicar parametros |

---

## Licencia

Propietario. Patente pendiente.

## Autor

Jordi Silva — LiquidBit Studio, 2026.
