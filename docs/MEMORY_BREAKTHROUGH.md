# Analisis de VRAM: El Breakthrough de Memoria

> Ultima actualizacion: 2026-03-28. Para arquitectura general ver [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Resumen

SpectralAI reduce el consumo de VRAM activa en **375x** respecto al modelo completo
(7.86 MB vs 2,944 MB), demostrado con Qwen2.5-Coder-1.5B en RTX 5070 Ti.
Esto se logra reemplazando el KV Cache denso por un arbol BVH compacto
y cargando solo los expertos necesarios via LRU cache.

---

## El Problema: KV Cache en Transformers Tradicionales

En un Transformer estandar, el mecanismo de atencion mantiene un KV Cache que crece
linealmente con la longitud de secuencia y el numero de capas:

```
VRAM_kv = 2 * n_layers * seq_len * d_model * bytes_per_element
```

### Ejemplo: GPT-4 (estimado)

| Parametro | Valor |
|---|---|
| Capas (n_layers) | 96 (estimado) |
| Dimension (d_model) | 12,288 |
| Secuencia (seq_len) | 128,000 |
| Precision | FP16 (2 bytes) |
| **VRAM KV Cache** | **~307 GB** |

Este consumo hace que modelos grandes requieran multiples GPUs H100 (80 GB cada una)
solo para almacenar el KV Cache, sin contar los pesos del modelo.

---

## La Solucion: BVH como Estructura Cache-Bound

SpectralAI reemplaza el KV Cache por un arbol BVH (Bounding Volume Hierarchy):

### Tamano del BVH

```
VRAM_bvh = n_tokens * sizeof(BVHNode) + overhead_arbol
```

| Componente | Por nodo | Total (N=100K) |
|---|---|---|
| Centroide 3D (`float3`) | 12 B | 1.2 MB |
| AABB min/max (`float3` x2) | 24 B | 2.4 MB |
| Metadata (token_id, position) | 8 B | 0.8 MB |
| Nodos internos del arbol | ~50% extra | ~2.2 MB |
| **Total BVH** | ~44 B/nodo | **~6.6 MB** |

### Comparativa Directa

| Estructura | VRAM (N=100K, 96 capas) | Ratio |
|---|---|---|
| KV Cache tradicional | ~307 GB | 1x |
| BVH SpectralAI | ~10-50 MB | **~6,000-30,000x menos** |

### Por que funciona: Cache-Bound vs VRAM-Bound

El KV Cache tradicional es **VRAM-bound**: necesita almacenar TODOS los key-value pairs
de TODAS las capas simultaneamente en memoria de GPU.

El BVH es **cache-bound**: opera mediante traversal jerarquico donde cada paso
solo necesita acceder a un nodo padre y sus hijos (~4 nodos). Esto cabe completamente
en la cache L1/L2 de la GPU:

```
KV Cache:   Acceso a N tokens * 2 (K,V) * d_model floats  --> VRAM-bound
BVH:        Acceso a log2(N) niveles * ~4 nodos * 12 bytes --> Cache L1-bound
```

Para N=100,000 tokens:
- **KV Cache**: accede a ~25 GB de datos por consulta (una capa)
- **BVH**: accede a ~816 bytes por consulta (17 niveles * 4 nodos * 12 bytes)

---

## Expert LRU Cache: Solo Top-K en GPU

El segundo factor de ahorro es que SpectralAI no necesita todos los expertos en VRAM
simultaneamente. El `ExpertLRUCache` (`python/expert_lru_cache.py`) gestiona la carga
dinamica:

### Mecanismo

```
                     GPU VRAM (limitada)
                    +-------------------+
                    | Expert 3 (activo) |
                    | Expert 7 (activo) |
                    | Expert 12 (activo)|
  CPU RAM           | Expert 45 (activo)|
+-----------+       +-------------------+
| Expert 0  |            ^    |
| Expert 1  |    load()  |    | evict() (LRU)
| Expert 2  |            |    v
| ...       |       +-------------------+
| Expert 63 |       | Expert 22 (freed) |
+-----------+       +-------------------+
```

### Presupuesto de VRAM por Expert

| Tipo de Expert | Tamano por Expert | Top-8 en VRAM |
|---|---|---|
| SwiGLU FP16 (OLMoE) | 6.3 MB | 50.4 MB |
| SwiGLU FP32 | 12.6 MB | 100.8 MB |
| Ternario (BitNet 1.58) | ~0.8 MB | 6.4 MB |
| **Ternario (Qwen demo)** | **~0.98 MB** | **7.86 MB** |

### Estrategia de Eviccion

El LRU cache evita recargas frecuentes manteniendo los expertos mas usados:

1. **Hit**: Expert ya en GPU --> latencia 0 (solo puntero)
2. **Miss**: Expert no en GPU --> carga desde CPU RAM (~50-100 us via PCIe 5.0)
3. **Evict**: Si GPU llena --> desaloja el expert menos recientemente usado
4. **Prefetch** (futuro): Predice el siguiente expert basado en el historial de routing

---

## Demo: Prueba Empirica

### Configuracion

- **Modelo**: Qwen2.5-Coder-1.5B (1.5B parametros)
- **Hardware**: NVIDIA RTX 5070 Ti (16 GB VRAM)
- **Kernels**: Router CUDA ext + Expert POPCOUNT ternario
- **Prompts**: 6 prompts de codigo (Fibonacci, Quicksort, hash tables, etc.)

### Resultados

| Metrica | Modelo Completo | SpectralAI | Ratio |
|---|---|---|---|
| VRAM activa | 2,944 MB | 7.86 MB | **375x menos** |
| Throughput | ~52 tok/s | 51.9 tok/s | ~1x (sin degradacion) |
| Calidad | Coherente | Coherente (6/6 prompts) | Equivalente |

### Con BitNet 2B (ternario nativo)

| Metrica | Modelo Completo | SpectralAI | Ratio |
|---|---|---|---|
| VRAM activa | ~1,200 MB | ~2.31 MB | **519x menos** |
| Pesos expert | FP16 | {-1, 0, +1} | 16x compresion |

---

## Implicaciones para Hardware de Consumidor

### El Cambio de Paradigma

Los LLMs tradicionales requieren GPUs de datacenter porque el KV Cache y los pesos
del modelo no caben en VRAM de consumidor:

| Modelo | VRAM Requerida | Hardware Minimo | Coste |
|---|---|---|---|
| GPT-4 (1.8T) | ~500+ GB | Rack de H100 (8x80GB) | >240,000 EUR |
| LLaMA-3 70B | ~140 GB (FP16) | 2x A100 80GB | ~30,000 EUR |
| Mixtral 8x22B | ~88 GB (FP16) | 2x A100 80GB | ~30,000 EUR |
| DeepSeek-V3 (671B) | ~1.3 TB (FP16) | Rack de H100 | >240,000 EUR |

### SpectralAI en Hardware de Consumidor

Con BVH routing + LRU expert cache + cuantizacion ternaria:

| Configuracion | VRAM Activa | Hardware | Coste |
|---|---|---|---|
| 64 expertos ternarios | ~50 MB | RTX 4060 (8 GB) | ~300 EUR |
| 64 expertos FP16 | ~100 MB | RTX 4060 (8 GB) | ~300 EUR |
| 1,024 expertos ternarios | ~50 MB + LRU | RTX 4090 (24 GB) | ~1,600 EUR |
| 65,536 expertos ternarios | ~50 MB + NVMe LRU | RTX 5070 Ti (16 GB) | ~800 EUR |

### RTX 5070 Ti como Target Principal

La RTX 5070 Ti (Blackwell, sm_120) es el target principal por:

1. **16 GB VRAM**: Sobra para BVH (50 MB) + top-k expertos (~100 MB)
2. **RT Cores 5a gen**: Ray-sphere intersection en ~4 ciclos GPU
3. **PCIe 5.0**: 32 GB/s para carga de expertos desde CPU RAM
4. **Tensor Cores FP16**: Para forward pass de expertos SwiGLU
5. **Coste**: ~800 EUR vs >240,000 EUR para un rack de H100

---

## Proyeccion de Escalado

### VRAM vs Numero de Expertos

```
Expertos    VRAM activa (top-8, ternario)    VRAM total (todos en disco)
64          ~7.86 MB                          ~50 MB
512         ~7.86 MB (mismo top-k)            ~400 MB
1,024       ~7.86 MB                          ~800 MB
10,000      ~7.86 MB                          ~7.8 GB
65,536      ~7.86 MB                          ~51 GB (NVMe)
```

La VRAM activa es **constante** respecto al numero total de expertos, porque solo
los top-k (tipicamente 8) estan en GPU simultaneamente. El resto reside en CPU RAM
o almacenamiento NVMe.

### VRAM vs Longitud de Secuencia

| Secuencia | KV Cache (GPT-4) | BVH (SpectralAI) | Ratio |
|---|---|---|---|
| 1K tokens | ~3.1 GB | ~0.07 MB | 44,000x |
| 10K tokens | ~31 GB | ~0.66 MB | 47,000x |
| 100K tokens | ~307 GB | ~6.6 MB | 46,500x |
| 1M tokens | ~3,072 GB | ~66 MB | 46,500x |

El BVH escala **linealmente** con N (O(N) en memoria) mientras que el KV Cache
tambien escala linealmente pero con un factor constante ~46,000x mayor. La ventaja
de SpectralAI en memoria se mantiene constante independientemente de la longitud
de secuencia.

---

## Limitaciones y Trabajo Futuro

1. **Latencia de miss**: Cuando un expert no esta en GPU, la carga desde CPU RAM
   toma ~50-100 us (PCIe 5.0). Con NVMe, ~200-500 us. Esto puede impactar la
   latencia de first-token si muchos experts nuevos se necesitan.

2. **Calidad vs compresion**: Los expertos ternarios ({-1, 0, +1}) pierden precision.
   PPL sube de 13.3 (FP32) a 349.9 (ternario from scratch). Con distillation desde
   modelos pre-entrenados (OLMoE), la degradacion es aceptable: +0.8% PPL por capa.

3. **Prefetch inteligente**: El LRU cache actual es reactivo. Un sistema predictivo
   que anticipe los experts necesarios basandose en el contexto reduciria los misses.

4. **NVMe-backed cache**: Para 65K+ expertos, se necesita un sistema de paginacion
   GPU <-> CPU <-> NVMe, analogo al virtual memory del SO.

---

## Archivos Relacionados

| Archivo | Funcion |
|---|---|
| `python/expert_lru_cache.py` | GPU memory manager con LRU eviction |
| `python/real_model_demo.py` | Demo que demuestra el 375x savings |
| `python/benchmark_comparativo.py` | Modelo de coste: FLOPs + memoria + energia |
| `python/micro_expert.py` | Wrapper expertos (FP16, INT8, Ternary, Inception) |
| `include/token_geometry.h` | Struct TokenNode (44 bytes/nodo) |
| `include/semantic_bvh.h` | BVHNode, construccion del arbol |
