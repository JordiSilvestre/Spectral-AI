# CUDA BVH Router — Documentacion Tecnica

> Ultima actualizacion: 2026-03-28. Para arquitectura general ver [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Resumen

El BVH Router CUDA es un kernel fusionado de 3 niveles que reemplaza el routing PyTorch
del `BVHRouter` con una implementacion nativa en CUDA. Utiliza memoria constante para los
centros del BVH, operaciones a nivel de warp para reduccion, y zero-copy con PyTorch
para eliminar overhead de transferencia CPU-GPU.

**Resultado clave:** 8.83 us/batch (batch=256), 105x mas rapido que PyTorch, 28.9M tok/s.

---

## Arquitectura del Kernel

### Jerarquia de 3 Niveles

El router mapea hidden states a expert IDs mediante traversal geometrico de un arbol BVH:

```
Nivel 1 (dims 1-3):  4 dominios       --> Science, Code, Humanities, General
Nivel 2 (dims 4-6):  4 subdominios    --> por dominio (16 total)
Nivel 3 (dims 7-9):  4 conceptos      --> por subdominio (64 total = n_experts)
```

Cada nivel calcula distancias euclidianas entre el hidden state proyectado y 4 centros,
seleccionando el mas cercano. El indice final es:

```
expert_id = idx_nivel1 * 16 + idx_nivel2 * 4 + idx_nivel3
```

### Flujo del Kernel

```
1. Carga:     hidden_state[tid] desde global memory
2. Proyecto:  D-dim --> 3D via pesos W_proj en constant memory
3. Nivel 1:   distancia a 4 centros L1 --> argmin --> domain_id
4. Nivel 2:   distancia a 4 centros L2[domain_id] --> argmin --> subdomain_id
5. Nivel 3:   distancia a 4 centros L3[domain_id][subdomain_id] --> argmin --> concept_id
6. Salida:    expert_ids[tid] = combined index, top-k weights via softmax local
```

### Operaciones a Nivel de Warp

El kernel utiliza `__shfl_down_sync()` para reduccion intra-warp al calcular distancias,
evitando shared memory para los 4 centros por nivel. Con solo 4 candidatos por nivel,
la comparacion cabe en un unico warp sin necesidad de sincronizacion de bloque.

---

## Layout de Memoria

### `__constant__` Memory (64 KB, cache L1 broadcast)

| Dato | Tamano | Descripcion |
|---|---|---|
| `bvh_centers_L1[4][3]` | 48 B | 4 centros x 3 floats (nivel 1) |
| `bvh_centers_L2[4][4][3]` | 192 B | 4x4 centros x 3 floats (nivel 2) |
| `bvh_centers_L3[4][4][4][3]` | 768 B | 4x4x4 centros x 3 floats (nivel 3) |
| `W_proj[D][3]` | D*12 B | Pesos de proyeccion (D=128 tipico) |
| **Total** | ~2.5 KB | << 64 KB limite constant memory |

La memoria constante es ideal para este caso: todos los threads del warp leen los mismos
centros simultaneamente (broadcast), resultando en un unico acceso a cache L1.

### Global Memory

| Dato | Acceso | Tamano (batch=256, D=128) |
|---|---|---|
| `hidden_states[B][D]` | Lectura coalescente | 128 KB |
| `expert_ids[B]` | Escritura coalescente | 1 KB |
| `expert_weights[B][top_k]` | Escritura coalescente | 4 KB |

El patron de acceso es completamente coalescente: thread `i` lee `hidden_states[i]` y
escribe `expert_ids[i]`, maximizando throughput de memoria global.

---

## Rendimiento

### Benchmark Aislado (Kernel Micro)

| Metrica | Valor |
|---|---|
| Latencia por batch (B=256) | 8.83 us |
| Throughput | 28.9M tok/s |
| Occupancy | >90% (compute-bound, no memory-bound) |
| Determinismo | Bit-exact entre ejecuciones |
| Tests pasados | 5/5 |

### Benchmark End-to-End (Orchestrator completo)

| Sistema | Latencia routing (B=256) | Speedup vs PyTorch |
|---|---|---|
| PyTorch BVHRouter | 1,003 us | 1x |
| CUDA Extension (zero-copy) | 10 us | **105x** |
| CUDA kernel micro (aislado) | 8.83 us | **113x** |

### Benchmark Orchestrator (routing + backbone, B=1)

| Sistema | Latencia E2E | Speedup |
|---|---|---|
| Orchestrator PyTorch puro | 1,793 us | 1x |
| Orchestrator + CUDA Extension | 949 us | **1.89x** |

El speedup E2E es menor (1.89x vs 105x) porque el cuello de botella se mueve al backbone
(expert forward pass), que domina la latencia total.

---

## Tres Backends de Routing

El sistema ofrece tres backends con fallback automatico:

| Prioridad | Backend | Latencia (B=256) | Interfaz | Requisito |
|---|---|---|---|---|
| 1 (mejor) | `bvh_router_ext` | ~170 us (E2E con bridge) | PyTorch Extension (pybind11) | CUDA + `build_ext.py` |
| 2 | `CUDABVHRouter` | ~957 us (E2E con bridge) | ctypes a `libbvh_router.so` | CUDA + Makefile |
| 3 (fallback) | `BVHRouter` PyTorch | ~1,313 us | Python puro | Solo PyTorch |

### Porque la diferencia 170 us vs 8.83 us?

La latencia del kernel aislado es 8.83 us. Los 170 us del backend 1 incluyen:
- Sincronizacion PyTorch (`torch.cuda.synchronize()`)
- Overhead de llamada pybind11
- Conversion de tensores (si necesaria)
- Copia de resultados al grafo de autograd

El kernel CUDA puro sigue ejecutandose en ~9 us dentro de esos 170 us.

---

## Compilacion e Instalacion

### Prerequisitos

- CUDA Toolkit 12.x+
- PyTorch con soporte CUDA
- Compilador C++ compatible (GCC 9+, MSVC 19+)

### Build de la Extension PyTorch (Recomendado)

```bash
# Desde WSL2 o Linux
cd /path/to/spectral-ai
python cuda/v5/build_ext.py
```

El script usa `torch.utils.cpp_extension.load()` para JIT-compilar la extension.
El resultado se cachea en `~/.cache/torch_extensions/bvh_router_ext/`.

### Build del .so via ctypes (Alternativa)

```bash
cd cuda/v5
make bvh_router  # Genera libbvh_router.so
```

### Verificacion

```python
from python.bvh_router_bridge import HybridBVHRouter

router = HybridBVHRouter(hidden_dim=128, n_experts=64)
router.eval()          # IMPORTANTE: modo eval activa CUDA
router.sync_to_torch_ext()  # Sincroniza pesos a la extension

# Verificar backend activo
print(router.active_backend)  # "bvh_router_ext" si compilation exitosa
```

---

## API: HybridBVHRouter

El punto de entrada principal es `HybridBVHRouter` en `python/bvh_router_bridge.py`.

### Constructor

```python
HybridBVHRouter(
    hidden_dim: int = 128,    # Dimension del hidden state de entrada
    n_experts: int = 64,      # Numero total de expertos (4x4x4)
    top_k: int = 8,           # Expertos seleccionados por token
    levels: int = 3,          # Niveles del arbol BVH
    branches: int = 4,        # Ramas por nivel
)
```

### Metodos Clave

| Metodo | Descripcion |
|---|---|
| `forward(hidden_states)` | Retorna `(expert_ids, expert_weights)` |
| `sync_to_torch_ext()` | Copia pesos PyTorch al kernel CUDA |
| `eval()` | Activa modo inferencia (requisito para CUDA) |
| `train()` | Activa modo training (usa Gumbel-Softmax en PyTorch) |

### Ejemplo de Uso

```python
import torch
from python.bvh_router_bridge import HybridBVHRouter

# Inicializar
router = HybridBVHRouter(hidden_dim=2048, n_experts=64, top_k=8)
router = router.cuda().eval()
router.sync_to_torch_ext()

# Routing
hidden = torch.randn(256, 2048, device="cuda")  # batch=256
expert_ids, weights = router(hidden)
# expert_ids: [256, 8] -- top-8 expert indices per token
# weights:    [256, 8] -- softmax weights per expert
```

### Notas Importantes

1. **`.eval()` es obligatorio** para que el backend CUDA se active. En modo `.train()`,
   usa Gumbel-Softmax diferenciable en PyTorch.
2. **`sync_to_torch_ext()`** debe llamarse despues de cargar pesos o actualizar el router.
3. **`norm_topk_prob=False`** es critico para compatibilidad con OLMoE: los pesos son
   raw softmax, NO normalizados sobre los top-k.

---

## Archivos Relacionados

| Archivo | Funcion |
|---|---|
| `cuda/v5/bvh_router_kernel.cu` | Kernel CUDA principal (constant mem, warp ops) |
| `cuda/v5/bvh_torch_ext.cu` | Extension PyTorch zero-copy (pybind11) |
| `cuda/v5/build_ext.py` | Script JIT para compilar la extension |
| `cuda/v5/bvh_router_deep.cu` | Variante escalable 3-8 niveles (65K expertos) |
| `python/bvh_router.py` | Implementacion PyTorch (training + fallback) |
| `python/bvh_router_bridge.py` | HybridBVHRouter: auto-seleccion de backend |
| `python/bvh_router_cuda.py` | Binding ctypes legacy (supersedido por ext) |
| `python/benchmark_e2e_final.py` | Benchmark definitivo de los 3 backends |

---

## Escalado Futuro

| Configuracion | Expertos | Kernel | Estado |
|---|---|---|---|
| 4x4x4 (actual) | 64 | `bvh_router_kernel.cu` | Validado |
| 8x8x8 | 512 | `bvh_router_deep.cu` | Compilado |
| 8 niveles | 65,536 | `bvh_router_deep.cu` | Compilado, sin training |

Con 65K expertos, el BVH mantiene O(log N) = ~16 pasos de traversal.
El cuello de botella se mueve a la carga de expertos desde CPU/NVMe a GPU VRAM.
