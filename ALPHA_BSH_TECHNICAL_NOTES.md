# Alpha BSH - Notas Técnicas de Implementación

**Fecha:** 2026-03-24
**Estado:** Prototipo completado (sin OptiX .ptx compilado)

---

## Decisiones de Diseño

### 1. Un Único Rayo en Fase A vs Miles de Rayos (Original)

**Decisión:** Emitir un SOLO rayo por query

**Justificación:**
- Original emitía K rayos (K >> N, típicamente 4096+) para acumular atención multi-head
- Alpha BSH cambia el objetivo: NO acumular distribuciones, SELECCIONAR contexto
- Un rayo es suficiente si el árbol BSH está bien construido (topología semántica)
- **Reducción de complejidad:** O(K·N log N) → O(N log N)

**Trade-off:**
- Pierden contexto multi-perspectiva (cada rayo representaba una "dimension of thought")
- Ganancia: 1000x+ aceleración en Fase A

**Mitigación:**
- La Fase B (MatMul en esfera) proporciona transformación densa que puede recuperar complejidad
- Posible extensión futura: emitir K rayos pero solo en la esfera seleccionada (Fase A+, antes de B)

---

### 2. Traversal Greedy vs Exhaustivo

**Implementación:** Greedy (elige mejor hijo en cada nodo)

**Pseudocódigo:**
```
current = root
while not is_leaf:
    best_child = argmax_child( similarity(ray, child) )
    current = best_child
return current
```

**Ventajas:**
- O(log N) garantizado (8 hijos por nodo = árbol octal)
- No requiere cálculos cuadráticos
- Simple, rápido

**Desventajas:**
- Puede perder contextos en "ramas hermanas" (no explora todos los hijos)
- Solución: Usar múltiples rayos con diversificación (future work)

**Validación:**
- Si el árbol está bien construido (esferas cercanas = similares), greedy es óptimo
- Si hay errores en construcción, se pierde contexto
- Mitigación: Validar árbol en `validateTreeStructure()`

---

### 3. Construcción del Árbol: Greedy Nearest-Neighbor

**Algoritmo actual:**
```
for each sphere i (except root):
    parent = argmin_sphere( distance(i, sphere) )
    assign_as_child(parent, i)
depth[i] = depth[parent] + 1
```

**Complejidad:** O(N² log N) — cuadrático en número de esferas

**Problemas:**
- Lento para N > 100K
- No garantiza árbol balanceado
- Posibles desbalances (algunos nodos con 8 hijos, otros con 0)

**Soluciones alternativas:**
1. **KD-tree 3D**: O(N log N) para construcción, mejor distribución
   - Dividir espacio en octantes recursivamente
   - Garantiza profundidad O(log N)

2. **Hierarchical K-means**: O(N log K) para K clusters
   - Más balanceado que greedy
   - Requiere iteración (convergencia)

3. **Octree espacial regular**: O(N log L) donde L es profundidad máxima
   - Determinista, sin iteración
   - Requiere normalizar embeddings a rango [0, 1]³

**Recomendación:** Implementar KD-tree 3D antes de escalar a datos reales

---

### 4. FP16 con Tensor Cores vs FP32

**Decisión:** FP16 en cuBLAS (Tensor Cores)

**Beneficios:**
- **Velocidad:** 2-10x más rápido que FP32 en Tensor Cores
- **Memoria:** 2x menos VRAM (importante para cargas lazy)
- **Hardware:** Dedicado (no compete con float cores)

**Riesgos:**
- **Precisión:** FP16 tiene ~3-4 dígitos decimales (vs 7 para FP32)
- **Overflow:** Rango dinámico limitado (~3.3e-4 a 6.5e4)
- **Underflow:** Valores muy pequeños se redondean a 0

**Mitigación implementada:**
- Flag `config.use_fp32_fallback` para detectar NaN/Inf y reintentar en FP32
- Normalización de inputs si es necesario (rescale por factor)
- Acumulación en FP32 si es posible, conversión al final

**Validación necesaria:**
- Comparar resultados FP16 vs FP32 en benchmarks
- Medir degradación de accuracy vs ganancia de velocidad

---

### 5. Carga Lazy vs Pre-carga

**Implementación:** Configurable (`config.lazy_load_matrices`)

**Lazy (default):**
- MatrixBlocks se cargan desde disco solo cuando esfera es seleccionada
- **Ventaja:** Memoria GPU limitada no es cuello de botella
- **Desventaja:** Latencia en primer acceso a esfera (I/O blocking)

**Pre-carga (opcional):**
- Todas las matrices se cargan al construir árbol
- **Ventaja:** Sin latencia posterior
- **Desventaja:** Requiere VRAM enormous (100K esferas × 4KB cada una ≈ 400 GB)

**Solución intermedia (futura):**
- Mantener caché LRU de K esferas más recientes
- Usar cudaMemcpyAsync() para overlapping de I/O con compute

---

### 6. MatrixBlock: Dos Capas vs Una Sola

**Diseño:** Dos capas densas (W1 + GELU + W2)

**Justificación:**
- Mimics FFN (Feed-Forward Network) de Transformer estándar
- Típicamente: W1 [dim × 4·dim], W2 [4·dim × dim]
- Capas adicionales permiten transformaciones complejas

**Alternative:** Una única capa
- Más rápido (una sola cuBLAS call)
- Menos expresivo (no hay no-linealidad intermediada)

**Decisión:** Mantener dos capas para permitir arquitecturas flexibles

---

### 7. Similarity Metric: Exponencial Decay vs Otros

**Fórmula implementada:**
```
similarity = exp(-lambda * distance)
```

**Alternativas consideradas:**

| Métrica | Fórmula | Ventajas | Desventajas |
|---|---|---|---|
| Exponencial | exp(-λ·d) | Smooth, derivable | Computacionalmente caro |
| Gaussiana | exp(-d²) | Más pronunciado | Requiere normalización |
| Sigmoide | 1/(1+d) | Simple | Menos pronunciado |
| Inversa | 1/d | Muy rápido | Singularidad en d=0 |
| Linear | max(0, 1-d) | Muy rápido | No diferenciable |

**Selección:** Exponencial decay se usa porque:
- Preserva comportamiento físico (Lambert-Beer en óptica)
- Analógico a attention decay en Transformers
- Suave, derivable (importante para futuro entrenamiento)

---

### 8. OptiX vs CUDA Kernels Puros

**Estado actual:** Kernels CUDA puros (fallback greedy)

**OptiX en futuro:**
- Programas OptiX compilados a .ptx (pseudocódigo incluido)
- Hardware Raytracing (RT Cores) ejecuta intersecciones más rápido
- Potencial 10-100x speedup en ray-sphere tests

**Pseudocódigo vs Implementación Real:**
- Los 4 programas OptiX están documentados como comentarios
- Requieren OptiX SDK 8.x + CMake configuration
- Cada programa (raygen, intersection, closesthit, miss) es un "shader"

**Plan para compilación:**
1. Reescribir pseudocódigo a CUDA C++ compatible OptiX
2. Configurar pipeline en JSON (ver optix_pipeline.json)
3. Compilar con optixir (OptiX IR compiler) → .ptx
4. Linkear en executable CUDA

---

## Desafíos Encontrados

### 1. Gestión de Memoria GPU

**Problema:** Múltiples allocations/deallocations pueden fragmentar memoria

**Solución:** Usar `cudaMallocAsync()` (CUDA 11.2+) para mejor reutilización
- O mantener pool de memoria pre-alocada
- Actualmente: simple cudaMalloc/cudaFree (aceptable para prototipo)

### 2. Precisión en Proyección D→3D

**Problema:** Reducción de 768D (BERT) → 3D pierde información

**Mitigación:**
- Usar PCA esférica (preserva ~95% varianza de métrica coseno)
- No es transformación lineal simple; requiere descomposición SVD
- Para prototipo: usar primeros 3 componentes normalizados (subóptimo)

**Trabajo futuro:** Implementar PCA esférica real

### 3. Traversal Greedy Can Get Stuck in Local Minima

**Problema:** Si árbol mal construido, rayo elige hijo "incorrecto" temprano

**Ejemplo:**
```
root
├─ sphere_A (muy mala, pero hermana de query)
│  └─ sphere_A1 (mejor que root, pero no la mejor global)
├─ sphere_B (mejor global, pero no visible desde root)
```
Rayo elige A por greedy → no ve B

**Mitigación:**
1. Construir árbol correctamente (KD-tree, no greedy nearest-neighbor)
2. Multi-rayo: emitir K rayos con diferentes semillas aleatorias
3. Backtracking: si energía muy baja, ir a hermano de padre

---

## Optimizaciones Implementadas

✅ **Normalización de vectors** para evitar NaN/Inf en operaciones
✅ **Clamping de similitud** a [0.0, 1.0]
✅ **Kernel GELU aproximado** en lugar de tablas lookup (más rápido)
✅ **cudaEvent_t** para profiling preciso (sub-ms)
✅ **Async I/O ready** (código preparado para cudaMemcpyAsync)

---

## Optimizaciones Futuras

⏳ **KD-tree 3D** - Reemplazar greedy nearest-neighbor
⏳ **Multi-rayo selectivo** - Emitir K rayos solo en esfera ganadora
⏳ **Caching LRU** - Mantener K matrices recientes en VRAM
⏳ **Overlapping I/O** - cudaMemcpyAsync concurrente con MatMul
⏳ **Diferenciabilidad** - Soft BSH para gradientes durante entrenamiento
⏳ **Batch processing** - Paralelizar múltiples queries simultáneamente

---

## Validación y Testing

**Tests recomendados:**

1. **Unitarios:**
   - `test_projection()` - Verificar proyección D→3D preserva distancias
   - `test_bsh_traversal()` - Verificar que greedy siempre encuentra un hoja
   - `test_matrix_block_loading()` - Verificar carga lazy desde disco

2. **Integración:**
   - `test_full_pipeline()` - Ejecutar Fase A + B completo
   - `test_multiple_spheres()` - Árbol con 100-1000 esferas

3. **Benchmarks:**
   - Medir tiempo Fase A (vs expectativa O(log N))
   - Medir tiempo Fase B (vs cuBLAS baseline)
   - Comparar vs Transformer attention estándar
   - Comparar vs LiquidBit original

4. **Robustez:**
   - Entrada negativa (queries vacías, matrices sin inicializar)
   - Límites (máximo depth, máximo children)
   - Recuperación de errores CUDA/cuBLAS

---

## Integración con LEARNINGS.md

Sugerir agregar a LEARNINGS.md:

```markdown
## Alpha BSH - 2026-03-24

### Decisiones Clave
- Un rayo en Fase A (vs miles en original): 1000x aceleración, trade-off contexto
- Greedy traversal: O(log N) pero puede fallar si árbol mal construido
- FP16 con Tensor Cores: 2-10x speedup, requiere validación numérica

### Conocidos Problemas
- Construcción O(N²) debe reemplazarse con KD-tree O(N log N)
- Proyección D→3D subóptima (usar PCA esférica real en futuro)
- Greedy traversal stuck en mínimos locales si árbol desbalanceado

### Próximos Pasos
1. Compilar OptiX .ptx (actualmente pseudocódigo)
2. Implementar KD-tree 3D para construcción
3. Benchmarks vs Transformer + LiquidBit original
4. Trabajo en diferenciabilidad para entrenamiento
```

---

**Fin de Notas Técnicas**

---

## Ficheros de Referencia

- `/include/alpha_bsh.h` - Definiciones públicas (670 líneas)
- `/cuda/alpha_phase_a.cu` - Kernel Fase A (514 líneas)
- `/cuda/alpha_phase_b.cu` - Kernel Fase B + cuBLAS (570 líneas)
- `/src/alpha_bsh.cpp` - Implementación host (614 líneas)
- `/ALPHA_BSH_SUMMARY.md` - Resumen ejecutivo
- `ALPHA_BSH_TECHNICAL_NOTES.md` - Este archivo
