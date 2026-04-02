# SpectralAI — Build Instructions

## Requisitos

- **CMake** ≥ 3.28
- **CUDA Toolkit** ≥ 12.0
- **OptiX** ≥ 8.0 (recomendado, opcional para prototipo)
- **GCC/Clang** (C++17) o **MSVC** (Visual Studio 2019+)
- **Python 3.8+** (opcional, para embedding_bridge.py)

## Hardware Target

- **NVIDIA RTX 4090** (Ada Lovelace, SM_89)
- **NVIDIA RTX 5070 Ti** (Blackwell, SM_100)

## Instalación rápida

### 1. Preparar entorno

```bash
# Linux / macOS
export CUDA_PATH=/usr/local/cuda  # Ajustar según tu instalación
export PATH=$CUDA_PATH/bin:$PATH

# Windows (PowerShell)
$env:CUDA_PATH="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"
$env:PATH="$env:CUDA_PATH\bin;$env:PATH"
```

### 2. Clonar y configurar

```bash
cd SpectralAI\ Zero-Matrix
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

Si OptiX está instalado:
```bash
cmake .. -DOptiX_INSTALL_DIR=/path/to/optix -DCMAKE_BUILD_TYPE=Release
```

### 3. Compilar

```bash
cmake --build . --config Release -j$(nproc)
```

O en Windows:
```bash
cmake --build . --config Release
```

### 4. Ejecutar tests

```bash
# Benchmark ray tracing vs MatMul
./tests/spectral_benchmark
# o en Windows:
.\tests\Release\spectral_benchmark.exe
```

## Python Scripts

### Usar embedding_bridge.py

```bash
cd python

# Crear tokens desde vocabulario de muestra (código)
python3 embedding_bridge.py --sample-vocab 50 --output tokens.bin --visualize

# Cargar embeddings GloVe
python3 embedding_bridge.py --load-glove embeddings.txt --output tokens.bin --max-vocab 10000

# Generar archivo binario sin visualización
python3 embedding_bridge.py --sample-vocab 100 --output token_nodes.bin
```

## Estructura del build

Después de `cmake --build .`:

```
build/
├── CMakeFiles/
├── CMakeLists.txt
├── Makefile (Linux/macOS) o Visual Studio files (Windows)
├── libspectral_core.a      # Librería estática C++
├── tests/
│   └── spectral_benchmark  # Ejecutable de benchmark
└── bin/
    └── embedding_bridge.py  # Script Python (instalado)
```

## Opciones de CMake

```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPECTRAL_BUILD_TESTS=ON \
    -DSPECTRAL_BUILD_PYTHON=OFF \
    -DSPECTRAL_ENABLE_DEBUG=OFF \
    -DSPECTRAL_ENABLE_LINEINFO=ON \
    -DOptiX_INSTALL_DIR=/path/to/optix
```

| Opción | Default | Descripción |
|--------|---------|-------------|
| `SPECTRAL_BUILD_TESTS` | ON | Compilar benchmarks y tests |
| `SPECTRAL_BUILD_PYTHON` | OFF | Compilar bindings Python (requiere pybind11) |
| `SPECTRAL_ENABLE_DEBUG` | ON | Incluir símbolos de debug (-g) |
| `SPECTRAL_ENABLE_LINEINFO` | ON | Incluir info de líneas en kernels CUDA |

## Troubleshooting

### "CUDA not found"

```bash
# Verificar instalación CUDA
nvcc --version
which nvcc

# En Linux, asegurarse de que está en PATH
export PATH=/usr/local/cuda/bin:$PATH
```

### "OptiX include directory not found"

```bash
# Descargar OptiX desde https://developer.nvidia.com/optix/downloads
# O compilar sin OptiX (prototipo funcional sin ray tracing real)
cmake .. -DOptiX_INSTALL_DIR=""
```

### "Compute capability mismatch"

Si tu GPU no es Ada (SM_89) o Blackwell (SM_100):

```bash
# Editar CMakeLists.txt, línea "set(CUDA_ARCHITECTURES ...)"
# Ejemplo para RTX 3090 (Ampere, SM_86):
# set(CUDA_ARCHITECTURES 86)
```

### "NVCC version mismatch"

```bash
# Verificar versiones compatibles
nvcc --version
cmake --version

# CMake debe encontrar la misma versión de CUDA que 'nvcc'
```

## Validación post-build

```bash
# 1. Ejecutar benchmark
./tests/spectral_benchmark

# Salida esperada:
# ╔══════════════════════════════════════════════════╗
# ║ SpectralAI: Ray Tracing vs MatMul    ║
# ║ Benchmark Suite                                 ║
# ╚══════════════════════════════════════════════════╝
#
# N      │ MatMul (ms) │ Optical (ms) │ Speedup
# 1000   │ ...         │ ...          │ ...
# 5000   │ ...         │ ...          │ ...
# ...

# 2. Crear tokens de muestra
cd python
python3 embedding_bridge.py --sample-vocab 50 --output ../tokens.bin

# Verificar archivo binario creado
ls -lh ../tokens.bin
```

## Notas de desarrollo

### Estructura de includes

```cpp
// En tu código:
#include "token_geometry.h"   // Está en ../include/
#include "semantic_bvh.h"     // (a crear)
```

CMakeLists.txt configura automáticamente los paths de include.

### Linking manual

Si necesitas compilar código que depende de spectral_core:

```bash
g++ -std=c++17 \
    -I../include \
    -L./build \
    my_code.cpp \
    -lspectral_core \
    -lcudart \
    -o my_app
```

### CUDA kernel compilation flags

Los flags CUDA se configuran en CMakeLists.txt:
- `-O3`: Optimización
- `-arch=sm_89`: Para Ada (RTX 4090)
- `--lineinfo`: Info de debug

Para cambiar, editar:
```cmake
# CMakeLists.txt, alrededor de línea 120
list(APPEND CMAKE_CUDA_FLAGS "-O3")
```

## Próximos pasos

1. **Completar kernels CUDA** (`ray_attention.cu`, `ray_generation.cu`)
2. **Implementar OptiX pipeline** (ray tracing real)
3. **Agregar tests unitarios** (GoogleTest)
4. **Python bindings** (pybind11)
5. **Entrenamiento** (custom autograd para BVH)

## Referencias

- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- OptiX: https://developer.nvidia.com/optix
- CMake: https://cmake.org/
- LLVM/Clang: https://clang.llvm.org/

---

**Última actualización**: 2026-03-24
**Status**: Prototipo funcional ✓
