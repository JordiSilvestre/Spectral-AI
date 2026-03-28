# ============================================================================
# CMakeLists.txt — Sistema de build para LiquidBit Zero-Matrix
#
# Este archivo configura la compilación de:
#   - Librería core C++ (liquidbit_core)
#   - Kernels CUDA y OptiX (liquidbit_cuda)
#   - Benchmarks y tests
#   - Python bindings (futuro)
#
# Build:
#     mkdir build && cd build
#     cmake ..
#     cmake --build . --config Release
#
# Target mínimo: NVIDIA RTX 4090 / 5070 Ti (sm_89 o sm_100)
# ============================================================================

cmake_minimum_required(VERSION 3.28)
project(LiquidBitZeroMatrix LANGUAGES CXX CUDA)

# ============================================================================
# Versión del proyecto
# ============================================================================

set(LIQUIDBIT_VERSION_MAJOR 0)
set(LIQUIDBIT_VERSION_MINOR 1)
set(LIQUIDBIT_VERSION_PATCH 0)

set(LIQUIDBIT_VERSION "${LIQUIDBIT_VERSION_MAJOR}.${LIQUIDBIT_VERSION_MINOR}.${LIQUIDBIT_VERSION_PATCH}")
message(STATUS "LiquidBit Zero-Matrix v${LIQUIDBIT_VERSION}")

# ============================================================================
# Opciones de configuración
# ============================================================================

option(LIQUIDBIT_BUILD_TESTS "Build tests and benchmarks" ON)
option(LIQUIDBIT_BUILD_PYTHON "Build Python bindings" OFF)
option(LIQUIDBIT_ENABLE_DEBUG "Enable debug symbols" ON)
option(LIQUIDBIT_ENABLE_LINEINFO "Enable CUDA line info" ON)

# ============================================================================
# Configuración de C++
# ============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Flags de compilación C++
if(MSVC)
    add_compile_options(/W4 /WX)
    if(LIQUIDBIT_ENABLE_DEBUG)
        add_compile_options(/Zi /Od)
    endif()
else()
    add_compile_options(-Wall -Wextra -Wno-unused-parameter)
    if(LIQUIDBIT_ENABLE_DEBUG)
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3)
    endif()
endif()

# ============================================================================
# Búsqueda de CUDA
# ============================================================================

# Asegurar que CUDA está disponible
find_package(CUDAToolkit REQUIRED)

message(STATUS "CUDA Toolkit version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Include path: ${CUDAToolkit_INCLUDE_DIRS}")

# ============================================================================
# Búsqueda de OptiX (con HINTS personalizados)
# ============================================================================

# OptiX típicamente no tiene FindOptiX.cmake estándar.
# Se proporcionan HINTS para ubicaciones comunes.

set(OptiX_INSTALL_DIR "" CACHE PATH "OptiX installation directory")

if(OptiX_INSTALL_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${OptiX_INSTALL_DIR}")
endif()

# Intentar encontrar OptiX via include/optix.h
find_path(OptiX_INCLUDE_DIR optix.h
    HINTS
        ${OptiX_INSTALL_DIR}/include
        /opt/nvidia/optix
        /usr/local/optix
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0/include"
)

if(NOT OptiX_INCLUDE_DIR)
    message(WARNING "OptiX include directory not found. Ray tracing features may not compile.")
    message(STATUS "To fix: set OptiX_INSTALL_DIR to your OptiX installation path")
else()
    message(STATUS "OptiX include directory: ${OptiX_INCLUDE_DIR}")
endif()

# ============================================================================
# Directorio de include principal
# ============================================================================

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

if(OptiX_INCLUDE_DIR)
    include_directories(${OptiX_INCLUDE_DIR})
endif()

include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ============================================================================
# Configuración de CUDA
# ============================================================================

# Especificar compute capabilities (SM versions)
# sm_89 = Ada Lovelace (RTX 4090, RTX 6000 Ada)
# sm_100 = Blackwell (RTX 5070 Ti, futuro)

set(CUDA_ARCHITECTURES 89 100)

enable_language(CUDA)

# Flags de compilación CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(LIQUIDBIT_ENABLE_DEBUG)
    list(APPEND CMAKE_CUDA_FLAGS "-g")
endif()

if(LIQUIDBIT_ENABLE_LINEINFO)
    list(APPEND CMAKE_CUDA_FLAGS "--lineinfo")
endif()

# Optimización por defecto
list(APPEND CMAKE_CUDA_FLAGS "-O3")

# Warnings CUDA
list(APPEND CMAKE_CUDA_FLAGS "-Xcompiler=-Wall")

message(STATUS "CUDA architectures: ${CUDA_ARCHITECTURES}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")

# ============================================================================
# Librería Core C++ (token_geometry, semantic_bvh, etc.)
# ============================================================================

add_library(liquidbit_core STATIC
    src/token_geometry.cpp
    src/semantic_bvh.cpp
)

target_include_directories(liquidbit_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compilación con C++17
target_compile_features(liquidbit_core PUBLIC cxx_std_17)

# Enlazar CUDA runtime
target_link_libraries(liquidbit_core PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

message(STATUS "Created target: liquidbit_core")

# ============================================================================
# Kernels CUDA (ray tracing, atención óptica)
# ============================================================================

# Por ahora, stub de kernels CUDA compilados como object files
# En producción, esto incluiría ray_attention.cu, ray_generation.cu, etc.

# Crear una librería virtual para compatibilidad
add_library(liquidbit_cuda OBJECT)

target_link_libraries(liquidbit_cuda PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
)

# Si OptiX está disponible, añadir flags adicionales
if(OptiX_INCLUDE_DIR)
    target_include_directories(liquidbit_cuda PUBLIC ${OptiX_INCLUDE_DIR})
endif()

message(STATUS "Created target: liquidbit_cuda (stub)")

# ============================================================================
# Tests y Benchmarks (compilación condicional)
# ============================================================================

if(LIQUIDBIT_BUILD_TESTS)
    # Benchmark CUDA: comparativa ray tracing vs matmul
    add_executable(liquidbit_benchmark tests/benchmark.cu)

    target_include_directories(liquidbit_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    # Compilar como CUDA
    set_target_properties(liquidbit_benchmark PROPERTIES LANGUAGE CUDA)

    target_link_libraries(liquidbit_benchmark
        liquidbit_core
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cufft
        CUDA::cublas
    )

    message(STATUS "Created target: liquidbit_benchmark")

    enable_testing()
    add_test(NAME BenchmarkRayTracing COMMAND liquidbit_benchmark)
endif()

# ============================================================================
# Python Bindings (futuro)
# ============================================================================

if(LIQUIDBIT_BUILD_PYTHON)
    message(STATUS "Python bindings enabled (pybind11 required)")
    # TODO: find_package(pybind11) y crear módulo Python
endif()

# ============================================================================
# Instalación
# ============================================================================

# Headers públicos
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
    DESTINATION include/liquidbit
    FILES_MATCHING PATTERN "*.h"
)

# Librerías
install(TARGETS liquidbit_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Scripts Python (si existen)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/
    DESTINATION bin
    FILES_MATCHING PATTERN "*.py"
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

message(STATUS "Installation targets configured")

# ============================================================================
# Resumen de configuración
# ============================================================================

message(STATUS "")
message(STATUS "LiquidBit Zero-Matrix Configuration:")
message(STATUS "  Build tests:        ${LIQUIDBIT_BUILD_TESTS}")
message(STATUS "  Build Python:       ${LIQUIDBIT_BUILD_PYTHON}")
message(STATUS "  Debug symbols:      ${LIQUIDBIT_ENABLE_DEBUG}")
message(STATUS "  CUDA line info:     ${LIQUIDBIT_ENABLE_LINEINFO}")
message(STATUS "  C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "  CUDA Toolkit:       ${CUDAToolkit_VERSION}")
message(STATUS "  OptiX:              ${OptiX_INCLUDE_DIR}")
message(STATUS "")
