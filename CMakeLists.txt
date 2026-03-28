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
# Target mínimo: NVIDIA RTX 4090 / 5070 Ti (sm_89 / sm_120)
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
    add_compile_options(/W3)  # W4+WX too strict for prototype
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
        "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include"
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
# sm_89  = Ada Lovelace (RTX 4090, RTX 6000 Ada)
# sm_120 = Blackwell (RTX 5070 Ti) — CUDA 12.8+ / PyTorch reports as compute 12.0

set(CUDA_ARCHITECTURES 89 120)

enable_language(CUDA)

# Flags de compilación CUDA
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# On MSVC the CUDA MSBuild targets already inject /Zi (debug) and /O2 (opt)
# based on the build configuration. We only add these manually on non-MSVC to
# avoid double-injection that breaks nvcc's command-line parsing on Windows.
if(NOT MSVC)
    if(LIQUIDBIT_ENABLE_DEBUG)
        string(APPEND CMAKE_CUDA_FLAGS " -g")
    endif()

    if(LIQUIDBIT_ENABLE_LINEINFO)
        string(APPEND CMAKE_CUDA_FLAGS " --lineinfo")
    endif()

    # Optimización
    string(APPEND CMAKE_CUDA_FLAGS " -O3")

    # Host compiler warnings (GCC/Clang only)
    string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler=-Wall")
    # MSVC: lineinfo added as nvcc-native flag (single dash form for MSVC mode)
    if(LIQUIDBIT_ENABLE_LINEINFO)
        string(APPEND CMAKE_CUDA_FLAGS " -lineinfo")
    endif()
endif()

message(STATUS "CUDA architectures: ${CUDA_ARCHITECTURES}")
message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")

# ============================================================================
# Librería Core C++ (token_geometry, semantic_bvh, etc.)
# ============================================================================

add_library(liquidbit_core STATIC
    src/token_geometry.cpp
    src/semantic_bvh.cpp
    src/alpha_bsh.cpp
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
# OptiX Pipeline — PTX compilation + host code + inception_runner
# ============================================================================

if(OptiX_INCLUDE_DIR)
    option(LIQUIDBIT_BUILD_OPTIX "Build OptiX ray tracing pipeline" ON)
else()
    option(LIQUIDBIT_BUILD_OPTIX "Build OptiX ray tracing pipeline" OFF)
endif()

if(LIQUIDBIT_BUILD_OPTIX AND OptiX_INCLUDE_DIR)
    message(STATUS "OptiX build ENABLED — compiling shaders and host code")

    # ── PTX compilation: .cu → .ptx ──────────────────────────────
    # OptiX shaders must be compiled to PTX (not object files).
    # nvcc --ptx produces device-only code that OptiX loads at runtime.

    set(OPTIX_SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cuda)
    set(PTX_OUTPUT_DIR ${CMAKE_BINARY_DIR}/ptx)
    file(MAKE_DIRECTORY ${PTX_OUTPUT_DIR})

    set(OPTIX_SHADERS
        ray_generation.cu
        closest_hit.cu
        miss.cu
        ray_attention.cu
        optix_router_raygen.cu
        optix_router_hitgroup.cu
    )

    set(PTX_FILES "")
    foreach(SHADER ${OPTIX_SHADERS})
        get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
        set(PTX_FILE ${PTX_OUTPUT_DIR}/${SHADER_NAME}.ptx)
        list(APPEND PTX_FILES ${PTX_FILE})

        add_custom_command(
            OUTPUT "${PTX_FILE}"
            COMMAND "${CUDAToolkit_NVCC_EXECUTABLE}"
                --ptx
                -O3
                --use_fast_math
                -std=c++17
                "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
                "-I${OptiX_INCLUDE_DIR}"
                "-I${CUDAToolkit_INCLUDE_DIRS}"
                -gencode=arch=compute_89,code=compute_89
                "${OPTIX_SHADER_DIR}/${SHADER}"
                -o "${PTX_FILE}"
            DEPENDS "${OPTIX_SHADER_DIR}/${SHADER}"
            COMMENT "Compiling OptiX shader ${SHADER} -> ${SHADER_NAME}.ptx"
            VERBATIM
        )
    endforeach()

    # Custom target to build all PTX files
    add_custom_target(optix_ptx ALL DEPENDS ${PTX_FILES})
    message(STATUS "Created target: optix_ptx (${OPTIX_SHADERS})")

    # ── Copy PTX to build root for easy access ─────────────────
    # The Python bridge (optix_router_bridge.py) loads PTX from build/
    set(COMBINED_PTX "${CMAKE_BINARY_DIR}/liquidbit_kernels.ptx")
    add_custom_command(
        OUTPUT "${COMBINED_PTX}"
        COMMAND ${CMAKE_COMMAND} -E copy "${PTX_OUTPUT_DIR}/ray_generation.ptx" "${COMBINED_PTX}"
        DEPENDS ${PTX_FILES}
        COMMENT "Copying ray_generation.ptx as liquidbit_kernels.ptx"
        VERBATIM
    )
    add_custom_target(combined_ptx ALL DEPENDS "${COMBINED_PTX}")

    # ── OptiX host code library ──────────────────────────────────
    add_library(liquidbit_optix STATIC
        cuda/optix_host.cpp
    )

    target_include_directories(liquidbit_optix PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OptiX_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(liquidbit_optix PUBLIC
        liquidbit_core
        CUDA::cudart
        CUDA::cuda_driver
    )

    # optix_host.cpp uses relative include "include/token_geometry.h"
    # Fix: also search from project root
    target_include_directories(liquidbit_optix PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_dependencies(liquidbit_optix optix_ptx)
    message(STATUS "Created target: liquidbit_optix")

    # ── inception_runner executable ───────────────────────────────
    # Main executable that loads PTX, builds BVH, and runs RT routing
    # Used by optix_router_bridge.py to perform routing via RT Cores

    # Check if we have a main entry point (add one if needed)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/inception_runner_main.cpp)
        add_executable(inception_runner
            cuda/inception_runner_main.cpp
        )
    else()
        # Generate a minimal main that uses LiquidBitOptixContext
        file(WRITE ${CMAKE_BINARY_DIR}/inception_runner_main.cpp
"// Auto-generated main for inception_runner
#include <iostream>
#include <fstream>
#include <string>

// Forward declare from optix_host.cpp (will be linked)
// For now, just verify the pipeline can be created
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << \"Usage: inception_runner <ptx_path> <scene.bin> <results.bin>\" << std::endl;
        return 1;
    }
    std::cout << \"[inception_runner] PTX: \" << argv[1] << std::endl;
    std::cout << \"[inception_runner] Scene: \" << argv[2] << std::endl;
    std::cout << \"[inception_runner] Results: \" << argv[3] << std::endl;
    std::cout << \"[inception_runner] TODO: integrate with LiquidBitOptixContext\" << std::endl;
    return 0;
}
")
        add_executable(inception_runner
            ${CMAKE_BINARY_DIR}/inception_runner_main.cpp
        )
    endif()

    target_link_libraries(inception_runner
        liquidbit_optix
        liquidbit_core
        CUDA::cudart
        CUDA::cuda_driver
    )

    add_dependencies(inception_runner combined_ptx)
    message(STATUS "Created target: inception_runner")

    # ── RT Core Router library ───────────────────────────────────
    # Minimal OptiX pipeline for expert selection via hardware RT Cores
    # Uses optix_router_raygen.ptx + optix_router_hitgroup.ptx

    add_library(liquidbit_rt_router STATIC
        cuda/optix_router_host.cpp
    )

    target_include_directories(liquidbit_rt_router PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${OptiX_INCLUDE_DIR}
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    target_link_libraries(liquidbit_rt_router PUBLIC
        liquidbit_core
        CUDA::cudart
        CUDA::cuda_driver
    )

    add_dependencies(liquidbit_rt_router optix_ptx)
    message(STATUS "Created target: liquidbit_rt_router (RT Core expert routing)")

    # ── RT Router benchmark executable ───────────────────────────
    file(WRITE ${CMAKE_BINARY_DIR}/rt_router_benchmark_main.cpp
"// Auto-generated benchmark runner for RT Core Router
#include <iostream>
#include <string>
#include <cstdlib>

extern \"C\" bool rtcore_router_benchmark(
    unsigned int num_experts, unsigned int batch_size,
    unsigned int num_warmup, unsigned int num_iters,
    const char* ptx_raygen, const char* ptx_hitgroup);

int main(int argc, char** argv) {
    unsigned int num_experts = 64;
    unsigned int batch_size  = 256;
    unsigned int num_warmup  = 10;
    unsigned int num_iters   = 100;

    std::string ptx_dir = \".\";
    if (argc > 1) ptx_dir = argv[1];
    if (argc > 2) batch_size = static_cast<unsigned int>(std::atoi(argv[2]));
    if (argc > 3) num_iters = static_cast<unsigned int>(std::atoi(argv[3]));

    std::string ptx_raygen   = ptx_dir + \"/ptx/optix_router_raygen.ptx\";
    std::string ptx_hitgroup = ptx_dir + \"/ptx/optix_router_hitgroup.ptx\";

    std::cout << \"RT Core Router Benchmark\" << std::endl;
    std::cout << \"PTX raygen:   \" << ptx_raygen << std::endl;
    std::cout << \"PTX hitgroup: \" << ptx_hitgroup << std::endl;

    if (!rtcore_router_benchmark(num_experts, batch_size, num_warmup,
                                  num_iters, ptx_raygen.c_str(),
                                  ptx_hitgroup.c_str())) {
        std::cerr << \"Benchmark FAILED\" << std::endl;
        return 1;
    }

    return 0;
}
")

    add_executable(rt_router_benchmark
        ${CMAKE_BINARY_DIR}/rt_router_benchmark_main.cpp
    )

    target_link_libraries(rt_router_benchmark
        liquidbit_rt_router
        CUDA::cudart
        CUDA::cuda_driver
    )

    add_dependencies(rt_router_benchmark optix_ptx)
    message(STATUS "Created target: rt_router_benchmark")

else()
    message(STATUS "OptiX build DISABLED — set OptiX_INSTALL_DIR or install OptiX SDK 9.1")

    # Fallback: empty OBJECT library for compatibility
    add_library(liquidbit_cuda OBJECT)
    target_link_libraries(liquidbit_cuda PUBLIC CUDA::cudart CUDA::cuda_driver)
    message(STATUS "Created target: liquidbit_cuda (stub, no OptiX)")
endif()

# ============================================================================
# Tests y Benchmarks (compilación condicional)
# ============================================================================

if(LIQUIDBIT_BUILD_TESTS AND NOT MSVC)
    # Benchmark CUDA: comparativa ray tracing vs matmul
    # NOTE: Excluded on MSVC due to a CUDA 13.2 MSBuild toolchain bug where
    # the HostCommandLineTemplate in CUDA 13.2.props generates a malformed
    # -Xcompiler="/EHsc -Ob2" flag (space inside quotes) that nvcc splits
    # into two arguments, causing "single input file required" fatal error.
    # The benchmark is a development tool; the core libraries build fine.
    add_executable(liquidbit_benchmark tests/benchmark.cu)

    target_include_directories(liquidbit_benchmark PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )

    # Compilar como CUDA
    set_target_properties(liquidbit_benchmark PROPERTIES LANGUAGE CUDA)

    set_target_properties(liquidbit_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION OFF
    )

    target_link_libraries(liquidbit_benchmark
        liquidbit_core
        CUDA::cudart
        CUDA::cuda_driver
        CUDA::cufft
        CUDA::cublas
    )

    message(STATUS "Created target: liquidbit_benchmark")

    # Test: OptiX RT pipeline integration test
    add_executable(test_optix_pipeline tests/test_optix_pipeline.cpp)

    target_include_directories(test_optix_pipeline PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
        ${OptiX_INCLUDE}
    )

    target_link_libraries(test_optix_pipeline
        liquidbit_core
        CUDA::cudart
        CUDA::cuda_driver
    )

    message(STATUS "Created target: test_optix_pipeline")

    enable_testing()
    add_test(NAME BenchmarkRayTracing COMMAND liquidbit_benchmark)
    add_test(NAME OptixPipeline COMMAND test_optix_pipeline)
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
