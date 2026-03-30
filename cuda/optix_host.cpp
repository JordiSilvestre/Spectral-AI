/**
 * @file optix_host.cpp
 * @brief Host code completo para inicialización, gestión y ejecución del pipeline OptiX 8.x
 *
 * Este archivo contiene la implementación de la clase SpectralAIOptixContext que encapsula
 * toda la lógica de host code para el pipeline de ray tracing de SpectralAI Zero-Matrix.
 *
 * FUNCIONALIDADES:
 * ================
 *   1. Inicialización de OptiX (optixInit, optixDeviceContextCreate)
 *   2. Compilación de módulos PTX desde los kernels CUDA (.cu)
 *   3. Creación del pipeline OptiX con los 4 programas (raygen, closest-hit, miss, any-hit)
 *   4. Construcción del Shader Binding Table (SBT) para conectar programas con datos
 *   5. Aceleración geométrica: construcción del BVH/BSH desde TokenNodes
 *   6. Lanzamiento del pipeline: optixLaunch
 *   7. Liberación ordenada de recursos
 *
 * NOTAS DE COMPILACIÓN:
 * ======================
 *   - Requiere OptiX 8.x SDK instalado
 *   - Requiere CUDA 12.x
 *   - Las funciones CUDA devem ser compiladas a PTX antes (nvcc -ptx)
 *   - Incluye <optix_stubs.h> para las funciones del runtime
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>  // Defines g_optixFunctionTable symbol
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>

#include "token_geometry.h"
#include "alpha_bsh.h"

// ============================================================================
// CONSTANTES Y CONFIGURACIÓN GLOBAL
// ============================================================================

/// Tamaño máximo del SBT (Shader Binding Table) en bytes
constexpr size_t MAX_SBT_SIZE = 10 * 1024 * 1024;  // 10 MB

/// Visibilidad mask para rayos (255 = todos visibles)
constexpr unsigned int RAY_VISIBILITY_MASK = 255;

/// Número de tipos de rayos (para SBT)
constexpr unsigned int NUM_RAY_TYPES = 2;  // RADIANCE (0) y otra (1)

/// Índice del tipo de rayo primario
constexpr unsigned int RAY_TYPE_RADIANCE = 0;

// ============================================================================
// PLANTILLA SBT: Patrón estándar de OptiX SDK (no existe en el SDK como tipo)
// ============================================================================

/**
 * @brief Registro del SBT con cabecera alineada y datos de usuario.
 *
 * La cabecera debe ser rellenada con optixSbtRecordPackHeader().
 * El campo data contiene los datos específicos del programa.
 */
template <typename T>
struct OptixSbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

// ============================================================================
// ESTRUCTURA AUXILIAR: Handle Logger
// ============================================================================

/**
 * @brief Logger simple para errores y mensajes de OptiX.
 *
 * Implementa el callback de logger de OptiX que se llama cuando hay errores
 * o mensajes de debug en el pipeline.
 */
class OptixLogger {
public:
    static void log(unsigned int level, const char* tag, const char* message) {
        std::string levelStr;
        switch (level) {
            case 0: levelStr = "[FATAL]"; break;
            case 1: levelStr = "[ERROR]"; break;
            case 2: levelStr = "[WARN]"; break;
            case 3: levelStr = "[INFO]"; break;
            case 4: levelStr = "[DEBUG]"; break;
            default: levelStr = "[UNKNOWN]"; break;
        }
        std::cerr << levelStr << " [" << tag << "] " << message << std::endl;
    }

    static void callback(unsigned int level, const char* tag, const char* message, void* cbdata) {
        reinterpret_cast<OptixLogger*>(cbdata)->log(level, tag, message);
    }
};

// ============================================================================
// CLASE PRINCIPAL: SpectralAIOptixContext
// ============================================================================

/**
 * @class SpectralAIOptixContext
 * @brief Gestor completo del contexto OptiX para SpectralAI Zero-Matrix.
 *
 * Responsabilidades:
 *   1. Inicialización y gestión del contexto OptiX
 *   2. Compilación del pipeline desde PTX
 *   3. Construcción y gestión del SBT
 *   4. Construcción de estructuras de aceleración (BVH)
 *   5. Lanzamiento del pipeline de ray tracing
 *   6. Gestión de memoria GPU y liberación de recursos
 */
class SpectralAIOptixContext {
public:
    // ========================================================================
    // CONSTRUCTOR / DESTRUCTOR
    // ========================================================================

    /**
     * @brief Constructor. Inicializa OptiX pero no compila el pipeline.
     *
     * Pasos:
     *   1. Inicializa CUDA en el device 0
     *   2. Llama optixInit() para inicializar OptiX
     *   3. Crea el contexto de dispositivo OptiX
     *   4. Configura el logger de OptiX
     */
    SpectralAIOptixContext()
        : cuda_context_(nullptr),
          optix_context_(nullptr),
          pipeline_(nullptr),
          module_raygen_(nullptr),
          module_closest_hit_(nullptr),
          module_miss_(nullptr),
          pipeline_compile_options_{},
          d_gas_output_buffer_(0),
          gas_output_buffer_size_(0),
          d_sbt_buffer_(0),
          sbt_buffer_size_(0) {

        if (!initializeCUDA()) {
            throw std::runtime_error("Failed to initialize CUDA");
        }

        if (!initializeOptiX()) {
            throw std::runtime_error("Failed to initialize OptiX");
        }

        std::cout << "[OptiX] Context initialized successfully" << std::endl;
    }

    /**
     * @brief Destructor. Libera todos los recursos OptiX y CUDA.
     *
     * Orden de liberación:
     *   1. Pipeline OptiX (si existe)
     *   2. Módulo OptiX (si existe)
     *   3. Buffers GPU (BVH, SBT)
     *   4. Contexto de dispositivo OptiX
     *   5. CUDA context
     */
    ~SpectralAIOptixContext() {
        cleanup();
    }

    // ========================================================================
    // MÉTODOS PÚBLICOS: INICIALIZACIÓN Y CONSTRUCCIÓN
    // ========================================================================

    /**
     * @brief Compila el pipeline OptiX a partir de strings PTX.
     *
     * Pasos:
     *   1. Crea un módulo OptiX desde el PTX compilado
     *   2. Crea los 4 programas de shader (raygen, closest_hit, miss, any_hit)
     *   3. Agrupa los programas en un pipeline
     *   4. Compila el pipeline (genera código nativo)
     *   5. Construye el SBT (Shader Binding Table)
     *
     * @param ptx_raygen String PTX compilado del programa raygen
     * @param ptx_closest_hit String PTX compilado del programa closest-hit
     * @param ptx_miss String PTX compilado del programa miss
     * @param ptx_any_hit String PTX compilado del programa any-hit
     *
     * @return true si la compilación fue exitosa, false si hubo error.
     */
    bool createPipeline(
        const char* ptx_raygen,
        const char* ptx_closest_hit,
        const char* ptx_miss) {

        if (!createModule(ptx_raygen, ptx_closest_hit, ptx_miss)) {
            std::cerr << "[OptiX] Failed to create module" << std::endl;
            return false;
        }

        if (!createPrograms()) {
            std::cerr << "[OptiX] Failed to create programs" << std::endl;
            return false;
        }

        if (!buildPipeline()) {
            std::cerr << "[OptiX] Failed to build pipeline" << std::endl;
            return false;
        }

        if (!buildShaderBindingTable()) {
            std::cerr << "[OptiX] Failed to build SBT" << std::endl;
            return false;
        }

        std::cout << "[OptiX] Pipeline created and compiled successfully" << std::endl;
        return true;
    }

    /**
     * @brief Construye una estructura de aceleración (BVH) a partir de TokenNodes.
     *
     * Pasos:
     *   1. Convierte TokenNodes a primitivas (AABBs)
     *   2. Crea OptixBuildInput describiendo los AABBs
     *   3. Llama optixAccelBuild() para compilar el BVH en GPU
     *   4. Almacena el puntero GPU para uso en lanzamiento
     *
     * @param tokens Array host-side de TokenNodes
     * @param num_tokens Número de tokens
     *
     * @return true si la construcción fue exitosa
     */
    bool buildAccelerationStructure(const TokenNode* tokens, uint32_t num_tokens) {
        if (!tokens || num_tokens == 0) {
            std::cerr << "[OptiX] Invalid token input" << std::endl;
            return false;
        }

        // Convertir TokenNodes a AABBs (primitivas para OptiX)
        std::vector<OptixAabb> aabbs;
        aabbs.reserve(num_tokens);

        for (uint32_t i = 0; i < num_tokens; ++i) {
            const TokenNode& token = tokens[i];
            OptixAabb aabb;
            aabb.minX = token.aabb_min.x;
            aabb.minY = token.aabb_min.y;
            aabb.minZ = token.aabb_min.z;
            aabb.maxX = token.aabb_max.x;
            aabb.maxY = token.aabb_max.y;
            aabb.maxZ = token.aabb_max.z;
            aabbs.push_back(aabb);
        }

        // Copiar AABBs a GPU
        CUdeviceptr d_aabbs;
        if (cudaMalloc(reinterpret_cast<void**>(&d_aabbs), aabbs.size() * sizeof(OptixAabb)) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate AABB buffer on GPU" << std::endl;
            return false;
        }

        if (cudaMemcpy(
            reinterpret_cast<void*>(d_aabbs),
            aabbs.data(),
            aabbs.size() * sizeof(OptixAabb),
            cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to copy AABBs to GPU" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_aabbs));
            return false;
        }

        // Crear build input
        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &d_aabbs;
        build_input.customPrimitiveArray.numPrimitives = num_tokens;
        build_input.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
        build_input.customPrimitiveArray.flags = &custom_primitive_flags_;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        // Configuración de compilación
        OptixAccelBuildOptions accel_options = {};
        accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
        accel_options.motionOptions.numKeys = 0;

        // Calcular requisitos de memoria
        OptixAccelBufferSizes gas_buffer_sizes;
        if (optixAccelComputeMemoryUsage(
            optix_context_,
            &accel_options,
            &build_input,
            1,  // num_build_inputs
            &gas_buffer_sizes) != OPTIX_SUCCESS) {
            std::cerr << "[OptiX] Failed to compute GAS memory usage" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_aabbs));
            return false;
        }

        // Alocar buffers: temp (construcción) y output (resultado)
        CUdeviceptr d_temp_buffer;
        if (cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate temp buffer for GAS" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_aabbs));
            return false;
        }

        // Liberar buffer de salida anterior si existe
        if (d_gas_output_buffer_ != 0) {
            cudaFree(reinterpret_cast<void*>(d_gas_output_buffer_));
        }

        if (cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer_), gas_buffer_sizes.outputSizeInBytes) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate output buffer for GAS" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_temp_buffer));
            cudaFree(reinterpret_cast<void*>(d_aabbs));
            return false;
        }

        gas_output_buffer_size_ = gas_buffer_sizes.outputSizeInBytes;

        // Construir la estructura de aceleración
        OptixTraversableHandle gas_handle;
        if (optixAccelBuild(
            optix_context_,
            nullptr,  // stream (nullptr = usar stream por defecto)
            &accel_options,
            &build_input,
            1,  // num_build_inputs
            d_temp_buffer,
            gas_buffer_sizes.tempSizeInBytes,
            d_gas_output_buffer_,
            gas_buffer_sizes.outputSizeInBytes,
            &gas_handle,
            nullptr,  // emitted_properties (nullptr = no necesario)
            0) != OPTIX_SUCCESS) {
            std::cerr << "[OptiX] Failed to build acceleration structure" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_temp_buffer));
            cudaFree(reinterpret_cast<void*>(d_aabbs));
            return false;
        }

        // Almacenar el handle traversable para use en optixLaunch
        gas_handle_ = gas_handle;

        // Liberar buffers temporales
        cudaFree(reinterpret_cast<void*>(d_temp_buffer));
        cudaFree(reinterpret_cast<void*>(d_aabbs));

        std::cout << "[OptiX] Acceleration structure built successfully ("
                  << num_tokens << " primitives, "
                  << gas_output_buffer_size_ << " bytes)" << std::endl;
        return true;
    }

    /**
     * @brief Lanza el pipeline OptiX para procesar rayos.
     *
     * Pasos:
     *   1. Prepara los buffers de entrada (rayos)
     *   2. Prepara los buffers de salida (resultados)
     *   3. Copian datos host → GPU
     *   4. Llama optixLaunch() para ejecutar el pipeline
     *   5. Copia resultados GPU → host
     *
     * @param rays Array host-side de rayos de entrada
     * @param num_rays Número de rayos
     * @param output Array host-side para resultados (debe tener espacio para num_rays resultados)
     * @param output_size Tamaño del buffer de salida en bytes
     *
     * @return true si el lanzamiento fue exitoso
     */
    bool launch(
        const void* rays,
        uint32_t num_rays,
        void* output,
        uint32_t output_size) {

        if (!pipeline_ || !module_raygen_ || gas_handle_ == 0) {
            std::cerr << "[OptiX] Pipeline or acceleration structure not initialized" << std::endl;
            return false;
        }

        // Alocar buffers GPU para entrada y salida
        CUdeviceptr d_rays;
        if (cudaMalloc(reinterpret_cast<void**>(&d_rays), num_rays * sizeof(SemanticRay)) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate rays buffer" << std::endl;
            return false;
        }

        CUdeviceptr d_output;
        if (cudaMalloc(reinterpret_cast<void**>(&d_output), output_size) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate output buffer" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_rays));
            return false;
        }

        // Copiar rayos a GPU
        if (cudaMemcpy(
            reinterpret_cast<void*>(d_rays),
            rays,
            num_rays * sizeof(SemanticRay),
            cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to copy rays to GPU" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_rays));
            cudaFree(reinterpret_cast<void*>(d_output));
            return false;
        }

        // Preparar parámetros de lanzamiento (launch params)
        // NOTA: Los parámetros exactos dependen de tu programa raygen
        struct LaunchParams {
            OptixTraversableHandle traversable;
            CUdeviceptr rays;
            CUdeviceptr output;
            uint32_t num_rays;
        } launch_params = {
            gas_handle_,
            d_rays,
            d_output,
            num_rays
        };

        CUdeviceptr d_launch_params;
        if (cudaMalloc(reinterpret_cast<void**>(&d_launch_params), sizeof(LaunchParams)) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate launch params buffer" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_rays));
            cudaFree(reinterpret_cast<void*>(d_output));
            return false;
        }

        if (cudaMemcpy(
            reinterpret_cast<void*>(d_launch_params),
            &launch_params,
            sizeof(LaunchParams),
            cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to copy launch params to GPU" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_rays));
            cudaFree(reinterpret_cast<void*>(d_output));
            cudaFree(reinterpret_cast<void*>(d_launch_params));
            return false;
        }

        // Ejecutar el pipeline
        // Parámetros: context, stream, launch_params_ptr, launch_params_size,
        //            SBT, launch_width, launch_height, launch_depth
        if (optixLaunch(
            pipeline_,
            0,  // stream (0 = stream por defecto)
            d_launch_params,
            sizeof(LaunchParams),
            &sbt_,
            1,  // launch_width (1 porque procesamos 1 rayo por thread)
            num_rays,  // launch_height
            1) != OPTIX_SUCCESS) {
            std::cerr << "[OptiX] optixLaunch failed" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_rays));
            cudaFree(reinterpret_cast<void*>(d_output));
            cudaFree(reinterpret_cast<void*>(d_launch_params));
            return false;
        }

        // Sincronizar GPU
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::cerr << "[OptiX] Failed to synchronize after launch" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_rays));
            cudaFree(reinterpret_cast<void*>(d_output));
            cudaFree(reinterpret_cast<void*>(d_launch_params));
            return false;
        }

        // Copiar resultados a host
        if (cudaMemcpy(
            output,
            reinterpret_cast<void*>(d_output),
            output_size,
            cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to copy output from GPU" << std::endl;
            cudaFree(reinterpret_cast<void*>(d_rays));
            cudaFree(reinterpret_cast<void*>(d_output));
            cudaFree(reinterpret_cast<void*>(d_launch_params));
            return false;
        }

        // Liberar buffers temporales
        cudaFree(reinterpret_cast<void*>(d_rays));
        cudaFree(reinterpret_cast<void*>(d_output));
        cudaFree(reinterpret_cast<void*>(d_launch_params));

        std::cout << "[OptiX] Launch completed successfully (" << num_rays << " rays)" << std::endl;
        return true;
    }

    /**
     * @brief Libera todos los recursos y limpia el contexto.
     *
     * Esta función es segura de llamar múltiples veces.
     */
    void cleanup() {
        if (pipeline_ != nullptr) {
            optixPipelineDestroy(pipeline_);
            pipeline_ = nullptr;
        }

        if (module_raygen_ != nullptr) {
            optixModuleDestroy(module_raygen_);
            module_raygen_ = nullptr;
        }
        if (module_closest_hit_ != nullptr) {
            optixModuleDestroy(module_closest_hit_);
            module_closest_hit_ = nullptr;
        }
        if (module_miss_ != nullptr) {
            optixModuleDestroy(module_miss_);
            module_miss_ = nullptr;
        }

        if (d_gas_output_buffer_ != 0) {
            cudaFree(reinterpret_cast<void*>(d_gas_output_buffer_));
            d_gas_output_buffer_ = 0;
        }

        if (d_sbt_buffer_ != 0) {
            cudaFree(reinterpret_cast<void*>(d_sbt_buffer_));
            d_sbt_buffer_ = 0;
        }

        if (optix_context_ != nullptr) {
            optixDeviceContextDestroy(optix_context_);
            optix_context_ = nullptr;
        }

        if (cuda_context_ != nullptr) {
            cuCtxDestroy(cuda_context_);
            cuda_context_ = nullptr;
        }

        std::cout << "[OptiX] Resources cleaned up" << std::endl;
    }

    // ========================================================================
    // MÉTODOS GETTER
    // ========================================================================

    OptixDeviceContext getOptixContext() const { return optix_context_; }
    OptixPipeline getPipeline() const { return pipeline_; }
    OptixShaderBindingTable getSBT() const { return sbt_; }

private:
    // ========================================================================
    // MIEMBROS PRIVADOS
    // ========================================================================

    CUcontext cuda_context_;                    ///< Contexto CUDA
    OptixDeviceContext optix_context_;          ///< Contexto de dispositivo OptiX
    OptixPipeline pipeline_;                    ///< Pipeline compilado OptiX
    OptixModule module_raygen_;                  ///< Módulo PTX: raygen
    OptixModule module_closest_hit_;             ///< Módulo PTX: closest-hit + any-hit + intersection
    OptixModule module_miss_;                    ///< Módulo PTX: miss
    OptixPipelineCompileOptions pipeline_compile_options_;  ///< Opciones de compilación (shared)

    CUdeviceptr d_gas_output_buffer_;           ///< Buffer GPU: estructura de aceleración (BVH)
    size_t gas_output_buffer_size_;             ///< Tamaño del BVH en bytes

    CUdeviceptr d_sbt_buffer_;                  ///< Buffer GPU: Shader Binding Table
    size_t sbt_buffer_size_;                    ///< Tamaño del SBT en bytes

    OptixShaderBindingTable sbt_;               ///< Descriptor del SBT
    OptixTraversableHandle gas_handle_;         ///< Handle para el BVH en shaders

    OptixProgramGroup pg_raygen_;               ///< Program group: raygen
    OptixProgramGroup pg_closest_hit_;          ///< Program group: closest-hit
    OptixProgramGroup pg_miss_;                 ///< Program group: miss
    OptixProgramGroup pg_any_hit_;              ///< Program group: any-hit

    unsigned int custom_primitive_flags_;       ///< Flags para primitivas custom (AABBs)

    OptixLogger logger_;                        ///< Logger para mensajes OptiX

    // ========================================================================
    // MÉTODOS PRIVADOS: INICIALIZACIÓN
    // ========================================================================

    /**
     * @brief Inicializa CUDA en el device 0.
     */
    bool initializeCUDA() {
        CUdevice device;
        if (cuDeviceGet(&device, 0) != CUDA_SUCCESS) {
            std::cerr << "[CUDA] Failed to get device 0" << std::endl;
            return false;
        }

        // cuCtxCreate was remapped to cuCtxCreate_v4 in CUDA 12.x+
        // Pass nullptr for CUctxCreateParams to use default settings
        if (cuCtxCreate_v4(&cuda_context_, nullptr, 0, device) != CUDA_SUCCESS) {
            std::cerr << "[CUDA] Failed to create context" << std::endl;
            return false;
        }

        return true;
    }

    /**
     * @brief Inicializa OptiX.
     *
     * Pasos:
     *   1. Llama optixInit() (debe ser primera llamada OptiX)
     *   2. Crea OptixDeviceContext
     *   3. Configura logger
     */
    bool initializeOptiX() {
        if (optixInit() != OPTIX_SUCCESS) {
            std::cerr << "[OptiX] optixInit failed" << std::endl;
            return false;
        }

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = &OptixLogger::callback;
        options.logCallbackData = &logger_;
        options.logCallbackLevel = 4;  // DEBUG level

        if (optixDeviceContextCreate(cuda_context_, &options, &optix_context_) != OPTIX_SUCCESS) {
            std::cerr << "[OptiX] optixDeviceContextCreate failed" << std::endl;
            return false;
        }

        // Inicializar flags para primitivas custom
        custom_primitive_flags_ = OPTIX_GEOMETRY_FLAG_NONE;

        return true;
    }

    /**
     * @brief Helper: creates a single OptiX module from a PTX string.
     */
    bool createSingleModule(const char* ptx, size_t ptx_size,
                            const OptixModuleCompileOptions& mod_opts,
                            OptixModule& out_module, const char* label) {
        char log[2048];
        size_t sizeof_log = sizeof(log);

        if (optixModuleCreate(
            optix_context_,
            &mod_opts,
            &pipeline_compile_options_,
            ptx, ptx_size,
            log, &sizeof_log,
            &out_module) != OPTIX_SUCCESS) {
            std::cerr << "[OptiX] Module creation failed (" << label << "):\n"
                      << log << std::endl;
            return false;
        }

        if (sizeof_log > 1) {
            std::cout << "[OptiX] Module " << label << " log:\n" << log << std::endl;
        }
        return true;
    }

    /**
     * @brief Crea módulos OptiX separados para cada PTX shader.
     *
     * Cada .cu se compila a su propio .ptx, y cada PTX se carga como un
     * módulo OptiX independiente. Concatenar PTX es inválido.
     */
    bool createModule(
        const char* ptx_raygen,
        const char* ptx_closest_hit,
        const char* ptx_miss) {

        OptixModuleCompileOptions module_compile_options = {};
        module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        // Store pipeline compile options as member (needed by buildPipeline)
        pipeline_compile_options_ = {};
        pipeline_compile_options_.usesMotionBlur = false;
        pipeline_compile_options_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_options_.numPayloadValues = 8;
        pipeline_compile_options_.numAttributeValues = 2;
        pipeline_compile_options_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipeline_compile_options_.pipelineLaunchParamsVariableName = "params";
        pipeline_compile_options_.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

        // Create separate module for each PTX shader
        if (!createSingleModule(ptx_raygen, strlen(ptx_raygen),
                                module_compile_options, module_raygen_, "raygen"))
            return false;

        if (!createSingleModule(ptx_closest_hit, strlen(ptx_closest_hit),
                                module_compile_options, module_closest_hit_, "hitgroup"))
            return false;

        if (!createSingleModule(ptx_miss, strlen(ptx_miss),
                                module_compile_options, module_miss_, "miss"))
            return false;

        return true;
    }

    /**
     * @brief Crea los 4 programas de shader (program groups).
     */
    bool createPrograms() {
        // Program Group: Raygen
        {
            OptixProgramGroupOptions pg_options = {};
            OptixProgramGroupDesc pg_desc = {};
            pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            pg_desc.raygen.module = module_raygen_;
            pg_desc.raygen.entryFunctionName = "__raygen__rg_optical_attention";

            char log[2048];
            size_t sizeof_log = sizeof(log);

            if (optixProgramGroupCreate(
                optix_context_,
                &pg_desc,
                1,
                &pg_options,
                log,
                &sizeof_log,
                &pg_raygen_) != OPTIX_SUCCESS) {
                std::cerr << "[OptiX] Raygen program creation failed:\n" << log << std::endl;
                return false;
            }
        }

        // Program Group: Closest Hit (con intersection)
        {
            OptixProgramGroupOptions pg_options = {};
            OptixProgramGroupDesc pg_desc = {};
            pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            pg_desc.hitgroup.moduleCH = module_closest_hit_;
            pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch_optical_attention";
            // No intersection program — using built-in AABB intersection
            pg_desc.hitgroup.moduleIS = nullptr;
            pg_desc.hitgroup.entryFunctionNameIS = nullptr;
            // No any-hit program — all hits are processed in closest-hit
            pg_desc.hitgroup.moduleAH = nullptr;
            pg_desc.hitgroup.entryFunctionNameAH = nullptr;

            char log[2048];
            size_t sizeof_log = sizeof(log);

            if (optixProgramGroupCreate(
                optix_context_,
                &pg_desc,
                1,
                &pg_options,
                log,
                &sizeof_log,
                &pg_closest_hit_) != OPTIX_SUCCESS) {
                std::cerr << "[OptiX] Hitgroup program creation failed:\n" << log << std::endl;
                return false;
            }
        }

        // Program Group: Miss
        {
            OptixProgramGroupOptions pg_options = {};
            OptixProgramGroupDesc pg_desc = {};
            pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            pg_desc.miss.module = module_miss_;
            pg_desc.miss.entryFunctionName = "__miss__ms_optical_attention";

            char log[2048];
            size_t sizeof_log = sizeof(log);

            if (optixProgramGroupCreate(
                optix_context_,
                &pg_desc,
                1,
                &pg_options,
                log,
                &sizeof_log,
                &pg_miss_) != OPTIX_SUCCESS) {
                std::cerr << "[OptiX] Miss program creation failed:\n" << log << std::endl;
                return false;
            }
        }

        return true;
    }

    /**
     * @brief Construye el pipeline a partir de los program groups.
     */
    bool buildPipeline() {
        std::vector<OptixProgramGroup> program_groups = {
            pg_raygen_,
            pg_closest_hit_,
            pg_miss_
        };

        OptixPipelineLinkOptions pipeline_link_options = {};
        pipeline_link_options.maxTraceDepth = 16;
        // debugLevel was removed in OptiX 8.0; debug level is now set in module compile options

        char log[2048];
        size_t sizeof_log = sizeof(log);

        if (optixPipelineCreate(
            optix_context_,
            &pipeline_compile_options_,
            &pipeline_link_options,
            program_groups.data(),
            program_groups.size(),
            log,
            &sizeof_log,
            &pipeline_) != OPTIX_SUCCESS) {
            std::cerr << "[OptiX] Pipeline creation failed:\n" << log << std::endl;
            return false;
        }

        // Configurar el stack de rayos
        if (optixPipelineSetStackSize(
            pipeline_,
            2 * 1024,  // direct_callable_stack_size_from_traversal
            2 * 1024,  // direct_callable_stack_size_from_state
            2 * 1024,  // continuation_stack_size_from_traversal
            16) != OPTIX_SUCCESS) {  // continuation_stack_size_from_state
            std::cerr << "[OptiX] Failed to set pipeline stack size" << std::endl;
            return false;
        }

        return true;
    }

    /**
     * @brief Construye el Shader Binding Table (SBT).
     *
     * El SBT conecta los program groups con los datos geométricos.
     * Estructura:
     *   [RaygenRecord] [HitgroupRecord] [MissRecord]
     */
    bool buildShaderBindingTable() {
        // Tamaños fijos de registro SBT (header 32 bytes + datos alineados a 16)
        size_t raygen_record_size    = sizeof(OptixSbtRecord<RayGenRecord>);
        size_t hitgroup_record_size  = sizeof(OptixSbtRecord<HitGroupRecord>);
        size_t miss_record_size      = sizeof(OptixSbtRecord<MissRecord>);

        size_t sbt_size = raygen_record_size + hitgroup_record_size + miss_record_size;

        // Alocar buffer SBT en GPU
        if (d_sbt_buffer_ != 0) {
            cudaFree(reinterpret_cast<void*>(d_sbt_buffer_));
        }

        if (cudaMalloc(reinterpret_cast<void**>(&d_sbt_buffer_), sbt_size) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to allocate SBT buffer" << std::endl;
            return false;
        }

        sbt_buffer_size_ = sbt_size;

        // Rellenar registros del SBT en host
        std::vector<unsigned char> sbt_host(sbt_size, 0);

        // Raygen record
        {
            OptixSbtRecord<RayGenRecord> raygen_record = {};
            optixSbtRecordPackHeader(pg_raygen_, &raygen_record);
            raygen_record.data.num_rays = 1;  // Alpha genera 1 rayo

            std::memcpy(sbt_host.data(), &raygen_record, sizeof(raygen_record));
        }

        // Hitgroup record
        {
            OptixSbtRecord<HitGroupRecord> hitgroup_record = {};
            optixSbtRecordPackHeader(pg_closest_hit_, &hitgroup_record);
            hitgroup_record.data.sphere_id = 0;  // Será actualizado según tokens

            std::memcpy(
                sbt_host.data() + raygen_record_size,
                &hitgroup_record,
                sizeof(hitgroup_record));
        }

        // Miss record
        {
            OptixSbtRecord<MissRecord> miss_record = {};
            optixSbtRecordPackHeader(pg_miss_, &miss_record);

            std::memcpy(
                sbt_host.data() + raygen_record_size + hitgroup_record_size,
                &miss_record,
                sizeof(miss_record));
        }

        // Copiar SBT a GPU
        if (cudaMemcpy(
            reinterpret_cast<void*>(d_sbt_buffer_),
            sbt_host.data(),
            sbt_size,
            cudaMemcpyHostToDevice) != cudaSuccess) {
            std::cerr << "[OptiX] Failed to copy SBT to GPU" << std::endl;
            return false;
        }

        // Configurar descriptores del SBT
        sbt_.raygenRecord = d_sbt_buffer_;
        sbt_.hitgroupRecordBase = d_sbt_buffer_ + raygen_record_size;
        sbt_.hitgroupRecordStrideInBytes = hitgroup_record_size;
        sbt_.hitgroupRecordCount = 1;
        sbt_.missRecordBase = d_sbt_buffer_ + raygen_record_size + hitgroup_record_size;
        sbt_.missRecordStrideInBytes = miss_record_size;
        sbt_.missRecordCount = 1;

        std::cout << "[OptiX] SBT created (" << sbt_size << " bytes)" << std::endl;
        return true;
    }

    // ========================================================================
    // ESTRUCTURAS AUXILIARES PARA SBT
    // ========================================================================

    /**
     * @brief Estructura de datos embebida en el registro Raygen del SBT.
     */
    struct RayGenRecord {
        uint32_t num_rays;
    };

    /**
     * @brief Estructura de datos embebida en el registro Hitgroup del SBT.
     */
    struct HitGroupRecord {
        uint32_t sphere_id;
    };

    /**
     * @brief Estructura de datos embebida en el registro Miss del SBT.
     */
    struct MissRecord {
        // Puede estar vacío o contener datos globales
    };
};

// ============================================================================
// UTILIDAD: Carga un archivo PTX desde disco
// ============================================================================

/**
 * @brief Lee un archivo .ptx completo y devuelve su contenido como string.
 *
 * @param filepath Ruta al archivo .ptx
 * @param out_content String donde se almacena el contenido PTX
 * @return true si la lectura fue exitosa
 */
bool loadPTXFile(const std::string& filepath, std::string& out_content) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "[OptiX] Failed to open PTX file: " << filepath << std::endl;
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    out_content.resize(static_cast<size_t>(size));
    if (!file.read(out_content.data(), size)) {
        std::cerr << "[OptiX] Failed to read PTX file: " << filepath << std::endl;
        return false;
    }

    std::cout << "[OptiX] Loaded PTX: " << filepath
              << " (" << size << " bytes)" << std::endl;
    return true;
}

// ============================================================================
// FUNCIÓN GLOBAL: Factory desde strings PTX
// ============================================================================

SpectralAIOptixContext* createSpectralAIOptixContext(
    const char* ptx_raygen,
    const char* ptx_closest_hit,
    const char* ptx_miss) {

    try {
        SpectralAIOptixContext* context = new SpectralAIOptixContext();

        if (!context->createPipeline(ptx_raygen, ptx_closest_hit, ptx_miss)) {
            delete context;
            return nullptr;
        }

        return context;
    } catch (const std::exception& e) {
        std::cerr << "[OptiX] Exception during context creation: " << e.what() << std::endl;
        return nullptr;
    }
}

// ============================================================================
// FUNCIÓN GLOBAL: Factory desde archivos PTX en disco
// ============================================================================

/**
 * @brief Crea un contexto OptiX cargando .ptx directamente desde archivos.
 *
 * Uso típico:
 *   auto* ctx = createSpectralAIOptixContextFromFiles(
 *       "build/ptx/ray_generation.ptx",
 *       "build/ptx/closest_hit.ptx",
 *       "build/ptx/miss.ptx"
 *   );
 */
SpectralAIOptixContext* createSpectralAIOptixContextFromFiles(
    const std::string& ptx_raygen_path,
    const std::string& ptx_closest_hit_path,
    const std::string& ptx_miss_path) {

    std::string ptx_raygen, ptx_closest_hit, ptx_miss;

    if (!loadPTXFile(ptx_raygen_path, ptx_raygen) ||
        !loadPTXFile(ptx_closest_hit_path, ptx_closest_hit) ||
        !loadPTXFile(ptx_miss_path, ptx_miss)) {
        return nullptr;
    }

    return createSpectralAIOptixContext(
        ptx_raygen.c_str(),
        ptx_closest_hit.c_str(),
        ptx_miss.c_str());
}

// ============================================================================
// FUNCIÓN GLOBAL: Destructor seguro
// ============================================================================

/**
 * @brief Destruye de forma segura un contexto OptiX.
 *
 * @param context Puntero a SpectralAIOptixContext (puede ser nullptr)
 */
void destroySpectralAIOptixContext(SpectralAIOptixContext* context) {
    if (context != nullptr) {
        context->cleanup();
        delete context;
    }
}
