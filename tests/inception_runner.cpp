/**
 * @file tests/inception_runner.cpp
 * @brief Runner del pipeline OptiX String-Inception con escena cargada desde archivo.
 *
 * Lee un archivo de escena binario generado por python/inference.py,
 * ejecuta el pipeline OptiX real, y escribe los resultados a un archivo binario.
 *
 * USO:
 *   inception_runner.exe <ptx_path> <scene.bin> <results.bin>
 *
 * FORMATO scene.bin:
 *   Header (28 bytes):
 *     uint32 magic      = 0x4C425354 ('LBST')
 *     uint32 version    = 1
 *     uint32 numSpheres
 *     uint32 numStrings
 *     uint32 numPortals
 *     uint32 numRays
 *     float  baseOmega
 *   Data:
 *     SemanticSphere[numSpheres]  (32 bytes cada uno)
 *     SemanticString[numStrings]  (128 bytes cada uno)
 *     AffinePortal[numPortals]    (64 bytes cada uno)
 *
 * FORMATO results.bin:
 *   Header (8 bytes):
 *     uint32 magic      = 0x4C425253 ('LBRS')
 *     uint32 numResults
 *   Data:
 *     SpectralAttentionResult[numResults] (32 bytes cada uno)
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "liquidbit_resonance.h"
#include "token_geometry.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cassert>
#include <stdexcept>

// ============================================================================
// Constantes
// ============================================================================

static constexpr uint32_t SCENE_MAGIC   = 0x4C425354u;  // 'LBST'
static constexpr uint32_t SCENE_VERSION = 1u;
static constexpr uint32_t RESULTS_MAGIC = 0x4C425253u;  // 'LBRS'

// ============================================================================
// Utilidades
// ============================================================================

static void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error " << file << ":" << line
                  << " — " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}
#define CUDA_CHECK(x) checkCuda((x), __FILE__, __LINE__)

static void checkOptix(OptixResult res, const char* file, int line) {
    if (res != OPTIX_SUCCESS) {
        std::cerr << "OptiX error " << file << ":" << line
                  << " — code " << static_cast<int>(res) << "\n";
        std::exit(1);
    }
}
#define OPTIX_CHECK(x) checkOptix((x), __FILE__, __LINE__)

static void optixLog(unsigned level, const char* tag, const char* msg, void*) {
    if (level <= 2)
        std::cerr << "[OptiX][" << tag << "] " << msg << "\n";
}

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T    data;
};
struct RaygenData  { uint32_t numRays; };
struct HitgroupData{ uint32_t pad; };
struct MissData    { uint32_t pad; };

// ============================================================================
// Carga de escena
// ============================================================================

struct Scene {
    std::vector<SemanticSphere> spheres;
    std::vector<SemanticString> strings;
    std::vector<AffinePortal>   portals;
    uint32_t                    numRays;
    float                       baseOmega;
};

static Scene loadScene(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::string("No se puede abrir escena: ") + path);

    // Header
    uint32_t magic, version, numSpheres, numStrings, numPortals, numRays;
    float    baseOmega;
    f.read(reinterpret_cast<char*>(&magic),      4);
    f.read(reinterpret_cast<char*>(&version),    4);
    f.read(reinterpret_cast<char*>(&numSpheres), 4);
    f.read(reinterpret_cast<char*>(&numStrings), 4);
    f.read(reinterpret_cast<char*>(&numPortals), 4);
    f.read(reinterpret_cast<char*>(&numRays),    4);
    f.read(reinterpret_cast<char*>(&baseOmega),  4);

    if (magic != SCENE_MAGIC)
        throw std::runtime_error("scene.bin: magic incorrecto");
    if (version != SCENE_VERSION)
        throw std::runtime_error("scene.bin: versión incompatible");

    Scene s;
    s.numRays   = numRays;
    s.baseOmega = baseOmega;

    // Esferas
    s.spheres.resize(numSpheres);
    f.read(reinterpret_cast<char*>(s.spheres.data()),
           numSpheres * sizeof(SemanticSphere));

    // Strings
    s.strings.resize(numStrings);
    f.read(reinterpret_cast<char*>(s.strings.data()),
           numStrings * sizeof(SemanticString));

    // Portales
    s.portals.resize(numPortals);
    f.read(reinterpret_cast<char*>(s.portals.data()),
           numPortals * sizeof(AffinePortal));

    std::cout << "[scene] " << numSpheres << " esferas, "
              << numStrings << " strings, "
              << numPortals << " portales, "
              << numRays    << " rayos, ω₀=" << baseOmega << "\n";
    return s;
}

// ============================================================================
// Guarda resultados
// ============================================================================

static void saveResults(const char* path,
                        const std::vector<SpectralAttentionResult>& results) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::string("No se puede escribir resultados: ") + path);

    uint32_t magic = RESULTS_MAGIC;
    uint32_t count = static_cast<uint32_t>(results.size());
    f.write(reinterpret_cast<const char*>(&magic), 4);
    f.write(reinterpret_cast<const char*>(&count), 4);
    f.write(reinterpret_cast<const char*>(results.data()),
            count * sizeof(SpectralAttentionResult));
}

// ============================================================================
// PTX desde archivo
// ============================================================================

static std::string loadPTX(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::string("No se puede abrir PTX: ") + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Uso: " << argv[0]
                  << " <ptx_path> <scene.bin> <results.bin>\n";
        return 1;
    }

    const char* ptxPath     = argv[1];
    const char* scenePath   = argv[2];
    const char* resultsPath = argv[3];

    try {
        // ──────────────────────────────────────────────────
        // 0. CUDA + OptiX
        // ──────────────────────────────────────────────────
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(0));
        CUcontext cuCtx = nullptr;
        cuCtxGetCurrent(&cuCtx);

        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions ctxOpts = {};
        ctxOpts.logCallbackFunction = optixLog;
        ctxOpts.logCallbackLevel    = 4;
        OptixDeviceContext optixCtx = nullptr;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctxOpts, &optixCtx));

        // ──────────────────────────────────────────────────
        // 1. Cargar escena y PTX
        // ──────────────────────────────────────────────────
        Scene scene = loadScene(scenePath);
        std::string ptxSrc = loadPTX(ptxPath);
        std::cout << "[ptx] " << ptxSrc.size() << " bytes\n";

        // ──────────────────────────────────────────────────
        // 2. Módulo OptiX
        // ──────────────────────────────────────────────────
        OptixModuleCompileOptions modOpts = {};
        modOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        modOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        modOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        OptixPipelineCompileOptions pipeOpts = {};
        pipeOpts.usesMotionBlur                   = false;
        pipeOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeOpts.numPayloadValues                 = 4;
        pipeOpts.numAttributeValues               = 2;
        pipeOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
        pipeOpts.pipelineLaunchParamsVariableName = "c_params";
        pipeOpts.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

        char log[4096];
        size_t logSz = sizeof(log);

        OptixModule mod = nullptr;
        OPTIX_CHECK(optixModuleCreate(optixCtx, &modOpts, &pipeOpts,
            ptxSrc.c_str(), ptxSrc.size(), log, &logSz, &mod));
        if (logSz > 1) std::cerr << "[mod] " << log << "\n";

        // ──────────────────────────────────────────────────
        // 3. Program groups
        // ──────────────────────────────────────────────────
        OptixProgramGroupOptions pgOpts = {};
        OptixProgramGroup pgRaygen = nullptr, pgHit = nullptr, pgMiss = nullptr;

        { // Raygen
            OptixProgramGroupDesc d = {};
            d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            d.raygen.module = mod;
            d.raygen.entryFunctionName = "__raygen__liquidbit";
            logSz = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &d, 1, &pgOpts, log, &logSz, &pgRaygen));
        }
        { // HitGroup
            OptixProgramGroupDesc d = {};
            d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            d.hitgroup.moduleCH = mod;
            d.hitgroup.entryFunctionNameCH = "__closesthit__semantic_portal";
            d.hitgroup.moduleIS = mod;
            d.hitgroup.entryFunctionNameIS = "__intersection__sphere";
            logSz = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &d, 1, &pgOpts, log, &logSz, &pgHit));
        }
        { // Miss
            OptixProgramGroupDesc d = {};
            d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            d.miss.module = mod;
            d.miss.entryFunctionName = "__miss__inception";
            logSz = sizeof(log);
            OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &d, 1, &pgOpts, log, &logSz, &pgMiss));
        }

        // ──────────────────────────────────────────────────
        // 4. Pipeline
        // ──────────────────────────────────────────────────
        OptixProgramGroup pgs[] = { pgRaygen, pgHit, pgMiss };
        OptixPipelineLinkOptions linkOpts = {};
        linkOpts.maxTraceDepth = 4;
        logSz = sizeof(log);
        OptixPipeline pipeline = nullptr;
        OPTIX_CHECK(optixPipelineCreate(optixCtx, &pipeOpts, &linkOpts,
            pgs, 3, log, &logSz, &pipeline));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 0, 0, 1024, 1));

        // ──────────────────────────────────────────────────
        // 5. Buffers GPU
        // ──────────────────────────────────────────────────
        const uint32_t N = static_cast<uint32_t>(scene.spheres.size());

        SemanticSphere* dSpheres = nullptr;
        SemanticString* dStrings = nullptr;
        AffinePortal*   dPortals = nullptr;
        SpectralAttentionResult* dResults = nullptr;

        CUDA_CHECK(cudaMalloc(&dSpheres, N * sizeof(SemanticSphere)));
        CUDA_CHECK(cudaMalloc(&dStrings, scene.strings.size() * sizeof(SemanticString)));
        CUDA_CHECK(cudaMalloc(&dPortals, scene.portals.size() * sizeof(AffinePortal)));
        CUDA_CHECK(cudaMalloc(&dResults, scene.numRays * sizeof(SpectralAttentionResult)));

        CUDA_CHECK(cudaMemcpy(dSpheres, scene.spheres.data(),
            N * sizeof(SemanticSphere), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dStrings, scene.strings.data(),
            scene.strings.size() * sizeof(SemanticString), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPortals, scene.portals.data(),
            scene.portals.size() * sizeof(AffinePortal), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(dResults, 0,
            scene.numRays * sizeof(SpectralAttentionResult)));

        // ──────────────────────────────────────────────────
        // 6. GAS
        // ──────────────────────────────────────────────────
        std::vector<OptixAabb> aabbs(N);
        for (uint32_t i = 0; i < N; ++i) {
            const float r = scene.spheres[i].radius;
            const float3& c = scene.spheres[i].center;
            aabbs[i] = { c.x-r, c.y-r, c.z-r, c.x+r, c.y+r, c.z+r };
        }

        CUdeviceptr dAabbs = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAabbs), N * sizeof(OptixAabb)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dAabbs), aabbs.data(),
            N * sizeof(OptixAabb), cudaMemcpyHostToDevice));

        unsigned int geomFlag = OPTIX_GEOMETRY_FLAG_NONE;
        OptixBuildInput bi = {};
        bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        bi.customPrimitiveArray.aabbBuffers   = &dAabbs;
        bi.customPrimitiveArray.numPrimitives = N;
        bi.customPrimitiveArray.strideInBytes = sizeof(OptixAabb);
        bi.customPrimitiveArray.flags         = &geomFlag;
        bi.customPrimitiveArray.numSbtRecords = 1;

        OptixAccelBuildOptions accelOpts = {};
        accelOpts.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes gasSz = {};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(optixCtx, &accelOpts, &bi, 1, &gasSz));

        CUdeviceptr dTemp = 0, dGas = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTemp), gasSz.tempSizeInBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGas),  gasSz.outputSizeInBytes));

        OptixTraversableHandle gasHandle = 0;
        OPTIX_CHECK(optixAccelBuild(optixCtx, nullptr, &accelOpts, &bi, 1,
            dTemp, gasSz.tempSizeInBytes, dGas, gasSz.outputSizeInBytes,
            &gasHandle, nullptr, 0));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dTemp)));
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dAabbs)));
        std::cout << "[gas] " << N << " esferas, " << gasSz.outputSizeInBytes << " bytes\n";

        // ──────────────────────────────────────────────────
        // 7. SBT
        // ──────────────────────────────────────────────────
        SbtRecord<RaygenData>   sbtRg  = {};
        SbtRecord<HitgroupData> sbtHit = {};
        SbtRecord<MissData>     sbtMs  = {};
        OPTIX_CHECK(optixSbtRecordPackHeader(pgRaygen, sbtRg.header));
        OPTIX_CHECK(optixSbtRecordPackHeader(pgHit,    sbtHit.header));
        OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss,   sbtMs.header));
        sbtRg.data.numRays = scene.numRays;

        CUdeviceptr dSbtRg = 0, dSbtHit = 0, dSbtMs = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbtRg),  sizeof(sbtRg)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbtHit), sizeof(sbtHit)));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbtMs),  sizeof(sbtMs)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbtRg),  &sbtRg,  sizeof(sbtRg),  cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbtHit), &sbtHit, sizeof(sbtHit), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbtMs),  &sbtMs,  sizeof(sbtMs),  cudaMemcpyHostToDevice));

        OptixShaderBindingTable sbt = {};
        sbt.raygenRecord                = dSbtRg;
        sbt.hitgroupRecordBase          = dSbtHit;
        sbt.hitgroupRecordStrideInBytes = sizeof(sbtHit);
        sbt.hitgroupRecordCount         = 1;
        sbt.missRecordBase              = dSbtMs;
        sbt.missRecordStrideInBytes     = sizeof(sbtMs);
        sbt.missRecordCount             = 1;

        // ──────────────────────────────────────────────────
        // 8. Launch params → c_params
        // ──────────────────────────────────────────────────
        InceptionLaunchParams hParams = {};
        hParams.topLevelIAS = gasHandle;
        hParams.spheres     = dSpheres;
        hParams.portals     = dPortals;
        hParams.strings     = dStrings;
        hParams.results     = dResults;
        hParams.baseOmega   = scene.baseOmega;
        hParams.numRays     = scene.numRays;
        hParams.numStrings  = static_cast<uint32_t>(scene.strings.size());

        CUdeviceptr dParams = 0;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParams), sizeof(InceptionLaunchParams)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dParams), &hParams,
            sizeof(InceptionLaunchParams), cudaMemcpyHostToDevice));

        // ──────────────────────────────────────────────────
        // 9. optixLaunch
        // ──────────────────────────────────────────────────
        std::cout << "[launch] " << scene.numRays << " rayos...\n";
        OPTIX_CHECK(optixLaunch(pipeline, nullptr,
            dParams, sizeof(InceptionLaunchParams),
            &sbt, scene.numRays, 1, 1));
        CUDA_CHECK(cudaDeviceSynchronize());

        // ──────────────────────────────────────────────────
        // 10. Leer resultados
        // ──────────────────────────────────────────────────
        std::vector<SpectralAttentionResult> results(scene.numRays);
        CUDA_CHECK(cudaMemcpy(results.data(), dResults,
            scene.numRays * sizeof(SpectralAttentionResult), cudaMemcpyDeviceToHost));

        // Tabla resumen
        int hits = 0;
        for (uint32_t i = 0; i < scene.numRays; ++i) {
            if (results[i].traversalDepth > 0) hits++;
        }
        std::cout << "[results] " << hits << "/" << scene.numRays
                  << " rayos con hit\n";

        // Guardar archivo de resultados
        saveResults(resultsPath, results);
        std::cout << "[ok] Resultados escritos → " << resultsPath << "\n";

        // ──────────────────────────────────────────────────
        // 11. Cleanup
        // ──────────────────────────────────────────────────
        cudaFree(reinterpret_cast<void*>(dSbtRg));
        cudaFree(reinterpret_cast<void*>(dSbtHit));
        cudaFree(reinterpret_cast<void*>(dSbtMs));
        cudaFree(reinterpret_cast<void*>(dParams));
        cudaFree(reinterpret_cast<void*>(dGas));
        cudaFree(dSpheres); cudaFree(dStrings);
        cudaFree(dPortals); cudaFree(dResults);

        optixPipelineDestroy(pipeline);
        optixProgramGroupDestroy(pgRaygen);
        optixProgramGroupDestroy(pgHit);
        optixProgramGroupDestroy(pgMiss);
        optixModuleDestroy(mod);
        optixDeviceContextDestroy(optixCtx);

        return (hits > 0) ? 0 : 2;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
}
