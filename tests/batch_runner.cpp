/**
 * @file tests/batch_runner.cpp
 * @brief Benchmark de latencia real: N escenas con OptiX vivo (sin reinit).
 *
 * A diferencia de inception_runner (una escena por invocación),
 * batch_runner inicializa OptiX UNA VEZ y procesa N escenas en batch.
 * Esto mide latencia real de traversal, no overhead de inicialización.
 *
 * FORMATO batch_scenes.bin:
 *   Header (12 bytes):
 *     uint32 magic      = 0x4C424243 ('LBBC')
 *     uint32 version    = 1
 *     uint32 numScenes
 *   Para cada escena (misma estructura que scene.bin):
 *     uint32 numSpheres
 *     uint32 numStrings
 *     uint32 numPortals
 *     uint32 numRays
 *     float  baseOmega
 *     SemanticSphere[numSpheres]
 *     SemanticString[numStrings]
 *     AffinePortal[numPortals]
 *
 * FORMATO batch_results.bin:
 *   Header (12 bytes):
 *     uint32 magic      = 0x4C424252 ('LBBR')
 *     uint32 numScenes
 *     uint32 totalRays
 *   Para cada escena:
 *     uint32 numRays
 *     float  launch_ms    -- tiempo de optixLaunch (CUDA events)
 *     float  build_ms     -- tiempo de GAS build
 *     uint32 hits
 *     SpectralAttentionResult[numRays]
 *
 * USO:
 *   batch_runner.exe <ptx_path> <batch_scenes.bin> <batch_results.bin>
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
#include <vector>
#include <string>
#include <stdexcept>
#include <cstring>

// ============================================================================
// Macros
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
struct RaygenData   { uint32_t numRays; };
struct HitgroupData { uint32_t pad; };
struct MissData     { uint32_t pad; };

// ============================================================================
// Constantes
// ============================================================================

static constexpr uint32_t BATCH_MAGIC   = 0x4C424243u;  // 'LBBC'
static constexpr uint32_t BATCH_VERSION = 1u;
static constexpr uint32_t BRESULT_MAGIC = 0x4C424252u;  // 'LBBR'
static constexpr uint32_t RESULTS_MAGIC = 0x4C425253u;  // 'LBRS' (compatibilidad)

// ============================================================================
// Scene por escena
// ============================================================================

struct SceneData {
    std::vector<SemanticSphere> spheres;
    std::vector<SemanticString> strings;
    std::vector<AffinePortal>   portals;
    uint32_t                    numRays;
    float                       baseOmega;
};

struct SceneResult {
    float    launch_ms;
    float    build_ms;
    uint32_t hits;
    std::vector<SpectralAttentionResult> results;
};

// ============================================================================
// Carga de batch
// ============================================================================

static std::vector<SceneData> loadBatch(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::string("No se puede abrir batch: ") + path);

    uint32_t magic, version, numScenes;
    f.read(reinterpret_cast<char*>(&magic),     4);
    f.read(reinterpret_cast<char*>(&version),   4);
    f.read(reinterpret_cast<char*>(&numScenes), 4);

    if (magic != BATCH_MAGIC)
        throw std::runtime_error("batch: magic incorrecto");

    std::vector<SceneData> scenes(numScenes);
    for (uint32_t i = 0; i < numScenes; ++i) {
        uint32_t numSpheres, numStrings, numPortals, numRays;
        float    baseOmega;
        f.read(reinterpret_cast<char*>(&numSpheres), 4);
        f.read(reinterpret_cast<char*>(&numStrings), 4);
        f.read(reinterpret_cast<char*>(&numPortals), 4);
        f.read(reinterpret_cast<char*>(&numRays),    4);
        f.read(reinterpret_cast<char*>(&baseOmega),  4);

        scenes[i].numRays   = numRays;
        scenes[i].baseOmega = baseOmega;

        scenes[i].spheres.resize(numSpheres);
        f.read(reinterpret_cast<char*>(scenes[i].spheres.data()),
               numSpheres * sizeof(SemanticSphere));

        scenes[i].strings.resize(numStrings);
        f.read(reinterpret_cast<char*>(scenes[i].strings.data()),
               numStrings * sizeof(SemanticString));

        scenes[i].portals.resize(numPortals);
        f.read(reinterpret_cast<char*>(scenes[i].portals.data()),
               numPortals * sizeof(AffinePortal));
    }

    std::cout << "[batch] Cargadas " << numScenes << " escenas\n";
    return scenes;
}

// ============================================================================
// Carga del PTX
// ============================================================================

static std::string loadPTX(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error(std::string("No se puede abrir PTX: ") + path);
    auto sz = f.tellg();
    f.seekg(0);
    std::string s(sz, '\0');
    f.read(s.data(), sz);
    return s;
}

// ============================================================================
// GAS builder (reutilizado por cada escena)
// ============================================================================

static OptixTraversableHandle buildGAS(
    OptixDeviceContext ctx,
    const std::vector<SemanticSphere>& spheres,
    CUdeviceptr& dGas)
{
    const uint32_t N = static_cast<uint32_t>(spheres.size());

    // AABB list
    std::vector<OptixAabb> aabbs(N);
    for (uint32_t i = 0; i < N; ++i) {
        const SemanticSphere& sp = spheres[i];
        float r = sp.radius;
        aabbs[i].minX = sp.center.x - r;
        aabbs[i].minY = sp.center.y - r;
        aabbs[i].minZ = sp.center.z - r;
        aabbs[i].maxX = sp.center.x + r;
        aabbs[i].maxY = sp.center.y + r;
        aabbs[i].maxZ = sp.center.z + r;
    }

    CUdeviceptr dAabb = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAabb), N * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dAabb), aabbs.data(),
                          N * sizeof(OptixAabb), cudaMemcpyHostToDevice));

    OptixBuildInput bi = {};
    bi.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    bi.customPrimitiveArray.aabbBuffers   = &dAabb;
    bi.customPrimitiveArray.numPrimitives = N;
    uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE;
    bi.customPrimitiveArray.flags         = &flags;
    bi.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBuildOptions opts = {};
    opts.buildFlags = OPTIX_BUILD_FLAG_NONE;
    opts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes sizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(ctx, &opts, &bi, 1, &sizes));

    CUdeviceptr dTemp = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTemp), sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGas), sizes.outputSizeInBytes));

    OptixTraversableHandle handle = 0;
    OPTIX_CHECK(optixAccelBuild(ctx, nullptr, &opts, &bi, 1,
                                dTemp, sizes.tempSizeInBytes,
                                dGas, sizes.outputSizeInBytes,
                                &handle, nullptr, 0));

    cudaFree(reinterpret_cast<void*>(dTemp));
    cudaFree(reinterpret_cast<void*>(dAabb));
    return handle;
}

// ============================================================================
// Procesar una escena (OptiX context ya inicializado)
// ============================================================================

static SceneResult processScene(
    OptixDeviceContext     optixCtx,
    OptixPipeline          pipeline,
    OptixProgramGroup      pgRaygen,
    OptixProgramGroup      pgHit,
    OptixProgramGroup      pgMiss,
    const SceneData&       scene,
    cudaEvent_t            evStart,
    cudaEvent_t            evStop)
{
    const uint32_t N = static_cast<uint32_t>(scene.spheres.size());

    // GAS
    CUdeviceptr dGas = 0;
    cudaEventRecord(evStart);
    OptixTraversableHandle gasHandle = buildGAS(optixCtx, scene.spheres, dGas);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(evStop);
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float buildMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&buildMs, evStart, evStop));

    // Buffers GPU
    CUdeviceptr dSpheres = 0, dStrings = 0, dPortals = 0, dResults = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSpheres),
                          N * sizeof(SemanticSphere)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dStrings),
                          scene.strings.size() * sizeof(SemanticString)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPortals),
                          scene.portals.size() * sizeof(AffinePortal)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dResults),
                          scene.numRays * sizeof(SpectralAttentionResult)));

    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSpheres), scene.spheres.data(),
                          N * sizeof(SemanticSphere), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dStrings), scene.strings.data(),
                          scene.strings.size() * sizeof(SemanticString), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dPortals), scene.portals.data(),
                          scene.portals.size() * sizeof(AffinePortal), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(reinterpret_cast<void*>(dResults), 0,
                          scene.numRays * sizeof(SpectralAttentionResult)));

    // SBT
    using RaygenRecord   = SbtRecord<RaygenData>;
    using HitgroupRecord = SbtRecord<HitgroupData>;
    using MissRecord     = SbtRecord<MissData>;

    RaygenRecord   sbtRg  = {};
    HitgroupRecord sbtHit = {};
    MissRecord     sbtMs  = {};
    sbtRg.data.numRays = scene.numRays;

    OPTIX_CHECK(optixSbtRecordPackHeader(pgRaygen, &sbtRg));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgHit,    &sbtHit));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss,   &sbtMs));

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

    // Launch params
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

    // optixLaunch — medir con CUDA events (GPU time real)
    CUDA_CHECK(cudaEventRecord(evStart));
    OPTIX_CHECK(optixLaunch(pipeline, nullptr,
                            dParams, sizeof(InceptionLaunchParams),
                            &sbt, scene.numRays, 1, 1));
    CUDA_CHECK(cudaEventRecord(evStop));
    CUDA_CHECK(cudaEventSynchronize(evStop));
    float launchMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&launchMs, evStart, evStop));

    // Resultado
    SceneResult sr;
    sr.launch_ms = launchMs;
    sr.build_ms  = buildMs;
    sr.results.resize(scene.numRays);
    CUDA_CHECK(cudaMemcpy(sr.results.data(), reinterpret_cast<void*>(dResults),
                          scene.numRays * sizeof(SpectralAttentionResult),
                          cudaMemcpyDeviceToHost));

    sr.hits = 0;
    for (const auto& r : sr.results)
        if (r.traversalDepth > 0) sr.hits++;

    // Cleanup por escena
    cudaFree(reinterpret_cast<void*>(dSbtRg));
    cudaFree(reinterpret_cast<void*>(dSbtHit));
    cudaFree(reinterpret_cast<void*>(dSbtMs));
    cudaFree(reinterpret_cast<void*>(dParams));
    cudaFree(reinterpret_cast<void*>(dGas));
    cudaFree(reinterpret_cast<void*>(dSpheres));
    cudaFree(reinterpret_cast<void*>(dStrings));
    cudaFree(reinterpret_cast<void*>(dPortals));
    cudaFree(reinterpret_cast<void*>(dResults));

    return sr;
}

// ============================================================================
// Guardar resultados del batch
// ============================================================================

static void saveBatchResults(const char* path,
                             const std::vector<SceneResult>& results,
                             uint32_t totalRays)
{
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::string("No se puede escribir: ") + path);

    uint32_t magic     = BRESULT_MAGIC;
    uint32_t numScenes = static_cast<uint32_t>(results.size());
    f.write(reinterpret_cast<char*>(&magic),     4);
    f.write(reinterpret_cast<char*>(&numScenes), 4);
    f.write(reinterpret_cast<char*>(&totalRays), 4);

    for (const auto& r : results) {
        uint32_t nr    = static_cast<uint32_t>(r.results.size());
        f.write(reinterpret_cast<const char*>(&nr),           4);
        f.write(reinterpret_cast<const char*>(&r.launch_ms),  4);
        f.write(reinterpret_cast<const char*>(&r.build_ms),   4);
        f.write(reinterpret_cast<const char*>(&r.hits),       4);
        f.write(reinterpret_cast<const char*>(r.results.data()),
                nr * sizeof(SpectralAttentionResult));
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Uso: batch_runner.exe <ptx_path> <batch_scenes.bin> <batch_results.bin>\n";
        return 1;
    }

    const char* ptxPath      = argv[1];
    const char* batchPath    = argv[2];
    const char* resultsPath  = argv[3];

    try {
        // ── CUDA + OptiX init (UNA SOLA VEZ) ──────────────────────────────
        CUDA_CHECK(cudaSetDevice(0));
        CUDA_CHECK(cudaFree(0));
        CUcontext cuCtx = nullptr;
        cuCtxGetCurrent(&cuCtx);

        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions ctxOpts = {};
        ctxOpts.logCallbackFunction = optixLog;
        ctxOpts.logCallbackLevel    = 2;  // solo errores, no info
        OptixDeviceContext optixCtx = nullptr;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctxOpts, &optixCtx));

        // ── Cargar batch y PTX ────────────────────────────────────────────
        std::vector<SceneData> scenes = loadBatch(batchPath);
        std::string ptxSrc = loadPTX(ptxPath);
        std::cout << "[ptx] " << ptxSrc.size() << " bytes\n";

        // ── Compilar módulo y programa groups (UNA SOLA VEZ) ──────────────
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
        std::cout << "[mod] Modulo compilado\n";

        OptixProgramGroupOptions pgOpts = {};
        OptixProgramGroup pgRaygen = nullptr, pgHit = nullptr, pgMiss = nullptr;

        { OptixProgramGroupDesc d = {};
          d.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
          d.raygen.module = mod;
          d.raygen.entryFunctionName = "__raygen__liquidbit";
          logSz = sizeof(log);
          OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &d, 1, &pgOpts, log, &logSz, &pgRaygen)); }

        { OptixProgramGroupDesc d = {};
          d.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
          d.hitgroup.moduleCH = mod;
          d.hitgroup.entryFunctionNameCH = "__closesthit__semantic_portal";
          d.hitgroup.moduleIS = mod;
          d.hitgroup.entryFunctionNameIS = "__intersection__sphere";
          logSz = sizeof(log);
          OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &d, 1, &pgOpts, log, &logSz, &pgHit)); }

        { OptixProgramGroupDesc d = {};
          d.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
          d.miss.module = mod;
          d.miss.entryFunctionName = "__miss__inception";
          logSz = sizeof(log);
          OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &d, 1, &pgOpts, log, &logSz, &pgMiss)); }

        OptixProgramGroup pgs[] = { pgRaygen, pgHit, pgMiss };
        OptixPipelineLinkOptions linkOpts = {};
        linkOpts.maxTraceDepth = 4;
        logSz = sizeof(log);
        OptixPipeline pipeline = nullptr;
        OPTIX_CHECK(optixPipelineCreate(optixCtx, &pipeOpts, &linkOpts,
            pgs, 3, log, &logSz, &pipeline));
        OPTIX_CHECK(optixPipelineSetStackSize(pipeline, 0, 0, 1024, 1));

        std::cout << "[pipeline] Listo. Procesando " << scenes.size() << " escenas...\n";

        // ── CUDA events para timing preciso ──────────────────────────────
        cudaEvent_t evStart, evStop;
        CUDA_CHECK(cudaEventCreate(&evStart));
        CUDA_CHECK(cudaEventCreate(&evStop));

        // ── Procesar batch ────────────────────────────────────────────────
        std::vector<SceneResult> allResults;
        allResults.reserve(scenes.size());

        uint32_t totalRays = 0;
        float totalLaunchMs = 0.0f, totalBuildMs = 0.0f;
        uint32_t totalHits = 0;

        // Warmup: primera escena no cuenta (GPU warm-up)
        if (!scenes.empty()) {
            processScene(optixCtx, pipeline, pgRaygen, pgHit, pgMiss,
                         scenes[0], evStart, evStop);
            std::cout << "[warmup] Primera escena descartada (GPU warm-up)\n";
        }

        for (size_t i = 0; i < scenes.size(); ++i) {
            SceneResult sr = processScene(optixCtx, pipeline, pgRaygen, pgHit, pgMiss,
                                          scenes[i], evStart, evStop);

            totalRays     += scenes[i].numRays;
            totalLaunchMs += sr.launch_ms;
            totalBuildMs  += sr.build_ms;
            totalHits     += sr.hits;
            allResults.push_back(std::move(sr));

            if ((i + 1) % 10 == 0 || i == scenes.size() - 1) {
                std::cout << "[" << i+1 << "/" << scenes.size() << "] "
                          << "launch=" << allResults.back().launch_ms << " ms, "
                          << "build="  << allResults.back().build_ms  << " ms, "
                          << "hits="   << allResults.back().hits      << "/"
                          << scenes[i].numRays << "\n";
            }
        }

        // ── Guardar y mostrar resumen ─────────────────────────────────────
        saveBatchResults(resultsPath, allResults, totalRays);

        uint32_t ns = static_cast<uint32_t>(scenes.size());
        std::cout << "\n=== RESUMEN BATCH ===\n";
        std::cout << "Escenas procesadas:  " << ns << "\n";
        std::cout << "Rayos totales:       " << totalRays << "\n";
        std::cout << "Launch ms (media):   " << totalLaunchMs / ns << "\n";
        std::cout << "Build  ms (media):   " << totalBuildMs  / ns << "\n";
        std::cout << "Hits (media/escena): " << (float)totalHits / ns << "\n";
        std::cout << "Resultados:          " << resultsPath << "\n";

        // ── Cleanup global ────────────────────────────────────────────────
        CUDA_CHECK(cudaEventDestroy(evStart));
        CUDA_CHECK(cudaEventDestroy(evStop));
        optixPipelineDestroy(pipeline);
        optixProgramGroupDestroy(pgRaygen);
        optixProgramGroupDestroy(pgHit);
        optixProgramGroupDestroy(pgMiss);
        optixModuleDestroy(mod);
        optixDeviceContextDestroy(optixCtx);

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[FATAL] " << e.what() << "\n";
        return 1;
    }
}
