/**
 * @file tests/test_inception_pipeline.cpp
 * @brief Test de integración del pipeline OptiX String-Inception real.
 *
 * Este test:
 *   1. Carga spectral_kernels.ptx desde disco (generado por CMake)
 *   2. Crea el pipeline OptiX con los 3 entry points del Inception Engine:
 *        __raygen__spectral, __closesthit__semantic_portal, __miss__inception,
 *        __intersection__sphere
 *   3. Construye un GAS con 5 SemanticSpheres en posiciones conocidas
 *   4. Lanza 16 rayos desde el origen con omega = π/4
 *   5. Verifica que al menos un rayo golpeó una esfera (accumulated > 0)
 *   6. Imprime tabla de resultados
 *
 * RESULTADO ESPERADO:
 *   - Rayos dirigidos hacia las esferas → attentionWeight > 0
 *   - Rayos que no golpean nada → attentionWeight = 0, energyRemaining = 1.0
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "spectral_resonance.h"
#include "token_geometry.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <cassert>

// ============================================================================
// Utilidades
// ============================================================================

static void checkCuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " — " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}
#define CUDA_CHECK(x) checkCuda((x), __FILE__, __LINE__)

static void checkOptix(OptixResult res, const char* file, int line) {
    if (res != OPTIX_SUCCESS) {
        std::cerr << "OptiX error at " << file << ":" << line
                  << " — code " << (int)res << "\n";
        std::exit(1);
    }
}
#define OPTIX_CHECK(x) checkOptix((x), __FILE__, __LINE__)

// Callback de log de OptiX — imprime solo warnings y errores
static void optixLogCb(unsigned level, const char* tag,
                       const char* msg, void* /*cbdata*/) {
    if (level <= 2) {
        std::cerr << "[OptiX][" << tag << "] " << msg << "\n";
    }
}

// ============================================================================
// SBT records
// ============================================================================

template <typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T    data;
};

struct RaygenData  { uint32_t numRays; };
struct HitgroupData{ uint32_t pad; };   // datos de primitiva vienen de c_params.spheres
struct MissData    { uint32_t pad; };

// ============================================================================
// Carga PTX desde archivo
// ============================================================================

static std::string loadPTX(const char* path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error(std::string("No se puede abrir: ") + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    // ----------------------------------------------------------
    // Ruta al PTX: argumento o ruta por defecto junto al exe
    // ----------------------------------------------------------
    const char* ptxPath = (argc > 1) ? argv[1] : "spectral_kernels.ptx";

    std::cout << "=== SpectralAI String-Inception Pipeline Test ===\n";
    std::cout << "PTX: " << ptxPath << "\n\n";

    // ----------------------------------------------------------
    // 0. CUDA + OptiX init
    // ----------------------------------------------------------
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));  // fuerza inicialización del runtime

    CUcontext cuCtx = nullptr;
    cuCtxGetCurrent(&cuCtx);  // el runtime ya creó el contexto

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions ctxOpts = {};
    ctxOpts.logCallbackFunction = optixLogCb;
    ctxOpts.logCallbackLevel    = 4;

    OptixDeviceContext optixCtx = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctxOpts, &optixCtx));
    std::cout << "[OK] OptiX context creado\n";

    // ----------------------------------------------------------
    // 1. Cargar PTX
    // ----------------------------------------------------------
    std::string ptxSrc = loadPTX(ptxPath);
    std::cout << "[OK] PTX cargado (" << ptxSrc.size() << " bytes)\n";

    // ----------------------------------------------------------
    // 2. Crear módulo
    // ----------------------------------------------------------
    OptixModuleCompileOptions modOpts = {};
    modOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    modOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    modOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipeOpts = {};
    pipeOpts.usesMotionBlur                   = false;
    pipeOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeOpts.numPayloadValues                 = 4;   // omega, accumulated, depth, hitCount
    pipeOpts.numAttributeValues               = 2;
    pipeOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    pipeOpts.pipelineLaunchParamsVariableName = "c_params";  // coincide con spectral_kernels.cu
    pipeOpts.usesPrimitiveTypeFlags           = OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    char log[4096];
    size_t logSz = sizeof(log);

    OptixModule mod = nullptr;
    OptixResult modRes = optixModuleCreate(
        optixCtx, &modOpts, &pipeOpts,
        ptxSrc.c_str(), ptxSrc.size(),
        log, &logSz, &mod);
    if (logSz > 1) std::cout << "[ModuleLog] " << log << "\n";
    OPTIX_CHECK(modRes);
    std::cout << "[OK] Módulo OptiX compilado\n";

    // ----------------------------------------------------------
    // 3. Program groups
    // ----------------------------------------------------------
    OptixProgramGroup pgRaygen = nullptr, pgHit = nullptr, pgMiss = nullptr;
    // OptiX 9.1: options NO acepta nullptr — pasar struct vacía (valores por defecto)
    OptixProgramGroupOptions pgOpts = {};

    // Raygen
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module         = mod;
        desc.raygen.entryFunctionName = "__raygen__spectral";
        logSz = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &desc, 1,
            &pgOpts, log, &logSz, &pgRaygen));
        if (logSz > 1) std::cerr << "[PG raygen] " << log << "\n";
    }

    // HitGroup: closesthit + intersection
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                          = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH             = mod;
        desc.hitgroup.entryFunctionNameCH  = "__closesthit__semantic_portal";
        desc.hitgroup.moduleIS             = mod;
        desc.hitgroup.entryFunctionNameIS  = "__intersection__sphere";
        logSz = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &desc, 1,
            &pgOpts, log, &logSz, &pgHit));
        if (logSz > 1) std::cerr << "[PG hitgroup] " << log << "\n";
    }

    // Miss
    {
        OptixProgramGroupDesc desc = {};
        desc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module           = mod;
        desc.miss.entryFunctionName = "__miss__inception";
        logSz = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &desc, 1,
            &pgOpts, log, &logSz, &pgMiss));
        if (logSz > 1) std::cerr << "[PG miss] " << log << "\n";
    }
    std::cout << "[OK] Program groups creados (raygen, hitgroup, miss)\n";

    // ----------------------------------------------------------
    // 4. Pipeline
    // ----------------------------------------------------------
    OptixProgramGroup pgs[] = { pgRaygen, pgHit, pgMiss };
    OptixPipelineLinkOptions linkOpts = {};
    linkOpts.maxTraceDepth = 4;

    OptixPipeline pipeline = nullptr;
    logSz = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(optixCtx, &pipeOpts, &linkOpts,
        pgs, 3, log, &logSz, &pipeline));
    if (logSz > 1) std::cout << "[PipelineLog] " << log << "\n";

    OPTIX_CHECK(optixPipelineSetStackSize(pipeline,
        0,      // directCallableStackSizeFromTraversal
        0,      // directCallableStackSizeFromState
        1024,   // continuationStackSize
        1));    // maxTraversableGraphDepth (1 = solo GAS, sin IAS anidados en este test)

    std::cout << "[OK] Pipeline compilado\n";

    // ----------------------------------------------------------
    // 5. Datos de prueba: 5 esferas semánticas conocidas
    // ----------------------------------------------------------
    const uint32_t NUM_SPHERES  = 5;
    const uint32_t NUM_RAYS     = 16;
    const float    BASE_OMEGA   = 3.14159265f / 4.0f;  // π/4

    // Posiciones conocidas: a lo largo de los 5 ejes principales
    // Los rayos del fibonacci sphere los cubren bien
    SemanticSphere spheres[NUM_SPHERES] = {};
    const float3 centers[NUM_SPHERES] = {
        { 2.0f,  0.0f,  0.0f},  // +X
        {-2.0f,  0.0f,  0.0f},  // -X
        { 0.0f,  2.0f,  0.0f},  // +Y
        { 0.0f, -2.0f,  0.0f},  // -Y
        { 0.0f,  0.0f,  2.0f},  // +Z
    };

    for (uint32_t i = 0; i < NUM_SPHERES; ++i) {
        spheres[i].center        = centers[i];
        spheres[i].radius        = 1.5f;  // Radio grande → garantiza hits con 16 rayos fibonacci
        spheres[i].instanceId    = i;
        spheres[i].childIAS      = 0;       // hojas (sin IAS hijo)
        spheres[i].depth         = 3;       // depth == INCEPTION_MAX_DEPTH-1 → activa resonancia
        spheres[i].frequencyBias = (float)i * 0.3f;
    }

    // SemanticStrings (una por esfera, relación 1:1 con primitiveIdx)
    SemanticString strings[NUM_SPHERES] = {};
    for (uint32_t i = 0; i < NUM_SPHERES; ++i) {
        strings[i].stringId      = i;
        strings[i].position      = centers[i];
        // Coeficientes Fourier simples: a_k = 1/k, b_k = 0
        for (uint32_t k = 0; k < RESONANCE_NUM_MODES; ++k) {
            strings[i].resonance.a[k] = 1.0f / (float)(k + 1);
            strings[i].resonance.b[k] = 0.0f;
        }
        strings[i].resonance.numModes   = RESONANCE_NUM_MODES;
        strings[i].resonance.outputScale = 1.0f;
        strings[i].resonance.semanticTag = i;
    }

    // Portales afines (identidad para este test)
    AffinePortal portals[INCEPTION_MAX_DEPTH] = {};
    for (int d = 0; d < INCEPTION_MAX_DEPTH; ++d) {
        portals[d].rows[0] = make_float4(1, 0, 0, 0);
        portals[d].rows[1] = make_float4(0, 1, 0, 0);
        portals[d].rows[2] = make_float4(0, 0, 1, 0);
        portals[d].rows[3] = make_float4(0, 0, 0, 1);
    }

    // ----------------------------------------------------------
    // 6. Buffers GPU
    // ----------------------------------------------------------
    // Esferas
    SemanticSphere* dSpheres = nullptr;
    CUDA_CHECK(cudaMalloc(&dSpheres, NUM_SPHERES * sizeof(SemanticSphere)));
    CUDA_CHECK(cudaMemcpy(dSpheres, spheres,
        NUM_SPHERES * sizeof(SemanticSphere), cudaMemcpyHostToDevice));

    // Strings
    SemanticString* dStrings = nullptr;
    CUDA_CHECK(cudaMalloc(&dStrings, NUM_SPHERES * sizeof(SemanticString)));
    CUDA_CHECK(cudaMemcpy(dStrings, strings,
        NUM_SPHERES * sizeof(SemanticString), cudaMemcpyHostToDevice));

    // Portales
    AffinePortal* dPortals = nullptr;
    CUDA_CHECK(cudaMalloc(&dPortals, INCEPTION_MAX_DEPTH * sizeof(AffinePortal)));
    CUDA_CHECK(cudaMemcpy(dPortals, portals,
        INCEPTION_MAX_DEPTH * sizeof(AffinePortal), cudaMemcpyHostToDevice));

    // Resultados
    SpectralAttentionResult* dResults = nullptr;
    CUDA_CHECK(cudaMalloc(&dResults, NUM_RAYS * sizeof(SpectralAttentionResult)));
    CUDA_CHECK(cudaMemset(dResults, 0, NUM_RAYS * sizeof(SpectralAttentionResult)));

    // ----------------------------------------------------------
    // 7. Construir GAS (Geometry Acceleration Structure)
    // ----------------------------------------------------------

    // AABBs de las esferas
    std::vector<OptixAabb> aabbs(NUM_SPHERES);
    for (uint32_t i = 0; i < NUM_SPHERES; ++i) {
        const float r = spheres[i].radius;  // 1.5 → AABBs grandes, fáciles de alcanzar
        const float3& c = spheres[i].center;
        aabbs[i] = { c.x-r, c.y-r, c.z-r, c.x+r, c.y+r, c.z+r };
    }

    CUdeviceptr dAabbs = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dAabbs),
        NUM_SPHERES * sizeof(OptixAabb)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dAabbs), aabbs.data(),
        NUM_SPHERES * sizeof(OptixAabb), cudaMemcpyHostToDevice));

    unsigned int geomFlags = OPTIX_GEOMETRY_FLAG_NONE;
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    buildInput.customPrimitiveArray.aabbBuffers    = &dAabbs;
    buildInput.customPrimitiveArray.numPrimitives  = NUM_SPHERES;
    buildInput.customPrimitiveArray.strideInBytes  = sizeof(OptixAabb);
    buildInput.customPrimitiveArray.flags          = &geomFlags;
    buildInput.customPrimitiveArray.numSbtRecords  = 1;

    OptixAccelBuildOptions accelOpts = {};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gasSizes = {};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optixCtx, &accelOpts, &buildInput, 1, &gasSizes));

    CUdeviceptr dTemp = 0, dGasOutput = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dTemp),    gasSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dGasOutput), gasSizes.outputSizeInBytes));

    OptixTraversableHandle gasHandle = 0;
    OPTIX_CHECK(optixAccelBuild(
        optixCtx, nullptr,
        &accelOpts, &buildInput, 1,
        dTemp, gasSizes.tempSizeInBytes,
        dGasOutput, gasSizes.outputSizeInBytes,
        &gasHandle, nullptr, 0));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dTemp)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(dAabbs)));
    std::cout << "[OK] GAS construido (" << NUM_SPHERES << " esferas, "
              << gasSizes.outputSizeInBytes << " bytes)\n";

    // ----------------------------------------------------------
    // 8. Shader Binding Table
    // ----------------------------------------------------------
    SbtRecord<RaygenData>   sbtRaygen  = {};
    SbtRecord<HitgroupData> sbtHit     = {};
    SbtRecord<MissData>     sbtMiss    = {};

    OPTIX_CHECK(optixSbtRecordPackHeader(pgRaygen, sbtRaygen.header));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgHit,    sbtHit.header));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss,   sbtMiss.header));
    sbtRaygen.data.numRays = NUM_RAYS;

    CUdeviceptr dSbtRaygen = 0, dSbtHit = 0, dSbtMiss = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbtRaygen), sizeof(sbtRaygen)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbtHit),    sizeof(sbtHit)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSbtMiss),   sizeof(sbtMiss)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbtRaygen), &sbtRaygen, sizeof(sbtRaygen), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbtHit),    &sbtHit,    sizeof(sbtHit),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dSbtMiss),   &sbtMiss,   sizeof(sbtMiss),   cudaMemcpyHostToDevice));

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord                = dSbtRaygen;
    sbt.hitgroupRecordBase          = dSbtHit;
    sbt.hitgroupRecordStrideInBytes = sizeof(sbtHit);
    sbt.hitgroupRecordCount         = 1;
    sbt.missRecordBase              = dSbtMiss;
    sbt.missRecordStrideInBytes     = sizeof(sbtMiss);
    sbt.missRecordCount             = 1;

    std::cout << "[OK] SBT construida\n";

    // ----------------------------------------------------------
    // 9. Launch params (InceptionLaunchParams → c_params en device)
    // ----------------------------------------------------------
    InceptionLaunchParams hParams = {};
    hParams.topLevelIAS = gasHandle;
    hParams.spheres     = dSpheres;
    hParams.portals     = dPortals;
    hParams.strings     = dStrings;
    hParams.results     = dResults;
    hParams.baseOmega   = BASE_OMEGA;
    hParams.numRays     = NUM_RAYS;
    hParams.numStrings  = NUM_SPHERES;

    CUdeviceptr dParams = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParams), sizeof(InceptionLaunchParams)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dParams), &hParams,
        sizeof(InceptionLaunchParams), cudaMemcpyHostToDevice));

    // ----------------------------------------------------------
    // 10. optixLaunch
    // ----------------------------------------------------------
    std::cout << "\nLanzando " << NUM_RAYS << " rayos con ω = π/4...\n";

    OPTIX_CHECK(optixLaunch(
        pipeline,
        nullptr,    // stream
        dParams, sizeof(InceptionLaunchParams),
        &sbt,
        NUM_RAYS,   // width
        1,          // height
        1));        // depth

    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "[OK] optixLaunch completado\n\n";

    // ----------------------------------------------------------
    // 11. Leer y mostrar resultados
    // ----------------------------------------------------------
    std::vector<SpectralAttentionResult> results(NUM_RAYS);
    CUDA_CHECK(cudaMemcpy(results.data(), dResults,
        NUM_RAYS * sizeof(SpectralAttentionResult), cudaMemcpyDeviceToHost));

    std::cout << "Ray | attentionWeight | finalOmega | depth | hits\n";
    std::cout << "----|-----------------|-----------:|------:|-----\n";

    int hitsTotal = 0;
    for (uint32_t i = 0; i < NUM_RAYS; ++i) {
        const auto& r = results[i];
        // hitCount está en energyRemaining: (hitCount>0) → 1/hitCount, else 1.0f
        // Detectamos hit por traversalDepth > 0 (más fiable que attentionWeight que puede ser negativo)
        const bool isHit = (r.traversalDepth > 0);
        if (isHit) hitsTotal++;
        std::cout << "  " << i
                  << " |  " << r.attentionWeight
                  << "  |  " << r.finalOmega
                  << "  |  " << r.traversalDepth
                  << "  |  " << (isHit ? "HIT" : "miss")
                  << "\n";
    }

    std::cout << "\nRayos con hit: " << hitsTotal << " / " << NUM_RAYS << "\n";

    // ----------------------------------------------------------
    // 12. Verificación mínima
    // ----------------------------------------------------------
    if (hitsTotal == 0) {
        std::cerr << "\n[FAIL] Ningún rayo golpeó una esfera.\n";
        std::cerr << "       Posible error en el pipeline o en la escena de prueba.\n";
    } else {
        std::cout << "[PASS] Pipeline String-Inception funcional — "
                  << hitsTotal << " hits con resonancia Fourier calculada.\n";
    }

    // ----------------------------------------------------------
    // 13. Cleanup
    // ----------------------------------------------------------
    cudaFree(reinterpret_cast<void*>(dSbtRaygen));
    cudaFree(reinterpret_cast<void*>(dSbtHit));
    cudaFree(reinterpret_cast<void*>(dSbtMiss));
    cudaFree(reinterpret_cast<void*>(dParams));
    cudaFree(reinterpret_cast<void*>(dGasOutput));
    cudaFree(dSpheres);
    cudaFree(dStrings);
    cudaFree(dPortals);
    cudaFree(dResults);

    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(pgRaygen);
    optixProgramGroupDestroy(pgHit);
    optixProgramGroupDestroy(pgMiss);
    optixModuleDestroy(mod);
    optixDeviceContextDestroy(optixCtx);

    std::cout << "[OK] Cleanup completo\n";
    return (hitsTotal > 0) ? 0 : 1;
}
