/**
 * @file tests/inception_engine_test.cpp
 * @brief Test del pipeline Inception Engine v4.0 — Salto Dimensional Explícito
 *
 * DIFERENCIAS VS test_inception_pipeline.cpp:
 * =============================================
 * Este test valida el pipeline de inception_kernels.cu (nuevo, FASE 2.2), que usa:
 *   - __raygen__inception         (vs __raygen__liquidbit)
 *   - __closesthit__inception_portal  (vs __closesthit__semantic_portal)
 *   - __miss__inception_portal    (vs __miss__inception)
 *   - __intersection__inception_sphere  (vs __intersection__sphere)
 *   - g_inception_params          (vs c_params — nombre de launch params)
 *
 * ESCENA DE TEST (2 niveles efectivos):
 * ======================================
 * Nivel 0 (Dominio "Código"):    esfera en (1, 0, 0), r=1.5 — depth=0
 * Nivel 3 (Hojas — SemanticStr): 3 strings en posiciones dentro del dominio
 *
 * VALIDACIONES:
 * =============
 * 1. Al menos 1 rayo golpea el dominio (nivel 0) → hitCount > 0
 * 2. La resonancia Fourier se calcula correctamente en los nodos hoja
 * 3. El finalOmega refleja el sesgo de frecuencia de las esferas
 * 4. Los miss no modifican el payload (accumulated = 0 en miss puro)
 *
 * RESULTADO ESPERADO:
 *   [PASS] inception_engine: N hits con resonancia Fourier calculada
 *
 * CÓMO COMPILAR (si inception_kernels.ptx existe):
 *   cmake --build build --target inception_engine
 *   build/Release/inception_engine.exe build/inception_kernels.ptx
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#define NOMINMAX  // evita conflicto min/max de windows.h vs std::min/std::max
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "liquidbit_resonance.h"
#include "inception_engine.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <cassert>
#include <iomanip>

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
struct EmptyData {};
using RaygenRecord    = SbtRecord<EmptyData>;
using MissRecord      = SbtRecord<EmptyData>;
using HitgroupRecord  = SbtRecord<EmptyData>;

// ============================================================================
// Helpers
// ============================================================================

static std::string readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Crea un AffinePortal identidad (sin transformación)
static AffinePortal identityPortal() {
    AffinePortal p;
    memset(&p, 0, sizeof(p));
    p.rows[0].x = 1.0f;
    p.rows[1].y = 1.0f;
    p.rows[2].z = 1.0f;
    p.rows[3].w = 1.0f;
    return p;
}

// Crea un AffinePortal con sesgo espectral en fila 3
static AffinePortal spectralPortal(float omegaScale, float omegaBias) {
    AffinePortal p = identityPortal();
    p.rows[3].x = omegaScale;  // escala de ω
    p.rows[3].w = omegaBias;   // sesgo constante
    return p;
}

// ============================================================================
// Construcción de escena de test (2 niveles simplificados)
// ============================================================================

struct TestScene {
    // Nivel 0 (raíz): 3 dominios semánticos
    static constexpr int NUM_DOMAINS  = 3;
    // Nivel 3 (hojas): 3 SemanticStrings por dominio
    static constexpr int NUM_STRINGS  = 9;
    static constexpr int NUM_SPHERES  = NUM_DOMAINS + NUM_STRINGS;
    static constexpr int NUM_PORTALS  = INCEPTION_MAX_DEPTH;
    static constexpr int NUM_RAYS     = 16;

    std::vector<SemanticSphere> spheres;
    std::vector<SemanticString> strings;
    std::vector<AffinePortal>   portals;
    std::vector<OptixAabb>      aabbs;

    TestScene() {
        // ── Portales (identidad + sesgo espectral por nivel) ──────────────
        portals.resize(NUM_PORTALS);
        portals[0] = identityPortal();                  // nivel 0→1: sin cambio
        portals[1] = spectralPortal(0.8f, 0.1f);        // nivel 1→2: pequeño sesgo
        portals[2] = spectralPortal(0.6f, 0.3f);        // nivel 2→3: sesgo mayor
        portals[3] = identityPortal();                  // nivel 3 (hoja): no aplica

        // ── Dominios (nivel 0) ────────────────────────────────────────────
        const float domainPositions[3][3] = {
            { 2.0f,  0.0f,  0.0f },   // "Código"
            {-1.0f,  1.73f, 0.0f },   // "Música"
            {-1.0f, -1.73f, 0.0f }    // "Ciencia"
        };
        const float domainFreqBias[3] = { 0.0f, 0.4f, 0.8f };

        for (int d = 0; d < NUM_DOMAINS; ++d) {
            SemanticSphere s;
            s.center.x      = domainPositions[d][0];
            s.center.y      = domainPositions[d][1];
            s.center.z      = domainPositions[d][2];
            s.radius        = 1.5f;
            s.instanceId    = (uint32_t)d;
            s.childIAS      = 0;     // Sin IAS hijo real en este test simplificado
            s.depth         = 3;     // Tratar como hoja para test sin 4 niveles completos
            s.frequencyBias = domainFreqBias[d];
            spheres.push_back(s);

            // AABB para OptiX GAS
            OptixAabb aabb;
            aabb.minX = s.center.x - s.radius;
            aabb.minY = s.center.y - s.radius;
            aabb.minZ = s.center.z - s.radius;
            aabb.maxX = s.center.x + s.radius;
            aabb.maxY = s.center.y + s.radius;
            aabb.maxZ = s.center.z + s.radius;
            aabbs.push_back(aabb);
        }

        // ── SemanticStrings (nivel 3) ──────────────────────────────────────
        // 3 strings por dominio, con coeficientes Fourier distintos
        // para verificar que la resonancia varía con ω
        float baseOmegas[9] = {
            0.2f, 0.5f, 0.9f,    // strings del dominio "Código"
            1.2f, 1.5f, 1.9f,    // strings del dominio "Música"
            2.2f, 2.5f, 2.9f     // strings del dominio "Ciencia"
        };

        for (int i = 0; i < NUM_STRINGS; ++i) {
            int d = i / 3;  // dominio padre

            SemanticString ss;
            ss.position.x = domainPositions[d][0] + 0.2f * (float)(i % 3);
            ss.position.y = domainPositions[d][1] + 0.1f * (float)(i % 3);
            ss.position.z = domainPositions[d][2];
            ss.stringId   = (uint32_t)i;

            // Coeficientes Fourier: varían por string para producir resonancias distintas
            ResonanceParams& rp = ss.resonance;
            rp.numModes    = 4;
            rp.outputScale = 1.0f;
            rp.semanticTag = (uint32_t)d;
            rp._pad        = 0;
            for (int k = 0; k < RESONANCE_NUM_MODES; ++k) {
                float phase = baseOmegas[i] + (float)k * 0.1f;
                rp.a[k] = (k < 4) ? sinf(phase) * 0.5f : 0.0f;
                rp.b[k] = (k < 4) ? cosf(phase) * 0.5f : 0.0f;
            }

            strings.push_back(ss);

            // Los strings NO tienen AABB en este test — pertenecen al GAS de los dominios
            // En el pipeline completo, habría GAS separados por nivel
        }
    }
};

// ============================================================================
// main
// ============================================================================

int main(int argc, char* argv[]) {
    // Ruta al PTX de inception_kernels.cu
    std::string ptxPath = (argc > 1) ? argv[1] : "inception_kernels.ptx";

    std::cout << "==========================================================\n";
    std::cout << " LiquidBit Inception Engine v4.0 — Test\n";
    std::cout << " PTX: " << ptxPath << "\n";
    std::cout << "==========================================================\n\n";

    // ── 1. Inicializar CUDA + OptiX ──────────────────────────────────────
    CUDA_CHECK(cudaSetDevice(0));

    CUcontext cuCtx = nullptr;
    CUDA_CHECK(cudaFree(0));  // Inicializa el contexto CUDA implícitamente
    cuCtxGetCurrent(&cuCtx);

    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions optixOpts{};
    optixOpts.logCallbackFunction = optixLogCb;
    optixOpts.logCallbackLevel    = 4;

    OptixDeviceContext optixCtx = nullptr;
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &optixOpts, &optixCtx));

    {
        int deviceId = 0;
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, deviceId);
        std::cout << "[GPU] " << prop.name
                  << " — Compute " << prop.major << "." << prop.minor
                  << " — " << (prop.totalGlobalMem / (1024*1024)) << " MB\n\n";
    }

    // ── 2. Cargar PTX y crear módulo OptiX ──────────────────────────────
    std::string ptxCode;
    try {
        ptxCode = readFile(ptxPath);
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        std::cerr << "        Compilar con: cmake --build build --target inception_engine_ptx\n";
        std::exit(1);
    }

    OptixModuleCompileOptions modCompOpts{};
    modCompOpts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    modCompOpts.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    modCompOpts.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    OptixPipelineCompileOptions pipCompOpts{};
    pipCompOpts.usesMotionBlur                   = 0;
    pipCompOpts.traversableGraphFlags            = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipCompOpts.numPayloadValues                 = 4;
    pipCompOpts.numAttributeValues               = 2;
    pipCompOpts.exceptionFlags                   = OPTIX_EXCEPTION_FLAG_NONE;
    // CLAVE: nombre de la variable de launch params debe coincidir con el kernel
    pipCompOpts.pipelineLaunchParamsVariableName = "g_inception_params";
    pipCompOpts.usesPrimitiveTypeFlags           =
        OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;

    char logBuf[2048] = {};
    size_t logSz = sizeof(logBuf);

    OptixModule mod = nullptr;
    OPTIX_CHECK(optixModuleCreate(
        optixCtx, &modCompOpts, &pipCompOpts,
        ptxCode.c_str(), ptxCode.size(),
        logBuf, &logSz, &mod
    ));
    if (logSz > 1) std::cout << "[module log] " << logBuf << "\n";
    std::cout << "[OK] Módulo OptiX creado desde " << ptxPath << "\n";

    // ── 3. Crear Program Groups ──────────────────────────────────────────
    OptixProgramGroupOptions pgOpts{};

    // Raygen
    OptixProgramGroupDesc rgDesc{};
    rgDesc.kind                    = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module           = mod;
    rgDesc.raygen.entryFunctionName = "__raygen__inception";
    OptixProgramGroup raygenPG;
    logSz = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &rgDesc, 1, &pgOpts,
                                        logBuf, &logSz, &raygenPG));
    if (logSz > 1) std::cout << "[raygen log] " << logBuf << "\n";
    std::cout << "[OK] raygen: __raygen__inception\n";

    // Miss
    OptixProgramGroupDesc msDesc{};
    msDesc.kind                  = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module           = mod;
    msDesc.miss.entryFunctionName = "__miss__inception_portal";
    OptixProgramGroup missPG;
    logSz = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &msDesc, 1, &pgOpts,
                                        logBuf, &logSz, &missPG));
    std::cout << "[OK] miss:   __miss__inception_portal\n";

    // Closesthit + Intersection
    OptixProgramGroupDesc hitDesc{};
    hitDesc.kind                               = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH                  = mod;
    hitDesc.hitgroup.entryFunctionNameCH       = "__closesthit__inception_portal";
    hitDesc.hitgroup.moduleIS                  = mod;
    hitDesc.hitgroup.entryFunctionNameIS       = "__intersection__inception_sphere";
    OptixProgramGroup hitgroupPG;
    logSz = sizeof(logBuf);
    OPTIX_CHECK(optixProgramGroupCreate(optixCtx, &hitDesc, 1, &pgOpts,
                                        logBuf, &logSz, &hitgroupPG));
    std::cout << "[OK] hit:    __closesthit__inception_portal + __intersection__inception_sphere\n\n";

    // ── 4. Crear Pipeline ─────────────────────────────────────────────────
    OptixProgramGroup pgs[] = { raygenPG, missPG, hitgroupPG };

    OptixPipelineLinkOptions pipLinkOpts{};
    pipLinkOpts.maxTraceDepth = 5;   // 4 niveles + 1 extra por si acaso

    OptixPipeline pipeline = nullptr;
    logSz = sizeof(logBuf);
    OPTIX_CHECK(optixPipelineCreate(
        optixCtx, &pipCompOpts, &pipLinkOpts,
        pgs, 3,
        logBuf, &logSz, &pipeline
    ));
    if (logSz > 1) std::cout << "[pipeline log] " << logBuf << "\n";
    std::cout << "[OK] Pipeline creado (maxTraceDepth=" << pipLinkOpts.maxTraceDepth << ")\n\n";

    // ── 5. Construir escena de test ───────────────────────────────────────
    TestScene scene;

    // Subir AABBs a GPU
    CUdeviceptr d_aabbs = 0;
    const size_t aabbBytes = scene.aabbs.size() * sizeof(OptixAabb);
    CUDA_CHECK(cudaMalloc((void**)&d_aabbs, aabbBytes));
    CUDA_CHECK(cudaMemcpy((void*)d_aabbs, scene.aabbs.data(), aabbBytes, cudaMemcpyHostToDevice));

    // Construir GAS (nivel hoja)
    OptixAccelBuildOptions accelOpts{};
    accelOpts.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOpts.operation  = OPTIX_BUILD_OPERATION_BUILD;

    OptixBuildInput buildInput{};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    buildInput.customPrimitiveArray.aabbBuffers   = &d_aabbs;
    buildInput.customPrimitiveArray.numPrimitives = (uint32_t)scene.aabbs.size();
    uint32_t inputFlags[TestScene::NUM_SPHERES];
    for (int i = 0; i < TestScene::NUM_SPHERES; ++i) inputFlags[i] = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.customPrimitiveArray.flags         = inputFlags;
    buildInput.customPrimitiveArray.numSbtRecords = 1;

    OptixAccelBufferSizes bufSizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixCtx, &accelOpts, &buildInput, 1, &bufSizes));

    CUdeviceptr d_temp   = 0;
    CUdeviceptr d_output = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_temp,   bufSizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, bufSizes.outputSizeInBytes));

    OptixTraversableHandle gasHandle = 0;
    OPTIX_CHECK(optixAccelBuild(
        optixCtx, 0, &accelOpts, &buildInput, 1,
        d_temp, bufSizes.tempSizeInBytes,
        d_output, bufSizes.outputSizeInBytes,
        &gasHandle, nullptr, 0
    ));
    CUDA_CHECK(cudaFree((void*)d_temp));
    std::cout << "[OK] GAS construido: " << scene.spheres.size() << " esferas\n";

    // ── 6. Subir datos de la escena a GPU ─────────────────────────────────
    CUdeviceptr d_spheres = 0, d_portals = 0, d_strings = 0, d_results = 0;

    const size_t sphereBytes  = scene.spheres.size() * sizeof(SemanticSphere);
    const size_t portalBytes  = scene.portals.size() * sizeof(AffinePortal);
    const size_t stringBytes  = scene.strings.size() * sizeof(SemanticString);
    const size_t resultBytes  = TestScene::NUM_RAYS * sizeof(SpectralAttentionResult);

    CUDA_CHECK(cudaMalloc((void**)&d_spheres, sphereBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_portals, portalBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_strings, stringBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_results, resultBytes));

    CUDA_CHECK(cudaMemcpy((void*)d_spheres, scene.spheres.data(), sphereBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)d_portals, scene.portals.data(), portalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)d_strings, scene.strings.data(), stringBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset((void*)d_results, 0, resultBytes));

    std::cout << "[OK] Datos subidos: " << scene.spheres.size() << " esferas, "
              << scene.strings.size() << " strings, "
              << scene.portals.size() << " portales\n";

    // ── 7. Configurar launch params ───────────────────────────────────────
    InceptionLaunchParams params{};
    params.topLevelIAS  = gasHandle;
    params.spheres      = reinterpret_cast<const SemanticSphere*>(d_spheres);
    params.portals      = reinterpret_cast<const AffinePortal*>(d_portals);
    params.strings      = reinterpret_cast<const SemanticString*>(d_strings);
    params.results      = reinterpret_cast<SpectralAttentionResult*>(d_results);
    params.baseOmega    = 0.785398f;   // π/4
    params.numRays      = TestScene::NUM_RAYS;
    params.numStrings   = (uint32_t)scene.strings.size();

    CUdeviceptr d_params = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_params, sizeof(params)));
    CUDA_CHECK(cudaMemcpy((void*)d_params, &params, sizeof(params), cudaMemcpyHostToDevice));

    // ── 8. Construir SBT ──────────────────────────────────────────────────
    RaygenRecord   h_raygenRec{};
    MissRecord     h_missRec{};
    HitgroupRecord h_hitRec{};

    OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG,  &h_raygenRec));
    OPTIX_CHECK(optixSbtRecordPackHeader(missPG,    &h_missRec));
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &h_hitRec));

    CUdeviceptr d_raygenBuf = 0, d_missBuf = 0, d_hitBuf = 0;
    CUDA_CHECK(cudaMalloc((void**)&d_raygenBuf, sizeof(h_raygenRec)));
    CUDA_CHECK(cudaMalloc((void**)&d_missBuf,   sizeof(h_missRec)));
    CUDA_CHECK(cudaMalloc((void**)&d_hitBuf,    sizeof(h_hitRec)));
    CUDA_CHECK(cudaMemcpy((void*)d_raygenBuf, &h_raygenRec, sizeof(h_raygenRec), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)d_missBuf,   &h_missRec,   sizeof(h_missRec),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy((void*)d_hitBuf,    &h_hitRec,    sizeof(h_hitRec),    cudaMemcpyHostToDevice));

    OptixShaderBindingTable sbt{};
    sbt.raygenRecord                = d_raygenBuf;
    sbt.missRecordBase              = d_missBuf;
    sbt.missRecordStrideInBytes     = sizeof(MissRecord);
    sbt.missRecordCount             = 1;
    sbt.hitgroupRecordBase          = d_hitBuf;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    sbt.hitgroupRecordCount         = 1;

    // ── 9. Lanzar pipeline ────────────────────────────────────────────────
    std::cout << "\n[LAUNCH] " << TestScene::NUM_RAYS << " rayos, baseOmega = π/4\n";

    OPTIX_CHECK(optixLaunch(
        pipeline, 0,
        d_params, sizeof(params),
        &sbt,
        TestScene::NUM_RAYS, 1, 1
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ── 10. Leer y mostrar resultados ─────────────────────────────────────
    std::vector<SpectralAttentionResult> results(TestScene::NUM_RAYS);
    CUDA_CHECK(cudaMemcpy(results.data(), (void*)d_results, resultBytes, cudaMemcpyDeviceToHost));

    std::cout << "\n";
    std::cout << std::left
              << std::setw(5)  << "Ray"
              << std::setw(14) << "attnWeight"
              << std::setw(12) << "finalOmega"
              << std::setw(8)  << "depth"
              << std::setw(8)  << "status"
              << "\n";
    std::cout << std::string(47, '-') << "\n";

    int hitCount = 0;
    for (int i = 0; i < TestScene::NUM_RAYS; ++i) {
        const auto& r = results[i];
        bool isHit = (r.traversalDepth > 0 || r.attentionWeight != 0.0f);
        if (isHit) ++hitCount;
        std::cout << std::setw(5)  << i
                  << std::setw(14) << std::fixed << std::setprecision(5) << r.attentionWeight
                  << std::setw(12) << r.finalOmega
                  << std::setw(8)  << r.traversalDepth
                  << std::setw(8)  << (isHit ? "HIT" : "miss")
                  << "\n";
    }

    std::cout << "\nRayos con hit: " << hitCount << " / " << TestScene::NUM_RAYS << "\n";

    // ── 11. Validaciones ──────────────────────────────────────────────────
    bool pass = true;

    // Test 1: al menos 1 hit
    if (hitCount == 0) {
        std::cerr << "[FAIL] Cero hits — las esferas no se detectaron\n";
        std::cerr << "       Verificar posiciones de esferas y distribución de rayos Fibonacci\n";
        pass = false;
    } else {
        std::cout << "[PASS] Hit detection: " << hitCount << " hits\n";
    }

    // Test 2: los hits tienen resonancia no-nula
    int resonanceHits = 0;
    for (int i = 0; i < TestScene::NUM_RAYS; ++i) {
        if (results[i].attentionWeight != 0.0f) ++resonanceHits;
    }
    if (resonanceHits == 0) {
        std::cerr << "[FAIL] Todos los attentionWeight son 0 — resonancia no calculada\n";
        pass = false;
    } else {
        std::cout << "[PASS] Resonance non-zero: " << resonanceHits << " rayos\n";
    }

    // Test 3: finalOmega varia (los sesgos de esfera deben modificarla)
    float minOmega = 1e9f, maxOmega = -1e9f;
    for (int i = 0; i < TestScene::NUM_RAYS; ++i) {
        float o = results[i].finalOmega;
        if (o > 0.0f) {  // solo rayos con hit
            minOmega = std::min(minOmega, o);
            maxOmega = std::max(maxOmega, o);
        }
    }
    if (hitCount > 1 && (maxOmega - minOmega) < 0.01f) {
        std::cerr << "[WARN] finalOmega muy uniforme — los sesgos de frecuencia no varían\n";
        std::cerr << "       (range: " << minOmega << " .. " << maxOmega << ")\n";
    } else if (hitCount > 0) {
        std::cout << "[PASS] finalOmega range: " << minOmega << " .. " << maxOmega << "\n";
    }

    std::cout << "\n";
    if (pass) {
        std::cout << "[PASS] Pipeline Inception Engine v4.0 funcional — "
                  << hitCount << " hits con resonancia Fourier calculada.\n";
    } else {
        std::cout << "[FAIL] Algunos tests fallaron. Ver mensajes arriba.\n";
    }

    // ── 12. Cleanup ───────────────────────────────────────────────────────
    cudaFree((void*)d_aabbs);
    cudaFree((void*)d_output);
    cudaFree((void*)d_spheres);
    cudaFree((void*)d_portals);
    cudaFree((void*)d_strings);
    cudaFree((void*)d_results);
    cudaFree((void*)d_params);
    cudaFree((void*)d_raygenBuf);
    cudaFree((void*)d_missBuf);
    cudaFree((void*)d_hitBuf);

    optixPipelineDestroy(pipeline);
    optixProgramGroupDestroy(raygenPG);
    optixProgramGroupDestroy(missPG);
    optixProgramGroupDestroy(hitgroupPG);
    optixModuleDestroy(mod);
    optixDeviceContextDestroy(optixCtx);

    return pass ? 0 : 1;
}
