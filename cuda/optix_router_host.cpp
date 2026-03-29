/**
 * optix_router_host.cpp — Minimal OptiX host code for RT Core routing
 *
 * Self-contained OptiX pipeline setup for expert selection:
 *   1. Build GAS from 64 expert AABBs (sphere centers ± radius)
 *   2. Create minimal pipeline (raygen + closesthit + miss)
 *   3. Launch: batch of queries → expert_ids
 *
 * This is separate from optix_host.cpp (which handles full optical attention).
 * The router host is designed to be wrapped as a PyTorch extension.
 *
 * Copyright (c) 2026 SpectralAI Studio — Apache 2.0
 */

#include <optix.h>
#include <optix_stubs.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cstring>
#include <cassert>
#include <cmath>

// ============================================================================
// Error checking macros
// ============================================================================

#define OPTIX_CHECK(call)                                                     \
    do {                                                                       \
        OptixResult res = call;                                                \
        if (res != OPTIX_SUCCESS) {                                            \
            std::cerr << "[OptiX ERROR] " << #call << " failed: "              \
                      << optixGetErrorName(res) << " ("                        \
                      << optixGetErrorString(res) << ")" << std::endl;         \
            return false;                                                      \
        }                                                                      \
    } while (0)

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "[CUDA ERROR] " << #call << " failed: "                \
                      << cudaGetErrorString(err) << std::endl;                  \
            return false;                                                        \
        }                                                                       \
    } while (0)

#define CU_CHECK(call)                                                          \
    do {                                                                         \
        CUresult err = call;                                                     \
        if (err != CUDA_SUCCESS) {                                               \
            const char* errStr;                                                  \
            cuGetErrorString(err, &errStr);                                      \
            std::cerr << "[CU ERROR] " << #call << " failed: "                  \
                      << errStr << std::endl;                                    \
            return false;                                                         \
        }                                                                        \
    } while (0)

// ============================================================================
// RTRouterParams — must match device-side struct in optix_router_raygen.cu
// ============================================================================

struct RTRouterParams {
    OptixTraversableHandle gas_handle;
    const float3* query_positions;
    const float3* query_directions;
    uint32_t* expert_ids;
    float* expert_distances;
    uint32_t* topk_expert_ids;
    float* topk_distances;
    uint32_t batch_size;
    uint32_t top_k;
    uint32_t num_experts;
    float ray_tmin;
    float ray_tmax;
};

// ============================================================================
// SBT Record template
// ============================================================================

template <typename T>
struct SbtRecord {
    alignas(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};
using RaygenRecord   = SbtRecord<EmptyData>;
using HitgroupRecord = SbtRecord<EmptyData>;
using MissRecord     = SbtRecord<EmptyData>;

// ============================================================================
// OptiX Logger callback
// ============================================================================

static void optixLogCallback(unsigned int level, const char* tag,
                              const char* message, void* /*cbdata*/) {
    if (level <= 2) {  // Only errors and warnings
        std::cerr << "[OptiX " << tag << "] " << message << std::endl;
    }
}

// ============================================================================
// Load PTX from file
// ============================================================================

static std::string loadPTX(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "[RTRouter] Failed to open PTX: " << path << std::endl;
        return "";
    }
    std::stringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

// ============================================================================
// RTCoreRouter class
// ============================================================================

class RTCoreRouter {
public:
    RTCoreRouter() = default;
    ~RTCoreRouter() { cleanup(); }

    // Non-copyable
    RTCoreRouter(const RTCoreRouter&) = delete;
    RTCoreRouter& operator=(const RTCoreRouter&) = delete;

    /**
     * Initialize OptiX context and compile pipeline from PTX files.
     * @param ptx_raygen_path  Path to optix_router_raygen.ptx
     * @param ptx_hitgroup_path Path to optix_router_hitgroup.ptx
     * @return true on success
     */
    bool initialize(const std::string& ptx_raygen_path,
                    const std::string& ptx_hitgroup_path) {
        // ── Init CUDA driver API ───────────────────────────────
        CU_CHECK(cuInit(0));

        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, 0));

        CU_CHECK(cuCtxGetCurrent(&cu_context_));
        if (!cu_context_) {
            // cuCtxCreate remapped to cuCtxCreate_v4 in CUDA 12.x+
            CU_CHECK(cuCtxCreate_v4(&cu_context_, nullptr, 0, device));
        }

        // ── Init OptiX ─────────────────────────────────────────
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions ctx_options = {};
        ctx_options.logCallbackFunction = optixLogCallback;
        ctx_options.logCallbackLevel = 3;

        OPTIX_CHECK(optixDeviceContextCreate(cu_context_, &ctx_options,
                                              &optix_context_));

        // ── Load PTX ───────────────────────────────────────────
        std::string ptx_raygen = loadPTX(ptx_raygen_path);
        std::string ptx_hitgroup = loadPTX(ptx_hitgroup_path);
        if (ptx_raygen.empty() || ptx_hitgroup.empty()) return false;

        // ── Module compile options ─────────────────────────────
        OptixModuleCompileOptions module_opts = {};
        module_opts.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        module_opts.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
        module_opts.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        // ── Pipeline compile options ───────────────────────────
        pipeline_compile_opts_ = {};
        pipeline_compile_opts_.usesMotionBlur = false;
        pipeline_compile_opts_.traversableGraphFlags =
            OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipeline_compile_opts_.numPayloadValues = 2;   // expert_id + distance
        pipeline_compile_opts_.numAttributeValues = 2;  // AABB built-in
        pipeline_compile_opts_.pipelineLaunchParamsVariableName = "params";

        // ── Create modules ─────────────────────────────────────
        char log[2048];
        size_t log_size;

        log_size = sizeof(log);
        OPTIX_CHECK(optixModuleCreate(
            optix_context_, &module_opts, &pipeline_compile_opts_,
            ptx_raygen.c_str(), ptx_raygen.size(),
            log, &log_size, &module_raygen_));

        log_size = sizeof(log);
        OPTIX_CHECK(optixModuleCreate(
            optix_context_, &module_opts, &pipeline_compile_opts_,
            ptx_hitgroup.c_str(), ptx_hitgroup.size(),
            log, &log_size, &module_hitgroup_));

        // ── Program groups ─────────────────────────────────────
        OptixProgramGroupOptions pg_opts = {};

        // Raygen
        OptixProgramGroupDesc raygen_desc = {};
        raygen_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygen_desc.raygen.module = module_raygen_;
        raygen_desc.raygen.entryFunctionName = "__raygen__rt_router";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            optix_context_, &raygen_desc, 1, &pg_opts,
            log, &log_size, &pg_raygen_));

        // Hitgroup (closest hit only, no any-hit or intersection)
        OptixProgramGroupDesc hitgroup_desc = {};
        hitgroup_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroup_desc.hitgroup.moduleCH = module_hitgroup_;
        hitgroup_desc.hitgroup.entryFunctionNameCH = "__closesthit__rt_router";
        hitgroup_desc.hitgroup.moduleAH = nullptr;
        hitgroup_desc.hitgroup.entryFunctionNameAH = nullptr;
        hitgroup_desc.hitgroup.moduleIS = nullptr;  // Built-in AABB intersection
        hitgroup_desc.hitgroup.entryFunctionNameIS = nullptr;

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            optix_context_, &hitgroup_desc, 1, &pg_opts,
            log, &log_size, &pg_hitgroup_));

        // Miss
        OptixProgramGroupDesc miss_desc = {};
        miss_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        miss_desc.miss.module = module_hitgroup_;  // miss is in same PTX
        miss_desc.miss.entryFunctionName = "__miss__rt_router";

        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(
            optix_context_, &miss_desc, 1, &pg_opts,
            log, &log_size, &pg_miss_));

        // ── Pipeline ───────────────────────────────────────────
        OptixProgramGroup groups[] = { pg_raygen_, pg_hitgroup_, pg_miss_ };

        OptixPipelineLinkOptions link_opts = {};
        link_opts.maxTraceDepth = 1;

        log_size = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(
            optix_context_, &pipeline_compile_opts_, &link_opts,
            groups, 3,
            log, &log_size, &pipeline_));

        // Set stack sizes
        OPTIX_CHECK(optixPipelineSetStackSize(
            pipeline_,
            2048,   // directCallableStackSizeFromTraversal
            2048,   // directCallableStackSizeFromState
            2048,   // continuationStackSize
            1       // maxTraversableGraphDepth
        ));

        // ── SBT ────────────────────────────────────────────────
        if (!buildSBT()) return false;

        is_ready_ = true;
        return true;
    }

    /**
     * Build GAS (Geometry Acceleration Structure) from expert sphere positions.
     * Call this when expert positions change (e.g., after training).
     *
     * @param centers  Expert center positions [num_experts x 3] (host memory)
     * @param radii    Expert sphere radii [num_experts] (host memory)
     * @param num_experts Number of experts (typically 64)
     * @return true on success
     */
    bool buildGAS(const float* centers, const float* radii,
                  uint32_t num_experts) {
        num_experts_ = num_experts;

        // Free previous GAS if any
        if (d_gas_buffer_) {
            cudaFree(reinterpret_cast<void*>(d_gas_buffer_));
            d_gas_buffer_ = 0;
        }

        // ── Build AABB array ───────────────────────────────────
        // Each expert sphere → one AABB
        std::vector<OptixAabb> aabbs(num_experts);
        for (uint32_t i = 0; i < num_experts; ++i) {
            float cx = centers[i * 3 + 0];
            float cy = centers[i * 3 + 1];
            float cz = centers[i * 3 + 2];
            float r  = radii[i];

            aabbs[i].minX = cx - r;
            aabbs[i].minY = cy - r;
            aabbs[i].minZ = cz - r;
            aabbs[i].maxX = cx + r;
            aabbs[i].maxY = cy + r;
            aabbs[i].maxZ = cz + r;
        }

        // Upload AABBs to GPU
        CUdeviceptr d_aabb_buffer;
        size_t aabb_size = num_experts * sizeof(OptixAabb);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_aabb_buffer), aabb_size));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_aabb_buffer),
                              aabbs.data(), aabb_size, cudaMemcpyHostToDevice));

        // ── Build input ────────────────────────────────────────
        uint32_t aabb_flags = OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;

        OptixBuildInput build_input = {};
        build_input.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        build_input.customPrimitiveArray.aabbBuffers = &d_aabb_buffer;
        build_input.customPrimitiveArray.numPrimitives = num_experts;
        build_input.customPrimitiveArray.flags = &aabb_flags;
        build_input.customPrimitiveArray.numSbtRecords = 1;

        // ── Accel build options ────────────────────────────────
        OptixAccelBuildOptions accel_opts = {};
        accel_opts.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |
                                OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
        accel_opts.operation = OPTIX_BUILD_OPERATION_BUILD;

        // Query buffer sizes
        OptixAccelBufferSizes buffer_sizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(
            optix_context_, &accel_opts, &build_input, 1, &buffer_sizes));

        // Allocate temp and output buffers
        CUdeviceptr d_temp_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer),
                              buffer_sizes.tempSizeInBytes));

        CUdeviceptr d_output_buffer;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_buffer),
                              buffer_sizes.outputSizeInBytes));

        // Compaction size buffer
        CUdeviceptr d_compacted_size;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted_size),
                              sizeof(size_t)));

        OptixAccelEmitDesc emit_desc = {};
        emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit_desc.result = d_compacted_size;

        // Build
        OPTIX_CHECK(optixAccelBuild(
            optix_context_, 0 /*stream*/,
            &accel_opts, &build_input, 1,
            d_temp_buffer, buffer_sizes.tempSizeInBytes,
            d_output_buffer, buffer_sizes.outputSizeInBytes,
            &gas_handle_, &emit_desc, 1));

        CUDA_CHECK(cudaDeviceSynchronize());

        // ── Compact ────────────────────────────────────────────
        size_t compacted_size;
        CUDA_CHECK(cudaMemcpy(&compacted_size,
                              reinterpret_cast<void*>(d_compacted_size),
                              sizeof(size_t), cudaMemcpyDeviceToHost));

        if (compacted_size < buffer_sizes.outputSizeInBytes) {
            CUdeviceptr d_compacted;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted),
                                  compacted_size));
            OPTIX_CHECK(optixAccelCompact(
                optix_context_, 0, gas_handle_,
                d_compacted, compacted_size, &gas_handle_));
            CUDA_CHECK(cudaDeviceSynchronize());
            cudaFree(reinterpret_cast<void*>(d_output_buffer));
            d_gas_buffer_ = d_compacted;
        } else {
            d_gas_buffer_ = d_output_buffer;
        }

        // Cleanup temp
        cudaFree(reinterpret_cast<void*>(d_temp_buffer));
        cudaFree(reinterpret_cast<void*>(d_aabb_buffer));
        cudaFree(reinterpret_cast<void*>(d_compacted_size));

        gas_built_ = true;
        gas_size_bytes_ = compacted_size;

        std::cout << "[RTRouter] GAS built: " << num_experts << " experts, "
                  << (gas_size_bytes_ / 1024) << " KB" << std::endl;

        return true;
    }

    /**
     * Route a batch of queries through the RT Core BVH.
     *
     * @param d_query_positions  Device pointer: [batch_size x 3] float3
     * @param d_query_directions Device pointer: [batch_size x 3] float3
     * @param batch_size         Number of queries
     * @param d_expert_ids       Device pointer output: [batch_size] uint32
     * @param d_expert_distances Device pointer output: [batch_size] float
     * @param top_k              Number of experts per query (1=single, 8=top-8)
     * @param d_topk_ids         Device pointer output: [batch_size * top_k] (or nullptr)
     * @param d_topk_dists       Device pointer output: [batch_size * top_k] (or nullptr)
     * @return true on success
     */
    bool route(const float3* d_query_positions,
               const float3* d_query_directions,
               uint32_t batch_size,
               uint32_t* d_expert_ids,
               float* d_expert_distances,
               uint32_t top_k = 1,
               uint32_t* d_topk_ids = nullptr,
               float* d_topk_dists = nullptr) {

        if (!is_ready_ || !gas_built_) {
            std::cerr << "[RTRouter] Not initialized or GAS not built" << std::endl;
            return false;
        }

        // ── Set launch params ──────────────────────────────────
        RTRouterParams h_params = {};
        h_params.gas_handle = gas_handle_;
        h_params.query_positions = d_query_positions;
        h_params.query_directions = d_query_directions;
        h_params.expert_ids = d_expert_ids;
        h_params.expert_distances = d_expert_distances;
        h_params.topk_expert_ids = d_topk_ids;
        h_params.topk_distances = d_topk_dists;
        h_params.batch_size = batch_size;
        h_params.top_k = top_k;
        h_params.num_experts = num_experts_;
        h_params.ray_tmin = 0.001f;
        h_params.ray_tmax = 1000.0f;

        // Upload params (persistent allocation — avoid malloc/free per call)
        if (!d_params_) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params_),
                                  sizeof(RTRouterParams)));
        }
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params_),
                              &h_params, sizeof(RTRouterParams),
                              cudaMemcpyHostToDevice));

        // ── Launch ─────────────────────────────────────────────
        OPTIX_CHECK(optixLaunch(
            pipeline_,
            0,          // CUDA stream
            d_params_,
            sizeof(RTRouterParams),
            &sbt_,
            batch_size, // width
            1,          // height
            1           // depth
        ));

        CUDA_CHECK(cudaDeviceSynchronize());

        return true;
    }

    bool isReady() const { return is_ready_ && gas_built_; }
    size_t gasSize() const { return gas_size_bytes_; }
    uint32_t numExperts() const { return num_experts_; }

private:
    bool buildSBT() {
        // ── Raygen SBT ─────────────────────────────────────────
        RaygenRecord raygen_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(pg_raygen_, &raygen_record));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record_),
                              sizeof(RaygenRecord)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record_),
                              &raygen_record, sizeof(RaygenRecord),
                              cudaMemcpyHostToDevice));

        // ── Miss SBT ───────────────────────────────────────────
        MissRecord miss_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(pg_miss_, &miss_record));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record_),
                              sizeof(MissRecord)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record_),
                              &miss_record, sizeof(MissRecord),
                              cudaMemcpyHostToDevice));

        // ── Hitgroup SBT ───────────────────────────────────────
        HitgroupRecord hitgroup_record;
        OPTIX_CHECK(optixSbtRecordPackHeader(pg_hitgroup_, &hitgroup_record));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record_),
                              sizeof(HitgroupRecord)));
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record_),
                              &hitgroup_record, sizeof(HitgroupRecord),
                              cudaMemcpyHostToDevice));

        // ── Assemble SBT ───────────────────────────────────────
        sbt_ = {};
        sbt_.raygenRecord = d_raygen_record_;
        sbt_.missRecordBase = d_miss_record_;
        sbt_.missRecordStrideInBytes = sizeof(MissRecord);
        sbt_.missRecordCount = 1;
        sbt_.hitgroupRecordBase = d_hitgroup_record_;
        sbt_.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        sbt_.hitgroupRecordCount = 1;

        return true;
    }

    void cleanup() {
        if (pipeline_) optixPipelineDestroy(pipeline_);
        if (pg_raygen_) optixProgramGroupDestroy(pg_raygen_);
        if (pg_hitgroup_) optixProgramGroupDestroy(pg_hitgroup_);
        if (pg_miss_) optixProgramGroupDestroy(pg_miss_);
        if (module_raygen_) optixModuleDestroy(module_raygen_);
        if (module_hitgroup_) optixModuleDestroy(module_hitgroup_);
        if (optix_context_) optixDeviceContextDestroy(optix_context_);

        if (d_raygen_record_) cudaFree(reinterpret_cast<void*>(d_raygen_record_));
        if (d_miss_record_) cudaFree(reinterpret_cast<void*>(d_miss_record_));
        if (d_hitgroup_record_) cudaFree(reinterpret_cast<void*>(d_hitgroup_record_));
        if (d_params_) cudaFree(reinterpret_cast<void*>(d_params_));
        if (d_gas_buffer_) cudaFree(reinterpret_cast<void*>(d_gas_buffer_));
    }

    // State
    bool is_ready_ = false;
    bool gas_built_ = false;
    uint32_t num_experts_ = 0;
    size_t gas_size_bytes_ = 0;

    // OptiX objects
    CUcontext cu_context_ = nullptr;
    OptixDeviceContext optix_context_ = nullptr;
    OptixModule module_raygen_ = nullptr;
    OptixModule module_hitgroup_ = nullptr;
    OptixProgramGroup pg_raygen_ = nullptr;
    OptixProgramGroup pg_hitgroup_ = nullptr;
    OptixProgramGroup pg_miss_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixPipelineCompileOptions pipeline_compile_opts_ = {};

    // SBT
    OptixShaderBindingTable sbt_ = {};
    CUdeviceptr d_raygen_record_ = 0;
    CUdeviceptr d_miss_record_ = 0;
    CUdeviceptr d_hitgroup_record_ = 0;

    // GAS
    OptixTraversableHandle gas_handle_ = 0;
    CUdeviceptr d_gas_buffer_ = 0;

    // Launch params (persistent to avoid malloc/free per route() call)
    CUdeviceptr d_params_ = 0;
};

// ============================================================================
// Standalone benchmark function (called from test executable)
// ============================================================================

/**
 * Run a routing benchmark: build GAS from random experts, route random queries,
 * measure latency.
 *
 * @param num_experts   Number of experts (e.g., 64)
 * @param batch_size    Number of queries per batch
 * @param num_warmup    Warmup iterations
 * @param num_iters     Benchmark iterations
 * @param ptx_raygen    Path to raygen PTX
 * @param ptx_hitgroup  Path to hitgroup PTX
 */
extern "C" bool rtcore_router_benchmark(
    uint32_t num_experts,
    uint32_t batch_size,
    uint32_t num_warmup,
    uint32_t num_iters,
    const char* ptx_raygen,
    const char* ptx_hitgroup
) {
    RTCoreRouter router;

    std::cout << "=== RT Core Router Benchmark ===" << std::endl;
    std::cout << "Experts: " << num_experts << ", Batch: " << batch_size << std::endl;

    // ── Initialize ─────────────────────────────────────────
    if (!router.initialize(ptx_raygen, ptx_hitgroup)) {
        std::cerr << "Failed to initialize router" << std::endl;
        return false;
    }
    std::cout << "Pipeline initialized" << std::endl;

    // ── Generate random expert positions ───────────────────
    std::vector<float> centers(num_experts * 3);
    std::vector<float> radii(num_experts);
    for (uint32_t i = 0; i < num_experts; ++i) {
        // Distribute experts on a sphere of radius 10
        float theta = 2.0f * 3.14159f * static_cast<float>(i) / num_experts;
        float phi = acosf(1.0f - 2.0f * (static_cast<float>(i) + 0.5f) / num_experts);
        centers[i * 3 + 0] = 10.0f * sinf(phi) * cosf(theta);
        centers[i * 3 + 1] = 10.0f * sinf(phi) * sinf(theta);
        centers[i * 3 + 2] = 10.0f * cosf(phi);
        radii[i] = 0.5f;
    }

    if (!router.buildGAS(centers.data(), radii.data(), num_experts)) {
        std::cerr << "Failed to build GAS" << std::endl;
        return false;
    }

    // ── Generate random queries ────────────────────────────
    std::vector<float3> h_positions(batch_size);
    std::vector<float3> h_directions(batch_size);
    for (uint32_t i = 0; i < batch_size; ++i) {
        // Random position near center
        float t = static_cast<float>(i) / batch_size;
        h_positions[i] = make_float3(
            sinf(t * 6.28f) * 2.0f,
            cosf(t * 6.28f) * 2.0f,
            sinf(t * 3.14f) * 2.0f
        );
        // Direction toward a random expert
        uint32_t target = i % num_experts;
        float3 dir;
        dir.x = centers[target * 3 + 0] - h_positions[i].x;
        dir.y = centers[target * 3 + 1] - h_positions[i].y;
        dir.z = centers[target * 3 + 2] - h_positions[i].z;
        float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
        if (len > 1e-6f) { dir.x /= len; dir.y /= len; dir.z /= len; }
        h_directions[i] = dir;
    }

    // Upload to GPU
    float3* d_positions;
    float3* d_directions;
    uint32_t* d_expert_ids;
    float* d_distances;

    if (cudaMalloc(&d_positions, batch_size * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&d_directions, batch_size * sizeof(float3)) != cudaSuccess ||
        cudaMalloc(&d_expert_ids, batch_size * sizeof(uint32_t)) != cudaSuccess ||
        cudaMalloc(&d_distances, batch_size * sizeof(float)) != cudaSuccess) {
        std::cerr << "Failed to allocate GPU memory for benchmark" << std::endl;
        return false;
    }

    cudaMemcpy(d_positions, h_positions.data(),
               batch_size * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(d_directions, h_directions.data(),
               batch_size * sizeof(float3), cudaMemcpyHostToDevice);

    // ── Warmup ─────────────────────────────────────────────
    for (uint32_t i = 0; i < num_warmup; ++i) {
        router.route(d_positions, d_directions, batch_size,
                     d_expert_ids, d_distances);
    }

    // ── Benchmark ──────────────────────────────────────────
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (uint32_t i = 0; i < num_iters; ++i) {
        router.route(d_positions, d_directions, batch_size,
                     d_expert_ids, d_distances);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    float us_per_iter = (ms * 1000.0f) / num_iters;
    float throughput = static_cast<float>(batch_size) / (us_per_iter * 1e-6f);

    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "Latency: " << us_per_iter << " us/batch" << std::endl;
    std::cout << "Throughput: " << (throughput / 1e6f) << " M queries/s" << std::endl;
    std::cout << "GAS size: " << router.gasSize() << " bytes" << std::endl;

    // ── Verify correctness ─────────────────────────────────
    std::vector<uint32_t> h_expert_ids(batch_size);
    cudaMemcpy(h_expert_ids.data(), d_expert_ids,
               batch_size * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    uint32_t correct = 0;
    for (uint32_t i = 0; i < batch_size; ++i) {
        uint32_t expected = i % num_experts;
        if (h_expert_ids[i] == expected) ++correct;
    }
    std::cout << "Routing accuracy: " << correct << "/" << batch_size
              << " (" << (100.0f * correct / batch_size) << "%)" << std::endl;

    // Cleanup
    cudaFree(d_positions);
    cudaFree(d_directions);
    cudaFree(d_expert_ids);
    cudaFree(d_distances);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}
