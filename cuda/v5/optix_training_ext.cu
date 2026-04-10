/**
 * optix_training_ext.cu — PyTorch Extension: OptiX RT Core Router for Training
 *
 * Exposes the C++ RTCoreRouter to Python via pybind11/torch extension.
 * All data stays on GPU (zero-copy): torch tensors → CUDA pointers → OptiX → torch tensors.
 *
 * Usage from Python:
 *   import optix_training_ext
 *   optix_training_ext.initialize(ptx_raygen, ptx_hitgroup)
 *   optix_training_ext.build_gas(centers_cpu, radii_cpu)
 *   expert_ids, distances = optix_training_ext.route(positions_gpu, directions_gpu)
 *   expert_ids_topk, dists_topk = optix_training_ext.route_topk(positions_gpu, directions_gpu, k=8)
 *
 * Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
 */

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <string>
#include <stdexcept>
#include <memory>

// ── Forward-declare RTCoreRouter from optix_router_host.cpp ───────────
// We link against spectral_rt_router library which provides this class.
// If building standalone, include the header directly.

#ifdef OPTIX_TRAINING_STANDALONE
// Standalone build: include RTCoreRouter definition inline
#include "../optix_router_host.cpp"
#else
// Library build: declare the class interface we need
struct float3 { float x, y, z; };

class RTCoreRouter {
public:
    RTCoreRouter();
    ~RTCoreRouter();

    RTCoreRouter(const RTCoreRouter&) = delete;
    RTCoreRouter& operator=(const RTCoreRouter&) = delete;

    bool initialize(const std::string& ptx_raygen_path,
                    const std::string& ptx_hitgroup_path);

    bool buildGAS(const float* centers, const float* radii, uint32_t num_experts);

    bool buildGAS_triangles(const float* centers, const float* radii, uint32_t num_experts);

    bool route(const float3* d_query_positions,
               const float3* d_query_directions,
               uint32_t batch_size,
               uint32_t* d_expert_ids,
               float* d_expert_distances,
               uint32_t top_k = 1,
               uint32_t* d_topk_ids = nullptr,
               float* d_topk_dists = nullptr);

    bool route_async(const float3* d_query_positions,
                     const float3* d_query_directions,
                     uint32_t batch_size,
                     uint32_t* d_expert_ids,
                     float* d_expert_distances,
                     uint32_t top_k = 1,
                     uint32_t* d_topk_ids = nullptr,
                     float* d_topk_dists = nullptr);

    bool sync();
    cudaStream_t getStream() const;

    bool isReady() const;
    size_t gasSize() const;
    uint32_t numExperts() const;
};
#endif

// ── Global singleton (one RTCoreRouter per process) ───────────────────
static std::unique_ptr<RTCoreRouter> g_router;
static bool g_initialized = false;

// ── Helper: verify tensor is CUDA, contiguous, correct dtype ─────────
static void check_tensor(const torch::Tensor& t, const char* name,
                          torch::ScalarType dtype, bool require_cuda = true) {
    if (require_cuda && !t.is_cuda()) {
        throw std::runtime_error(
            std::string(name) + " must be a CUDA tensor");
    }
    if (!require_cuda && t.is_cuda()) {
        throw std::runtime_error(
            std::string(name) + " must be a CPU tensor");
    }
    if (!t.is_contiguous()) {
        throw std::runtime_error(
            std::string(name) + " must be contiguous");
    }
    if (t.scalar_type() != dtype) {
        throw std::runtime_error(
            std::string(name) + " has wrong dtype");
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Python-facing functions
// ═══════════════════════════════════════════════════════════════════════

/**
 * Initialize OptiX context and compile the RT pipeline from PTX files.
 *
 * Args:
 *   ptx_raygen:   Path to optix_router_raygen.ptx
 *   ptx_hitgroup: Path to optix_router_hitgroup.ptx
 *
 * Returns: true on success
 */
bool initialize_impl(const std::string& ptx_raygen,
                     const std::string& ptx_hitgroup) {
    if (g_initialized && g_router) {
        return true;  // Already initialized
    }

    g_router = std::make_unique<RTCoreRouter>();
    bool ok = g_router->initialize(ptx_raygen, ptx_hitgroup);
    if (!ok) {
        g_router.reset();
        throw std::runtime_error("RTCoreRouter::initialize() failed. "
                                  "Check PTX paths and OptiX installation.");
    }
    g_initialized = true;
    return true;
}

/**
 * Build Geometry Acceleration Structure from expert sphere centers + radii.
 *
 * Args:
 *   centers: CPU tensor (num_experts, 3) float32
 *   radii:   CPU tensor (num_experts,)   float32
 *   use_triangles: if true, use octahedron GAS (more precise, slightly slower)
 *
 * Returns: true on success
 */
bool build_gas_impl(torch::Tensor centers, torch::Tensor radii,
                    bool use_triangles) {
    if (!g_initialized || !g_router) {
        throw std::runtime_error("Call initialize() first");
    }

    check_tensor(centers, "centers", torch::kFloat32, /*require_cuda=*/false);
    check_tensor(radii, "radii", torch::kFloat32, /*require_cuda=*/false);

    uint32_t num_experts = static_cast<uint32_t>(centers.size(0));
    TORCH_CHECK(centers.dim() == 2 && centers.size(1) == 3,
                "centers must be (N, 3)");
    TORCH_CHECK(radii.dim() == 1 && radii.size(0) == num_experts,
                "radii must be (N,)");

    const float* c_ptr = centers.data_ptr<float>();
    const float* r_ptr = radii.data_ptr<float>();

    bool ok;
    if (use_triangles) {
        ok = g_router->buildGAS_triangles(c_ptr, r_ptr, num_experts);
    } else {
        ok = g_router->buildGAS(c_ptr, r_ptr, num_experts);
    }

    if (!ok) {
        throw std::runtime_error("buildGAS() failed");
    }
    return true;
}

/**
 * Route a batch of queries through the OptiX BVH using RT Cores.
 *
 * Args:
 *   positions:  CUDA tensor (batch, 3) float32 — 3D query positions
 *   directions: CUDA tensor (batch, 3) float32 — ray directions
 *
 * Returns:
 *   Tuple of (expert_ids: int32 CUDA, distances: float32 CUDA)
 */
std::tuple<torch::Tensor, torch::Tensor>
route_impl(torch::Tensor positions, torch::Tensor directions) {
    if (!g_initialized || !g_router || !g_router->isReady()) {
        throw std::runtime_error("Router not ready. Call initialize() + build_gas() first.");
    }

    check_tensor(positions, "positions", torch::kFloat32, /*require_cuda=*/true);
    check_tensor(directions, "directions", torch::kFloat32, /*require_cuda=*/true);

    uint32_t batch_size = static_cast<uint32_t>(positions.size(0));
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                "positions must be (B, 3)");
    TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3,
                "directions must be (B, 3)");

    // Allocate output tensors on same device
    auto opts = torch::TensorOptions().device(positions.device());
    auto expert_ids = torch::empty({batch_size}, opts.dtype(torch::kInt32));
    auto distances  = torch::empty({batch_size}, opts.dtype(torch::kFloat32));

    // Cast data pointers — float3 is {float, float, float}, same layout as (B,3) float tensor
    const float3* d_pos = reinterpret_cast<const float3*>(positions.data_ptr<float>());
    const float3* d_dir = reinterpret_cast<const float3*>(directions.data_ptr<float>());
    uint32_t* d_ids     = reinterpret_cast<uint32_t*>(expert_ids.data_ptr<int32_t>());
    float* d_dists      = distances.data_ptr<float>();

    bool ok = g_router->route(d_pos, d_dir, batch_size, d_ids, d_dists);
    if (!ok) {
        throw std::runtime_error("route() failed");
    }

    return std::make_tuple(expert_ids, distances);
}

/**
 * Route with top-K expert selection (for MoE-style routing).
 *
 * Args:
 *   positions:  CUDA tensor (batch, 3) float32
 *   directions: CUDA tensor (batch, 3) float32
 *   top_k:      Number of experts per query (default 8)
 *
 * Returns:
 *   Tuple of (topk_ids: int32 (B,K), topk_dists: float32 (B,K))
 */
std::tuple<torch::Tensor, torch::Tensor>
route_topk_impl(torch::Tensor positions, torch::Tensor directions,
                int64_t top_k) {
    if (!g_initialized || !g_router || !g_router->isReady()) {
        throw std::runtime_error("Router not ready. Call initialize() + build_gas() first.");
    }

    check_tensor(positions, "positions", torch::kFloat32, /*require_cuda=*/true);
    check_tensor(directions, "directions", torch::kFloat32, /*require_cuda=*/true);

    uint32_t batch_size = static_cast<uint32_t>(positions.size(0));
    uint32_t k = static_cast<uint32_t>(top_k);

    auto opts = torch::TensorOptions().device(positions.device());
    auto expert_ids = torch::empty({batch_size}, opts.dtype(torch::kInt32));
    auto distances  = torch::empty({batch_size}, opts.dtype(torch::kFloat32));
    auto topk_ids   = torch::empty({batch_size, k}, opts.dtype(torch::kInt32));
    auto topk_dists = torch::empty({batch_size, k}, opts.dtype(torch::kFloat32));

    const float3* d_pos = reinterpret_cast<const float3*>(positions.data_ptr<float>());
    const float3* d_dir = reinterpret_cast<const float3*>(directions.data_ptr<float>());

    bool ok = g_router->route(
        d_pos, d_dir, batch_size,
        reinterpret_cast<uint32_t*>(expert_ids.data_ptr<int32_t>()),
        distances.data_ptr<float>(),
        k,
        reinterpret_cast<uint32_t*>(topk_ids.data_ptr<int32_t>()),
        topk_dists.data_ptr<float>()
    );

    if (!ok) {
        throw std::runtime_error("route_topk() failed");
    }

    return std::make_tuple(topk_ids, topk_dists);
}

/**
 * Query router state.
 */
bool is_ready_impl() {
    return g_initialized && g_router && g_router->isReady();
}

int64_t gas_size_impl() {
    return g_router ? static_cast<int64_t>(g_router->gasSize()) : 0;
}

int64_t num_experts_impl() {
    return g_router ? static_cast<int64_t>(g_router->numExperts()) : 0;
}

/**
 * Route async (no sync) — faster for pipelined inference.
 * Results may not be ready until sync() is called.
 */
std::tuple<torch::Tensor, torch::Tensor>
route_async_impl(torch::Tensor positions, torch::Tensor directions) {
    if (!g_initialized || !g_router || !g_router->isReady()) {
        throw std::runtime_error("Router not ready. Call initialize() + build_gas() first.");
    }

    check_tensor(positions, "positions", torch::kFloat32, /*require_cuda=*/true);
    check_tensor(directions, "directions", torch::kFloat32, /*require_cuda=*/true);

    uint32_t batch_size = static_cast<uint32_t>(positions.size(0));
    TORCH_CHECK(positions.dim() == 2 && positions.size(1) == 3,
                "positions must be (B, 3)");
    TORCH_CHECK(directions.dim() == 2 && directions.size(1) == 3,
                "directions must be (B, 3)");

    auto opts = torch::TensorOptions().device(positions.device());
    auto expert_ids = torch::empty({batch_size}, opts.dtype(torch::kInt32));
    auto distances  = torch::empty({batch_size}, opts.dtype(torch::kFloat32));

    const float3* d_pos = reinterpret_cast<const float3*>(positions.data_ptr<float>());
    const float3* d_dir = reinterpret_cast<const float3*>(directions.data_ptr<float>());
    uint32_t* d_ids     = reinterpret_cast<uint32_t*>(expert_ids.data_ptr<int32_t>());
    float* d_dists      = distances.data_ptr<float>();

    bool ok = g_router->route_async(d_pos, d_dir, batch_size, d_ids, d_dists);
    if (!ok) {
        throw std::runtime_error("route_async() failed");
    }

    return std::make_tuple(expert_ids, distances);
}

/**
 * Synchronize the OptiX router stream.
 * Call after route_async() when you need results.
 */
void sync_impl() {
    if (g_router) {
        g_router->sync();
    }
}

/**
 * Forward declaration for the standalone benchmark function.
 * Defined in optix_router_host.cpp (included via OPTIX_TRAINING_STANDALONE).
 */
#ifndef OPTIX_TRAINING_STANDALONE
extern "C" bool rtcore_router_benchmark(
    uint32_t, uint32_t, uint32_t, uint32_t,
    const char*, const char*);
#endif

/**
 * Run the full C++ benchmark (AABB sync/async + Triangle sync/async).
 */
bool benchmark_impl(int64_t num_experts, int64_t batch_size,
                    int64_t num_warmup, int64_t num_iters,
                    const std::string& ptx_raygen,
                    const std::string& ptx_hitgroup) {
    return rtcore_router_benchmark(
        static_cast<uint32_t>(num_experts),
        static_cast<uint32_t>(batch_size),
        static_cast<uint32_t>(num_warmup),
        static_cast<uint32_t>(num_iters),
        ptx_raygen.c_str(), ptx_hitgroup.c_str());
}

void shutdown_impl() {
    g_router.reset();
    g_initialized = false;
}

// ═══════════════════════════════════════════════════════════════════════
// Pybind11 module registration
// ═══════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SpectralAI OptiX RT Core Router — Training Extension";

    m.def("initialize", &initialize_impl,
          "Initialize OptiX context from PTX shader files",
          py::arg("ptx_raygen"), py::arg("ptx_hitgroup"));

    m.def("build_gas", &build_gas_impl,
          "Build GAS from expert sphere centers + radii (CPU tensors)",
          py::arg("centers"), py::arg("radii"),
          py::arg("use_triangles") = false);

    m.def("route", &route_impl,
          "Route batch through RT Cores. Returns (expert_ids, distances) on GPU",
          py::arg("positions"), py::arg("directions"));

    m.def("route_topk", &route_topk_impl,
          "Route with top-K selection. Returns (topk_ids, topk_dists) on GPU",
          py::arg("positions"), py::arg("directions"),
          py::arg("top_k") = 8);

    m.def("is_ready", &is_ready_impl,
          "Check if router is initialized and GAS built");

    m.def("gas_size", &gas_size_impl,
          "GAS memory in bytes");

    m.def("num_experts", &num_experts_impl,
          "Number of experts in current GAS");

    m.def("route_async", &route_async_impl,
          "Route async (no sync). Call sync() before reading results.",
          py::arg("positions"), py::arg("directions"));

    m.def("sync", &sync_impl,
          "Synchronize the OptiX router stream after route_async()");

    m.def("benchmark", &benchmark_impl,
          "Run full C++ benchmark (AABB/Triangle x sync/async)",
          py::arg("num_experts") = 64,
          py::arg("batch_size") = 256,
          py::arg("num_warmup") = 10,
          py::arg("num_iters") = 100,
          py::arg("ptx_raygen") = "",
          py::arg("ptx_hitgroup") = "");

    m.def("shutdown", &shutdown_impl,
          "Release OptiX resources");
}
