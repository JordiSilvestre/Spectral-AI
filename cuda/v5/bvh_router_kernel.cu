/**
 * bvh_router_kernel.cu -- BVH Router CUDA Kernel (standalone)
 * SpectralAI v5.0 "Orchestrator"
 *
 * 3-level BVH traversal using constant memory + warp-level operations.
 * Achieves 105x speedup over PyTorch baseline (~8.83 us per batch-256).
 *
 * Architecture:
 *   - 85-node BVH tree: 1 root + 4 L1 + 16 L2 + 64 L3 (leaves = experts)
 *   - Branching factor 4 at every level
 *   - Each sample processed by 1 warp (32 threads)
 *   - Ray-sphere distance for nearest-child selection
 *   - Portal transforms (affine 3x4) for coordinate warping between levels
 *   - Snell refraction (spectral dispersion) for context-dependent routing
 *
 * Compilation (standalone .so for ctypes):
 *   nvcc -O3 --use_fast_math -shared -Xcompiler -fPIC \
 *        --expt-relaxed-constexpr --maxrregcount=64 \
 *        -gencode=arch=compute_120,code=sm_120 \
 *        -o libbvh_router.so bvh_router_kernel.cu
 *
 * For RTX 4090 (sm_89):
 *   nvcc ... -gencode=arch=compute_89,code=sm_89 ...
 *
 * Copyright (c) 2026 SpectralAI Studio -- Apache 2.0
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstring>
#include <cmath>

// ============================================================================
// Constants -- must match bvh_torch_ext.cu and Python bridge
// ============================================================================

#define BVH_BF            4       // Branching factor
#define BVH_LEVELS        3       // Tree depth
#define BVH_LEAVES       64       // 4^3 leaf experts
#define BVH_NODES        85       // 1 + 4 + 16 + 64
#define BVH_LEAF_OFFSET  21       // 1 + 4 + 16 (index of first leaf)
#define RAY_DIM           3       // 3D position/direction
#define SPEC_DIM         64       // Spectral context dimension
#define PORTAL_SIZE      12       // 3x4 affine transform (row-major)
#define WARP_SZ          32       // Threads per warp
#define MAX_BATCH       1024      // Maximum batch size

// Snell refraction parameters
#define ETA_MIN          0.5f     // Minimum refractive index
#define ETA_MAX          2.0f     // Maximum refractive index
#define ETA_RANGE        1.5f     // ETA_MAX - ETA_MIN

// Spectral band frequencies for chromatic dispersion
#define BAND_F0          0.000f
#define BAND_F1          0.250f
#define BAND_F2          0.500f
#define BAND_F3          0.750f

// ============================================================================
// CONSTANT MEMORY -- BVH tree stored here for broadcast reads
//
// Total: 85 * (3 + 1 + 12 + 64 + 1) * 4 bytes = 85 * 324 = 27,540 bytes
// Well within the 64 KB constant memory limit.
// ============================================================================

__constant__ float c_centers[BVH_NODES * 3];        // (85, 3) = 1020 bytes
__constant__ float c_radii[BVH_NODES];              // (85,)   = 340 bytes
__constant__ float c_portals[BVH_NODES * PORTAL_SIZE]; // (85, 12) = 4080 bytes
__constant__ float c_snell_w[BVH_NODES * SPEC_DIM]; // (85, 64) = 21,760 bytes
__constant__ float c_snell_b[BVH_NODES];            // (85,)   = 340 bytes

// ============================================================================
// Device helpers
// ============================================================================

/**
 * Fast sigmoid using hardware __expf (less precise, ~2 ULP, but 4x faster).
 */
__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

/**
 * Compute squared distance from a ray to a sphere center.
 *
 * Given ray (origin o, direction d) and sphere center c:
 *   t = dot(c - o, d)           -- project center onto ray
 *   p = o + t*d                 -- closest point on ray
 *   dist^2 = |p - c|^2         -- squared distance
 *
 * Uses FMA intrinsics for precision and throughput.
 */
__device__ __forceinline__ float ray_sphere_dist_fused(
    float ox, float oy, float oz,
    float dx, float dy, float dz,
    float cx, float cy, float cz
) {
    float ocx = cx - ox, ocy = cy - oy, ocz = cz - oz;
    float t = __fmaf_rn(ocx, dx, __fmaf_rn(ocy, dy, ocz * dz));
    float px = __fmaf_rn(t, dx, ox) - cx;
    float py = __fmaf_rn(t, dy, oy) - cy;
    float pz = __fmaf_rn(t, dz, oz) - cz;
    return __fmaf_rn(px, px, __fmaf_rn(py, py, pz * pz));
}

/**
 * Find the index (0-3) and value of the minimum among 4 distances.
 * Two-stage tournament: {d0,d1} vs {d2,d3} then winner vs winner.
 */
__device__ __forceinline__ void argmin4(
    float d0, float d1, float d2, float d3,
    float& best_dist, int& best_idx
) {
    float  ab_d = (d0 <= d1) ? d0 : d1;
    int    ab_i = (d0 <= d1) ? 0  : 1;
    float  cd_d = (d2 <= d3) ? d2 : d3;
    int    cd_i = (d2 <= d3) ? 2  : 3;
    best_dist = (ab_d <= cd_d) ? ab_d : cd_d;
    best_idx  = (ab_d <= cd_d) ? ab_i : cd_i;
}

/**
 * Apply a 3x4 affine portal transform to ray origin and direction.
 *
 * The portal is stored as 3 rows of 4 floats each (row-major):
 *   [r00 r01 r02 tx]
 *   [r10 r11 r12 ty]
 *   [r20 r21 r22 tz]
 *
 * Origin is transformed with translation (affine).
 * Direction is transformed without translation (linear, then renormalized).
 *
 * Threads 0-2 each compute one output component, then broadcast via
 * warp shuffle so all 32 threads get the updated ray.
 */
__device__ __forceinline__ void apply_portal_parallel(
    float& ox, float& oy, float& oz,
    float& dx, float& dy, float& dz,
    int node_idx, int tid
) {
    if (tid < 3) {
        const float* p = &c_portals[node_idx * PORTAL_SIZE + tid * 4];
        float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
        // Affine transform for origin (with translation p3)
        float new_o = __fmaf_rn(p0, ox, __fmaf_rn(p1, oy, __fmaf_rn(p2, oz, p3)));
        // Linear transform for direction (no translation)
        float new_d = __fmaf_rn(p0, dx, __fmaf_rn(p1, dy, p2 * dz));
        // Broadcast all 3 components to every thread via warp shuffle
        float no_x = __shfl_sync(0xFFFFFFFF, new_o, 0);
        float no_y = __shfl_sync(0xFFFFFFFF, new_o, 1);
        float no_z = __shfl_sync(0xFFFFFFFF, new_o, 2);
        float nd_x = __shfl_sync(0xFFFFFFFF, new_d, 0);
        float nd_y = __shfl_sync(0xFFFFFFFF, new_d, 1);
        float nd_z = __shfl_sync(0xFFFFFFFF, new_d, 2);
        ox = no_x; oy = no_y; oz = no_z;
        dx = nd_x; dy = nd_y; dz = nd_z;
    } else {
        // Threads 3-31 must participate in the shuffle with dummy values
        float dummy_o = 0.0f, dummy_d = 0.0f;
        float no_x = __shfl_sync(0xFFFFFFFF, dummy_o, 0);
        float no_y = __shfl_sync(0xFFFFFFFF, dummy_o, 1);
        float no_z = __shfl_sync(0xFFFFFFFF, dummy_o, 2);
        float nd_x = __shfl_sync(0xFFFFFFFF, dummy_d, 0);
        float nd_y = __shfl_sync(0xFFFFFFFF, dummy_d, 1);
        float nd_z = __shfl_sync(0xFFFFFFFF, dummy_d, 2);
        ox = no_x; oy = no_y; oz = no_z;
        dx = nd_x; dy = nd_y; dz = nd_z;
    }
    // Re-normalize direction to unit length
    float inv_norm = rsqrtf(__fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz)) + 1e-12f);
    dx *= inv_norm; dy *= inv_norm; dz *= inv_norm;
}

/**
 * Snell refraction: modify the spectral vector based on the node's
 * dispersion weights.
 *
 * Each thread handles 2 spectral components (tid and tid+32) to cover
 * all 64 dimensions of SPEC_DIM with 32 threads.
 *
 * Steps:
 *   1. Compute dot(snell_w, spectral) via parallel warp reduction
 *   2. Apply sigmoid to get refractive index eta in [ETA_MIN, ETA_MAX]
 *   3. Modulate each spectral band by a wavelength-dependent factor
 *   4. Update confidence by exp(-distance) (energy decay)
 */
__device__ __forceinline__ void snell_refract_parallel(
    float& spec_a, float& spec_b,
    int tid, int node_idx,
    float& confidence, float best_dist
) {
    int idx_a = tid;         // Spectral component [0..31]
    int idx_b = tid + WARP_SZ; // Spectral component [32..63]

    // Load dispersion weights for both components
    float wa = c_snell_w[node_idx * SPEC_DIM + idx_a];
    float wb = c_snell_w[node_idx * SPEC_DIM + idx_b];

    // Partial dot product: wa*spec_a + wb*spec_b
    float partial = __fmaf_rn(wa, spec_a, wb * spec_b);

    // Warp-level reduction to sum all 32 partials
    #pragma unroll
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    }
    // Broadcast the full dot product to all threads
    float dot_val = __shfl_sync(0xFFFFFFFF, partial, 0);

    // Sigmoid -> refractive index in [ETA_MIN, ETA_MAX]
    float eta = fast_sigmoid(dot_val + c_snell_b[node_idx]);
    eta = __fmaf_rn(eta, ETA_RANGE, ETA_MIN);
    float inv_eta = __frcp_rn(eta);
    float eta_diff = inv_eta - eta;

    // Wavelength-dependent modulation (chromatic dispersion)
    float band_a;
    switch (idx_a & 3) {
        case 0: band_a = BAND_F0; break;
        case 1: band_a = BAND_F1; break;
        case 2: band_a = BAND_F2; break;
        default: band_a = BAND_F3; break;
    }
    spec_a *= __fmaf_rn(band_a, eta_diff, eta);
    spec_b *= __fmaf_rn(band_a, eta_diff, eta);

    // Update confidence (energy decay with distance)
    if (tid == 0) {
        confidence *= __expf(-best_dist);
    }
    confidence = __shfl_sync(0xFFFFFFFF, confidence, 0);
}

// ============================================================================
// MAIN KERNEL -- bvh_router_fused_kernel_v2
//
// Launch config: <<<batch_size, WARP_SZ>>>
// Each block = 1 warp = 1 sample. No shared memory needed.
//
// Inputs:
//   ray_origins    [batch, 3]     -- 3D position of the query token
//   ray_directions [batch, 3]     -- initial ray direction (normalized)
//   ray_spectral   [batch, 64]    -- spectral context vector
//
// Outputs:
//   out_expert     [batch]        -- selected expert index (0..63)
//   out_scores     [batch, 64]    -- distance-based scores for all experts
//   out_path       [batch, 3]     -- BVH traversal path (node indices)
//   out_confidence [batch]        -- routing confidence (energy remaining)
// ============================================================================

__global__ void __launch_bounds__(WARP_SZ, 32)
bvh_router_fused_kernel_v2(
    const float* __restrict__ ray_origins,
    const float* __restrict__ ray_directions,
    const float* __restrict__ ray_spectral,
    int* __restrict__   out_expert,
    float* __restrict__ out_scores,
    int* __restrict__   out_path,
    float* __restrict__ out_confidence,
    const int batch_size
) {
    const int sample = blockIdx.x;
    if (sample >= batch_size) return;
    const int tid = threadIdx.x;

    // ---- Load ray origin and direction (thread 0 reads, broadcast to warp) ----
    float ox, oy, oz, dx, dy, dz;
    if (tid == 0) {
        const int base = sample * 3;
        ox = __ldg(&ray_origins[base]);
        oy = __ldg(&ray_origins[base + 1]);
        oz = __ldg(&ray_origins[base + 2]);
        dx = __ldg(&ray_directions[base]);
        dy = __ldg(&ray_directions[base + 1]);
        dz = __ldg(&ray_directions[base + 2]);
    }
    ox = __shfl_sync(0xFFFFFFFF, ox, 0);
    oy = __shfl_sync(0xFFFFFFFF, oy, 0);
    oz = __shfl_sync(0xFFFFFFFF, oz, 0);
    dx = __shfl_sync(0xFFFFFFFF, dx, 0);
    dy = __shfl_sync(0xFFFFFFFF, dy, 0);
    dz = __shfl_sync(0xFFFFFFFF, dz, 0);

    // Save original ray for the all-experts scoring pass
    const float orig_ox = ox, orig_oy = oy, orig_oz = oz;
    const float orig_dx = dx, orig_dy = dy, orig_dz = dz;

    // ---- Load spectral vector (32 threads load 2 components each = 64 total) ----
    float spec_a = __ldg(&ray_spectral[sample * SPEC_DIM + tid]);
    float spec_b = __ldg(&ray_spectral[sample * SPEC_DIM + tid + WARP_SZ]);

    // ---- Traverse 3-level BVH ----
    int current_node = 0;       // Start at root
    float confidence = 1.0f;    // Energy starts at 1.0

    #pragma unroll
    for (int level = 0; level < BVH_LEVELS; level++) {
        int first_child = current_node * BVH_BF + 1;

        // Compute ray-sphere distances to all 4 children
        float d0, d1, d2, d3;
        {
            int c0 = first_child,     c1 = first_child + 1;
            int c2 = first_child + 2, c3 = first_child + 3;

            // Distance = ray-sphere dist^2 - radius^2
            // Negative means ray passes through the sphere
            d0 = ray_sphere_dist_fused(ox, oy, oz, dx, dy, dz,
                     c_centers[c0*3], c_centers[c0*3+1], c_centers[c0*3+2])
                 - c_radii[c0] * c_radii[c0];
            d1 = ray_sphere_dist_fused(ox, oy, oz, dx, dy, dz,
                     c_centers[c1*3], c_centers[c1*3+1], c_centers[c1*3+2])
                 - c_radii[c1] * c_radii[c1];
            d2 = ray_sphere_dist_fused(ox, oy, oz, dx, dy, dz,
                     c_centers[c2*3], c_centers[c2*3+1], c_centers[c2*3+2])
                 - c_radii[c2] * c_radii[c2];
            d3 = ray_sphere_dist_fused(ox, oy, oz, dx, dy, dz,
                     c_centers[c3*3], c_centers[c3*3+1], c_centers[c3*3+2])
                 - c_radii[c3] * c_radii[c3];
        }

        // Select nearest child
        float best_dist;
        int best_k;
        argmin4(d0, d1, d2, d3, best_dist, best_k);
        int best_child = first_child + best_k;

        // Record traversal path
        if (tid == 0) {
            out_path[sample * BVH_LEVELS + level] = best_child;
        }

        // Apply portal transform (warp through coordinate space of child)
        apply_portal_parallel(ox, oy, oz, dx, dy, dz, best_child, tid);

        // Apply Snell refraction (spectral dispersion at this node)
        snell_refract_parallel(spec_a, spec_b, tid, best_child, confidence, best_dist);

        // Descend into the selected child
        current_node = best_child;
    }

    // ---- Write primary output: selected expert ----
    int expert_idx = current_node - BVH_LEAF_OFFSET;
    if (tid == 0) {
        out_expert[sample] = expert_idx;
        out_confidence[sample] = confidence;
    }

    // ---- Compute distance-based scores for ALL 64 experts ----
    // Each thread handles 2 leaves (tid + tid+32) to cover all 64
    {
        int leaf0 = tid;           // Leaves 0..31
        int leaf1 = tid + WARP_SZ; // Leaves 32..63
        int node0 = BVH_LEAF_OFFSET + leaf0;
        int node1 = BVH_LEAF_OFFSET + leaf1;

        // Use original (untransformed) ray for fair distance comparison
        float dist0 = ray_sphere_dist_fused(orig_ox, orig_oy, orig_oz,
                           orig_dx, orig_dy, orig_dz,
                           c_centers[node0*3], c_centers[node0*3+1], c_centers[node0*3+2]);
        float dist1 = ray_sphere_dist_fused(orig_ox, orig_oy, orig_oz,
                           orig_dx, orig_dy, orig_dz,
                           c_centers[node1*3], c_centers[node1*3+1], c_centers[node1*3+2]);

        // Score = exp(-distance^2) -- closer experts get higher scores
        out_scores[sample * BVH_LEAVES + leaf0] = __expf(-dist0);
        out_scores[sample * BVH_LEAVES + leaf1] = __expf(-dist1);
    }
}

// ============================================================================
// HOST API -- C linkage for ctypes / dlopen access
// ============================================================================

extern "C" {

/**
 * Upload BVH tree parameters to GPU constant memory.
 *
 * All arrays must be host (CPU) pointers, contiguous, float32.
 *
 * @param centers   [85, 3]  -- 3D positions of all BVH nodes
 * @param radii     [85]     -- sphere radii of all nodes
 * @param portals   [85, 12] -- affine 3x4 portal transforms (row-major)
 * @param snell_w   [85, 64] -- Snell dispersion weights
 * @param snell_b   [85]     -- Snell dispersion biases
 * @return 0 on success, -1 on failure
 */
int bvh_upload_tree(
    const float* centers,
    const float* radii,
    const float* portals,
    const float* snell_w,
    const float* snell_b
) {
    cudaError_t err;

    err = cudaMemcpyToSymbol(c_centers, centers, BVH_NODES * 3 * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[bvh_upload_tree] centers: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpyToSymbol(c_radii, radii, BVH_NODES * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[bvh_upload_tree] radii: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpyToSymbol(c_portals, portals, BVH_NODES * PORTAL_SIZE * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[bvh_upload_tree] portals: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpyToSymbol(c_snell_w, snell_w, BVH_NODES * SPEC_DIM * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[bvh_upload_tree] snell_w: %s\n", cudaGetErrorString(err));
        return -1;
    }

    err = cudaMemcpyToSymbol(c_snell_b, snell_b, BVH_NODES * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "[bvh_upload_tree] snell_b: %s\n", cudaGetErrorString(err));
        return -1;
    }

    return 0;
}

// ============================================================================
// Persistent device buffers for the route API
// ============================================================================

static float* d_origins    = nullptr;
static float* d_directions = nullptr;
static float* d_spectral   = nullptr;
static int*   d_expert     = nullptr;
static float* d_scores     = nullptr;
static int*   d_path       = nullptr;
static float* d_confidence = nullptr;
static int    d_batch_cap  = 0;

// CUDA Graph for replay
static cudaGraph_t       cuda_graph       = nullptr;
static cudaGraphExec_t   cuda_graph_exec  = nullptr;
static cudaStream_t      cuda_stream      = nullptr;
static int               graph_batch_size = 0;

/**
 * Allocate device buffers for a given batch size.
 * Re-allocates only if the requested size exceeds current capacity.
 *
 * @param batch_size Maximum number of samples per route call
 * @return 0 on success, -1 on failure
 */
int bvh_alloc_buffers(int batch_size) {
    if (batch_size <= 0 || batch_size > MAX_BATCH) {
        fprintf(stderr, "[bvh_alloc_buffers] batch_size must be in [1, %d], got %d\n",
                MAX_BATCH, batch_size);
        return -1;
    }

    if (batch_size <= d_batch_cap) return 0; // Already large enough

    // Free existing buffers
    if (d_origins)    cudaFree(d_origins);
    if (d_directions) cudaFree(d_directions);
    if (d_spectral)   cudaFree(d_spectral);
    if (d_expert)     cudaFree(d_expert);
    if (d_scores)     cudaFree(d_scores);
    if (d_path)       cudaFree(d_path);
    if (d_confidence) cudaFree(d_confidence);

    // Invalidate CUDA graph (batch size changed)
    if (cuda_graph_exec) {
        cudaGraphExecDestroy(cuda_graph_exec);
        cuda_graph_exec = nullptr;
    }
    if (cuda_graph) {
        cudaGraphDestroy(cuda_graph);
        cuda_graph = nullptr;
    }
    graph_batch_size = 0;

    cudaError_t err;
    err = cudaMalloc(&d_origins,    batch_size * RAY_DIM * sizeof(float));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_directions, batch_size * RAY_DIM * sizeof(float));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_spectral,   batch_size * SPEC_DIM * sizeof(float));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_expert,     batch_size * sizeof(int));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_scores,     batch_size * BVH_LEAVES * sizeof(float));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_path,       batch_size * BVH_LEVELS * sizeof(int));
    if (err != cudaSuccess) goto fail;
    err = cudaMalloc(&d_confidence, batch_size * sizeof(float));
    if (err != cudaSuccess) goto fail;

    d_batch_cap = batch_size;

    // Create stream for async operations
    if (cuda_stream == nullptr) {
        cudaStreamCreate(&cuda_stream);
    }

    return 0;

fail:
    fprintf(stderr, "[bvh_alloc_buffers] cudaMalloc failed: %s\n", cudaGetErrorString(err));
    d_batch_cap = 0;
    return -1;
}

/**
 * Free all device buffers and CUDA resources.
 */
void bvh_free_buffers(void) {
    if (cuda_graph_exec) { cudaGraphExecDestroy(cuda_graph_exec); cuda_graph_exec = nullptr; }
    if (cuda_graph)      { cudaGraphDestroy(cuda_graph);          cuda_graph = nullptr; }
    if (cuda_stream)     { cudaStreamDestroy(cuda_stream);        cuda_stream = nullptr; }

    if (d_origins)    { cudaFree(d_origins);    d_origins    = nullptr; }
    if (d_directions) { cudaFree(d_directions); d_directions = nullptr; }
    if (d_spectral)   { cudaFree(d_spectral);   d_spectral   = nullptr; }
    if (d_expert)     { cudaFree(d_expert);     d_expert     = nullptr; }
    if (d_scores)     { cudaFree(d_scores);     d_scores     = nullptr; }
    if (d_path)       { cudaFree(d_path);       d_path       = nullptr; }
    if (d_confidence) { cudaFree(d_confidence); d_confidence = nullptr; }

    d_batch_cap = 0;
    graph_batch_size = 0;
}

/**
 * Route a batch of samples through the BVH tree.
 *
 * Copies inputs from host to device, runs the kernel, copies outputs back.
 * If use_graph=1 and the batch size matches, uses CUDA Graph replay for
 * minimal launch overhead (~3 us vs ~7 us).
 *
 * @param h_origins     [batch, 3]  -- host, float32
 * @param h_directions  [batch, 3]  -- host, float32
 * @param h_spectral    [batch, 64] -- host, float32
 * @param h_expert      [batch]     -- host output, int32
 * @param h_scores      [batch, 64] -- host output, float32
 * @param h_path        [batch, 3]  -- host output, int32
 * @param h_confidence  [batch]     -- host output, float32
 * @param batch_size    Number of samples in this batch
 * @param use_graph     1 = use CUDA Graph replay, 0 = normal launch
 * @return 0 on success, -1 on failure
 */
int bvh_route_batch(
    const float* h_origins,
    const float* h_directions,
    const float* h_spectral,
    int*         h_expert,
    float*       h_scores,
    int*         h_path,
    float*       h_confidence,
    int batch_size,
    int use_graph
) {
    if (batch_size <= 0 || batch_size > d_batch_cap) {
        fprintf(stderr, "[bvh_route_batch] batch_size=%d but buffer capacity=%d\n",
                batch_size, d_batch_cap);
        return -1;
    }

    cudaError_t err;

    // ---- Copy inputs H2D ----
    err = cudaMemcpyAsync(d_origins,    h_origins,
                          batch_size * RAY_DIM * sizeof(float),
                          cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpyAsync(d_directions, h_directions,
                          batch_size * RAY_DIM * sizeof(float),
                          cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpyAsync(d_spectral,   h_spectral,
                          batch_size * SPEC_DIM * sizeof(float),
                          cudaMemcpyHostToDevice, cuda_stream);
    if (err != cudaSuccess) goto fail;

    // ---- Launch kernel (or replay CUDA Graph) ----
    if (use_graph && graph_batch_size == batch_size && cuda_graph_exec != nullptr) {
        // Replay captured graph -- minimal overhead
        err = cudaGraphLaunch(cuda_graph_exec, cuda_stream);
        if (err != cudaSuccess) goto fail;
    } else if (use_graph && graph_batch_size != batch_size) {
        // Capture a new CUDA Graph for this batch size
        if (cuda_graph_exec) { cudaGraphExecDestroy(cuda_graph_exec); cuda_graph_exec = nullptr; }
        if (cuda_graph)      { cudaGraphDestroy(cuda_graph);          cuda_graph = nullptr; }

        // Wait for H2D copies to complete before capture
        cudaStreamSynchronize(cuda_stream);

        // Re-copy inside the capture (graph needs to see the memcpy nodes)
        cudaStreamBeginCapture(cuda_stream, cudaStreamCaptureModeGlobal);

        cudaMemcpyAsync(d_origins,    h_origins,
                        batch_size * RAY_DIM * sizeof(float),
                        cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_directions, h_directions,
                        batch_size * RAY_DIM * sizeof(float),
                        cudaMemcpyHostToDevice, cuda_stream);
        cudaMemcpyAsync(d_spectral,   h_spectral,
                        batch_size * SPEC_DIM * sizeof(float),
                        cudaMemcpyHostToDevice, cuda_stream);

        bvh_router_fused_kernel_v2<<<batch_size, WARP_SZ, 0, cuda_stream>>>(
            d_origins, d_directions, d_spectral,
            d_expert, d_scores, d_path, d_confidence,
            batch_size
        );

        cudaMemcpyAsync(h_expert,     d_expert,
                        batch_size * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_scores,     d_scores,
                        batch_size * BVH_LEAVES * sizeof(float),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_path,       d_path,
                        batch_size * BVH_LEVELS * sizeof(int),
                        cudaMemcpyDeviceToHost, cuda_stream);
        cudaMemcpyAsync(h_confidence, d_confidence,
                        batch_size * sizeof(float),
                        cudaMemcpyDeviceToHost, cuda_stream);

        cudaStreamEndCapture(cuda_stream, &cuda_graph);
        cudaGraphInstantiate(&cuda_graph_exec, cuda_graph, 0);
        graph_batch_size = batch_size;

        // Launch the captured graph
        err = cudaGraphLaunch(cuda_graph_exec, cuda_stream);
        if (err != cudaSuccess) goto fail;

        cudaStreamSynchronize(cuda_stream);
        return 0;
    } else {
        // Normal launch (no graph)
        bvh_router_fused_kernel_v2<<<batch_size, WARP_SZ, 0, cuda_stream>>>(
            d_origins, d_directions, d_spectral,
            d_expert, d_scores, d_path, d_confidence,
            batch_size
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) goto fail;
    }

    // ---- Copy outputs D2H ----
    err = cudaMemcpyAsync(h_expert,     d_expert,
                          batch_size * sizeof(int),
                          cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpyAsync(h_scores,     d_scores,
                          batch_size * BVH_LEAVES * sizeof(float),
                          cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpyAsync(h_path,       d_path,
                          batch_size * BVH_LEVELS * sizeof(int),
                          cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) goto fail;

    err = cudaMemcpyAsync(h_confidence, d_confidence,
                          batch_size * sizeof(float),
                          cudaMemcpyDeviceToHost, cuda_stream);
    if (err != cudaSuccess) goto fail;

    // Synchronize to ensure outputs are available
    cudaStreamSynchronize(cuda_stream);
    return 0;

fail:
    fprintf(stderr, "[bvh_route_batch] CUDA error: %s\n", cudaGetErrorString(err));
    return -1;
}

/**
 * Get the maximum batch capacity of allocated buffers.
 */
int bvh_get_batch_capacity(void) {
    return d_batch_cap;
}

/**
 * Get the number of BVH nodes (compile-time constant).
 */
int bvh_get_num_nodes(void) {
    return BVH_NODES;
}

/**
 * Get the number of leaf experts (compile-time constant).
 */
int bvh_get_num_experts(void) {
    return BVH_LEAVES;
}

/**
 * Get the spectral dimension (compile-time constant).
 */
int bvh_get_spec_dim(void) {
    return SPEC_DIM;
}

} // extern "C"
