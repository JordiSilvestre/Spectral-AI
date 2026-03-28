/**
 * bvh_torch_ext.cu — Extensión PyTorch Zero-Copy para BVH Router
 * LiquidBit v5.0 "Orchestrator"
 *
 * Envuelve bvh_router_fused_kernel_v2 para uso directo con tensores CUDA de PyTorch.
 * Los datos nunca salen de la GPU — zero CPU transfers en el hot path.
 *
 * Compilar con:
 *   python build_ext.py
 *
 * Usar:
 *   import bvh_router_ext
 *   bvh_router_ext.upload_tree(centers, radii, portals, snell_w, snell_b)
 *   expert_ids, scores, confidence = bvh_router_ext.route(origins, directions, spectral)
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// ============================================================================
// Constantes (deben coincidir EXACTAMENTE con bvh_router_kernel.cu)
// ============================================================================

#define BVH_BF            4
#define BVH_LEVELS        3
#define BVH_LEAVES       64
#define BVH_NODES        85
#define BVH_LEAF_OFFSET  21
#define RAY_DIM           3
#define SPEC_DIM         64
#define PORTAL_SIZE      12
#define WARP_SZ          32
#define MAX_BATCH       1024

#define ETA_MIN          0.5f
#define ETA_MAX          2.0f
#define ETA_RANGE        1.5f
#define BAND_F0          0.000f
#define BAND_F1          0.250f
#define BAND_F2          0.500f
#define BAND_F3          0.750f

// ============================================================================
// CONSTANT MEMORY — Copia local para esta TU (Translation Unit)
// Cada .cu compilado tiene su propio espacio de constant memory.
// ============================================================================

__constant__ float c_centers[BVH_NODES * 3];
__constant__ float c_radii[BVH_NODES];
__constant__ float c_portals[BVH_NODES * PORTAL_SIZE];
__constant__ float c_snell_w[BVH_NODES * SPEC_DIM];
__constant__ float c_snell_b[BVH_NODES];

// ============================================================================
// Device: helpers (copiados de bvh_router_kernel.cu)
// ============================================================================

__device__ __forceinline__ float fast_sigmoid(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

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

__device__ __forceinline__ void apply_portal_parallel(
    float& ox, float& oy, float& oz,
    float& dx, float& dy, float& dz,
    int node_idx, int tid
) {
    if (tid < 3) {
        const float* p = &c_portals[node_idx * PORTAL_SIZE + tid * 4];
        float p0 = p[0], p1 = p[1], p2 = p[2], p3 = p[3];
        float new_o = __fmaf_rn(p0, ox, __fmaf_rn(p1, oy, __fmaf_rn(p2, oz, p3)));
        float new_d = __fmaf_rn(p0, dx, __fmaf_rn(p1, dy, p2 * dz));
        float no_x = __shfl_sync(0xFFFFFFFF, new_o, 0);
        float no_y = __shfl_sync(0xFFFFFFFF, new_o, 1);
        float no_z = __shfl_sync(0xFFFFFFFF, new_o, 2);
        float nd_x = __shfl_sync(0xFFFFFFFF, new_d, 0);
        float nd_y = __shfl_sync(0xFFFFFFFF, new_d, 1);
        float nd_z = __shfl_sync(0xFFFFFFFF, new_d, 2);
        ox = no_x; oy = no_y; oz = no_z;
        dx = nd_x; dy = nd_y; dz = nd_z;
    } else {
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
    float inv_norm = rsqrtf(__fmaf_rn(dx, dx, __fmaf_rn(dy, dy, dz * dz)) + 1e-12f);
    dx *= inv_norm; dy *= inv_norm; dz *= inv_norm;
}

__device__ __forceinline__ void snell_refract_parallel(
    float& spec_a, float& spec_b,
    int tid, int node_idx,
    float& confidence, float best_dist
) {
    int idx_a = tid;
    int idx_b = tid + WARP_SZ;
    float wa = c_snell_w[node_idx * SPEC_DIM + idx_a];
    float wb = c_snell_w[node_idx * SPEC_DIM + idx_b];
    float partial = __fmaf_rn(wa, spec_a, wb * spec_b);

    #pragma unroll
    for (int offset = WARP_SZ / 2; offset > 0; offset >>= 1) {
        partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
    }
    float dot_val = __shfl_sync(0xFFFFFFFF, partial, 0);
    float eta = fast_sigmoid(dot_val + c_snell_b[node_idx]);
    eta = __fmaf_rn(eta, ETA_RANGE, ETA_MIN);
    float inv_eta = __frcp_rn(eta);
    float eta_diff = inv_eta - eta;

    float band_a;
    switch (idx_a & 3) {
        case 0: band_a = BAND_F0; break;
        case 1: band_a = BAND_F1; break;
        case 2: band_a = BAND_F2; break;
        default: band_a = BAND_F3; break;
    }
    spec_a *= __fmaf_rn(band_a, eta_diff, eta);
    spec_b *= __fmaf_rn(band_a, eta_diff, eta);

    if (tid == 0) {
        confidence *= __expf(-best_dist);
    }
    confidence = __shfl_sync(0xFFFFFFFF, confidence, 0);
}

// ============================================================================
// KERNEL — Copia exacta de bvh_router_fused_kernel_v2
// ============================================================================

__global__ void __launch_bounds__(WARP_SZ, 32)
bvh_router_fused_kernel(
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

    const float orig_ox = ox, orig_oy = oy, orig_oz = oz;
    const float orig_dx = dx, orig_dy = dy, orig_dz = dz;

    float spec_a = __ldg(&ray_spectral[sample * SPEC_DIM + tid]);
    float spec_b = __ldg(&ray_spectral[sample * SPEC_DIM + tid + WARP_SZ]);

    int current_node = 0;
    float confidence = 1.0f;

    #pragma unroll
    for (int level = 0; level < BVH_LEVELS; level++) {
        int first_child = current_node * BVH_BF + 1;

        float d0, d1, d2, d3;
        {
            int c0 = first_child, c1 = first_child + 1;
            int c2 = first_child + 2, c3 = first_child + 3;

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

        float best_dist;
        int best_k;
        argmin4(d0, d1, d2, d3, best_dist, best_k);
        int best_child = first_child + best_k;

        if (tid == 0) {
            out_path[sample * BVH_LEVELS + level] = best_child;
        }

        apply_portal_parallel(ox, oy, oz, dx, dy, dz, best_child, tid);
        snell_refract_parallel(spec_a, spec_b, tid, best_child, confidence, best_dist);
        current_node = best_child;
    }

    int expert_idx = current_node - BVH_LEAF_OFFSET;
    if (tid == 0) {
        out_expert[sample] = expert_idx;
        out_confidence[sample] = confidence;
    }

    {
        int leaf0 = tid;
        int leaf1 = tid + WARP_SZ;
        int node0 = BVH_LEAF_OFFSET + leaf0;
        int node1 = BVH_LEAF_OFFSET + leaf1;

        float dist0 = ray_sphere_dist_fused(orig_ox, orig_oy, orig_oz,
                           orig_dx, orig_dy, orig_dz,
                           c_centers[node0*3], c_centers[node0*3+1], c_centers[node0*3+2]);
        float dist1 = ray_sphere_dist_fused(orig_ox, orig_oy, orig_oz,
                           orig_dx, orig_dy, orig_dz,
                           c_centers[node1*3], c_centers[node1*3+1], c_centers[node1*3+2]);

        out_scores[sample * BVH_LEAVES + leaf0] = __expf(-dist0);
        out_scores[sample * BVH_LEAVES + leaf1] = __expf(-dist1);
    }
}

// ============================================================================
// HOST: Upload árbol BVH a constant memory
// ============================================================================

void upload_tree_impl(
    torch::Tensor centers,
    torch::Tensor radii,
    torch::Tensor portals,
    torch::Tensor snell_w,
    torch::Tensor snell_b
) {
    TORCH_CHECK(centers.is_cpu() && centers.is_contiguous() && centers.dtype() == torch::kFloat32,
        "centers: CPU contiguous float32 [85, 3]");
    TORCH_CHECK(radii.is_cpu() && radii.is_contiguous() && radii.dtype() == torch::kFloat32,
        "radii: CPU contiguous float32 [85]");
    TORCH_CHECK(portals.is_cpu() && portals.is_contiguous() && portals.dtype() == torch::kFloat32,
        "portals: CPU contiguous float32 [85, 12]");
    TORCH_CHECK(snell_w.is_cpu() && snell_w.is_contiguous() && snell_w.dtype() == torch::kFloat32,
        "snell_w: CPU contiguous float32 [85, 64]");
    TORCH_CHECK(snell_b.is_cpu() && snell_b.is_contiguous() && snell_b.dtype() == torch::kFloat32,
        "snell_b: CPU contiguous float32 [85]");

    cudaError_t err;
    err = cudaMemcpyToSymbol(c_centers, centers.data_ptr<float>(), BVH_NODES * 3 * sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "upload centers: ", cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(c_radii, radii.data_ptr<float>(), BVH_NODES * sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "upload radii: ", cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(c_portals, portals.data_ptr<float>(), BVH_NODES * PORTAL_SIZE * sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "upload portals: ", cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(c_snell_w, snell_w.data_ptr<float>(), BVH_NODES * SPEC_DIM * sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "upload snell_w: ", cudaGetErrorString(err));

    err = cudaMemcpyToSymbol(c_snell_b, snell_b.data_ptr<float>(), BVH_NODES * sizeof(float));
    TORCH_CHECK(err == cudaSuccess, "upload snell_b: ", cudaGetErrorString(err));
}

// ============================================================================
// HOST: Route — Zero-Copy con tensores PyTorch
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> route_impl(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor ray_spectral
) {
    TORCH_CHECK(ray_origins.is_cuda() && ray_origins.is_contiguous(),
        "ray_origins: CUDA contiguous float32 [batch, 3]");
    TORCH_CHECK(ray_directions.is_cuda() && ray_directions.is_contiguous(),
        "ray_directions: CUDA contiguous float32 [batch, 3]");
    TORCH_CHECK(ray_spectral.is_cuda() && ray_spectral.is_contiguous(),
        "ray_spectral: CUDA contiguous float32 [batch, 64]");

    const int batch_size = ray_origins.size(0);
    TORCH_CHECK(batch_size > 0 && batch_size <= MAX_BATCH,
        "batch_size debe estar en [1, ", MAX_BATCH, "], got ", batch_size);

    auto device = ray_origins.device();

    // Pre-alocar tensores de salida en el MISMO device (zero-copy out)
    auto expert_ids = torch::empty({batch_size}, torch::dtype(torch::kInt32).device(device));
    auto scores     = torch::empty({batch_size, BVH_LEAVES}, torch::dtype(torch::kFloat32).device(device));
    auto confidence = torch::empty({batch_size}, torch::dtype(torch::kFloat32).device(device));
    auto path       = torch::empty({batch_size, BVH_LEVELS}, torch::dtype(torch::kInt32).device(device));

    // Obtener stream de PyTorch (sincronización automática)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Lanzar kernel: 1 bloque = 1 warp = 1 sample
    bvh_router_fused_kernel<<<batch_size, WARP_SZ, 0, stream>>>(
        ray_origins.data_ptr<float>(),       // zero-copy in
        ray_directions.data_ptr<float>(),    // zero-copy in
        ray_spectral.data_ptr<float>(),      // zero-copy in
        expert_ids.data_ptr<int>(),          // zero-copy out
        scores.data_ptr<float>(),            // zero-copy out
        path.data_ptr<int>(),                // zero-copy out
        confidence.data_ptr<float>(),        // zero-copy out
        batch_size
    );

    // Verificar errores de lanzamiento
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "BVH kernel launch failed: ", cudaGetErrorString(err));

    return std::make_tuple(expert_ids, scores, confidence);
}

// ============================================================================
// HOST: Route síncrono (para benchmark — fuerza sincronización)
// ============================================================================

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> route_sync_impl(
    torch::Tensor ray_origins,
    torch::Tensor ray_directions,
    torch::Tensor ray_spectral
) {
    auto result = route_impl(ray_origins, ray_directions, ray_spectral);
    cudaDeviceSynchronize();
    return result;
}

// ============================================================================
// Pybind11: registrar módulo
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "LiquidBit BVH Router — Extensión PyTorch Zero-Copy";

    m.def("upload_tree", &upload_tree_impl,
          "Subir árbol BVH a constant memory GPU (tensores CPU)",
          py::arg("centers"), py::arg("radii"), py::arg("portals"),
          py::arg("snell_w"), py::arg("snell_b"));

    m.def("route", &route_impl,
          "Routing BVH zero-copy (tensores CUDA). Retorna (expert_ids, scores, confidence)",
          py::arg("ray_origins"), py::arg("ray_directions"), py::arg("ray_spectral"));

    m.def("route_sync", &route_sync_impl,
          "Routing BVH síncrono para benchmark (fuerza cudaDeviceSynchronize)",
          py::arg("ray_origins"), py::arg("ray_directions"), py::arg("ray_spectral"));
}
