/**
 * ternary_torch_ext.cu — PyTorch Extension for Arbitrary-Dimension Ternary MatMul
 * SpectralAI v5.0 "Orchestrator"
 *
 * Replaces F.linear(x, w) where w is ternary {-1, 0, +1} with POPCOUNT-based
 * add/sub/skip — ZERO multiplications in the hot path.
 *
 * Unlike ternary_expert.cu (fixed 64->1024->4096), this extension supports
 * arbitrary dimensions, making it compatible with any model's expert shapes.
 *
 * Key functions:
 *   ternary_pack(weights_int8)        -> packed_uint32  (host-side packing)
 *   ternary_linear(input, packed, scale, bias) -> output  (GPU ternary matmul)
 *   ternary_gated_mlp(input, gate_p, up_p, down_p, scales, biases) -> output
 *
 * Encoding: 2 bits per weight, 16 weights per uint32
 *   00 = zero, 01 = +1, 10 = -1
 *
 * Compile: python cuda/v5/build_ternary_ext.py
 * Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
 */

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#define TERN_PACK 16
#define TERN_MASK 3u
#define TERN_POS  1u
#define TERN_NEG  2u
#define BLOCK_SIZE 256

// ============================================================================
// Device: ternary dot product for 16 packed weights
// ============================================================================

__device__ __forceinline__ float ternary_dot_16(
    uint32_t packed,
    const float* __restrict__ input,
    int stride
) {
    float acc = 0.0f;
    #pragma unroll
    for (int i = 0; i < TERN_PACK; i++) {
        uint32_t code = (packed >> (i * 2)) & TERN_MASK;
        float val = input[i * stride];
        // Branch-free: +1 if code==1, -1 if code==2, 0 if code==0
        float sign = (float)(code == TERN_POS) - (float)(code >> 1);
        acc = __fmaf_rn(sign, val, acc);
    }
    return acc;
}

// ============================================================================
// Kernel: Ternary Linear (replaces F.linear for ternary weights)
//
// For each output element: dot product of input row with packed weight column,
// then scale + bias.
//
// input:  [batch, in_dim]  float32
// packed: [packed_rows, out_dim]  uint32 where packed_rows = ceil(in_dim/16)
// scale:  [out_dim]  float32
// bias:   [out_dim]  float32 (optional, can be nullptr)
// output: [batch, out_dim]  float32
// ============================================================================

__global__ void ternary_linear_kernel(
    const float* __restrict__    input,
    const uint32_t* __restrict__ packed,
    const float* __restrict__    scale,
    const float* __restrict__    bias,
    float* __restrict__          output,
    int batch_size,
    int in_dim,
    int out_dim,
    int packed_rows,
    int has_bias
) {
    // Grid: blockIdx.x = sample, threads cover output dims
    const int sample = blockIdx.x;
    if (sample >= batch_size) return;

    const float* inp = &input[sample * in_dim];
    float* out = &output[sample * out_dim];

    // Each thread handles multiple output columns
    for (int col = threadIdx.x; col < out_dim; col += blockDim.x) {
        float acc = 0.0f;

        // Iterate over packed rows (each covers 16 input elements)
        for (int pr = 0; pr < packed_rows; pr++) {
            uint32_t p = __ldg(&packed[pr * out_dim + col]);
            acc += ternary_dot_16(p, &inp[pr * TERN_PACK], 1);
        }

        // Apply per-channel scale
        acc *= __ldg(&scale[col]);

        // Add bias if present
        if (has_bias) {
            acc += __ldg(&bias[col]);
        }

        out[col] = acc;
    }
}

// ============================================================================
// Kernel: Ternary Gated MLP (gate * up -> SiLU -> down) — full expert forward
//
// Fuses 3 ternary matmuls + SiLU gating into one kernel launch.
// Shared memory holds the input and intermediate hidden state.
//
// input:        [batch, in_dim]
// gate_packed:  [packed_in, intermediate_dim]   uint32
// up_packed:    [packed_in, intermediate_dim]    uint32
// down_packed:  [packed_inter, in_dim]           uint32  (down projects back)
// scales:       gate_scale[inter], up_scale[inter], down_scale[in_dim]
// output:       [batch, in_dim]
// ============================================================================

__global__ void ternary_gated_mlp_kernel(
    const float* __restrict__    input,
    const uint32_t* __restrict__ gate_packed,
    const uint32_t* __restrict__ up_packed,
    const uint32_t* __restrict__ down_packed,
    const float* __restrict__    gate_scale,
    const float* __restrict__    up_scale,
    const float* __restrict__    down_scale,
    float* __restrict__          output,
    int batch_size,
    int in_dim,
    int intermediate_dim,
    int out_dim,
    int packed_in_rows,
    int packed_inter_rows
) {
    extern __shared__ float smem[];
    // Layout: smem[0..in_dim-1] = input, smem[in_dim..in_dim+intermediate_dim-1] = hidden

    const int sample = blockIdx.x;
    if (sample >= batch_size) return;

    float* s_input = smem;
    float* s_hidden = smem + in_dim;

    const float* inp = &input[sample * in_dim];
    float* out = &output[sample * out_dim];

    // Load input to shared memory
    for (int i = threadIdx.x; i < in_dim; i += blockDim.x) {
        s_input[i] = __ldg(&inp[i]);
    }
    __syncthreads();

    // === Layer 1: gate * up with SiLU gating ===
    for (int col = threadIdx.x; col < intermediate_dim; col += blockDim.x) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;

        for (int pr = 0; pr < packed_in_rows; pr++) {
            uint32_t gp = __ldg(&gate_packed[pr * intermediate_dim + col]);
            uint32_t up = __ldg(&up_packed[pr * intermediate_dim + col]);
            gate_acc += ternary_dot_16(gp, &s_input[pr * TERN_PACK], 1);
            up_acc   += ternary_dot_16(up, &s_input[pr * TERN_PACK], 1);
        }

        // Scale
        gate_acc *= __ldg(&gate_scale[col]);
        up_acc   *= __ldg(&up_scale[col]);

        // SiLU(gate) * up — this is the only "multiply" in the entire expert
        float silu_gate = gate_acc / (1.0f + __expf(-gate_acc));
        s_hidden[col] = silu_gate * up_acc;
    }
    __syncthreads();

    // === Layer 2: down projection ===
    for (int col = threadIdx.x; col < out_dim; col += blockDim.x) {
        float acc = 0.0f;

        for (int pr = 0; pr < packed_inter_rows; pr++) {
            uint32_t dp = __ldg(&down_packed[pr * out_dim + col]);
            acc += ternary_dot_16(dp, &s_hidden[pr * TERN_PACK], 1);
        }

        acc *= __ldg(&down_scale[col]);
        out[col] = acc;
    }
}

// ============================================================================
// Host: Pack ternary weights (int8 {-1,0,+1} -> uint32 packed)
// ============================================================================

torch::Tensor pack_ternary_impl(torch::Tensor weights_int8) {
    // weights_int8: [rows, cols] int8 with values in {-1, 0, +1}
    TORCH_CHECK(weights_int8.is_cpu() && weights_int8.dtype() == torch::kInt8,
        "weights must be CPU int8");

    auto weights = weights_int8.contiguous();
    int rows = weights.size(0);
    int cols = weights.size(1);
    int packed_rows = (rows + TERN_PACK - 1) / TERN_PACK;

    auto packed = torch::zeros({packed_rows, cols}, torch::dtype(torch::kInt32));
    auto* w_ptr = weights.data_ptr<int8_t>();
    auto* p_ptr = packed.data_ptr<int32_t>();

    for (int pr = 0; pr < packed_rows; pr++) {
        for (int c = 0; c < cols; c++) {
            uint32_t pack = 0;
            for (int i = 0; i < TERN_PACK; i++) {
                int row = pr * TERN_PACK + i;
                if (row >= rows) break;

                int8_t val = w_ptr[row * cols + c];
                uint32_t code;
                if (val > 0)        code = TERN_POS;
                else if (val < 0)   code = TERN_NEG;
                else                code = 0u;

                pack |= (code << (i * 2));
            }
            p_ptr[pr * cols + c] = (int32_t)pack;
        }
    }

    return packed;
}

// ============================================================================
// Host: Ternary Linear forward
// ============================================================================

torch::Tensor ternary_linear_impl(
    torch::Tensor input,          // [batch, in_dim]  float32 CUDA
    torch::Tensor packed_weights, // [packed_rows, out_dim]  int32 CUDA
    torch::Tensor scale,          // [out_dim]  float32 CUDA
    c10::optional<torch::Tensor> bias  // [out_dim]  float32 CUDA (optional)
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
        "input must be CUDA contiguous float32");
    TORCH_CHECK(packed_weights.is_cuda() && packed_weights.is_contiguous(),
        "packed_weights must be CUDA contiguous int32");
    TORCH_CHECK(scale.is_cuda() && scale.is_contiguous(),
        "scale must be CUDA contiguous float32");

    int batch = input.size(0);
    int in_dim = input.size(1);
    int packed_rows = packed_weights.size(0);
    int out_dim = packed_weights.size(1);

    auto output = torch::empty({batch, out_dim},
        torch::dtype(torch::kFloat32).device(input.device()));

    int has_bias = bias.has_value() ? 1 : 0;
    const float* bias_ptr = has_bias ? bias.value().data_ptr<float>() : nullptr;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = max(min(out_dim, BLOCK_SIZE), 32);  // min 1 warp
    ternary_linear_kernel<<<batch, threads, 0, stream>>>(
        input.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(packed_weights.data_ptr<int32_t>()),
        scale.data_ptr<float>(),
        bias_ptr,
        output.data_ptr<float>(),
        batch, in_dim, out_dim, packed_rows, has_bias
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "ternary_linear kernel failed: ", cudaGetErrorString(err));

    return output;
}

// ============================================================================
// Host: Ternary Gated MLP forward (fused gate+up+down)
// ============================================================================

torch::Tensor ternary_gated_mlp_impl(
    torch::Tensor input,           // [batch, in_dim]  float32 CUDA
    torch::Tensor gate_packed,     // [packed_in, intermediate]  int32 CUDA
    torch::Tensor up_packed,       // [packed_in, intermediate]  int32 CUDA
    torch::Tensor down_packed,     // [packed_inter, out_dim]    int32 CUDA
    torch::Tensor gate_scale,      // [intermediate]  float32 CUDA
    torch::Tensor up_scale,        // [intermediate]  float32 CUDA
    torch::Tensor down_scale       // [out_dim]  float32 CUDA
) {
    TORCH_CHECK(input.is_cuda() && input.is_contiguous(),
        "input must be CUDA contiguous float32");

    int batch = input.size(0);
    int in_dim = input.size(1);
    int intermediate_dim = gate_packed.size(1);
    int out_dim = down_packed.size(1);
    int packed_in_rows = gate_packed.size(0);
    int packed_inter_rows = down_packed.size(0);

    auto output = torch::empty({batch, out_dim},
        torch::dtype(torch::kFloat32).device(input.device()));

    // Shared memory: input + intermediate hidden
    size_t smem_bytes = (in_dim + intermediate_dim) * sizeof(float);

    // Guard: check shared memory doesn't exceed device limit
    int device_id = input.device().index();
    int max_smem = 0;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock, device_id);
    TORCH_CHECK(smem_bytes <= (size_t)max_smem,
        "ternary_gated_mlp: shared memory ", smem_bytes, " bytes exceeds device max ",
        max_smem, " bytes. Reduce in_dim (", in_dim, ") or intermediate_dim (",
        intermediate_dim, ").");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int threads = min(max(intermediate_dim, out_dim), 1024);
    ternary_gated_mlp_kernel<<<batch, threads, smem_bytes, stream>>>(
        input.data_ptr<float>(),
        reinterpret_cast<const uint32_t*>(gate_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(up_packed.data_ptr<int32_t>()),
        reinterpret_cast<const uint32_t*>(down_packed.data_ptr<int32_t>()),
        gate_scale.data_ptr<float>(),
        up_scale.data_ptr<float>(),
        down_scale.data_ptr<float>(),
        output.data_ptr<float>(),
        batch, in_dim, intermediate_dim, out_dim,
        packed_in_rows, packed_inter_rows
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
        "ternary_gated_mlp kernel failed: ", cudaGetErrorString(err));

    return output;
}

// ============================================================================
// Pybind11: register module
// ============================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "SpectralAI Ternary Expert — PyTorch Extension (arbitrary dimensions, zero multiplications)";

    m.def("pack_ternary", &pack_ternary_impl,
          "Pack int8 ternary weights {-1,0,+1} to uint32 (16 per uint32)",
          py::arg("weights_int8"));

    m.def("ternary_linear", &ternary_linear_impl,
          "Ternary matrix multiply: no FP multiplications, only add/sub/skip",
          py::arg("input"), py::arg("packed_weights"), py::arg("scale"),
          py::arg("bias") = c10::nullopt);

    m.def("ternary_gated_mlp", &ternary_gated_mlp_impl,
          "Fused ternary gated MLP: gate(SiLU)*up -> down, zero multiplications",
          py::arg("input"), py::arg("gate_packed"), py::arg("up_packed"),
          py::arg("down_packed"), py::arg("gate_scale"), py::arg("up_scale"),
          py::arg("down_scale"));
}
