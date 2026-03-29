/**
 * async_pipeline.cu — Tri-Core Async Pipeline for LiquidBit
 *
 * Core insight: Modern GPUs have 3 independent execution engines that can
 * overlap when given separate CUDA streams:
 *
 *   1. RT Cores    — BVH traversal for expert routing (OptiX)
 *   2. CUDA Cores  — Data preparation, index scatter, calibration
 *   3. Tensor Cores — Expert forward pass (cuBLAS GEMM)
 *
 * Pipeline design (triple buffer):
 *
 *   Time →  T0        T1        T2        T3
 *   RT:     [Route₀]  [Route₁]  [Route₂]  [Route₃]
 *   CUDA:             [Prep₀]   [Prep₁]   [Prep₂]
 *   Tensor:                     [Expert₀]  [Expert₁]
 *
 * After warmup (2 tokens), all 3 engines run simultaneously.
 * Steady-state latency ≈ max(route, prep, expert) instead of sum.
 *
 * Copyright (c) 2026 LiquidBit Studio — Apache 2.0
 */

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>

// ============================================================================
// Constants
// ============================================================================

static constexpr int NUM_EXPERTS = 64;
static constexpr int TOP_K = 8;
static constexpr int MAX_BATCH = 512;
static constexpr int NUM_BUFFERS = 3;  // Triple buffering

// ============================================================================
// Pipeline state — one per buffer slot
// ============================================================================

struct PipelineSlot {
    // Inputs (pinned host → device)
    float* d_hidden;          // [batch_size, hidden_dim]
    int batch_size;

    // Routing results (from RT Cores or CUDA kernel)
    uint32_t* d_expert_ids;   // [batch_size, top_k]
    float* d_expert_weights;  // [batch_size, top_k]

    // Scatter indices for expert dispatch
    int* d_dispatch_indices;  // [num_experts, max_tokens_per_expert]
    int* d_expert_counts;     // [num_experts]

    // Softmax output (separate from expert_weights to avoid overwrite)
    float* d_softmax_weights; // [batch_size, top_k]

    // Expert output
    float* d_output;          // [batch_size, hidden_dim]

    // Synchronization
    cudaEvent_t route_done;
    cudaEvent_t prep_done;
    cudaEvent_t expert_done;
};

// ============================================================================
// Kernel: Scatter — group tokens by expert assignment
// ============================================================================

/**
 * After routing, we know which experts each token goes to.
 * This kernel builds per-expert token lists for batched GEMM dispatch.
 *
 * Input:  expert_ids[batch_size][top_k] — which experts selected
 * Output: dispatch_indices[expert][slot] — which tokens go to each expert
 *         expert_counts[expert] — how many tokens per expert
 */
__global__ void scatter_by_expert_kernel(
    const uint32_t* __restrict__ expert_ids,
    const float* __restrict__ expert_weights,
    int* __restrict__ dispatch_indices,
    int* __restrict__ expert_counts,
    const int batch_size,
    const int top_k,
    const int max_tokens_per_expert
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * top_k) return;

    const int token_idx = tid / top_k;
    const uint32_t expert_id = expert_ids[tid];

    if (expert_id >= NUM_EXPERTS) return;  // sentinel from miss

    // Atomic increment to get slot in expert's token list
    const int slot = atomicAdd(&expert_counts[expert_id], 1);
    if (slot < max_tokens_per_expert) {
        dispatch_indices[expert_id * max_tokens_per_expert + slot] = token_idx;
    }
}

// ============================================================================
// Kernel: Weighted combine — merge top-K expert outputs
// ============================================================================

/**
 * After all top-K experts have been evaluated, combine their outputs
 * using the routing weights.
 *
 * output[token] = sum_k(weight[token][k] * expert_output[token][k])
 */
__global__ void weighted_combine_kernel(
    float* __restrict__ output,
    const float* __restrict__ expert_outputs,  // [batch_size, top_k, hidden_dim]
    const float* __restrict__ weights,         // [batch_size, top_k]
    const int batch_size,
    const int hidden_dim,
    const int top_k
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * hidden_dim) return;

    const int token_idx = tid / hidden_dim;
    const int dim_idx = tid % hidden_dim;

    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) {
        const float w = weights[token_idx * top_k + k];
        const float v = expert_outputs[(token_idx * top_k + k) * hidden_dim + dim_idx];
        acc += w * v;
    }
    output[tid] = acc;
}

// ============================================================================
// Kernel: Calibration apply — Linear(64,64) on raw BVH logits
// ============================================================================

/**
 * Applies the learned calibration layer to raw BVH router logits.
 * This is a small matmul (64x64) that corrects the weight distribution.
 *
 * calibrated[i] = sum_j(W[i][j] * raw[j]) + bias[i]
 * Then softmax + top-K selection.
 */
__global__ void apply_calibration_kernel(
    float* __restrict__ calibrated_logits,  // [batch_size, NUM_EXPERTS]
    const float* __restrict__ raw_logits,   // [batch_size, NUM_EXPERTS]
    const float* __restrict__ cal_weight,   // [NUM_EXPERTS, NUM_EXPERTS]
    const float* __restrict__ cal_bias,     // [NUM_EXPERTS]
    const int batch_size
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size * NUM_EXPERTS) return;

    const int token_idx = tid / NUM_EXPERTS;
    const int expert_idx = tid % NUM_EXPERTS;

    float acc = cal_bias[expert_idx];
    for (int j = 0; j < NUM_EXPERTS; ++j) {
        acc += cal_weight[expert_idx * NUM_EXPERTS + j]
             * raw_logits[token_idx * NUM_EXPERTS + j];
    }
    calibrated_logits[tid] = acc;
}

// ============================================================================
// Kernel: Softmax + Top-K selection
// ============================================================================

__global__ void softmax_topk_kernel(
    const float* __restrict__ logits,  // [batch_size, NUM_EXPERTS]
    uint32_t* __restrict__ topk_ids,   // [batch_size, top_k]
    float* __restrict__ topk_weights,  // [batch_size, top_k]
    const int batch_size,
    const int top_k
) {
    const int token_idx = blockIdx.x;
    if (token_idx >= batch_size) return;

    const float* row = logits + token_idx * NUM_EXPERTS;

    // Shared memory for this block's softmax
    __shared__ float s_logits[NUM_EXPERTS];
    __shared__ float s_probs[NUM_EXPERTS];

    // Load logits
    if (threadIdx.x < NUM_EXPERTS) {
        s_logits[threadIdx.x] = row[threadIdx.x];
    }
    __syncthreads();

    // Find max for numerical stability (thread 0)
    if (threadIdx.x == 0) {
        float max_val = s_logits[0];
        for (int i = 1; i < NUM_EXPERTS; ++i) {
            if (s_logits[i] > max_val) max_val = s_logits[i];
        }

        float sum_exp = 0.0f;
        for (int i = 0; i < NUM_EXPERTS; ++i) {
            s_probs[i] = expf(s_logits[i] - max_val);
            sum_exp += s_probs[i];
        }
        for (int i = 0; i < NUM_EXPERTS; ++i) {
            s_probs[i] /= sum_exp;
        }

        // Top-K selection (simple insertion sort for K=8)
        for (int k = 0; k < top_k; ++k) {
            float best_val = -1.0f;
            int best_idx = 0;
            for (int i = 0; i < NUM_EXPERTS; ++i) {
                if (s_probs[i] > best_val) {
                    best_val = s_probs[i];
                    best_idx = i;
                }
            }
            topk_ids[token_idx * top_k + k] = (uint32_t)best_idx;
            topk_weights[token_idx * top_k + k] = best_val;
            s_probs[best_idx] = -1.0f;  // Mark as taken
        }
    }
}

// ============================================================================
// Pipeline orchestrator
// ============================================================================

struct AsyncPipeline {
    // CUDA streams (3 priority levels)
    cudaStream_t stream_route;    // High priority — RT Core routing
    cudaStream_t stream_prep;     // Medium — scatter + calibration
    cudaStream_t stream_expert;   // Low priority — Tensor Core GEMM

    // Triple buffer slots
    PipelineSlot slots[NUM_BUFFERS];

    // Expert weights (shared, read-only during inference)
    float* d_expert_weights_all;  // [NUM_EXPERTS, hidden_dim, inter_dim] packed
    int hidden_dim;
    int inter_dim;

    // Calibration parameters (shared)
    float* d_cal_weight;  // [64, 64]
    float* d_cal_bias;    // [64]

    // cuBLAS handle for expert GEMM
    cublasHandle_t cublas_handle;

    bool initialize(int hidden_dim_, int inter_dim_) {
        hidden_dim = hidden_dim_;
        inter_dim = inter_dim_;

        // Create streams with priorities
        int least_priority, greatest_priority;
        cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);

        cudaStreamCreateWithPriority(&stream_route, cudaStreamNonBlocking,
                                     greatest_priority);      // highest
        cudaStreamCreateWithPriority(&stream_prep, cudaStreamNonBlocking,
                                     greatest_priority + 1);  // medium
        cudaStreamCreateWithPriority(&stream_expert, cudaStreamNonBlocking,
                                     least_priority);         // lowest

        // Initialize buffer slots
        for (int i = 0; i < NUM_BUFFERS; ++i) {
            auto& s = slots[i];
            cudaMalloc(&s.d_hidden, MAX_BATCH * hidden_dim * sizeof(float));
            cudaMalloc(&s.d_expert_ids, MAX_BATCH * TOP_K * sizeof(uint32_t));
            cudaMalloc(&s.d_expert_weights, MAX_BATCH * TOP_K * sizeof(float));
            cudaMalloc(&s.d_dispatch_indices,
                       NUM_EXPERTS * (MAX_BATCH / 4) * sizeof(int));
            cudaMalloc(&s.d_expert_counts, NUM_EXPERTS * sizeof(int));
            // Bug 2.7 fix: separate buffer for softmax output so logits are preserved
            cudaMalloc(&s.d_softmax_weights, MAX_BATCH * TOP_K * sizeof(float));
            cudaMalloc(&s.d_output, MAX_BATCH * hidden_dim * sizeof(float));
            cudaEventCreateWithFlags(&s.route_done, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&s.prep_done, cudaEventDisableTiming);
            cudaEventCreateWithFlags(&s.expert_done, cudaEventDisableTiming);
        }

        // Calibration buffers
        cudaMalloc(&d_cal_weight, NUM_EXPERTS * NUM_EXPERTS * sizeof(float));
        cudaMalloc(&d_cal_bias, NUM_EXPERTS * sizeof(float));

        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, stream_expert);

        return true;
    }

    /**
     * Run one step of the triple-buffered pipeline.
     *
     * @param step    Current pipeline step (0, 1, 2, ...)
     * @param d_input Device pointer to input hidden states
     * @param batch   Batch size for this step
     *
     * The pipeline overlaps 3 operations:
     *   - Route step N   (RT Cores, stream_route)
     *   - Prep step N-1  (CUDA Cores, stream_prep)
     *   - Expert step N-2 (Tensor Cores, stream_expert)
     */
    void pipeline_step(int step, const float* d_input, int batch) {
        const int route_slot  = step % NUM_BUFFERS;
        const int prep_slot   = (step - 1 + NUM_BUFFERS) % NUM_BUFFERS;
        const int expert_slot = (step - 2 + NUM_BUFFERS) % NUM_BUFFERS;

        // ── Stage 1: Route (RT Cores) ─────────────────────────────
        // Copy input to routing buffer
        auto& rs = slots[route_slot];
        rs.batch_size = batch;
        cudaMemcpyAsync(rs.d_hidden, d_input,
                        batch * hidden_dim * sizeof(float),
                        cudaMemcpyDeviceToDevice, stream_route);

        // TODO: Replace with optixLaunch() for actual RT Core routing
        // For now, use CUDA kernel routing as placeholder
        // optixLaunch(pipeline, stream_route, d_params, ...);

        cudaEventRecord(rs.route_done, stream_route);

        // ── Stage 2: Prep (CUDA Cores) ────────────────────────────
        if (step >= 1) {
            auto& ps = slots[prep_slot];

            // Wait for routing to finish for this slot
            cudaStreamWaitEvent(stream_prep, ps.route_done, 0);

            // Apply calibration (Linear 64→64)
            // This corrects the BVH logit distribution
            // apply_calibration_kernel<<<...>>>(stream_prep);

            // Softmax + top-K
            // Bug 2.7 fix: write softmax output to d_softmax_weights instead of
            // overwriting d_expert_weights (which holds the original logits)
            softmax_topk_kernel<<<ps.batch_size, 64, 0, stream_prep>>>(
                ps.d_expert_weights,    // input: raw logits (preserved)
                ps.d_expert_ids,        // output: top-K expert IDs
                ps.d_softmax_weights,   // output: top-K softmax weights
                ps.batch_size,
                TOP_K
            );

            // Scatter tokens to expert groups
            cudaMemsetAsync(ps.d_expert_counts, 0,
                           NUM_EXPERTS * sizeof(int), stream_prep);

            const int total_assignments = ps.batch_size * TOP_K;
            const int threads = 256;
            const int blocks = (total_assignments + threads - 1) / threads;
            scatter_by_expert_kernel<<<blocks, threads, 0, stream_prep>>>(
                ps.d_expert_ids,
                ps.d_softmax_weights,  // Bug 2.7: use softmax weights, not raw logits
                ps.d_dispatch_indices,
                ps.d_expert_counts,
                ps.batch_size,
                TOP_K,
                MAX_BATCH / 4
            );

            cudaEventRecord(ps.prep_done, stream_prep);
        }

        // ── Stage 3: Expert forward (Tensor Cores) ────────────────
        if (step >= 2) {
            auto& es = slots[expert_slot];

            // Wait for scatter to finish for this slot
            cudaStreamWaitEvent(stream_expert, es.prep_done, 0);

            // Batched expert forward pass via cuBLAS
            // Each active expert processes its assigned tokens
            // TODO(Bug 2.11): Implement expert forward pass using cublasSgemmBatched.
            // Without this, weighted_combine_kernel below combines raw hidden states
            // instead of expert-transformed outputs. The pipeline is incomplete:
            //   for each active expert e:
            //     expert_out[e] = GELU(W1_e * gathered_tokens[e] + b1_e)
            //     expert_out[e] = W2_e * expert_out[e] + b2_e
            //   Then scatter expert_out back to token positions before combine.

            // Weighted combine of top-K expert outputs
            const int total = es.batch_size * hidden_dim;
            const int threads = 256;
            const int blocks = (total + threads - 1) / threads;
            weighted_combine_kernel<<<blocks, threads, 0, stream_expert>>>(
                es.d_output,
                es.d_hidden,  // placeholder for expert outputs
                es.d_softmax_weights,  // Bug 2.7: use softmax weights
                es.batch_size,
                hidden_dim,
                TOP_K
            );

            cudaEventRecord(es.expert_done, stream_expert);
        }
    }

    /**
     * Get output from a completed pipeline step.
     * Blocks until the expert stage for that step is done.
     */
    float* get_output(int step) {
        const int slot = (step - 2 + NUM_BUFFERS) % NUM_BUFFERS;
        cudaEventSynchronize(slots[slot].expert_done);
        return slots[slot].d_output;
    }

    void destroy() {
        for (int i = 0; i < NUM_BUFFERS; ++i) {
            auto& s = slots[i];
            cudaFree(s.d_hidden);
            cudaFree(s.d_expert_ids);
            cudaFree(s.d_expert_weights);
            cudaFree(s.d_dispatch_indices);
            cudaFree(s.d_expert_counts);
            cudaFree(s.d_softmax_weights);
            cudaFree(s.d_output);
            cudaEventDestroy(s.route_done);
            cudaEventDestroy(s.prep_done);
            cudaEventDestroy(s.expert_done);
        }
        cudaFree(d_cal_weight);
        cudaFree(d_cal_bias);
        cudaStreamDestroy(stream_route);
        cudaStreamDestroy(stream_prep);
        cudaStreamDestroy(stream_expert);
        cublasDestroy(cublas_handle);
    }
};

// ============================================================================
// Benchmark: measure pipeline overlap efficiency
// ============================================================================

extern "C" void benchmark_async_pipeline(
    int hidden_dim,
    int inter_dim,
    int batch_size,
    int num_steps
) {
    AsyncPipeline pipeline;
    pipeline.initialize(hidden_dim, inter_dim);

    // Synthetic input
    // Bug 2.9 fix: Initialize with random normalized vectors instead of zeros.
    // Zero vectors produce degenerate behavior in BVH routing (zero-length
    // directions) and don't reflect real-world performance.
    float* d_input;
    cudaMalloc(&d_input, batch_size * hidden_dim * sizeof(float));
    {
        // Create random input on host and copy to device
        float* h_input = new float[batch_size * hidden_dim];
        for (int i = 0; i < batch_size * hidden_dim; ++i) {
            // Simple LCG pseudo-random in [-1, 1], then normalize per-row below
            h_input[i] = (float)((i * 1103515245 + 12345) % 1000) / 500.0f - 1.0f;
        }
        // Normalize each row to unit length
        for (int b = 0; b < batch_size; ++b) {
            float norm = 0.0f;
            for (int d = 0; d < hidden_dim; ++d) {
                norm += h_input[b * hidden_dim + d] * h_input[b * hidden_dim + d];
            }
            norm = sqrtf(norm + 1e-8f);
            for (int d = 0; d < hidden_dim; ++d) {
                h_input[b * hidden_dim + d] /= norm;
            }
        }
        cudaMemcpy(d_input, h_input, batch_size * hidden_dim * sizeof(float),
                   cudaMemcpyHostToDevice);
        delete[] h_input;
    }

    // Warmup
    for (int i = 0; i < 5; ++i) {
        pipeline.pipeline_step(i, d_input, batch_size);
    }
    cudaDeviceSynchronize();

    // Benchmark
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < num_steps; ++i) {
        pipeline.pipeline_step(i + 5, d_input, batch_size);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    printf("\n  === LiquidBit Async Pipeline Benchmark ===\n");
    printf("  Hidden dim:    %d\n", hidden_dim);
    printf("  Inter dim:     %d\n", inter_dim);
    printf("  Batch size:    %d\n", batch_size);
    printf("  Steps:         %d\n", num_steps);
    printf("  Total time:    %.2f ms\n", ms);
    printf("  Per step:      %.2f us\n", ms * 1000.0f / num_steps);
    printf("  Throughput:    %.0f tok/s\n",
           (float)num_steps * batch_size / (ms / 1000.0f));

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_input);
    pipeline.destroy();
}
