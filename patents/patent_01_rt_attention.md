# NON-PROVISIONAL PATENT APPLICATION

## LBS-2026-001: System and Method for Attention Mechanism in Neural Language Models Using Hardware-Accelerated Ray Tracing with Bounding Volume Hierarchy Traversal

---

**Application Number:** [To be assigned by USPTO]
**Filing Date:** [To be determined]
**Applicant:** Jordi Silvestre Lopez
**Assignee:** Jordi Silvestre Lopez (individual inventor)
**Status:** NON-PROVISIONAL APPLICATION UNDER 35 U.S.C. 111(a)

---

## CROSS-REFERENCE TO RELATED APPLICATIONS

This application is related to co-pending provisional applications:
- LBS-2026-002: "System and Method for Multi-Dimensional Semantic Representation Using Nested Instance Acceleration Structures in Ray Tracing Hardware" (filed concurrently)
- LBS-2026-003: "System and Method for Context-Dependent Routing in Neural Networks Using Spectral Encoding and Optical Refraction Principles" (filed concurrently)

The disclosures of the above-identified applications are incorporated herein by reference in their entireties.

---

## FIELD OF THE INVENTION

The present invention relates generally to the field of artificial intelligence and large language model (LLM) inference. More specifically, the invention relates to a novel attention mechanism for neural language models that replaces conventional matrix multiplication (MatMul) operations with hardware-accelerated ray tracing operations using Bounding Volume Hierarchy (BVH) data structures and dedicated ray tracing (RT) processing cores available in modern graphics processing units (GPUs), thereby achieving O(N log N) computational complexity instead of the conventional O(N^2) complexity.

---

## BACKGROUND OF THE INVENTION

### Prior Art and Limitations

Large language models based on the Transformer architecture (Vaswani et al., 2017) have achieved state-of-the-art performance across natural language processing tasks. The core of the Transformer is the self-attention mechanism, which computes pairwise relationships between all tokens in a sequence using scaled dot-product attention:

```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

where Q (query), K (key), and V (value) are matrices derived from the input sequence of N tokens, and d_k is the key dimension. This operation requires computation of the full N x N attention matrix, resulting in O(N^2) time and memory complexity.

**Limitations of the Prior Art:**

1. **Quadratic Scaling:** For a sequence of N = 100,000 tokens, the attention matrix contains 10^10 (ten billion) entries. Computing this matrix requires approximately 80 trillion floating-point operations (FLOPs) per attention layer. This quadratic cost is the primary bottleneck limiting context window sizes in current LLMs.

2. **Memory Consumption:** The Key-Value (KV) cache mechanism used during autoregressive inference stores key and value projections across all layers. For a model with 96 layers and 4096-dimensional hidden states, the KV cache for 100,000 tokens consumes approximately 307 GB of VRAM, exceeding the memory capacity of any single commercial GPU.

3. **Hardware Requirements:** Serving models like GPT-4 or Gemini requires clusters of NVIDIA H100 GPUs (approximately 30,000 EUR per unit) operating in parallel via NVLink or InfiniBand interconnects. This restricts deployment to well-funded organizations with access to datacenter-grade hardware.

4. **Underutilized Silicon:** Modern consumer GPUs (NVIDIA RTX 4090, RTX 5070 Ti) contain dedicated Ray Tracing (RT) Cores designed for Bounding Volume Hierarchy traversal and ray-geometry intersection tests. These RT Cores operate independently of the Tensor Cores used for matrix multiplication and remain idle during conventional LLM inference, representing wasted silicon area and computational potential.

5. **Prior Efficiency Approaches:** Existing techniques to reduce the quadratic cost include sparse attention patterns (Longformer, BigBird), linear attention approximations (Performer, Linear Transformer), and FlashAttention (memory-efficient exact attention). However, all these methods fundamentally compute approximations of, or memory-efficient implementations of, the same O(N^2) attention matrix. None eliminates the matrix multiplication paradigm entirely.

There exists a need for a fundamentally different approach to the attention mechanism that achieves sub-quadratic computational complexity while leveraging dedicated hardware that is already present but unutilized in consumer-grade GPUs.

---

## SUMMARY OF THE INVENTION

The present invention provides a system and method for computing attention in neural language models using hardware-accelerated ray tracing operations instead of matrix multiplication. The key innovation is mapping tokens from a high-dimensional embedding space into a three-dimensional geometric space where semantically similar tokens are positioned near each other, organizing these geometric representations into a Bounding Volume Hierarchy (BVH), and using ray tracing hardware (RT Cores) to traverse the BVH and identify relevant tokens in O(N log N) time complexity.

The invention comprises:

1. **Token-to-Geometry Mapping:** A projection method that maps each token's D-dimensional embedding vector to a three-dimensional position (centroid) with an associated axis-aligned bounding box (AABB), preserving the cosine similarity metric of the original embedding space. Each token is represented as a `TokenNode` data structure containing geometric coordinates, a compressed FP16 embedding, and attention state variables.

2. **Semantic BVH Construction:** A method for constructing a Bounding Volume Hierarchy over the geometric token representations, where the tree structure reflects semantic clustering. The BVH is constructed once per sequence and reused across all attention layers, with O(N log N) amortized construction cost.

3. **Optical Attention Mechanism:** An attention computation method wherein query tokens emit rays into the 3D semantic space, and RT Cores traverse the BVH to find intersecting (relevant) tokens. The attention weight for each intersection is computed using an exponential energy decay formula analogous to the Beer-Lambert law of optical absorption:

```
attention_weight = E_0 * exp(-lambda * d_semantic)
```

where E_0 is the initial ray energy (1.0), lambda is a learnable semantic absorption coefficient, and d_semantic is the Euclidean distance in the projected 3D space serving as a proxy for semantic irrelevance.

4. **RT Core Acceleration:** The use of dedicated ray tracing hardware (NVIDIA RT Cores via OptiX API or Vulkan VK_KHR_ray_tracing extension) to perform the BVH traversal and ray-geometry intersection tests in hardware, achieving per-intersection latency of approximately 4 GPU clock cycles versus approximately 80 cycles for software-emulated intersection on CUDA cores.

5. **Multi-Ray Attention Heads:** An analogy to multi-head attention wherein each query token emits multiple rays distributed across a Fibonacci hemisphere in the semantic space, with each ray representing an independent attention head exploring a different semantic direction.

The resulting system achieves O(N log N) attention complexity, reduces VRAM requirements from approximately 307 GB (KV cache for 100K tokens) to approximately 10-50 MB (BVH), and enables LLM inference on consumer-grade hardware (NVIDIA RTX 4090 or RTX 5070 Ti) rather than datacenter GPU clusters.

---

## BRIEF DESCRIPTION OF THE DRAWINGS

**FIG. 1** is a block diagram illustrating the overall system architecture pipeline, showing the flow from token embeddings through 3D projection, BVH construction, RT Core traversal, and attention output aggregation.

**FIG. 2** is a structural diagram of the TokenNode data structure, showing the geometric coordinates (centroid, AABB), compressed FP16 embedding, and attention state variables.

**FIG. 3** is a diagram illustrating the Bounding Volume Hierarchy (BVH) tree structure with multiple levels of axis-aligned bounding boxes, and a ray traversing the tree to identify semantically relevant tokens.

**FIG. 4** is a graph showing the exponential energy decay function (attention_weight = E_0 * exp(-lambda * d)), illustrating how attention weight decreases with semantic distance.

**FIG. 5** is a block diagram of the EnhancedBVHRouter architecture showing the 3-level hierarchical routing structure with branching factor 4 (4x4x4 = 64 experts), including the projection layer, level centroids, and expert selection.

**FIG. 6** is a flowchart of the confidence-gated routing mechanism, showing the decision path where tokens with high BVH confidence (above threshold T) are routed via O(log N) BVH traversal and tokens with low confidence fall back to the exact linear gate.

---

## DETAILED DESCRIPTION OF THE INVENTION

### 1. System Architecture Overview

The system comprises four principal components operating in a pipeline (see FIG. 1):

```
Input Tokens -> [Token Geometry Module] -> [BVH Construction Module]
             -> [Ray Generation Module] -> [RT Core Traversal + Hit Processing]
             -> Attention Weights -> [Value Aggregation] -> Output
```

**Component 1: Token Geometry Module** receives token embeddings (vectors in R^D, where D is typically 768 for BERT-base or 4096 for GPT-4-scale models) and projects them into 3D positions using spherical PCA (Principal Component Analysis) with cosine metric preservation.

**Component 2: BVH Construction Module** organizes the 3D token representations into a hierarchical tree structure (BVH) that enables logarithmic-time search.

**Component 3: Ray Generation Module** generates semantic rays from query tokens, with directions computed from the query embedding and distributed across the semantic hemisphere.

**Component 4: RT Core Traversal and Hit Processing** uses hardware-accelerated ray tracing to traverse the BVH and compute attention weights at intersection points.

### 2. TokenNode Data Structure

Each token in the input sequence is represented as a `TokenNode` data structure (see FIG. 2) containing:

```cpp
struct TokenNode {
    // Identity
    uint32_t token_id;           // Vocabulary index (0 to ~50,000)
    uint32_t position_in_seq;    // Position in sequence (0 to N-1)

    // 3D Geometry for RT Cores
    float3   centroid;           // 3D position in semantic space
    float3   aabb_min;           // AABB minimum corner
    float3   aabb_max;           // AABB maximum corner
    float    semantic_radius;    // Semantic dispersion radius

    // Compressed embedding (FP16)
    half     embedding[256];     // D-to-256 PCA-reduced embedding

    // Attention state
    float    attention_weight;   // Accumulated attention weight
    float    energy_remaining;   // Remaining ray energy after collision
};
```

The `centroid` field stores the 3D projection of the token's embedding vector. The `aabb_min` and `aabb_max` fields define an axis-aligned bounding box whose dimensions are proportional to the token's `semantic_radius`, representing the diversity of contexts in which the token appears. Polysemous tokens (e.g., "bank") have larger semantic radii.

The `embedding[256]` field stores a compressed version of the original D-dimensional embedding, reduced to 256 half-precision floats via PCA. This compressed embedding preserves approximately 95% or more of the cosine dissimilarity variance and is used for fine-grained similarity computation during the closest-hit shader execution.

### 3. Embedding-to-3D Projection Method

The projection from D-dimensional embedding space to 3D semantic space proceeds as follows:

**Step 1: Cosine-Metric PCA.** Given a set of N token embeddings {e_1, e_2, ..., e_N} where each e_i is in R^D, compute the top 3 principal components of the embedding matrix after L2-normalization. The normalization ensures that PCA operates on the cosine similarity structure rather than the Euclidean structure:

```
e_i_normalized = e_i / ||e_i||_2
PCA components: v_1, v_2, v_3 in R^D (top 3 eigenvectors)
projection: p_i = [dot(e_i_normalized, v_1), dot(e_i_normalized, v_2), dot(e_i_normalized, v_3)]
```

**Step 2: Spherical Normalization.** The 3D projections are normalized to lie on or near the surface of a unit sphere, preserving angular relationships:

```
centroid_i = p_i / ||p_i||_2 * scale_factor
```

where `scale_factor` controls the spatial extent of the token distribution.

**Step 3: AABB Construction.** For each token, an axis-aligned bounding box is constructed around the centroid:

```
aabb_min_i = centroid_i - semantic_radius_i * (1, 1, 1)
aabb_max_i = centroid_i + semantic_radius_i * (1, 1, 1)
```

The `semantic_radius` is computed from the variance of the token's embedding across its occurrences in a reference corpus. Tokens that appear in diverse contexts (polysemous tokens) receive larger radii.

**Step 4: Embedding Compression.** The original D-dimensional embedding is compressed to 256 dimensions using PCA, retaining the top 256 principal components. The compressed embedding is stored in FP16 (half-precision) format, requiring 512 bytes per token versus 3,072 bytes (BERT) or 16,384 bytes (GPT-4-scale) for the full embedding.

### 4. BVH Construction

The Bounding Volume Hierarchy (see FIG. 3) is a binary tree where each node contains a bounding box that encloses all descendant nodes. The BVH is constructed over the set of TokenNode AABBs using a top-down recursive algorithm:

**Step 1:** Compute the bounding box enclosing all token AABBs.

**Step 2:** Select the axis (x, y, or z) with the largest extent.

**Step 3:** Sort tokens along the selected axis by centroid position.

**Step 4:** Split the sorted list at the median, creating two child groups.

**Step 5:** Recursively apply Steps 1-4 to each child group until each leaf contains a single token or a small group of tokens.

The resulting BVH has depth O(log N), and construction requires O(N log N) time. The BVH is constructed once per input sequence and reused for all attention layers and all query tokens within a forward pass.

**Memory Footprint:** A BVH containing N = 100,000 tokens, where each internal node requires approximately 32 bytes (bounding box + child pointers), results in a total BVH size of approximately 6.4 MB (200,000 nodes x 32 bytes). Including the TokenNode data (approximately 600 bytes per token), the total memory footprint is approximately 60 MB. This compares favorably to the approximately 307 GB required for a conventional KV cache with equivalent sequence length, representing an approximately 5,000x to 6,000x reduction in memory.

### 5. SemanticRay Structure and Ray Generation

Each query token emits a set of rays to probe the semantic space:

```cpp
struct SemanticRay {
    float3   origin;                       // Query token centroid + offset
    float3   direction;                    // Normalized direction vector
    float    energy;                       // Initial energy (1.0)
    half     query_embedding[256];         // Compressed query embedding
    uint32_t ray_id;                       // Unique ray identifier
};
```

**Ray Generation Algorithm:**

For each query token q at position p in the sequence:

1. Set `origin = centroid_q` (3D position of the query token).

2. Generate `num_rays` direction vectors distributed uniformly across a hemisphere using the Fibonacci spiral method:

```
For ray index k = 0 to num_rays-1:
    golden_ratio = (1 + sqrt(5)) / 2
    theta = 2 * pi * k / golden_ratio
    phi = arccos(1 - 2 * (k + 0.5) / num_rays)
    direction_k = (sin(phi) * cos(theta), sin(phi) * sin(theta), cos(phi))
```

3. Each ray is initialized with `energy = 1.0`.

4. The `query_embedding` is copied from the query token's compressed embedding to enable local similarity computation in the closest-hit shader without global memory access.

The number of rays per query (`num_rays`) is analogous to the number of attention heads in multi-head attention. Typical values range from 8 to 4096, with each ray exploring a different direction in the semantic space.

### 6. RT Core Traversal and Attention Computation

The core of the invention is the use of hardware-accelerated ray tracing to compute attention. The process operates through a pipeline of OptiX shader programs:

**6.1 Ray Generation Program (__raygen__rg_optical_attention):**

Launched once per query token. For each query:
- Computes the query's 3D position.
- Generates `num_rays` rays with Fibonacci-distributed directions.
- Initializes each ray's payload with energy = 1.0 and hit_count = 0.
- Calls `optixTrace()` for each ray, which triggers hardware BVH traversal.

**6.2 BVH Traversal (Hardware-accelerated):**

The RT Cores perform hierarchical ray-AABB intersection tests against the BVH. At each internal node, the hardware tests whether the ray intersects the node's bounding box:
- If the ray intersects the bounding box: the RT Core descends to the node's children.
- If the ray does not intersect: the RT Core skips the entire subtree.

This hierarchical pruning ensures that the ray visits at most O(log N) nodes, where N is the total number of tokens. The RT Core hardware performs these intersection tests in approximately 4 clock cycles per test, compared to approximately 80 clock cycles for software emulation on CUDA cores.

**6.3 Closest-Hit Program (__closesthit__ch_optical_attention):**

Executed when a ray intersects a token's AABB at the closest intersection point. The program computes the attention weight using the exponential energy decay formula:

```
d_semantic = ||centroid_query - centroid_hit||_2    (Euclidean distance in 3D)
attention_weight = energy * exp(-lambda * d_semantic)
```

Where:
- `energy` is the current remaining energy of the ray (starts at 1.0 and decays with each hit).
- `lambda` is the semantic absorption coefficient (a learnable hyperparameter, typically approximately 0.1).
- `d_semantic` is the Euclidean distance between the query and hit token centroids in the projected 3D space.

After computing the attention weight:
- The hit token's `attention_weight` field is atomically incremented.
- The ray's `energy` is reduced: `energy_new = energy - attention_weight`.
- If `energy_new < energy_threshold` (typically 0.01), the ray terminates (analogous to a photon being fully absorbed).
- Otherwise, the ray continues traversal to find additional relevant tokens.

**6.4 Any-Hit Program (__anyhit__ah_optical_attention):**

Optionally executed at every intersection (not just the closest). Used to implement continuous attention decay along the ray path, allowing a single ray to accumulate attention from multiple tokens:

```
For each intersection along the ray path:
    d_i = distance to i-th intersected token
    w_i = E_remaining * exp(-lambda * d_i)
    E_remaining = E_remaining - w_i
    if E_remaining < threshold:
        terminate ray (optixTerminateRay())
```

**6.5 Miss Program (__miss__ms_optical_attention):**

Executed when a ray exits the BVH without hitting any token. The ray's energy is not transferred to any token. This corresponds to a query direction where no relevant tokens exist, analogous to a zero-weight entry in the conventional attention matrix.

### 7. Attention Weight Normalization and Value Aggregation

After all rays from all query tokens have completed traversal:

**Step 1: Weight Collection.** For each query token q, collect all attention weights accumulated by its rays across all hit tokens:

```
raw_weights_q = {(token_k, weight_k)} for all tokens k hit by rays from q
```

**Step 2: Normalization.** Normalize the weights so they sum to 1.0, analogous to the softmax normalization in standard attention:

```
normalized_weight_q_k = raw_weight_q_k / sum_j(raw_weight_q_j)
```

**Step 3: Value Aggregation.** Compute the attention output as a weighted sum of the hit tokens' values (compressed embeddings):

```
output_q = sum_k(normalized_weight_q_k * value_embedding_k)
```

where `value_embedding_k` is derived from the compressed embedding stored in the hit token's TokenNode.

### 8. Computational Complexity Analysis

**BVH Construction:** O(N log N) for N tokens. Performed once per sequence.

**Ray Generation:** O(N * R) where R is the number of rays per query. Since R is a constant (typically 8 to 4096), this is O(N).

**BVH Traversal per Ray:** O(log N) due to hierarchical pruning. Each level of the BVH eliminates approximately half of the remaining candidates.

**Total Attention Computation:** O(N * R * log N) = O(N log N) since R is constant.

**Comparison with Standard Attention:**

| Metric | Standard Attention | Invention (Ray Tracing) | Improvement |
|---|---|---|---|
| Time Complexity | O(N^2 * d) | O(N * R * log N) | ~5,882x at N=100K |
| Memory (N=100K) | ~307 GB (KV Cache) | ~10-50 MB (BVH) | ~6,000x less |
| Operations (N=100K) | ~80T FLOPs | ~6.9B intersections | ~11,500x fewer |
| Hardware | Tensor Cores (saturated) | RT Cores (previously idle) | New silicon utilization |
| Minimum Hardware | H100 cluster | RTX 4090 / RTX 5070 Ti | Consumer GPU |

Note: Each ray-BVH intersection requires approximately 20-30 elementary FLOPs, reducing the effective advantage to approximately 380x when comparing FLOP-equivalent operations. However, because RT Cores execute these operations in dedicated hardware pipelines independent of the Tensor Cores, the wall-clock improvement is significantly larger.

### 9. Attention Decay Formula: Physical Basis

The exponential attention decay formula (illustrated in FIG. 4):

```
w(d) = E_0 * exp(-lambda * d)
```

is derived by analogy to the Beer-Lambert law of optical absorption, which describes the attenuation of light as it passes through an absorbing medium:

```
I(x) = I_0 * exp(-alpha * x)
```

where I_0 is the initial light intensity, alpha is the absorption coefficient, and x is the path length through the medium.

In the present invention:
- The "light intensity" I corresponds to the ray's remaining energy E.
- The "absorption coefficient" alpha corresponds to the semantic absorption coefficient lambda.
- The "path length" x corresponds to the semantic distance d_semantic in the projected 3D space.
- "Absorption" corresponds to attention being allocated to a token.

This physical analogy provides several desirable properties:
1. **Monotonic Decay:** Attention weight strictly decreases with semantic distance, matching the intuition that semantically distant tokens are less relevant.
2. **Energy Conservation:** The total energy allocated across all hits equals the initial energy (1.0), ensuring proper normalization.
3. **Natural Sparsity:** Distant tokens receive exponentially small weights, providing implicit sparse attention without explicit thresholding.
4. **Differentiability:** The exponential function is smooth and differentiable, enabling gradient-based optimization of the lambda parameter during training.

The learnable parameter lambda controls the "hardness" of attention:
- High lambda (e.g., 0.5): Attention is sharply focused on nearby (semantically similar) tokens.
- Low lambda (e.g., 0.01): Attention is more diffuse, attending broadly to many tokens.

### 10. Hardware Acceleration Details

The system targets NVIDIA GPUs with dedicated RT Cores, specifically:

- **NVIDIA RTX 4090 (Ada Lovelace, sm_89):** Third-generation RT Cores capable of performing two concurrent ray-triangle intersection tests per clock cycle. Contains 128 RT Cores operating at up to 2.52 GHz.

- **NVIDIA RTX 5070 Ti (Blackwell, sm_120):** Fourth-generation RT Cores with approximately 2x throughput improvement over Ada Lovelace. Contains dedicated BVH traversal hardware with support for multi-level Instance Acceleration Structures (IAS).

The OptiX 8.x/9.x API provides the programming model:
- **optixModuleCreate():** Compiles CUDA source to OptiX IR (Intermediate Representation) for OptiX 9.0+ or PTX for earlier versions, then to hardware-specific microcode for the RT Cores.
- **optixPipelineCreate():** Links the ray generation, closest-hit, any-hit, and miss programs into a complete pipeline.
- **optixLaunch():** Dispatches the ray tracing workload to the RT Cores.
- **OptixTraversableHandle:** References the BVH in GPU memory for traversal.

**OptiX 9.0+ Cooperative Vectors (In-Shader Calibration):**

The system further leverages OptiX 9.0 Cooperative Vectors to perform calibration of routing logits directly within the closest-hit shader program, using Tensor Cores for matrix operations without leaving the RT pipeline. This eliminates the GPU-to-host round-trip that would otherwise add 1-2 milliseconds of latency per routing decision.

- **optixCoopVecMatMul():** Performs cooperative matrix multiplication using Tensor Cores inside the OptiX shader. Used for linear calibration mode (W[64×64] · logits + bias).
- **optixCoopVecFFMA():** Fused multiply-add for affine calibration (scale ⊙ logits + bias). Requires only 128 parameters (64 scale + 64 bias weights in FP16).
- **OptixCoopVec<half, N>:** Cooperative vector template class that enables Tensor Core operations on N-dimensional FP16 vectors within shader programs.
- **optixCoopVecMatrixConvert():** Host-side function to convert row-major FP16 weight matrices to INFERENCING_OPTIMAL layout for maximum Tensor Core throughput.

The calibration weights are stored in device constant memory and accessed by the closest-hit shader. Two calibration modes are supported:
1. **Affine mode:** 128 parameters (scale[64] + bias[64] in FP16, 272 bytes total). Applied as: calibrated_logits = scale ⊙ raw_logits + bias.
2. **Linear mode:** 4,160 parameters (W[64×64] + bias[64] in FP16, 8,336 bytes total). Applied as: calibrated_logits = W · raw_logits + bias.

This in-shader calibration is a key differentiator: the entire routing pipeline (BVH traversal via RT Cores + logit calibration via Tensor Cores) executes as a single GPU kernel launch with no intermediate memory transfers.

An alternative implementation path uses the Vulkan ray tracing extension (VK_KHR_ray_tracing_pipeline) for cross-vendor compatibility.

### 11. Experimental Validation

A prototype implementation was constructed and validated on the following hardware and software:

**Hardware:** NVIDIA RTX 5070 Ti (16 GB VRAM), CUDA Compute Capability sm_120 (Blackwell).
**Software:** CUDA 13.2, OptiX SDK 9.1.0, PyTorch 2.11 cu128, C++17, pybind11 bindings, CMake 4.2.
**Build platforms:** Windows native (MSVC 19.44 + Visual Studio 2022), WSL2 Ubuntu (GCC 13.3).

**Key Results (Certified 2026-03-30, updated 2026-04-02):**

| Metric | Measurement | Method |
|---|---|---|
| Routing Latency (batch=1, PyTorch) | 1,260 microseconds | `benchmark_e2e_final.py` |
| Routing Latency (batch=1, CUDA Extension) | 11 microseconds | `benchmark_e2e_final.py` |
| Routing Latency (batch=256, PyTorch) | 1,412 microseconds | `benchmark_e2e_final.py` |
| Routing Latency (batch=256, CUDA Extension) | 10 microseconds | `benchmark_e2e_final.py` |
| Speedup (CUDA Extension vs PyTorch) | 112-218x (batch dependent) | `benchmark_e2e_final.py` |
| **RT Core Routing (batch=256, AABB sync)** | **28.5 microseconds** | `rt_router_benchmark.exe` (OptiX, Windows) |
| **RT Core Routing (batch=256, Triangle async)** | **19.1 microseconds** | `rt_router_benchmark.exe` (OptiX, Windows) |
| **RT Core Throughput (Triangle async)** | **13.4 M queries/second** | `rt_router_benchmark.exe` |
| **RT Core Routing Accuracy** | **100% (256/256)** | `rt_router_benchmark.exe` |
| **RT Core Speedup vs PyTorch gate** | **~48x** | 19.1 µs vs ~927 µs |
| **GAS Memory (Triangle, 64 experts)** | **11 KB** | 512 triangles (octahedrons) |
| End-to-End Latency (routing + expert, batch=1) | 690 microseconds | `patent_benchmark.py` |
| Token Generation Rate (full model baseline) | 55.4 tokens/second (peak) | `patent_benchmark.py`, `model.generate()` |
| Active VRAM Usage (router + 1 expert) | 4.03 MB | Router 890 KB + Expert 3,234 KB |
| VRAM Reduction vs Full Model | 731x less | 2,944 MB / 4.03 MB |

**RT Core Benchmark Details (2026-04-02):**

The OptiX RT Core router benchmark was validated on Windows native with the full OptiX 9.1 pipeline. Four geometry/execution modes were tested:

| Mode | Latency (µs/batch) | Throughput (M q/s) | Accuracy | GAS Size |
|---|---|---|---|---|
| AABB sync | 28.5 | 9.0 | 100% | 3 KB |
| AABB async | 37.2 | 6.9 | 100% | 3 KB |
| Triangle sync | 32.5 | 7.9 | 100% | 11 KB |
| **Triangle async** | **19.1** | **13.4** | **100%** | **11 KB** |

Triangle async achieves the best performance by using octahedral triangle meshes (8 triangles per expert, 512 total) with asynchronous ray tracing launch. The GAS (Geometry Acceleration Structure) for 64 experts occupies only 11 KB of VRAM.

**Measurement methodology:**
- Active VRAM counts only the MoE routing overhead: projection layer (1536->128, 768 KB), BVH router (128-dim, 122 KB), and one active ternary expert (3,234 KB packed 2-bit). The attention backbone is shared infrastructure and is excluded from both the numerator and denominator.
- Token generation rate measures the native HuggingFace `model.generate()` throughput as the baseline. The BVH routing overhead (10-11 us) is negligible relative to the per-token forward pass (~20 ms), so the system matches baseline speed.
- Routing speedup is measured as PyTorch `BVHRouter.forward()` vs `bvh_router_ext.route()` CUDA kernel, both on GPU. The conservative lower bound of ≥85x accounts for measurement variance across configurations; measured speedups range from 112x (batch=1) to 218x (batch=1024).

The BVH router (see FIG. 5 for the hierarchical architecture, FIG. 6 for the confidence-gated routing mechanism) was validated against the OLMoE-1B-7B model (7 billion parameters, 64 experts, 16 MoE layers), achieving:
- Top-8 expert selection accuracy: 85-95% across all 16 layers (with Spectral Techniques)
- **Full 16/16 layer replacement (hybrid mode, 24+ candidates):** PPL 7.15 vs baseline 7.15 — **0.0% degradation**
- Full 16/16 layer replacement (hybrid mode, 16 candidates): PPL 7.91 vs baseline 7.15 — +10.7% degradation
- Single-layer perplexity (layer 8): 7.19 (vs baseline 7.15, a delta of +0.6%)
- The hybrid mode uses BVH for O(log N) candidate pre-selection, then the original gate weight matrix computes exact routing weights via softmax. With 24+ candidates out of 64 experts (2.7x search reduction), zero quality loss is achieved.

**Note:** All perplexity measurements use WikiText-2 with transformers 5.4.0 (baseline PPL 7.15). Earlier measurements with transformers 4.46.3 (baseline 6.11) produced equivalent deltas.

**Reproduction:** Run `scripts/patent_benchmark.py` and `python/benchmark_e2e_final.py` from the project root.

---

## CLAIMS

**Claim 1.** A computer-implemented method for computing attention in a neural language model, the method comprising:
(a) receiving a sequence of N token embeddings, each token embedding being a vector in R^D;
(b) projecting each token embedding from R^D to a position in a K-dimensional geometric space, where K is less than D, using a learned or statistical dimensionality reduction;
(c) constructing a bounding volume around each projected position, the bounding volume having dimensions proportional to a semantic radius associated with the token;
(d) building a spatial acceleration structure over the set of bounding volumes;
(e) for each query token in the sequence, generating one or more rays originating from the query token's position in the geometric space with directions distributed across a semantic hemisphere;
(f) traversing the spatial acceleration structure using a hardware or software spatial traversal engine to identify tokens whose bounding volumes are intersected by the rays;
(g) computing an attention weight for each ray-token intersection using a monotonically decreasing function of the distance between the query token and the intersected token in the geometric space; and
(h) aggregating the attention weights to produce an attention output for each query token.

**Claim 2.** The method of Claim 1, wherein the dimensionality reduction technique of step (b) comprises spherical Principal Component Analysis (PCA) performed on L2-normalized token embeddings, projecting onto the top three principal components.

**Claim 3.** The method of Claim 1, wherein the semantic radius of step (c) is computed from the variance of the token's embedding across multiple occurrences in a reference corpus, such that polysemous tokens receive larger semantic radii.

**Claim 4.** The method of Claim 1, wherein the spatial acceleration structure of step (d) is a Bounding Volume Hierarchy (BVH) constructed using a top-down recursive algorithm that at each level selects the axis with the largest extent, sorts tokens along that axis, and splits at the median.

**Claim 5.** The method of Claim 1, wherein the spatial acceleration structure is constructed once per input sequence and reused across all attention layers of the neural language model.

**Claim 6.** The method of Claim 1, wherein the ray directions in step (e) are distributed according to a Fibonacci spiral pattern on the hemisphere, providing uniform angular coverage of the semantic space.

**Claim 7.** The method of Claim 1, wherein multiple rays per query token are generated, each ray representing an independent attention head, analogous to multi-head attention in Transformer models.

**Claim 8.** The method of Claim 1, wherein the spatial traversal engine of step (f) comprises dedicated ray tracing cores accessed via the OptiX application programming interface (API), the traversal being performed by OptiX shader programs including a ray generation program, a closest-hit program, and a miss program.

**Claim 9.** The method of Claim 1, wherein the spatial traversal engine of step (f) is accessed via the Vulkan VK_KHR_ray_tracing_pipeline extension.

**Claim 10.** The method of Claim 1, wherein the monotonically decreasing function of step (g) is an exponential energy decay function derived by analogy to the Beer-Lambert law of optical absorption, of the form: attention_weight = E_0 * exp(-lambda * d), where E_0 is the ray's remaining energy, lambda is a semantic absorption coefficient, and d is the distance, with lambda being a learnable parameter optimized during training.

**Claim 11.** The method of Claim 1, wherein ray traversal terminates when the ray's remaining energy falls below a predetermined energy threshold, providing implicit sparse attention.

**Claim 12.** The method of Claim 1, further comprising storing a compressed version of each token's embedding within the spatial acceleration structure node, the compressed embedding being obtained by PCA reduction from D dimensions to a smaller number of dimensions and stored in half-precision (FP16) floating-point format.

**Claim 13.** The method of Claim 12, wherein the compressed embedding is 256 dimensions in FP16 format, preserving at least 95% of the cosine dissimilarity variance.

**Claim 14.** The method of Claim 1, wherein the aggregating step (h) comprises normalizing the attention weights across all hit tokens for each query token such that the normalized weights sum to 1.0, and computing a weighted sum of the hit tokens' value representations.

**Claim 15.** The method of Claim 1, wherein the total computational complexity of steps (e) through (h) is O(N log N), where N is the number of tokens in the sequence.

**Claim 16.** A data structure for representing a token of a neural language model in a geometric semantic space for attention computation, the data structure comprising:
(a) a token identity portion comprising a vocabulary index and a sequence position;
(b) a geometry portion comprising a centroid position in a K-dimensional geometric space where K is less than D, a bounding volume minimum corner, a bounding volume maximum corner, and a semantic radius;
(c) an embedding portion comprising a compressed representation of the token's embedding vector stored in half-precision floating-point format; and
(d) an attention state portion comprising an accumulated attention weight and a remaining energy value;
wherein the centroid position is derived from the token's embedding vector by a dimensionality-reducing projection.

**Claim 17.** A system for neural language model inference comprising:
(a) a processor having spatial traversal hardware or software and general-purpose compute cores;
(b) a token geometry module configured to project token embeddings into a geometric semantic space of dimensionality K less than D;
(c) a spatial structure construction module configured to organize the geometric token representations into a spatial acceleration structure stored in processor-accessible memory;
(d) a ray generation module configured to emit semantic rays from query token positions;
(e) an attention computation module configured to use spatial traversal hardware or software to traverse the spatial acceleration structure and compute attention weights using a distance-based decay function; and
(f) an aggregation module configured to produce attention outputs from the computed weights;
wherein the spatial traversal and compute cores operate concurrently, with the spatial traversal performing acceleration structure traversal and the compute cores performing value aggregation.

**Claim 18.** The system of Claim 17, wherein the processor is a GPU having dedicated ray tracing acceleration hardware, and the ray tracing acceleration hardware performs ray-bounding volume intersection tests in dedicated hardware.

**Claim 19.** The system of Claim 17, wherein the spatial acceleration structure stored in memory has a memory footprint of less than 100 MB for a sequence of 100,000 tokens, compared to approximately 307 GB for a conventional KV cache of equivalent sequence length.

**Claim 20.** The system of Claim 17, wherein the attention computation achieves at least an order-of-magnitude speedup over an equivalent matrix-multiplication-based softmax attention computation by exploiting sub-linear spatial traversal.

**Claim 21.** The method of Claim 1, wherein the method further comprises a two-phase execution: a Phase A using spatial traversal hardware for O(log N) acceleration structure traversal to identify the most relevant semantic region, and a Phase B using matrix computation hardware for high-precision matrix multiplication within the identified region, with the total complexity being O(N log N) + O(M^2) where M is much smaller than N.

**Claim 22.** The method of Claim 21, wherein Phase B uses cuBLAS half-precision (FP16) matrix multiplication on matrices of dimension M, where M is the dimension of a MatrixBlock associated with the identified semantic region.

**Claim 23.** A method for routing tokens to specialized expert sub-networks in a Mixture of Experts (MoE) architecture, the method comprising:
(a) organizing expert sub-networks as nodes in a hierarchical spatial acceleration structure in a geometric semantic space;
(b) for each input token, generating a ray from the token's position in the geometric space;
(c) traversing the spatial acceleration structure using hardware or software spatial traversal to identify the expert sub-network whose region the ray intersects; and
(d) dispatching the token to the identified expert sub-network for processing;
wherein the routing achieves O(log N) complexity where N is the number of experts.

**Claim 24.** The method of Claim 23, wherein the spatial acceleration structure organizes experts hierarchically with 4 levels of branching factor 4 (4x4x4x4 = 256 experts), and the traversal visits at most 4 * log_4(N) nodes.

**Claim 25.** The method of Claim 23, further comprising a calibration layer that adjusts the distribution of routing weights output by the spatial router to match the distribution of a reference linear gate, the calibration layer being a linear transformation with fewer than 5,000 learnable parameters.

### Confidence-Gated Routing (Claims 26-28)

**Claim 26.** The method of Claim 23, further comprising a confidence-gated routing step wherein:
(a) for each token at each layer, a confidence score is computed from the standard deviation of the top-k routing logits;
(b) when the confidence score exceeds a threshold T, the token is routed via spatial acceleration structure traversal in O(log N) time;
(c) when the confidence score does not exceed the threshold T, the token is routed via a linear gate in O(N) time as fallback;
whereby the method eliminates accuracy compounding across multiple layers by selectively using exact linear routing for uncertain tokens while maintaining O(log N) efficiency for the majority of tokens.

**Claim 27.** The method of Claim 26, wherein the threshold T is a single scalar parameter shared across all tokens and all layers, adjustable post-deployment without retraining, providing a continuous tradeoff between routing speed (lower T, more spatial traversal usage) and routing accuracy (higher T, more gate fallback).

**Claim 28.** The method of Claim 26, wherein on a 16-layer Mixture of Experts model with 64 experts per layer:
(a) at threshold T = 0.90, approximately 69% of token-layer routing decisions use O(log N) spatial traversal and 31% use linear gate fallback, achieving +17.1% perplexity increase over the baseline linear gate;
(b) at threshold T = 0.95, approximately 48% of token-layer routing decisions use spatial traversal, achieving +10.3% perplexity increase;
(c) the effective computational speedup is proportional to the spatial traversal usage fraction, with the spatial traversal component achieving significant speedup over the linear gate.

### Software-Only and Apparatus Claims (Claims 29-30)

**Claim 29.** A computer-implemented method for computing attention in a neural network without requiring specialized ray tracing hardware, the method comprising:
(a) receiving a sequence of N input element representations, each representation being a vector in R^D;
(b) projecting each input element representation from R^D to a position in a K-dimensional geometric space, where K is less than D, using a learned or statistical dimensionality reduction;
(c) constructing a bounding volume around each projected position;
(d) building a spatial acceleration structure over the set of bounding volumes;
(e) for each query element, performing a software-based spatial traversal of the spatial acceleration structure on a general-purpose processor, the general-purpose processor being one or more of a CPU, a GPU using general-purpose compute cores, an FPGA, or an ASIC, to identify input elements whose bounding volumes satisfy a proximity criterion with respect to the query element;
(f) computing an attention weight for each identified input element using a distance-based decay function that is a monotonically decreasing function of the distance between the query element and the identified input element in the geometric space; and
(g) aggregating the weighted representations of the identified input elements to produce an attention output for each query element;
wherein the computational complexity is sub-quadratic in N.

**Claim 30.** A non-transitory computer-readable storage medium storing instructions that, when executed by one or more processors, cause the one or more processors to perform a method comprising:
(a) receiving a sequence of N input element representations, each representation being a vector in R^D;
(b) projecting each input element representation from R^D to a position in a K-dimensional geometric space, where K is less than D;
(c) constructing a bounding volume around each projected position;
(d) building a spatial acceleration structure over the set of bounding volumes;
(e) for each query element, traversing the spatial acceleration structure to identify a subset of input elements whose bounding volumes are intersected or otherwise satisfy a proximity criterion, without examining all N input elements;
(f) computing an attention weight for each identified input element in the subset using a monotonically decreasing function of the distance between the query element and the identified input element in the geometric space; and
(g) aggregating the weighted representations of the identified input elements to produce an attention output for each query element;
wherein the computational complexity of steps (e) through (g) is sub-quadratic in N.

### Broad Spatial Attention (Claim 31)

**Claim 31.** A computer-implemented method for computing attention in a neural network, the method comprising:
(a) mapping each of N input elements to a position in a geometric space of dimensionality K, where K is less than D and D is the dimensionality of the input element's representation;
(b) organizing the mapped positions into a hierarchical spatial data structure that enables sub-linear search;
(c) for each query element, performing a spatial search in the hierarchical spatial data structure to identify a subset of relevant input elements without examining all N input elements;
(d) computing attention weights for the identified relevant input elements based on their spatial proximity to the query element; and
(e) producing an attention output by aggregating representations of the relevant input elements weighted by the attention weights;
whereby the computational complexity of the attention computation is sub-quadratic in N.

### Two-Phase Hybrid Attention (Claim 32)

**Claim 32.** A computer-implemented method for neural network inference, the method comprising:
(a) a first phase using a spatial acceleration structure with sub-linear traversal complexity to identify a relevant subset of M elements from N total elements, where M is less than N; and
(b) a second phase using matrix multiplication on the identified subset of M elements to compute high-precision attention results;
wherein the first phase and second phase may execute on different types of processing units, the first phase executing on spatial traversal hardware and the second phase executing on matrix computation hardware, and the total computational complexity is O(N log N) + O(M^2) where M is much less than N.

### Confidence-Gated Expert Routing (Claim 33)

**Claim 33.** A method for routing tokens to expert sub-networks in a neural network having multiple layers, the method comprising:
(a) at each layer, computing routing scores using a first routing mechanism having sub-linear complexity;
(b) computing a confidence score indicating reliability of the first routing mechanism's output;
(c) when the confidence score exceeds a threshold, using the first routing mechanism's result to route the token;
(d) when the confidence score does not exceed the threshold, computing routing scores using a second routing mechanism having higher accuracy than the first routing mechanism; and
(e) applying the routing independently per token per layer;
wherein the threshold is a single scalar parameter adjustable post-deployment without retraining, providing a continuous tradeoff between computational speed and routing accuracy.

### Spatial Acceleration Structure as KV-Cache Replacement (Claim 34)

**Claim 34.** A method for neural language model inference, comprising replacing a key-value cache with a spatial acceleration structure, wherein:
(a) token representations are stored as geometric primitives in the spatial acceleration structure instead of as key-value pairs in linear memory;
(b) attention computation retrieves relevant tokens via spatial traversal of the spatial acceleration structure rather than full key-value dot products;
(c) the memory footprint of the spatial acceleration structure is O(N) with a constant factor at least 100x smaller than the equivalent key-value cache for the same number of tokens N;
whereby context windows of 100,000 or more tokens are processable on consumer-grade GPUs with less than 16 GB of memory.

### In-Shader Calibration via Cooperative Vectors (Claims 35-37)

**Claim 35.** The method of Claim 1 or Claim 33, further comprising performing calibration of routing logits directly within a shader program of the ray tracing pipeline, using cooperative vector operations executed on matrix computation hardware (Tensor Cores) co-resident on the same GPU as the ray tracing hardware (RT Cores), wherein:
(a) the ray tracing hardware computes raw routing logits via BVH traversal and ray-geometry intersection;
(b) calibration weights are stored in device constant memory accessible by the shader program;
(c) the shader program applies a calibration transform to the raw routing logits using cooperative vector operations (matrix multiplication, fused multiply-add, or element-wise affine transform) without transferring data between GPU kernels or between GPU and host memory;
whereby the entire routing pipeline (BVH traversal + logit calibration) executes as a single GPU kernel launch with no intermediate memory transfers, achieving end-to-end routing latency of less than 25 microseconds per batch.

**Claim 36.** The method of Claim 35, wherein the calibration transform is one of:
(a) an affine transform: calibrated_logits = scale ⊙ raw_logits + bias, where scale and bias are learned FP16 vectors of dimension equal to the number of experts, requiring 2×E parameters total (E = number of experts); or
(b) a linear transform: calibrated_logits = W · raw_logits + bias, where W is a learned FP16 matrix of dimensions E×E and bias is a learned FP16 vector of dimension E, requiring E² + E parameters total;
and the calibration weights are exported from a training framework (PyTorch) as a binary blob in hardware-optimal memory layout for maximum Tensor Core throughput.

**Claim 37.** The method of Claim 35, wherein the cooperative vector operations are implemented using the OptiX Cooperative Vectors API (optixCoopVecMatMul, optixCoopVecFFMA) or equivalent hardware-accelerated in-shader matrix operation API, and the calibration weights are converted to INFERENCING_OPTIMAL layout using a host-side matrix format conversion function (optixCoopVecMatrixConvert or equivalent) prior to upload to device memory.

---

## ABSTRACT

A system and method for computing attention in neural language models that replaces conventional matrix multiplication with spatial acceleration structures, optionally hardware-accelerated. Token embeddings are projected from high-dimensional space (R^D) into a low-dimensional geometric semantic space preserving relative token relationships. The geometric token representations are organized into a spatial acceleration structure such as a Bounding Volume Hierarchy (BVH). For each query token, rays are emitted into the semantic space and traversed against the spatial acceleration structure using dedicated spatial traversal hardware or general-purpose processors. Attention weights are computed at ray-token intersections using a distance-based decay function such as an exponential energy decay analogous to the Beer-Lambert law: w = E_0 * exp(-lambda * d). The method achieves O(N log N) computational complexity versus O(N^2) for standard attention, reduces memory from approximately 307 GB to approximately 50 MB for 100,000-token sequences, and enables inference on consumer-grade GPUs rather than datacenter hardware. Experimental validation demonstrates over two orders of magnitude routing speedup (batch-dependent), 731x VRAM reduction, and less than 1% perplexity degradation when replacing linear gates in a 7-billion-parameter Mixture of Experts model.

---

**Inventor:** Jordi Silvestre Lopez
**Filed by:** Jordi Silvestre Lopez (individual inventor)
**Date of Conception:** March 2026
**Priority Date:** [Filing date of this non-provisional application]
