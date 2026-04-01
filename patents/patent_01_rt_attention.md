# PROVISIONAL PATENT APPLICATION

## LBS-2026-001: System and Method for Attention Mechanism in Neural Language Models Using Hardware-Accelerated Ray Tracing with Bounding Volume Hierarchy Traversal

---

**Application Number:** [To be assigned by USPTO]
**Filing Date:** [To be determined]
**Applicant:** Jordi Silvestre Lopez
**Assignee:** LiquidBit Studio
**Status:** PROVISIONAL APPLICATION UNDER 35 U.S.C. 111(b)

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
- **optixModuleCreate():** Compiles CUDA source to PTX and then to hardware-specific microcode for the RT Cores.
- **optixPipelineCreate():** Links the ray generation, closest-hit, any-hit, and miss programs into a complete pipeline.
- **optixLaunch():** Dispatches the ray tracing workload to the RT Cores.
- **OptixTraversableHandle:** References the BVH in GPU memory for traversal.

An alternative implementation path uses the Vulkan ray tracing extension (VK_KHR_ray_tracing_pipeline) for cross-vendor compatibility.

### 11. Experimental Validation

A prototype implementation was constructed and validated on the following hardware and software:

**Hardware:** NVIDIA RTX 5070 Ti (16 GB VRAM), CUDA Compute Capability sm_120.
**Software:** CUDA 13.2, PyTorch 2.11 cu128, C++17, pybind11 bindings, WSL2 Ubuntu.

**Key Results (Certified 2026-03-30):**

| Metric | Measurement | Method |
|---|---|---|
| Routing Latency (batch=1, PyTorch) | 1,260 microseconds | `benchmark_e2e_final.py` |
| Routing Latency (batch=1, CUDA Extension) | 11 microseconds | `benchmark_e2e_final.py` |
| Routing Latency (batch=256, PyTorch) | 1,412 microseconds | `benchmark_e2e_final.py` |
| Routing Latency (batch=256, CUDA Extension) | 10 microseconds | `benchmark_e2e_final.py` |
| Speedup (CUDA Extension vs PyTorch) | 112-218x (batch dependent) | `benchmark_e2e_final.py` |
| End-to-End Latency (routing + expert, batch=1) | 690 microseconds | `patent_benchmark.py` |
| Token Generation Rate (full model baseline) | 55.4 tokens/second (peak) | `patent_benchmark.py`, `model.generate()` |
| Active VRAM Usage (router + 1 expert) | 4.03 MB | Router 890 KB + Expert 3,234 KB |
| VRAM Reduction vs Full Model | 731x less | 2,944 MB / 4.03 MB |

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
(b) projecting each token embedding from R^D to a three-dimensional position in R^3 using a dimensionality reduction technique that preserves cosine similarity;
(c) constructing an axis-aligned bounding box (AABB) around each projected three-dimensional position, the AABB having dimensions proportional to a semantic radius associated with the token;
(d) building a Bounding Volume Hierarchy (BVH) over the set of AABBs;
(e) for each query token in the sequence, generating one or more rays originating from the query token's three-dimensional position with directions distributed across a semantic hemisphere;
(f) traversing the BVH using hardware-accelerated ray tracing cores (RT Cores) of a graphics processing unit to identify tokens whose AABBs are intersected by the rays;
(g) computing an attention weight for each ray-token intersection using an exponential energy decay function of the form: attention_weight = E_0 * exp(-lambda * d), where E_0 is the ray's remaining energy, lambda is a semantic absorption coefficient, and d is the distance between the query token and the intersected token in the three-dimensional space; and
(h) aggregating the attention weights to produce an attention output for each query token.

**Claim 2.** The method of Claim 1, wherein the dimensionality reduction technique of step (b) comprises spherical Principal Component Analysis (PCA) performed on L2-normalized token embeddings, projecting onto the top three principal components.

**Claim 3.** The method of Claim 1, wherein the semantic radius of step (c) is computed from the variance of the token's embedding across multiple occurrences in a reference corpus, such that polysemous tokens receive larger semantic radii.

**Claim 4.** The method of Claim 1, wherein the BVH of step (d) is constructed using a top-down recursive algorithm that at each level selects the axis with the largest extent, sorts tokens along that axis, and splits at the median.

**Claim 5.** The method of Claim 1, wherein the BVH is constructed once per input sequence and reused across all attention layers of the neural language model.

**Claim 6.** The method of Claim 1, wherein the ray directions in step (e) are distributed according to a Fibonacci spiral pattern on the hemisphere, providing uniform angular coverage of the semantic space.

**Claim 7.** The method of Claim 1, wherein multiple rays per query token are generated, each ray representing an independent attention head, analogous to multi-head attention in Transformer models.

**Claim 8.** The method of Claim 1, wherein the hardware-accelerated ray tracing cores of step (f) are NVIDIA RT Cores accessed via the OptiX application programming interface (API), the traversal being performed by OptiX shader programs including a ray generation program, a closest-hit program, and a miss program.

**Claim 9.** The method of Claim 1, wherein the hardware-accelerated ray tracing cores of step (f) are accessed via the Vulkan VK_KHR_ray_tracing_pipeline extension.

**Claim 10.** The method of Claim 1, wherein the exponential energy decay function of step (g) is derived by analogy to the Beer-Lambert law of optical absorption, with the semantic absorption coefficient lambda being a learnable parameter optimized during training.

**Claim 11.** The method of Claim 1, wherein ray traversal terminates when the ray's remaining energy falls below a predetermined energy threshold, providing implicit sparse attention.

**Claim 12.** The method of Claim 1, further comprising storing a compressed version of each token's embedding within the BVH node, the compressed embedding being obtained by PCA reduction from D dimensions to a smaller number of dimensions and stored in half-precision (FP16) floating-point format.

**Claim 13.** The method of Claim 12, wherein the compressed embedding is 256 dimensions in FP16 format, preserving at least 95% of the cosine dissimilarity variance.

**Claim 14.** The method of Claim 1, wherein the aggregating step (h) comprises normalizing the attention weights across all hit tokens for each query token such that the normalized weights sum to 1.0, and computing a weighted sum of the hit tokens' value representations.

**Claim 15.** The method of Claim 1, wherein the total computational complexity of steps (e) through (h) is O(N log N), where N is the number of tokens in the sequence.

**Claim 16.** A data structure for representing a token of a neural language model in a three-dimensional semantic space for hardware-accelerated attention computation, the data structure comprising:
(a) a token identity portion comprising a vocabulary index and a sequence position;
(b) a geometry portion comprising a three-dimensional centroid position, an axis-aligned bounding box minimum corner, an axis-aligned bounding box maximum corner, and a semantic radius;
(c) an embedding portion comprising a compressed representation of the token's embedding vector stored in half-precision floating-point format; and
(d) an attention state portion comprising an accumulated attention weight and a remaining energy value;
wherein the three-dimensional centroid position is derived from the token's embedding vector by a cosine-similarity-preserving projection.

**Claim 17.** A system for neural language model inference comprising:
(a) a graphics processing unit (GPU) having dedicated ray tracing cores (RT Cores) and general-purpose compute cores (CUDA Cores);
(b) a token geometry module configured to project token embeddings into a three-dimensional semantic space;
(c) a BVH construction module configured to organize the three-dimensional token representations into a Bounding Volume Hierarchy stored in GPU memory;
(d) a ray generation module configured to emit semantic rays from query token positions;
(e) an attention computation module configured to use the RT Cores to traverse the BVH and compute attention weights using an exponential energy decay function; and
(f) an aggregation module configured to produce attention outputs from the computed weights;
wherein the RT Cores and CUDA Cores operate concurrently, with the RT Cores performing BVH traversal and the CUDA Cores performing value aggregation.

**Claim 18.** The system of Claim 17, wherein the GPU is an NVIDIA RTX 4090 or RTX 5070 Ti, and the RT Cores perform ray-AABB intersection tests in dedicated hardware at approximately 4 clock cycles per intersection.

**Claim 19.** The system of Claim 17, wherein the BVH stored in GPU memory has a memory footprint of less than 100 MB for a sequence of 100,000 tokens, compared to approximately 307 GB for a conventional KV cache of equivalent sequence length.

**Claim 20.** The system of Claim 17, wherein the attention computation achieves a routing latency of less than 20 microseconds for a batch of 256 tokens, representing at least an 85x speedup over an equivalent PyTorch-based softmax attention computation, with measured speedups ranging from 112x to 218x depending on batch size.

**Claim 21.** The method of Claim 1, wherein the method further comprises a two-phase execution: a Phase A using RT Cores for O(log N) BVH traversal to identify the most relevant semantic region, and a Phase B using Tensor Cores for high-precision matrix multiplication within the identified region, with the total complexity being O(N log N) + O(M^2) where M is much smaller than N.

**Claim 22.** The method of Claim 21, wherein Phase B uses cuBLAS half-precision (FP16) matrix multiplication on matrices of dimension M, where M is the dimension of a MatrixBlock associated with the identified semantic region.

**Claim 23.** A method for routing tokens to specialized expert sub-networks in a Mixture of Experts (MoE) architecture, the method comprising:
(a) organizing expert sub-networks as nodes in a hierarchical BVH in a three-dimensional semantic space;
(b) for each input token, generating a ray from the token's three-dimensional position;
(c) traversing the BVH using hardware RT Cores to identify the expert sub-network whose region the ray intersects; and
(d) dispatching the token to the identified expert sub-network for processing;
wherein the routing achieves O(log N) complexity where N is the number of experts.

**Claim 24.** The method of Claim 23, wherein the BVH organizes experts hierarchically with 4 levels of branching factor 4 (4x4x4x4 = 256 experts), and the hardware RT Core traversal visits at most 4 * log_4(N) nodes.

**Claim 25.** The method of Claim 23, further comprising a calibration layer that adjusts the distribution of routing weights output by the BVH router to match the distribution of a reference linear gate, the calibration layer being a linear transformation with fewer than 5,000 learnable parameters.

### Confidence-Gated BVH Routing (Claims 26-28)

**Claim 26.** The method of Claim 23, further comprising a confidence-gated routing step wherein:
(a) for each token at each layer, a confidence score is computed from the standard deviation of the top-k BVH routing logits;
(b) when the confidence score exceeds a threshold T, the token is routed via BVH traversal in O(log N) time;
(c) when the confidence score does not exceed the threshold T, the token is routed via a linear gate in O(N) time as fallback;
whereby the method eliminates accuracy compounding across multiple layers by selectively using exact linear routing for uncertain tokens while maintaining O(log N) efficiency for the majority of tokens.

**Claim 27.** The method of Claim 26, wherein the threshold T is a single scalar parameter shared across all tokens and all layers, adjustable post-deployment without retraining, providing a continuous tradeoff between routing speed (lower T, more BVH usage) and routing accuracy (higher T, more gate fallback).

**Claim 28.** The method of Claim 26, wherein on a 16-layer Mixture of Experts model with 64 experts per layer:
(a) at threshold T = 0.90, approximately 69% of token-layer routing decisions use O(log N) BVH traversal and 31% use linear gate fallback, achieving +17.1% perplexity increase over the baseline linear gate;
(b) at threshold T = 0.95, approximately 48% of token-layer routing decisions use BVH traversal, achieving +10.3% perplexity increase;
(c) the effective computational speedup is proportional to the BVH usage fraction, with the BVH component achieving 85-170x speedup over the linear gate.

---

## ABSTRACT

A system and method for computing attention in neural language models that replaces conventional matrix multiplication with hardware-accelerated ray tracing. Token embeddings are projected from high-dimensional space (R^D) into a three-dimensional semantic space preserving cosine similarity. The three-dimensional token representations are organized into a Bounding Volume Hierarchy (BVH). For each query token, rays are emitted into the semantic space and traversed against the BVH using dedicated RT Cores present in modern NVIDIA GPUs. Attention weights are computed at ray-token intersections using an exponential energy decay function analogous to the Beer-Lambert law: w = E_0 * exp(-lambda * d). The method achieves O(N log N) computational complexity versus O(N^2) for standard attention, reduces memory from approximately 307 GB to approximately 50 MB for 100,000-token sequences, and enables inference on consumer-grade GPUs (RTX 4090, RTX 5070 Ti) rather than datacenter hardware. Experimental validation demonstrates 112-218x routing speedup (batch-dependent), 731x VRAM reduction, and less than 1% perplexity degradation when replacing linear gates in a 7-billion-parameter Mixture of Experts model.

---

**Inventor:** Jordi Silvestre Lopez
**Organization:** LiquidBit Studio
**Date of Conception:** March 2026
**Priority Date:** [Filing date of this provisional application]
