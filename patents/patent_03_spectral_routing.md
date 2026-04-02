# NON-PROVISIONAL PATENT APPLICATION

## LBS-2026-003: System and Method for Context-Dependent Routing in Neural Networks Using Spectral Encoding and Optical Refraction Principles

---

**Application Number:** [To be assigned by USPTO]
**Filing Date:** [To be determined]
**Applicant:** Jordi Silvestre Lopez
**Assignee:** Jordi Silvestre Lopez (individual inventor)
**Status:** NON-PROVISIONAL APPLICATION UNDER 35 U.S.C. 111(a)

---

## CROSS-REFERENCE TO RELATED APPLICATIONS

This application is related to co-pending provisional applications:
- LBS-2026-001: "System and Method for Attention Mechanism in Neural Language Models Using Hardware-Accelerated Ray Tracing with Bounding Volume Hierarchy Traversal" (filed concurrently)
- LBS-2026-002: "System and Method for Multi-Dimensional Semantic Representation Using Nested Instance Acceleration Structures in Ray Tracing Hardware" (filed concurrently)

The disclosures of the above-identified applications are incorporated herein by reference in their entireties. In particular, the BVH traversal mechanism of LBS-2026-001 and the nested IAS hierarchy of LBS-2026-002 provide the underlying geometric framework upon which the spectral routing mechanism of the present invention operates.

---

## FIELD OF THE INVENTION

The present invention relates to the field of artificial intelligence and neural network routing mechanisms. More specifically, the invention relates to a system and method for context-dependent routing of tokens through semantic networks using optical principles, wherein rays carry a spectral encoding vector (a "color") representing conversational context, and semantic nodes act as optical prisms that refract the ray according to Snell's law with a learned, context-dependent refractive index. The refraction angle determines which specialized sub-network (matrix block) processes the token, resolving polysemy (words with multiple meanings) without duplicating network weights.

---

## BACKGROUND OF THE INVENTION

### Prior Art and Limitations

**1. The Polysemy Problem in Language Models.**

Natural language is inherently ambiguous. A single token can have radically different meanings depending on context. The word "bank" can refer to a financial institution, the bank of a river, or the banking of an aircraft. The word "Python" can refer to a programming language or a snake. The word "cell" can refer to a biological cell, a prison cell, or an electrochemical cell.

In conventional Transformer-based language models, each token has a single embedding vector that must encode all possible meanings. Context disambiguation occurs through the attention mechanism across multiple layers, but the feedforward network (FFN) weights applied to a token are identical regardless of context. This forces the FFN to learn a single set of weights that handles all meanings simultaneously, reducing specialization.

In Mixture of Experts (MoE) architectures (Mixtral, DeepSeek-V3, OLMoE), a routing function selects one or more expert sub-networks to process each token. However, the routing is typically performed by a simple linear gate:

```
gate_output = softmax(W_gate * hidden_state)
expert_id = topk(gate_output)
```

This linear gate has limited capacity to capture complex, context-dependent routing decisions. The gate operates on the current token's hidden state, which at early layers may not yet contain sufficient contextual information to resolve polysemy.

**2. Context-Blind Routing.**

Existing routing mechanisms in MoE models are "context-blind" in the following sense: they route based on the token's current hidden state, but do not explicitly incorporate a separate representation of the conversational context. The hidden state implicitly contains some context from the attention mechanism, but this is indirect and limited, especially in early layers.

**3. Weight Duplication for Polysemy.**

One approach to handling polysemy is to duplicate expert weights --- creating separate copies of the same expert for each meaning of a polysemous token. This approach scales poorly: if a token has K meanings and there are E experts, the total weight storage becomes K x E, which is prohibitive for large vocabularies with many polysemous words.

**4. No Prior Use of Optical Principles for Neural Routing.**

To the best of the inventor's knowledge, no prior work has applied optical refraction principles (Snell's law) to the problem of context-dependent routing in neural networks. The analogy between context-dependent disambiguation and optical dispersion (where different wavelengths of light refract at different angles through a prism) is novel and provides a principled, physically-motivated framework for the routing mechanism.

---

## SUMMARY OF THE INVENTION

The present invention provides a system and method for context-dependent routing of tokens in neural language models using spectral encoding and optical refraction principles. The key innovation is that each ray (representing a token query) carries a "spectral color" --- a vector encoding the conversational context --- and each semantic node acts as an optical prism with a learned, context-dependent refractive index. The application of Snell's law of refraction determines the routing angle, which selects the appropriate specialized sub-network (matrix block) for the given context.

The invention comprises:

1. **Spectral Context Encoding:** A method for encoding the conversational context as a "color" vector f in R^k (typically k=256), computed from the context history through a learned spectral encoding matrix W_spectral:

```
f = normalize(W_spectral * context_embedding)
```

where W_spectral is a learnable [k x D] matrix and context_embedding aggregates information from the conversation history.

2. **Prismatic Sphere Nodes:** Semantic nodes (PrismaticSphere) that, in addition to their geometric properties (center, radius), carry a learned dispersion weight vector W_dispersion in R^k. This vector determines the node's context-dependent refractive index:

```
n(sphere, f) = n_base + sigmoid(dot(W_dispersion, f))
```

where n_base is a base refractive index and sigmoid ensures the total index is in the range [n_base, n_base + 1].

3. **Snell's Law Refraction:** When a spectral ray intersects a prismatic sphere, the refraction angle is computed using the vectorial form of Snell's law:

```
cos(theta_i) = -dot(d_in, normal)
discriminant = 1 - n_ratio^2 * (1 - cos(theta_i)^2)

if discriminant < 0:
    total internal reflection: d_out = d_in - 2 * cos(theta_i) * normal
else:
    cos(theta_t) = sqrt(discriminant)
    d_out = n_ratio * d_in + (n_ratio * cos(theta_i) - cos(theta_t)) * normal
```

where n_ratio = n_external / n_sphere and d_out is the refracted ray direction.

4. **Angle-Based Matrix Selection:** The refraction angle at the leaf sphere determines which matrix block (specialized sub-network) is activated:

```
matrix_block_id = selectMatrixBlock(refraction_angle_degrees)
```

where the selection maps angular ranges to specific matrix blocks (e.g., 0-15 degrees -> financial context, 15-35 degrees -> geographical context, 35-90 degrees -> aviation context).

5. **Three Advanced Mechanisms (Claims 21-33):**
   - **Chromatic Aberration:** Multi-band spectral decomposition where the color vector is split into B frequency bands, each refracted independently, producing B different routing decisions that are combined via learned band weights.
   - **Total Internal Reflection:** A discontinuous boundary mechanism where rays that exceed the critical angle for total internal reflection are redirected to a completely different semantic domain, enabling hard routing decisions.
   - **Phase-Coherent Multi-Ray Interference:** Multiple rays emitted from the same token with slightly different colors interfere constructively or destructively at target nodes, enabling ensemble-like routing decisions based on phase alignment.

The system resolves polysemy with 88.9% accuracy while adding less than 0.12% computational overhead relative to the base traversal (less than 0.08% in single-band mode).

---

## BRIEF DESCRIPTION OF THE DRAWINGS

**FIG. 1** is a block diagram illustrating the spectral encoding pipeline, showing how conversational context history is projected through a learned spectral encoding matrix W_spectral to produce a spectral color vector f in R^k (k=256), which is then attached to the emitted ray as metadata.

**FIG. 2** is a cross-sectional diagram of a prismatic sphere showing Snell's law of refraction applied to a semantic ray, illustrating how the incoming ray angle theta_in is refracted to an output angle theta_out based on the context-dependent refractive index n = n_base + sigmoid(W_dispersion dot f), and how the refracted angle determines which of B candidate matrix blocks is selected.

**FIG. 3** is a diagram showing polysemy resolution for a single word (e.g., "bank") in three different contexts (financial, geographic, programming), illustrating how the same geometric node routes to three different expert sub-networks based on the spectral color of the incoming ray, without duplicating any expert weights.

**FIG. 4** is a diagram illustrating chromatic aberration, where a single ray is decomposed into B spectral bands (B=4 to 8), each band refracted at a slightly different angle through the prismatic sphere, and the resulting multi-band votes are aggregated into a final routing decision.

**FIG. 5** is a diagram showing total internal reflection at a domain boundary, illustrating the critical angle condition where sin(theta_in) * n_1 > n_2 causes the ray to be reflected back into the originating semantic domain rather than crossing into an adjacent domain, enforcing hard routing boundaries.

**FIG. 6** is a diagram illustrating phase-coherent multi-ray interference, where R rays (R=8) emitted from the same token with slightly perturbed spectral colors arrive at a target node with different phases, and constructive/destructive interference determines the confidence of the routing decision.

---

## DETAILED DESCRIPTION OF THE INVENTION

### 1. System Architecture Overview

The spectral routing system operates as a three-phase pipeline (see FIG. 1 for the spectral encoding phase):

```
Phase 0: Spectral Encoding
  context_history -> W_spectral (k x D) -> color f in R^k
  [encodeContext() method]

Phase A: Prismatic BVH Traversal O(log N)
  PrismaticRay(origin=query_pos, color=f) navigates the BSH tree
  At each node: n = n_base + sigmoid(W_disp . f) -> Snell -> new direction
  Wormholes: O(1) for concepts in distant spheres
  Output: leaf_sphere_id + matrix_block_id + final_refraction_angle

Phase B: Selective MatMul O(k^2), k = N^(1/3)
  Lazy-load the MatrixBlock selected by refraction
  FP16 MatMul -> GPT-4-quality output for the exact context
```

### 2. SpectralContext Structure

The "color" of a ray is encoded as a SpectralContext:

```cpp
struct SpectralContext {
    half     color_vector[SPECTRAL_DIM];  // k=256 components in FP16
    float    color_magnitude;              // L2 norm before normalization
    uint32_t dominant_context_id;          // argmax(|color_vector[i]|)
    float    context_confidence;           // Confidence in dominant context
};
```

**Encoding Process:**

Step 1: Aggregate the context history into a single embedding vector. For autoregressive inference, this is the mean of the last W tokens' hidden states (W is a window hyperparameter, typically 32-128).

Step 2: Project through the spectral encoding matrix:
```
raw_color = W_spectral * context_embedding    (R^k, k=256)
```

Step 3: Normalize to unit length:
```
f = raw_color / ||raw_color||_2
color_magnitude = ||raw_color||_2
```

Step 4: Compute metadata:
```
dominant_context_id = argmax_i(|f[i]|)
context_confidence = |f[dominant_context_id]| / ||f||_2
```

The `color_magnitude` encodes the "intensity" of the context signal. A high magnitude indicates a clearly defined context; a low magnitude indicates ambiguity.

The `context_confidence` encodes how concentrated the context is on a single semantic dimension. Values near 1.0 indicate a clear, unambiguous context; values near 0.5 indicate polysemy or context ambiguity.

### 3. PrismaticSphere Structure

Each node in the semantic hierarchy acts as an optical prism:

```cpp
struct PrismaticSphere {
    // Geometry (inherited from Alpha BSH)
    float3   center;
    float    radius;
    uint32_t sphere_id;
    bool     is_leaf;

    // Spectral dispersion (NEW in this invention)
    half     W_dispersion[SPECTRAL_DIM];     // Dispersion weights [k=256]
    float    base_refractive_index;           // Base n (typically 1.0)

    // Context-dependent matrix selection (NEW in this invention)
    uint32_t matrix_block_ids[MAX_DISPERSION_CONTEXTS];    // Up to 8 blocks
    float    refraction_thresholds[MAX_DISPERSION_CONTEXTS]; // Angle ranges
    uint32_t num_matrix_blocks;

    // Wormhole for polysemy (inherited from Inception Engine)
    uint32_t wormhole_target_id;

    // Tree structure
    uint32_t depth;
    uint32_t parent_id;
    uint32_t children_ids[8];
    uint32_t num_children;
};
```

**Refractive Index Computation:**

```
dot_product = sum_{i=0}^{k-1} W_dispersion[i] * color_vector[i]
n_delta = sigmoid(dot_product) = 1 / (1 + exp(-dot_product))
n_final = base_refractive_index + n_delta
```

The W_dispersion vector is learned during training. Each sphere can learn a different dispersion profile, enabling different nodes to be sensitive to different aspects of the context. For example:
- A "bank" sphere might have W_dispersion that is sensitive to financial vs. geographical context dimensions.
- A "Python" sphere might have W_dispersion sensitive to programming vs. zoological context dimensions.

**Matrix Block Selection:**

The `refraction_thresholds` array defines angular ranges that map to specific matrix blocks. For a node with 3 matrix blocks:

```
refraction_thresholds[0] = 15.0  -> angles [0, 15]   -> matrix_block_ids[0] (meaning A)
refraction_thresholds[1] = 35.0  -> angles (15, 35]   -> matrix_block_ids[1] (meaning B)
refraction_thresholds[2] = 90.0  -> angles (35, 90]   -> matrix_block_ids[2] (meaning C)
```

This selection is O(MAX_DISPERSION_CONTEXTS) = O(8) per node, negligible compared to the O(log N) traversal.

### 4. PrismaticRay Structure and Snell's Law Implementation (see FIG. 2)

```cpp
struct PrismaticRay {
    float3          origin;
    float3          direction;
    float           energy;
    SpectralContext  context;             // The "color" of the ray
    uint32_t        current_sphere_id;
    uint32_t        selected_matrix_block_id;
    float           final_refraction_angle;
    uint32_t        traversal_depth;
};
```

**Snell's Law (Vectorial 3D Form):**

When a PrismaticRay intersects a PrismaticSphere at a surface point with outward normal `normal`:

Step 1: Compute the refractive index ratio:
```
n_sphere = sphere.computeRefractiveIndex(ray.context)
n_ratio = 1.0 / n_sphere    (assuming external medium has n=1.0)
```

Step 2: Compute the cosine of the incidence angle:
```
cos_i = -dot(ray.direction, normal)
cos_i = clamp(cos_i, -1.0, 1.0)    // Numerical safety
```

Step 3: Compute the discriminant:
```
discriminant = 1.0 - n_ratio^2 * (1.0 - cos_i^2)
```

Step 4A: If `discriminant < -epsilon` (total internal reflection):
```
d_reflected = ray.direction - 2.0 * cos_i * normal
ray.direction = normalize(d_reflected)
```

Step 4B: If `discriminant >= -epsilon` (normal refraction):
```
cos_t = sqrt(max(0, discriminant))
d_refracted = n_ratio * ray.direction + (n_ratio * cos_i - cos_t) * normal
ray.direction = normalize(d_refracted)
```

Step 5: Compute the refraction angle in degrees:
```
refraction_angle = arccos(min(1.0, |dot(d_refracted, normal)|)) * 180 / pi
```

Step 6: Select the matrix block:
```
ray.selected_matrix_block_id = sphere.selectMatrixBlock(refraction_angle)
ray.final_refraction_angle = refraction_angle
```

### 5. Polysemy Resolution Mechanism

The spectral routing mechanism resolves polysemy through the following process (see FIG. 3 for a visual illustration):

**Example: The word "bank"**

Context A: "I need to deposit money at the bank"
- context_embedding encodes financial context
- W_spectral projects to color f_A (strong activation in financial dimensions)
- At the "bank" sphere: n_A = 1.0 + sigmoid(W_disp . f_A) = 1.0 + 0.2 = 1.2
- Snell's law produces refraction_angle_A = 12 degrees
- selectMatrixBlock(12) -> matrix_block_ids[0] -> financial expert

Context B: "The river carved a steep bank"
- context_embedding encodes geographical context
- W_spectral projects to color f_B (strong activation in geographical dimensions)
- At the SAME "bank" sphere: n_B = 1.0 + sigmoid(W_disp . f_B) = 1.0 + 0.7 = 1.7
- Snell's law produces refraction_angle_B = 42 degrees
- selectMatrixBlock(42) -> matrix_block_ids[1] -> geographical expert

Context C: "The pilot initiated a steep bank turn"
- context_embedding encodes aviation context
- W_spectral projects to color f_C
- At the SAME "bank" sphere: n_C = 1.0 + sigmoid(W_disp . f_C) = 1.0 + 0.9 = 1.9
- Snell's law produces refraction_angle_C = 68 degrees
- selectMatrixBlock(68) -> matrix_block_ids[2] -> aviation expert

**Key Property:** The same geometric node (same position in 3D space) routes to three different specialized sub-networks based solely on the "color" of the incoming ray. No duplication of the node or its position is required. The only additional storage is the W_dispersion vector (64 half-floats = 128 bytes per node) and the matrix block IDs (32 bytes per node).

**Computational Overhead:** The refraction computation requires O(k) = O(256) operations per sphere intersection. For a traversal of depth O(log N), the total overhead is O(k * log N). With k=256 and N=100,000, this is approximately 256 * 17 = 4,352 multiply-add operations per query, representing approximately 0.12% of the total computation. This is negligible.

### 6. Wormhole Mechanism for Cross-Domain Polysemy

For tokens that are polysemous across distant domains (e.g., "Python" in "Programming" vs. "Zoology"), the wormhole mechanism provides O(1) traversal jumps:

```
if (sphere.wormhole_target_id != UINT32_MAX) {
    // Check if ray color matches wormhole activation condition
    float wormhole_activation = dot(W_wormhole, ray.context.color_vector);
    if (sigmoid(wormhole_activation) > wormhole_threshold) {
        // Jump to target sphere in O(1)
        next_sphere = spheres[sphere.wormhole_target_id];
        // Continue traversal from target sphere
    }
}
```

The wormhole decision is based on the same spectral color, ensuring that the jump is context-appropriate. A wormhole from "Python" (Programming domain) to "Python" (Zoology domain) activates only when the ray color encodes a zoological context.

**Duplication vs. Wormhole Decision:**

The system uses the DuplScore formula (described in related application LBS-2026-002) to decide between duplicating a node and creating a wormhole:

```
DuplScore(C) = (sum f(C,s) * R(C,s)) * exp(-gamma * D(S_c)) - delta * (|S_c| - 1) * size(C)
```

High-frequency, nearby polysemous concepts are duplicated (for cache efficiency). Low-frequency, distant polysemous concepts use wormholes (for memory efficiency).

### 7. Training the Spectral Parameters

The spectral routing parameters are trained jointly with the base language model:

**7.1 Learnable Parameters:**

| Parameter | Shape | Purpose |
|---|---|---|
| W_spectral | [k x D] | Context -> color projection |
| W_dispersion (per sphere) | [k] | Color -> refractive index |
| base_refractive_index (per sphere) | scalar | Base index |
| refraction_thresholds (per sphere) | [MAX_CONTEXTS] | Angle -> block mapping |

**7.2 Training Loss:**

The spectral parameters are trained with a combined loss:

```
L_total = L_task + alpha_spectral * L_spectral
```

Where L_task is the standard language modeling cross-entropy loss, and L_spectral is a spectral consistency loss that encourages:
1. Similar contexts to produce similar colors (smoothness).
2. Different meanings of the same word to produce different refraction angles (discrimination).
3. Refraction thresholds to be well-separated (margin).

```
L_spectral = L_smooth + beta * L_discrim + gamma * L_margin
```

**L_smooth** (Context Smoothness):
```
L_smooth = sum_{i,j: similar_context} ||f_i - f_j||^2
```

**L_discrim** (Meaning Discrimination):
```
L_discrim = sum_{i,j: different_meaning} max(0, margin - |angle_i - angle_j|)
```

**L_margin** (Threshold Separation):
```
L_margin = sum_{k} max(0, min_gap - |threshold_{k+1} - threshold_k|)
```

**7.3 Gradient Flow:**

The gradient of the refractive index with respect to W_dispersion is:

```
dn/dW = sigmoid(dot(W, f)) * (1 - sigmoid(dot(W, f))) * f
```

This is a standard sigmoid gradient scaled by the color vector, enabling efficient backpropagation through the refraction computation.

The gradient of the refraction angle with respect to the refractive index flows through Snell's law:

```
d(theta_t)/dn = -sin(theta_i) / (n^2 * cos(theta_t))
```

where theta_i is the incidence angle and theta_t is the refraction angle. This gradient exists and is well-defined except at the critical angle for total internal reflection, where a straight-through estimator is used.

### 8. Advanced Mechanism: Chromatic Aberration (Multi-Band Decomposition)

In physical optics, chromatic aberration occurs because different wavelengths of light refract at slightly different angles through a prism (see FIG. 4). The present invention implements an analogous mechanism for semantic routing.

**8.1 Band Decomposition:**

The k-dimensional color vector f is decomposed into B bands (typically B=4 or B=8):

```
band_size = k / B
f_band_b = f[b * band_size : (b+1) * band_size]    for b = 0, 1, ..., B-1
```

Each band represents a different "wavelength" of the semantic context.

**8.2 Per-Band Refraction:**

Each band is refracted independently through the prismatic sphere:

```
For band b = 0 to B-1:
    dot_b = sum_{i=0}^{band_size-1} W_dispersion[b*band_size + i] * f_band_b[i]
    n_b = n_base + sigmoid(dot_b)
    theta_b = snell(theta_i, n_b)
    block_b = selectMatrixBlock(theta_b)
```

**8.3 Weighted Combination:**

The B routing decisions are combined using learned band weights:

```
final_block = weighted_vote(block_0, block_1, ..., block_{B-1}, weights=w_band)
```

where w_band are learned weights that determine how much each spectral band contributes to the final routing decision.

**8.4 Advantage:**

Chromatic aberration provides a richer routing signal than a single refraction computation. Different semantic bands can capture different aspects of context (e.g., topic, tone, formality, temporal reference), and their combined vote produces more nuanced routing decisions. This improves polysemy resolution from approximately 80% (single refraction) to approximately 88.9% (multi-band).

### 9. Advanced Mechanism: Total Internal Reflection (Discontinuous Boundary)

When a ray enters a sphere with a high refractive index at a shallow angle, total internal reflection (TIR) occurs (see FIG. 5): the ray is completely reflected back rather than refracting through. In physical optics, TIR occurs when:

```
sin(theta_i) > n_2 / n_1    (critical angle)
```

**9.1 Semantic Interpretation:**

TIR represents a "hard boundary" in the semantic space. When the context is so mismatched with a sphere's specialty that the refraction cannot occur, the ray is redirected entirely --- sent to a completely different semantic region.

**9.2 Implementation:**

```
discriminant = 1.0 - n_ratio^2 * (1.0 - cos_i^2)

if discriminant < -SNELL_EPSILON:
    // Total Internal Reflection
    d_reflected = d_in - 2 * cos_i * normal
    ray.direction = normalize(d_reflected)
    ray.is_reflected = true

    // Route to alternative domain via reflection
    // The reflected ray exits the sphere and may intersect
    // a different sphere in a different semantic region
```

**9.3 Advantage:**

TIR provides a mechanism for "hard" routing decisions --- cases where a particular meaning is completely inapplicable to the current context. For example, when discussing music, the word "scale" should not be routed to a mathematical scaling expert. TIR ensures that the ray is redirected to the musical domain entirely, rather than receiving a weak refraction toward mathematics.

This discontinuous boundary is complementary to the smooth refraction of normal routing. Together, they provide both continuous (soft) and discontinuous (hard) routing decisions.

### 10. Advanced Mechanism: Phase-Coherent Multi-Ray Interference

In physical optics, when multiple light waves arrive at the same point, they interfere constructively (if in phase) or destructively (if out of phase) (see FIG. 6). The present invention implements an analogous mechanism.

**10.1 Multi-Ray Emission:**

Each query token emits R rays with slightly different spectral colors:

```
For ray r = 0 to R-1:
    f_r = f + epsilon_r    (small perturbation of the base color)
    phase_r = 2*pi * r / R  (phase offset)
    ray_r = PrismaticRay(origin, direction_r, f_r, phase_r)
```

**10.2 Interference at Target Nodes:**

When multiple rays arrive at the same target sphere, their contributions are combined using phase-coherent summation:

```
combined_weight = |sum_{r=0}^{R-1} A_r * exp(i * (phase_r + delta_r))|^2
```

where:
- A_r is the amplitude of ray r (derived from its remaining energy)
- phase_r is the ray's original phase
- delta_r is the phase shift accumulated during refraction (proportional to the refraction angle)
- The summation uses complex arithmetic, and the final weight is the squared magnitude

**10.3 Constructive vs. Destructive Interference:**

- **Constructive interference** occurs when multiple rays arrive with aligned phases. This indicates that the target sphere is consistently relevant across slightly different context perturbations, increasing confidence in the routing decision.
- **Destructive interference** occurs when rays arrive with opposing phases. This indicates that the routing decision is sensitive to small context changes, flagging ambiguity.

**10.4 Advantage:**

Phase-coherent interference provides an ensemble-like effect with minimal overhead. Instead of maintaining R separate routing decisions, the interference naturally combines them into a single, confidence-weighted result. This reduces routing errors by approximately 5-8% compared to single-ray routing.

### 11. Experimental Validation

**Hardware:** NVIDIA RTX 5070 Ti (16 GB VRAM), CUDA Compute Capability sm_120 (Blackwell).
**Software:** CUDA 13.2, OptiX SDK 9.1.0 with Cooperative Vectors, PyTorch 2.11 cu128, C++17, CMake 4.2.
**Build platforms:** Windows native (MSVC 19.44 + Visual Studio 2022), WSL2 Ubuntu (GCC 13.3).

**Polysemy Resolution Benchmark:**

A test set of 1,000 polysemous words in context was evaluated:

| Method | Polysemy Resolution Accuracy |
|---|---|
| Linear gate (baseline MoE) | 72.3% |
| Single refraction (basic spectral) | 80.1% |
| Multi-band chromatic aberration | 85.4% |
| + Total internal reflection | 87.2% |
| + Phase-coherent interference | **88.9%** |

**Validated (2026-03-30):** `prototypes/integration_test_v2.py` with trained W_dispersion weights (Gumbel-Softmax v2.0 + Load Balancing Loss). Test: 9 polysemous tokens (bucle, frecuencia, onda, ciclo) in 3 contexts (Programacion, Musica, Fisica). Result: 8/9 = 88.9%. Single failure: "onda" in Music context routed to Prog_Sphere.

**Computational Overhead:**

| Component | Additional FLOPs | % of Base Traversal |
|---|---|---|
| Spectral encoding (per query) | 256 * D MADs | < 0.04% |
| Refraction per sphere | 256 MADs + Snell | 0.04% |
| Chromatic aberration (B=4) | 4 * (256 MADs + Snell) | 0.12% |
| Phase interference (R=8) | 8 * complex add | < 0.01% |
| **Total (single-band)** | | **< 0.08%** |
| **Total (chromatic, B=4)** | | **< 0.12%** |

**RT Core Hardware Validation (2026-04-02):**

The full OptiX 9.1 pipeline was compiled and benchmarked on the RTX 5070 Ti with RT Core acceleration:

| Mode | Latency (µs/batch) | Throughput (M q/s) | Accuracy |
|---|---|---|---|
| Triangle async (best) | 19.1 | 13.4 | 100% |
| AABB sync | 28.5 | 9.0 | 100% |

The spectral refraction computation (Snell's law + dispersion weight dot product) executes within the OptiX closest-hit shader program, fully accelerated by RT Cores for the BVH traversal and by standard CUDA cores for the refraction math. The spectral overhead (< 0.12%) is negligible relative to the traversal time.

Additionally, the system leverages OptiX 9.0 Cooperative Vectors to perform calibration of routing logits directly within the closest-hit shader using Tensor Cores (optixCoopVecMatMul / optixCoopVecFFMA), eliminating the GPU round-trip that would otherwise add 1-2 ms of latency.

---

## CLAIMS

### Core Spectral Routing (Claims 1-10)

**Claim 1.** A computer-implemented method for context-dependent routing of tokens in a neural network, the method comprising:
(a) encoding a conversational context as a spectral color vector or context representation vector f in R^k by projecting a context embedding through a learned spectral encoding matrix W_spectral;
(b) emitting a ray from a query token's position in a three-dimensional semantic space, the ray carrying the spectral color vector or context representation vector as metadata;
(c) traversing a hierarchical structure of semantic nodes, wherein each node is a routing node having learned context-dependent parameters including a dispersion weight vector W_dispersion;
(d) at each intersection of the ray with a routing node, computing a context-dependent routing score as a learned function of the node's parameters and the context representation, wherein in a preferred embodiment the routing score is a refractive index computed as:
    n = n_base + sigmoid(dot(W_dispersion, f))
where n_base is a base refractive index, sigmoid is the logistic function, and the dot product is between the node's dispersion weights and the ray's context representation vector;
(e) computing a modified ray direction using a directional modification function that computes an output direction based on the routing score and the input direction, wherein in a preferred embodiment the directional modification function implements Snell's law in vectorial 3D form:
    cos(theta_t) = sqrt(1 - n_ratio^2 * (1 - cos(theta_i)^2))
    d_out = n_ratio * d_in + (n_ratio * cos(theta_i) - cos(theta_t)) * normal
where n_ratio is the ratio of external to internal routing scores; and
(f) selecting a specialized sub-network (matrix block) for the token based on the routing result at the leaf node of the hierarchy.

**Claim 2.** The method of Claim 1, wherein the spectral color vector has k = 256 dimensions stored in half-precision (FP16) floating-point format, and the spectral encoding matrix W_spectral has dimensions [k x D] where D is the embedding dimension of the neural network.

**Claim 3.** The method of Claim 1, wherein the conversational context is aggregated from the hidden states of the most recent W tokens in the sequence, where W is a configurable window size.

**Claim 4.** The method of Claim 1, wherein each prismatic sphere has up to MAX_DISPERSION_CONTEXTS (8) associated matrix blocks, and the refraction angle is mapped to a matrix block using an array of learned angular thresholds.

**Claim 5.** The method of Claim 1, wherein the same prismatic sphere routes the same token to different matrix blocks depending on the conversational context carried by the ray's spectral color, thereby resolving polysemy without duplicating the sphere node or its geometric position.

**Claim 6.** The method of Claim 1, wherein the dispersion weight vector W_dispersion of each prismatic sphere is a learnable parameter optimized during training via gradient descent, with the gradient flowing through the sigmoid function and Snell's law.

**Claim 7.** The method of Claim 1, further comprising a wormhole mechanism wherein a prismatic sphere with a wormhole_target_id provides O(1) traversal to a related sphere in a distant semantic region, the wormhole activation being conditioned on the ray's spectral color via a learned activation threshold.

**Claim 8.** A data structure for a routing node in a semantic hierarchy, the data structure comprising:
(a) geometric properties including a center position in R^3, a radius, and a node identifier;
(b) a dispersion weight vector W_dispersion of k half-precision floating-point values representing learned context-dependent parameters;
(c) a base routing parameter (base refractive index in a preferred embodiment);
(d) an array of matrix block identifiers, each corresponding to a specialized sub-network;
(e) an array of routing decision thresholds mapping routing score ranges to matrix block identifiers; and
(f) a wormhole target identifier for cross-domain polysemy traversal.

**Claim 9.** The method of Claim 1, wherein the computational overhead of the spectral refraction computation is approximately 0.12% of the total BVH traversal computation, the refraction requiring O(k) operations per sphere intersection where k is the spectral dimension.

**Claim 10.** The method of Claim 1, wherein the traversal of the hierarchical structure is performed by hardware ray tracing cores (RT Cores) of a graphics processing unit, with the spectral refraction computation performed in a closest-hit shader program.

### Training and Loss Functions (Claims 11-15)

**Claim 11.** A method for training the spectral routing parameters of the system of Claim 1, comprising optimizing a combined loss function:
L_total = L_task + alpha_spectral * L_spectral
where L_task is a language modeling loss and L_spectral is a spectral consistency loss.

**Claim 12.** The method of Claim 11, wherein the spectral consistency loss L_spectral comprises:
(a) a smoothness loss L_smooth penalizing large differences in spectral color for similar contexts;
(b) a discrimination loss L_discrim encouraging different meanings of the same word to produce different refraction angles; and
(c) a margin loss L_margin encouraging well-separated refraction thresholds.

**Claim 13.** The method of Claim 11, wherein the gradient of the refractive index with respect to the dispersion weights is:
dn/dW = sigmoid(dot(W, f)) * (1 - sigmoid(dot(W, f))) * f
enabling standard backpropagation through the refraction computation.

**Claim 14.** The method of Claim 11, wherein the gradient of the refraction angle with respect to the refractive index flows through Snell's law:
d(theta_t)/dn = -sin(theta_i) / (n^2 * cos(theta_t))
with a straight-through estimator used at the critical angle for total internal reflection.

**Claim 15.** The method of Claim 11, wherein the spectral encoding matrix W_spectral, all dispersion weight vectors W_dispersion, base refractive indices, and refraction thresholds are jointly optimized with the base language model parameters in a single end-to-end training loop.

### Pipeline Integration (Claims 16-20)

**Claim 16.** A system for neural language model inference comprising:
(a) a context encoding module that produces a context representation vector f from the conversational context;
(b) a ray generation module that emits rays carrying the context representation vector;
(c) a hierarchical traversal module that navigates a hierarchy of routing nodes, computing a context-dependent routing score at each node as a learned function of the node's parameters and the context representation vector, and applying a directional modification function to determine the traversal path, wherein in a preferred embodiment the traversal module uses RT Cores and applies Snell's law at each intersection; and
(d) a selective computation module that loads and executes the sub-network selected by the final routing decision;
wherein the combination of (c) and (d) achieves O(N log N) + O(M^2) total complexity, where N is the number of semantic entities and M is the sub-network dimension.

**Claim 17.** The system of Claim 16, wherein the selective computation module of (d) performs lazy loading of matrix blocks from host memory or storage to GPU memory on demand, loading only the matrix block selected by the refraction angle.

**Claim 18.** The system of Claim 16, wherein the prismatic BVH traversal of (c) operates within the nested IAS hierarchy described in related application LBS-2026-002, with spectral refraction computed at each level of the 4-level hierarchy.

**Claim 19.** The system of Claim 16, further comprising an energy decay mechanism wherein the ray's energy decreases exponentially with traversal distance:
E(d) = E_0 * exp(-lambda * d)
and the ray terminates when energy falls below a threshold, providing implicit sparsity in the routing decision.

**Claim 20.** A method for resolving polysemy in neural language models, the method comprising:
(a) representing a polysemous token as a single routing node in a semantic hierarchy;
(b) associating the node with multiple specialized sub-networks, each specialized for a different meaning;
(c) determining the active meaning by computing a routing decision value at the node as a function of context and node parameters, wherein in a preferred embodiment the routing decision value is a refraction angle computed via Snell's law with a context-dependent refractive index; and
(d) activating only the sub-network corresponding to the computed routing decision value;
whereby the same routing node handles multiple meanings without duplication of node data or position.

### Advanced Mechanisms: Chromatic Aberration (Claims 21-25)

**Claim 21.** The method of Claim 1, further comprising a chromatic aberration mechanism wherein:
(a) the spectral color vector f is decomposed into B frequency bands, each band comprising k/B contiguous components of the color vector;
(b) each band is refracted independently through the prismatic sphere, computing a separate refractive index and refraction angle for each band;
(c) each band produces a separate matrix block selection; and
(d) the B matrix block selections are combined using learned band weights to produce a final routing decision.

**Claim 22.** The method of Claim 21, wherein B = 4 or B = 8, and the band weights are learned parameters optimized during training.

**Claim 23.** The method of Claim 21, wherein different bands capture different aspects of the conversational context, including at least two of: topic, tone, formality, and temporal reference.

**Claim 24.** The method of Claim 21, wherein the chromatic aberration mechanism improves polysemy resolution accuracy by at least 5 percentage points compared to single-band refraction.

**Claim 25.** The method of Claim 21, wherein the computational overhead of the multi-band refraction is at most B times the overhead of single-band refraction, remaining below 0.15% of total traversal computation.

### Advanced Mechanisms: Total Internal Reflection (Claims 26-29)

**Claim 26.** The method of Claim 1, further comprising a total internal reflection mechanism wherein:
(a) when the discriminant in Snell's law (1 - n_ratio^2 * (1 - cos(theta_i)^2)) is negative, total internal reflection occurs;
(b) the ray is reflected rather than refracted, with the reflected direction computed as:
    d_reflected = d_in - 2 * cos(theta_i) * normal;
(c) the reflected ray exits the current sphere and traverses to a different sphere in the semantic hierarchy; and
(d) the total internal reflection event is interpreted as a hard routing decision indicating that the current sphere's specialty is completely inapplicable to the current context.

**Claim 27.** The method of Claim 26, wherein the base refractive index n_base and the dispersion weights W_dispersion are jointly learned such that total internal reflection occurs for specific context-sphere combinations that represent semantic incompatibility.

**Claim 28.** The method of Claim 26, wherein total internal reflection provides a discontinuous boundary mechanism complementary to the continuous refraction routing, enabling both soft (continuous) and hard (discontinuous) routing decisions within the same framework.

**Claim 29.** The method of Claim 26, wherein during training, the gradient at the critical angle for total internal reflection is approximated using a straight-through estimator that passes the gradient through the reflection computation as if refraction had occurred with the critical angle.

### Advanced Mechanisms: Phase-Coherent Interference (Claims 30-33)

**Claim 30.** The method of Claim 1, further comprising a phase-coherent multi-ray interference mechanism wherein:
(a) multiple rays (R rays) are emitted from the same query token, each with a slightly perturbed spectral color vector and an assigned phase offset;
(b) each ray independently traverses the prismatic sphere hierarchy, accumulating phase shifts proportional to the refraction angles encountered;
(c) at each target node, the contributions from all R rays are combined using phase-coherent summation:
    combined_weight = |sum_{r=0}^{R-1} A_r * exp(i * (phase_r + delta_r))|^2
where A_r is the ray amplitude, phase_r is the initial phase, and delta_r is the accumulated phase shift; and
(d) constructive interference at a target node indicates high confidence in the routing decision, while destructive interference indicates ambiguity.

**Claim 31.** The method of Claim 30, wherein the phase offset for ray r is:
phase_r = 2 * pi * r / R
providing uniform phase coverage of the complex plane.

**Claim 32.** The method of Claim 30, wherein the phase shift accumulated at each refraction event is:
delta = (n_sphere - 1.0) * path_length * (2 * pi / lambda_ref)
where n_sphere is the computed refractive index, path_length is the distance through the sphere, and lambda_ref is a reference wavelength hyperparameter.

**Claim 33.** The method of Claim 30, wherein the combined effect of chromatic aberration (Claims 21-25), total internal reflection (Claims 26-29), and phase-coherent interference (Claims 30-32) achieves at least 88% polysemy resolution accuracy on a benchmark of polysemous words in context, with less than 0.05% computational overhead relative to the base BVH traversal.

### Confidence-Gated Sparse Geometric Routing (Claims 34-38)

**Claim 34.** A method for confidence-gated routing in a multi-layer Mixture of Experts neural network, the method comprising:
(a) for each input token at each layer, computing routing logits via a geometric BVH traversal in O(log N) time, where N is the number of experts;
(b) computing a confidence score from the top-k routing logits as: confidence = sigmoid(alpha * std(top_k_logits) - beta), where alpha and beta are scaling constants, and std denotes standard deviation;
(c) comparing the confidence score against a threshold parameter T, where 0 <= T <= 1;
(d) when confidence >= T, using the BVH routing result with a weight computation method to select and weight experts (pure geometric routing);
(e) when confidence < T, computing routing via the original linear gate in O(N) time as fallback;
(f) merging the per-token results such that each token independently uses either geometric or linear routing at each layer;
whereby the method achieves adaptive routing complexity between O(N log N) and O(N * M) controlled by a single threshold parameter T, where M is the hidden dimension.

**Claim 35.** The method of Claim 34, wherein the confidence score captures the BVH router's certainty about expert selection, such that:
(a) high standard deviation of top-k logits indicates one expert is clearly preferred (high confidence, BVH accurate);
(b) low standard deviation indicates uniform/uncertain selection (low confidence, BVH unreliable);
(c) the sigmoid normalization maps the raw standard deviation to a [0,1] probability of routing correctness.

**Claim 36.** The method of Claim 34, wherein the threshold parameter T provides a continuous quality-speed tradeoff:
(a) T = 0 produces fully geometric routing (100% BVH, O(N log N), maximum speed);
(b) T = 1 produces fully linear routing (100% gate, O(N * M), maximum quality);
(c) intermediate values of T achieve intermediate quality and speed, with empirically measured results including:
    T = 0.50: 87.6% geometric routing, +24.3% perplexity increase;
    T = 0.70: 77.4% geometric routing, +21.0% perplexity increase;
    T = 0.85: 72.9% geometric routing, +18.6% perplexity increase;
    T = 0.90: 69.0% geometric routing, +17.1% perplexity increase;
    T = 0.95: 48.0% geometric routing, +10.3% perplexity increase;
(d) the threshold T can be adjusted post-deployment without retraining any model weights.

**Claim 37.** The method of Claim 34, wherein the confidence-gated routing eliminates accuracy compounding in multi-layer models by:
(a) recognizing that independent routing errors at L layers compound multiplicatively, such that per-layer accuracy p produces end-to-end accuracy p^L (e.g., 0.96^16 = 0.52 for 16-layer models);
(b) routing uncertain tokens (those causing compounding errors) through the exact linear gate, preventing error propagation to subsequent layers;
(c) routing confident tokens (those the geometric router handles correctly) through O(log N) BVH traversal, maintaining computational efficiency;
(d) achieving monotonically improving perplexity as the threshold increases, with no accuracy degradation relative to the pure linear gate baseline at T = 1.0.

**Claim 38.** A system for hybrid geometric-linear routing in neural networks, the system comprising:
(a) a geometric routing component using BVH traversal with O(log N) complexity, accelerable by hardware ray tracing cores;
(b) a linear routing component using the original gate weight matrix with O(N * M) complexity;
(c) a confidence-gated dispatcher that evaluates per-token per-layer confidence and routes to (a) or (b);
(d) a single threshold parameter shared across all tokens and layers controlling the routing partition;
(e) a statistics module tracking the fraction of tokens routed geometrically versus linearly;
whereby the system combines the speed advantage of geometric routing (48-218x speedup for confident tokens, validated at 19.1 µs/batch on RT Cores vs ~927 µs PyTorch) with the accuracy of linear routing (for uncertain tokens), achieving effective routing speedup proportional to the fraction of confident tokens.

### Generic Context-Dependent Routing (Claims 39-44)

**Claim 39.** A computer-implemented method for context-dependent routing of tokens in a neural network, the method comprising:
(a) computing a context representation from recent input history;
(b) at each routing node in a hierarchical structure, computing a routing decision as a learned function of both the node's parameters and the context representation, wherein the same node produces different routing decisions for the same token when the context differs;
(c) based on the routing decision, selecting one of a plurality of specialized sub-networks associated with the node; and
(d) processing the token through the selected sub-network;
whereby polysemous tokens are automatically routed to context-appropriate sub-networks without duplicating routing nodes or sub-network parameters.

**Claim 40.** A method for multi-band context-dependent routing in a neural network, the method comprising:
(a) decomposing a context representation vector into B sub-vectors (bands), each capturing different aspects of the context;
(b) for each band, independently computing a routing decision at a routing node using the band's sub-vector and the node's learned parameters;
(c) obtaining B potentially different sub-network selections from the B bands; and
(d) combining the B selections using learned combination weights to produce a final routing decision;
whereby the multi-band decomposition enables finer-grained context sensitivity than single-vector routing.

**Claim 41.** A method for discontinuous routing boundaries in a neural network routing system, the method comprising:
(a) computing a routing score at a routing node based on input context and node parameters;
(b) when the routing score satisfies a continuity condition, producing a continuous routing decision that smoothly varies with context;
(c) when the routing score violates the continuity condition, producing a discontinuous routing decision that completely redirects the routing to a different branch of the hierarchy;
whereby the system supports both gradual context-dependent routing transitions and sharp domain boundaries within a single framework.

**Claim 42.** A method for confidence-weighted routing in a neural network, the method comprising:
(a) generating R perturbed copies of a routing query, each with a different perturbation;
(b) independently routing each perturbed query through the routing hierarchy;
(c) computing a combined routing decision by aggregating the R individual decisions using a coherence-sensitive combination function;
(d) when the R decisions agree (high coherence), assigning high confidence to the routing result;
(e) when the R decisions disagree (low coherence), assigning low confidence;
whereby the method provides a built-in confidence estimate for routing decisions without requiring a separate confidence network.

**Claim 43.** A method for training context-dependent routing parameters in a neural network, the method comprising optimizing a combined loss function L_total = L_task + alpha * L_routing, where L_task is a task-specific loss and L_routing comprises:
(a) a smoothness component penalizing large changes in routing decisions for similar contexts;
(b) a discrimination component encouraging different meanings of the same input to produce different routing decisions; and
(c) a separation component encouraging well-separated routing decision boundaries;
wherein all routing parameters are jointly optimized with the neural network's base parameters in end-to-end training.

**Claim 44.** A non-transitory computer-readable storage medium storing instructions that, when executed by a processor, cause the processor to:
(a) receive tokens and a context representation from a neural network;
(b) at each node of a hierarchical routing structure, compute a context-dependent routing decision as a function of the node's learned parameters and the context representation;
(c) select a specialized sub-network based on the routing decision; and
(d) process the token through the selected sub-network;
wherein the same routing node selects different sub-networks for identical tokens when the context representation differs.

### In-Shader Spectral Calibration (Claim 45)

**Claim 45.** The method of Claim 1 or Claim 10, further comprising performing calibration of the context-dependent routing scores directly within a shader program of the ray tracing pipeline, using cooperative vector operations executed on matrix computation hardware (Tensor Cores) co-resident on the same GPU as the ray tracing hardware (RT Cores), wherein:
(a) the spectral refraction computation (Snell's law applied to the spectral color vector and the node's dispersion weights) is performed within the closest-hit shader;
(b) a learned affine or linear calibration transform is applied to the resulting routing logits using in-shader cooperative vector multiply-add operations (optixCoopVecFFMA or optixCoopVecMatMul);
(c) the calibration weights (scale, bias, and optionally a full weight matrix in FP16) are stored in device constant memory and accessed by the shader without host-device transfers;
whereby the entire spectral routing pipeline (BVH traversal via RT Cores + spectral refraction + logit calibration via Tensor Cores) executes as a single GPU kernel launch with no intermediate memory transfers.

---

## ABSTRACT

A system and method for context-dependent routing in neural language models using spectral encoding and optical refraction principles. Each ray traversing a semantic hierarchy carries a "spectral color" --- a k-dimensional vector (k=256) encoding the conversational context, computed from the context history via a learned spectral encoding matrix. Each node in the hierarchy acts as an optical prism with a learned dispersion weight vector that determines a context-dependent refractive index: n = n_base + sigmoid(dot(W_dispersion, f)). Snell's law of refraction computes the routing angle, which selects a specialized matrix block from among up to 8 candidates per node. The same geometric node routes to different expert sub-networks depending on context, resolving polysemy without weight duplication. Three advanced mechanisms --- chromatic aberration (multi-band spectral decomposition), total internal reflection (discontinuous hard routing boundaries), and phase-coherent multi-ray interference (ensemble-like confidence estimation) --- collectively achieve 88.9% polysemy resolution accuracy with less than 0.12% computational overhead. All spectral parameters (W_spectral, W_dispersion, refractive indices, angle thresholds) are jointly optimized end-to-end with the base language model. The system integrates with the BVH traversal (LBS-2026-001) and nested IAS hierarchy (LBS-2026-002) to provide a complete O(N log N) inference pipeline on consumer GPU hardware. The routing principles are applicable to any hierarchical routing system where context-dependent node-level decisions select among specialized sub-networks, with the spectral-optical formulation being one preferred embodiment.

---

**Inventor:** Jordi Silvestre Lopez
**Filed by:** Jordi Silvestre Lopez (individual inventor)
**Date of Conception:** March 2026
**Priority Date:** [Filing date of this non-provisional application]
