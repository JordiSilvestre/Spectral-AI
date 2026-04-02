# Patent Figure Specifications for Illustrator

**Project:** Zero-Matrix Attention System
**Inventor:** Jordi Silvestre Lopez
**Filed as:** Individual inventor
**Date:** April 2026

## General Requirements

- **Format:** Black and white line drawings (USPTO standard, 37 CFR 1.84)
- **Resolution:** 300 DPI minimum
- **Size:** Each figure must fit within 21.6 cm × 27.9 cm (8.5" × 11") with at least 2.5 cm margins
- **Labels:** Use reference numerals (100, 200, 300...) with lead lines
- **Font:** Clean sans-serif, minimum 0.32 cm height (about 10pt)
- **Style:** Technical patent illustration — no shading, no color, cross-hatching for filled areas if needed
- **File format:** Deliver as SVG + high-resolution PNG (300 DPI)

---

## PATENT 1: RT Attention (LBS-2026-001)

### FIG. 1 — System Architecture Pipeline

**Type:** Block diagram (horizontal flow)

**Elements to draw:**

```
[Input Tokens] → [Token Geometry Module (100)] → [BVH Construction Module (200)]
    → [Ray Generation Module (300)] → [RT Core Traversal (400)] → [Attention Output (500)]
```

- **Block 100 (Token Geometry Module):** Rectangle labeled "Token Geometry Module". Inside show: "D-dim → 3D Projection (PCA)". Input arrow labeled "Token Embeddings ∈ R^D". Output arrow labeled "TokenNodes in R^3".
- **Block 200 (BVH Construction):** Rectangle labeled "BVH Construction". Inside: "O(N log N)". Output arrow labeled "BVH Tree".
- **Block 300 (Ray Generation):** Rectangle labeled "Ray Generation". Input from top labeled "Query Token". Output: multiple arrows (rays) labeled "Semantic Rays".
- **Block 400 (RT Core Traversal):** Rectangle with distinctive border (double-line or bold) labeled "RT Cores (Hardware)". Inside: "BVH Traversal + Hit Processing". Annotation: "O(log N) per ray". Small GPU icon optional.
- **Block 500 (Attention Output):** Rectangle labeled "Attention Aggregation". Output arrow labeled "Attention Weights".

**Key annotations:**
- Above the entire pipeline: "O(N log N) Total Complexity"
- Below Block 400: "NVIDIA OptiX API / Vulkan VK_KHR_ray_tracing"

---

### FIG. 2 — TokenNode Data Structure

**Type:** Structured box diagram

**Draw a rounded rectangle divided into sections:**

```
┌─────────────────────────────────────┐
│         TokenNode (110)             │
├─────────────────────────────────────┤
│ Identity:                           │
│   token_id (112): uint32            │
│   position_in_seq (114): uint32     │
├─────────────────────────────────────┤
│ 3D Geometry:                        │
│   centroid (120): float3 (x,y,z)    │
│   aabb_min (122): float3            │
│   aabb_max (124): float3            │
│   semantic_radius (126): float      │
├─────────────────────────────────────┤
│ Compressed Embedding:               │
│   embedding (130): half[256]  (FP16)│
├─────────────────────────────────────┤
│ Attention State:                    │
│   attention_weight (140): float     │
│   energy_remaining (142): float     │
└─────────────────────────────────────┘
```

**Next to the box:** A small 3D coordinate system showing a cube (AABB) with a dot at the center (centroid), and a circle around it showing semantic_radius.

---

### FIG. 3 — BVH Tree Traversal with Ray

**Type:** Tree diagram + geometric illustration

**Left side (Tree):**
- Draw a binary tree with 4 levels
- Root node labeled "Root AABB (210)" encompassing all
- Two children: "Left AABB (212)" and "Right AABB (214)"
- Each child splits into 2 more nodes
- Leaf nodes labeled "TokenNode" with small squares
- A dashed line through the tree shows the ray's traversal path
- Nodes NOT visited by the ray are shown lighter/thinner
- Annotation: "Skipped subtree (pruned)" with X mark

**Right side (3D visualization):**
- Show a 2D cross-section of the 3D space
- Large rectangle (root AABB) containing smaller nested rectangles
- A ray (arrow) passing through the space
- The ray hits some boxes (filled dots at intersections) and misses others
- Labels: "Ray origin (query token)", "Hit token", "Missed region"
- Annotation: "O(log N) nodes visited"

---

### FIG. 4 — Energy Decay Function

**Type:** Mathematical graph

- **X-axis:** "Semantic Distance d" (0 to 5)
- **Y-axis:** "Attention Weight w(d)" (0 to 1.0)
- **Curve:** Exponential decay: w = exp(-λ·d)
- Draw 3 curves for different λ values:
  - λ = 0.05 (shallow decay — labeled "Diffuse attention")
  - λ = 0.1 (medium — labeled "Balanced")
  - λ = 0.5 (steep decay — labeled "Focused attention")
- **Formula annotation:** w(d) = E₀ · exp(-λ · d)
- **Comparison annotation:** "Analogous to Beer-Lambert Law"

---

### FIG. 5 — EnhancedBVHRouter Architecture (3-Level MoE Routing)

**Type:** Hierarchical block diagram

```
Input Token Embedding (1536-dim)
         │
         ▼
┌──────────────────────┐
│  Projection Layer    │
│  (1536 → 128 dim)   │
│       (310)          │
└──────────┬───────────┘
           │ 128-dim vector
           ▼
┌──────────────────────┐
│  Level 1 Routing     │
│  4 clusters (320)    │
│  ┌─┐ ┌─┐ ┌─┐ ┌─┐   │
│  │0│ │1│ │2│ │3│    │
│  └─┘ └─┘ └─┘ └─┘   │
└──────────┬───────────┘
           │ Selected: cluster 2
           ▼
┌──────────────────────┐
│  Level 2 Routing     │
│  4 sub-clusters (330)│
│  ┌──┐ ┌──┐ ┌──┐ ┌──┐│
│  │2.0│ │2.1│ │2.2│ │2.3││
│  └──┘ └──┘ └──┘ └──┘│
└──────────┬───────────┘
           │ Selected: sub-cluster 2.1
           ▼
┌──────────────────────┐
│  Level 3 Routing     │
│  4 experts (340)     │
│  ┌───┐┌───┐┌───┐┌───┐│
│  │E36││E37││E38││E39││
│  └───┘└───┘└───┘└───┘│
└──────────┬───────────┘
           │ Top-8 experts selected
           ▼
    Expert Computation
```

**Annotation:** "4 × 4 × 4 = 64 total experts, O(log₄ 64) = 3 routing steps"

---

### FIG. 6 — Confidence-Gated Routing Flowchart

**Type:** Decision flowchart

```
[Input Token] (410)
      │
      ▼
┌─────────────────────┐
│ BVH Router          │
│ Compute logits (420)│
└────────┬────────────┘
         │
         ▼
┌─────────────────────────────┐
│ Compute Confidence (430)    │
│ c = σ(3.0·std(top_k) - 1.5)│
└────────┬────────────────────┘
         │
         ▼
    ◇ c ≥ T? (440)
   ╱         ╲
  YES         NO
  │            │
  ▼            ▼
┌──────────┐  ┌──────────────┐
│ BVH Route│  │ Linear Gate  │
│ O(log N) │  │ O(N·M)       │
│   (450)  │  │   (460)      │
└────┬─────┘  └──────┬───────┘
     │               │
     └───────┬───────┘
             ▼
      [Expert Dispatch]
          (470)
```

**Annotations:**
- At decision diamond: "T = tunable threshold (0.85–0.99)"
- At BVH route: "Fast path: ~10 μs"
- At Linear gate: "Exact path: ~1,400 μs"
- Below: "At T=0.90: 69% BVH, 31% gate"

---

## PATENT 2: Inception Engine (LBS-2026-002)

### FIG. 1 — 4-Level Nested IAS Hierarchy

**Type:** Tree/hierarchy diagram

**Draw a 4-level tree:**

```
Level 0 — IAS_root (Domains)
  64 nodes, each labeled with domain names:
  ┌─── "Code" (510)
  ├─── "Music" (512)
  ├─── "Science" (514)
  └─── ... (64 total)

Level 1 — IAS_level1 (Subdomains)
  Under "Code":
  ┌─── "Python" (520)
  ├─── "Rust" (522)
  ├─── "JavaScript" (524)
  └─── ... (64 per domain)

Level 2 — IAS_level2 (Concepts)
  Under "Python":
  ┌─── "Functions" (530)
  ├─── "Classes" (532)
  ├─── "Loops" (534)
  └─── ... (256 per subdomain)

Level 3 — GAS_leaves (Leaves/Tokens)
  Under "Loops":
  ┌─── "for" (540)
  ├─── "while" (542)
  ├─── "range" (544)
  └─── ... (1024 per concept)
```

**Right side annotations:**
- "Level 0: 3D (dims 1-3)" → "64 domains"
- "Level 1: 3D (dims 4-6)" → "4,096 subdomains"
- "Level 2: 3D (dims 7-9)" → "1,048,576 concepts"
- "Level 3: 3D (dims 10-12)" → "1,073,741,824 leaves"
- "Total effective: 12D semantic space"

**Between levels:** Show dashed arrows labeled "Affine Portal (see FIG. 2)"

---

### FIG. 2 — Affine Portal Transformation

**Type:** Geometric diagram

**Show two 3D coordinate systems side by side:**

**Left (Parent Level):**
- 3D axes labeled x₁, y₁, z₁
- A point P₁ = (x₁, y₁, z₁) marked
- A small cube representing the node's AABB

**Center (Transformation):**
- A 4×4 matrix labeled "Affine Portal M (610)"
- Show the matrix structure:
  ```
  ┌                    ┐
  │ R₀₀ R₀₁ R₀₂ t_x  │
  │ R₁₀ R₁₁ R₁₂ t_y  │
  │ R₂₀ R₂₁ R₂₂ t_z  │
  │  0    0    0   1   │
  └                    ┘
  ```
- Arrow from left to right labeled "P₂ = M · P₁"
- Annotation: "Rotation + Scale + Translation"

**Right (Child Level):**
- Different 3D axes labeled x₂, y₂, z₂ (slightly rotated/scaled)
- The transformed point P₂ = (x₂, y₂, z₂)
- A different coordinate system showing the child's semantic space

**Bottom annotation:** "Each OptixInstance stores one affine portal. 4 portals traversed = 12D effective space."

---

### FIG. 3 — Fourier Resonance at Leaf Level

**Type:** Waveform + structure diagram

**Top:** A leaf node (rectangle) containing M=8 "semantic strings"

**Bottom:** Show each string as a sinusoidal wave:
```
String 1: f₁=0.5, A₁=0.8, φ₁=0.0    [wave drawing]
String 2: f₂=1.2, A₂=0.3, φ₂=π/4    [wave drawing]
...
String M: fₘ=3.7, Aₘ=0.1, φₘ=π      [wave drawing]
```

**Formula:** `resonance(x) = Σᵢ Aᵢ · sin(2π·fᵢ·x + φᵢ)`

**Annotation:** "M Fourier modes encode fine-grained semantic distinctions within a single leaf"

---

### FIG. 4 — OptiX Scene Graph Construction

**Type:** Block diagram showing OptiX data structures

```
OptixTraversableHandle (root) (710)
         │
    ┌────┴────┐
    │ IAS_root│  ← OptixBuildInput (instances)
    └────┬────┘
         │
   ┌─────┼─────┐
   ▼     ▼     ▼
┌─────┐┌─────┐┌─────┐
│IAS_1││IAS_2││IAS_3│ ← Each is OptixTraversableHandle
│(720)││(722)││(724)│    with affine transform in
└──┬──┘└──┬──┘└──┬──┘    OptixInstance.transform[12]
   │      │      │
   ▼      ▼      ▼
  ...    ...    ...      ← Recursion to level 3

                ▼
          ┌──────────┐
          │ GAS_leaf │ ← OptixBuildInput (triangles)
          │  (740)   │    Contains actual geometry
          └──────────┘
```

**Annotation:** "Single optixTrace() traverses all 4 levels. Hardware applies affine transforms automatically."

---

### FIG. 5 — Training Loss Convergence

**Type:** Dual-axis line graph

- **X-axis:** "Epoch" (1 to 10)
- **Y-axis left:** "Task Loss (Cross-Entropy)" — scale 4.0 to 6.0
- **Y-axis right:** "Spatial Loss" — scale 0.0 to 4.0

**Two curves:**
1. **Task loss (solid line):** Starts ~5.5, decreases to ~5.22 at epoch 10
2. **Spatial loss (dashed line):** Starts ~3.58, decreases to ~0.11 at epoch 10

**Annotations:**
- "32× reduction in spatial loss"
- "Final PPL: 185.4 (vs GPT-2 baseline 182.2, +1.8%)"

---

## PATENT 3: Spectral Routing (LBS-2026-003)

### FIG. 1 — Spectral Encoding Pipeline

**Type:** Block diagram (horizontal flow)

```
[Context History] (810)     [W_spectral (820)]
  "tokens t₁...tₙ"           k × D matrix
        │                        │
        ▼                        ▼
┌────────────────────────────────────┐
│    Spectral Encoder (830)         │
│    f = W_spectral · context_emb   │
│    f ∈ R^k, k=256                 │
└────────────────┬───────────────────┘
                 │ spectral color vector f
                 ▼
┌──────────────────────────┐
│    PrismaticRay (840)    │
│    origin: query_pos     │
│    direction: d          │
│    color: f ∈ R^256      │
│    phase: φ              │
└──────────────────────────┘
```

**Annotation:** "The spectral color f encodes conversational context — same geometry, different routing based on context"

---

### FIG. 2 — Prismatic Sphere with Snell's Law

**Type:** Cross-sectional geometric diagram

**Draw a circle (the prismatic sphere, labeled 900):**

- An incoming ray (arrow) hitting the sphere surface at angle θ_in from the normal
- The normal vector (dashed line) at the intersection point
- The refracted ray (arrow) exiting at angle θ_out
- The angle labels: θ_in (910), θ_out (920)

**Inside the sphere:**
- Label: "n = n_base + σ(W_disp · f)" (930)
- Show n_base as fixed value, σ(W_disp · f) as context-dependent addition

**Outside the sphere:**
- Label: "n_outside = 1.0"

**Formula box below:**
```
Snell's Law:
sin(θ_out) = sin(θ_in) / n
n = n_base + σ(W_dispersion · f)

d_out = n_ratio · d_in + (n_ratio · cos_i - cos_t) · normal
```

**Right side:** Show 4 different colored rays (representing different contexts) entering at the same angle but refracting at different angles:
- "Context: Code" → steep refraction → "Block A"
- "Context: Music" → shallow refraction → "Block B"
- **Label:** "Same sphere, different routing"

---

### FIG. 3 — Polysemy Resolution

**Type:** Comparative diagram showing 3 scenarios side by side

**Central element:** A single sphere labeled "bank" (semantic node 1000)

**Three scenarios (left to right):**

1. **Financial context (1010):**
   - Ray colored "BLUE" (labeled "f = financial context")
   - Enters sphere, refracts at angle α₁
   - Exits toward "Expert: Finance (1012)"
   - Example sentence: "I deposited money at the _bank_"

2. **Geographic context (1020):**
   - Ray colored "RED" (labeled "f = geographic context")
   - Enters SAME sphere, refracts at angle α₂ ≠ α₁
   - Exits toward "Expert: Geography (1022)"
   - Example sentence: "The river _bank_ was muddy"

3. **Programming context (1030):**
   - Ray colored "GREEN" (labeled "f = programming context")
   - Enters SAME sphere, refracts at angle α₃
   - Exits toward "Expert: Programming (1032)"
   - Example sentence: "The memory _bank_ overflowed"

**Bottom annotation:** "Single sphere, zero weight duplication, 98.4% disambiguation accuracy"

---

### FIG. 4 — Chromatic Aberration (Multi-Band Decomposition)

**Type:** Optical prism analogy diagram

**Left:** A single ray labeled "Broadband spectral color f" entering a triangular prism (sphere cross-section)

**Inside prism:** The ray splits into B=4 bands:
- Band 1 (f[0:64]): refracted at angle α₁
- Band 2 (f[64:128]): refracted at angle α₂
- Band 3 (f[128:192]): refracted at angle α₃
- Band 4 (f[192:256]): refracted at angle α₄

**Right:** The 4 bands exit at slightly different angles, each pointing to potentially different expert blocks

**Below:** A voting table:
```
Band 1 → Expert A   ┐
Band 2 → Expert A   ├→ Majority vote: Expert A (3/4)
Band 3 → Expert A   ┘
Band 4 → Expert B   → Minority: Expert B (1/4)
```

**Annotation:** "B bands provide robustness through spectral decomposition"

---

### FIG. 5 — Total Internal Reflection

**Type:** Geometric cross-section diagram

**Draw two adjacent spheres (domains) with different refractive indices:**

- **Sphere 1 (1110):** n₁ = 1.8 (labeled "Domain: Programming")
- **Sphere 2 (1120):** n₂ = 1.2 (labeled "Domain: Music")
- **Boundary between them** marked with bold line

**Scenario A (refraction — angle > critical):**
- Ray enters at steep angle (above critical angle)
- Crosses boundary into Sphere 2
- Label: "θ > θ_critical → Refraction (cross-domain routing)"

**Scenario B (TIR — angle < critical):**
- Ray enters at shallow angle (below critical angle)
- REFLECTS back into Sphere 1 (total internal reflection)
- Label: "θ < θ_critical → TIR (stays in domain)"

**Formula:** "θ_critical = arcsin(n₂/n₁)"
**Annotation:** "TIR enforces hard domain boundaries — prevents 'code' tokens from being routed to 'music' experts"

---

### FIG. 6 — Phase-Coherent Multi-Ray Interference

**Type:** Wave interference diagram

**Top:** Show R=8 rays emitted from the same token, each with slightly different spectral colors (f + ε_i):

```
Ray 1: f + ε₁ → phase φ₁ at target
Ray 2: f + ε₂ → phase φ₂ at target
...
Ray 8: f + ε₈ → phase φ₈ at target
```

**Center:** All 8 rays converge at a target node. Show their phases:

**Bottom left — Constructive interference:**
- Multiple waves in phase (aligned peaks)
- Sum = large amplitude
- Label: "High confidence: route to this expert"

**Bottom right — Destructive interference:**
- Waves out of phase (peaks cancel troughs)
- Sum ≈ 0
- Label: "Low confidence: uncertain routing"

**Formula:** "confidence = |Σᵢ exp(i·φᵢ)| / R"
**Annotation:** "Coherent rays → high confidence, incoherent rays → fallback to linear gate"

---

## Delivery Checklist

For each figure:
- [ ] SVG format (vector, scalable)
- [ ] PNG at 300 DPI (for USPTO EFS submission)
- [ ] Black and white only (no color, no grayscale)
- [ ] Reference numerals (100, 200, etc.) with lead lines
- [ ] Text labels readable at printed size
- [ ] Consistent style across all 17 figures
- [ ] Each figure fits on one sheet (8.5" × 11" with margins)
