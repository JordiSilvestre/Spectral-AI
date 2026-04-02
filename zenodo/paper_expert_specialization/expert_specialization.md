# Expert Specialization in Mixture-of-Experts Language Models: Syntactic Roles Dominate Semantic Topics

**Jordi Silvestre Lopez**
Independent Researcher

**Date:** 2026-04-02
**DOI:** [To be assigned by Zenodo]
**License:** CC-BY 4.0

---

## Abstract

We present a comprehensive analysis of expert specialization patterns in OLMoE-1B-7B, a 7-billion-parameter Mixture-of-Experts (MoE) language model with 64 experts per layer across 16 MoE layers. Contrary to the common assumption that MoE experts specialize by semantic topic (e.g., "science expert", "code expert"), our analysis reveals that experts primarily specialize by **syntactic token type**: content words, function words, punctuation, and capitalized tokens. The most topic-specialized expert achieves only 6.8% activation rate for its primary topic (vs. 3.3% uniform baseline across 30 categories), indicating near-uniform topic distribution. We further discover that expert co-activation naturally forms 4 clusters of 16 experts per layer -- a structure that aligns with our BVH router's 4x4x4 hierarchy -- but these clusters are **not stable across layers** (20--31% inter-layer stability), and not all layers produce exactly 4 clusters (7/16 layers have a 5th outlier cluster of 1--2 experts), implying that each layer requires its own BVH organization. Expert selectivity follows a **U-shaped curve** across layers: high in early layers (0.517--0.609), low in middle layers (0.384--0.421), and high again in late layers (0.510--0.592).

---

## 1. Introduction

Mixture-of-Experts (MoE) models (Fedus et al., 2022; Jiang et al., 2024; Muennighoff et al., 2024) activate only a subset of parameters per token via a learned routing function. A natural question is: *what do individual experts learn?* Understanding expert specialization informs routing algorithm design. If experts specialize by topic, a semantic router could map tokens to topically-relevant experts. If they specialize by syntactic role, a different routing strategy is needed.

We analyze OLMoE-1B-7B (Muennighoff et al., 2024) -- 1B active parameters, 7B total, 64 experts per layer, top-8 routing, 16 MoE layers -- using 4,000 tokens spanning 30 semantic categories from diverse domains (STEM, code, humanities, professional, conversational).

---

## 2. Methodology

### 2.1 Semantic Categories

We defined 30 categories spanning:
- **STEM** (7): algebra, calculus, statistics, linear_algebra, physics, chemistry, biology
- **Code** (5): python_code, systems_code, web_code, devops, algorithms
- **Humanities** (4): narrative, literary_analysis, poetry, grammar
- **Social Sciences** (4): history, geography, law_politics, economics
- **Professional** (4): medical, legal_text, financial, formal_writing
- **Conversational** (3): casual_chat, instruction, emotional
- **Special** (3): punctuation, numbers_data, multilingual

Each category contains 100--212 tokens (total: 4,000 tokens), generated from 8 diverse prompts per category.

### 2.2 Analysis Pipeline

For each token, we record:
1. Which top-8 experts are activated (from the original OLMoE linear gate)
2. The token's syntactic type (content_word, function_word, punctuation, number, capitalized, code_syntax, whitespace)
3. The semantic category of the source text

We then compute per-expert activation distributions across categories (topic specialization), per-expert token type distributions (syntactic specialization), per-layer selectivity scores, and co-activation clustering.

Scripts: `python/analyze_experts.py`. Data: `results/expert_catalog_exhaustive.json`, `results/expert_deep_analysis.json`.

---

## 3. Results

### 3.1 Topic Specialization Is Weak

The most topic-specialized expert (Expert 37, "history") activates for only 6.8% of its primary category's tokens -- barely 2x the uniform baseline of 3.3% (= 100% / 30 categories). The top 10 most specialized experts:

| Expert | Primary Topic | Activation % | vs. Uniform |
|---|---|---|---|
| 37 | history | 6.8% | 2.1x |
| 50 | poetry | 6.6% | 2.0x |
| 49 | law_politics | 6.5% | 2.0x |
| 53 | emotional | 6.3% | 1.9x |
| 10 | poetry | 6.2% | 1.9x |
| 60 | multilingual | 6.2% | 1.9x |
| 4 | multilingual | 6.2% | 1.9x |
| 22 | poetry | 6.2% | 1.9x |
| 42 | devops | 6.1% | 1.8x |
| 40 | history | 6.1% | 1.8x |

The least specialized expert (Expert 54, "punctuation") activates at 4.1%, close to uniform. **Experts do not form clear topic specialists** -- they process tokens from all 30 categories with near-uniform frequency.

### 3.2 Syntactic Token Type Specialization

In contrast to the weak topic specialization, experts show strong syntactic type differentiation. At layer 0:

- **49 of 64 experts** are content-word dominant (>40% content_word tokens)
- The remaining experts handle function words, punctuation, and mixed types

Token type composition across the 4 natural co-activation clusters at L0:

| Cluster | Content Words | Function Words | Punctuation | Role |
|---|---|---|---|---|
| C0 | 53.4% | 15.1% | 13.4% | Content-heavy |
| C1 | 46.6% | 23.2% | 14.2% | Content + function |
| C2 | 46.5% | 16.2% | 19.6% | Content + punctuation |
| C3 | 46.4% | 24.0% | 13.8% | Content + function |

The dominant pattern is content-word processing, with clusters differentiating by their secondary type (function words vs. punctuation).

### 3.3 U-Shaped Selectivity Across Layers

Expert selectivity (measured as the standard deviation of per-expert activation rates across token types) follows a U-shaped curve:

| Layer Group | Layers | Mean Selectivity | Interpretation |
|---|---|---|---|
| Early | L0--L3 | 0.569 | High discrimination |
| Middle | L5--L8 | 0.399 | Uniform processing |
| Late | L12--L15 | 0.543 | High discrimination |

Per-layer selectivity:

| Layer | Selectivity | Layer | Selectivity |
|---|---|---|---|
| L0 | 0.573 | L8 | 0.388 |
| L1 | 0.609 | L9 | 0.399 |
| L2 | 0.578 | L10 | 0.446 |
| L3 | 0.517 | L11 | 0.478 |
| L4 | 0.458 | L12 | 0.514 |
| L5 | 0.421 | L13 | 0.592 |
| L6 | 0.402 | L14 | 0.558 |
| L7 | 0.384 | L15 | 0.510 |

Early layers perform initial token categorization (distinguishing content from function words). Middle layers process more uniformly (contextual blending). Late layers re-specialize for output prediction.

### 3.4 Co-Activation Clusters Are Per-Layer

Co-activation analysis reveals a predominantly 4-cluster structure at most layers (9/16 layers produce exactly 4 clusters of 16 experts each), aligning with our 4x4x4 BVH hierarchy. The remaining 7 layers produce a 5th outlier cluster of 1--2 experts. Cluster membership is **not stable across layers**:

| Layer Transition | Stability |
|---|---|
| L0 -> L1 | 20.3% |
| L1 -> L2 | 21.9% |
| L2 -> L3 | 31.2% |
| L3 -> L4 | 29.7% |

On average, only ~25% of experts remain in the same cluster between adjacent layers (measured by cluster ID overlap; pair-based Jaccard stability yields similar values: 23--27%). This means a single BVH tree cannot serve all layers -- each layer requires its own geometric organization.

### 3.5 Content Expert Evolution

The number of content-word dominant experts (>40% content_word activation) varies across layers:

| Layer | Content-Dominant Experts | Interpretation |
|---|---|---|
| L0 | 49/64 (76.6%) | Strong content focus |
| L9 | 35/64 (54.7%) | More diverse |
| L15 | 47/64 (73.4%) | Re-specialization |

---

## 4. Implications for BVH Routing

These findings have direct implications for our BVH-based MoE routing system (SpectralAI):

1. **Per-layer BVH construction:** Since co-activation clusters change between layers (20--31% stability), the BVH tree must be built independently for each layer using that layer's co-activation structure. A single global BVH will perform poorly.

2. **Syntactic-aware routing:** Since experts specialize by token type rather than topic, the BVH geometry should cluster experts by their syntactic processing role (content vs. function vs. punctuation), not by semantic domain.

3. **U-shaped optimization:** Middle layers (L5--L8) have lower selectivity, meaning routing accuracy matters less there. This aligns with our observation that L8 has the lowest BVH accuracy (89.3%) -- middle layers are inherently harder to route because experts process more uniformly.

4. **Natural 4-cluster hierarchy:** The predominantly 4 x 16 cluster structure (9/16 layers) validates the BVH branching factor of 4 used in our 3-level hierarchy (4 x 4 x 4 = 64 experts). The 7 layers with outlier clusters suggest some experts resist clean geometric partitioning.

---

## 5. Connection to SpectralAI Results

These expert analysis findings complement the main SpectralAI results:

| Metric | Value |
|---|---|
| BVH Router accuracy (mean, 16 layers) | 95.9% top-8 |
| Best layer accuracy | L15: 97.6% |
| Worst layer accuracy | L8: 89.3% |
| Pre-filter 48 cand. PPL | 6.79 (+1.5%) |
| RT Core latency | 19.1 us, 13.4M q/s, 100% accuracy |
| Routing speedup | 113--218x |
| VRAM reduction | 731x |
| HellaSwag: baseline | 53.1% (N=2,000) |
| HellaSwag: 3-layer | 52.2% (N=2,000) |
| HellaSwag: 16-layer | 52.0% (N=2,000) |
| Polysemy accuracy | 98.4% (80 words, 442 pairs) |

---

## 6. Conclusion

Our exhaustive analysis of OLMoE-1B-7B reveals that MoE expert specialization is primarily **syntactic, not semantic**. Experts differentiate by token type (content words, function words, punctuation) rather than by topic domain. Expert selectivity follows a U-shaped curve across layers, and co-activation clusters reorganize between layers (20--31% inter-layer stability). These findings directly inform BVH router design: per-layer BVH construction with syntactic-aware clustering outperforms global or topic-based organization.

---

## Data Availability

- Expert catalog: `results/expert_catalog_exhaustive.json` (30 categories, 4,000 tokens, 64 experts)
- Deep analysis: `results/expert_deep_analysis.json` (16 layers, per-layer clusters, selectivity, token types)
- Analysis script: `python/analyze_experts.py`

---

## References

Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers. *JMLR*, 23(120), 1--39.

Jiang, A. Q., et al. (2024). Mixtral of experts. *arXiv:2401.04088*.

Muennighoff, N., et al. (2024). OLMoE: Open Mixture-of-Experts Language Models. *arXiv*.

---

## Author

Jordi Silvestre Lopez, 2026.
