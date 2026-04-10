# Social Media Posts — SpectralAI Publication

> DOIs published:
> - SpectralAI: https://doi.org/10.5281/zenodo.19457288
> - Expert Specialization: https://doi.org/10.5281/zenodo.19457411
> - Spectral Routing: https://doi.org/10.5281/zenodo.19457473

---

## LinkedIn

**What if we used ray tracing cores for AI instead of gaming?**

I just published 3 research papers on Zenodo exploring a question nobody seems to be asking: NVIDIA GPUs have dedicated RT Cores for ray tracing that sit completely idle during LLM inference. What if we repurposed them?

**SpectralAI** replaces the O(N) linear gate in Mixture-of-Experts models with O(log N) BVH traversal on RT Cores:

- 113-218x routing speedup
- 731x VRAM reduction
- 95.9% mean accuracy across 16 layers
- Only +1.5% perplexity vs baseline
- Validated on OLMoE-1B-7B with a single RTX 5070 Ti

The key insight: finding which expert should process a token is geometrically equivalent to finding which object a ray hits. Same hardware, different problem.

Along the way, I discovered something unexpected about how MoE models actually work:

**MoE experts don't specialize by topic.** They specialize by syntactic token type (content words, function words, punctuation). This was validated across 3 models (OLMoE, Qwen, DeepSeek) with very different architectures. The "science expert" or "code expert" doesn't exist -- experts are closer to "content word processors" and "punctuation handlers."

I also developed **Spectral Routing**, which uses optical principles (Snell's law, chromatic aberration, total internal reflection) to resolve word polysemy at the routing level. "Bank" routes to different experts depending on whether you're talking about finance or rivers -- 98.4% accuracy, <0.12% overhead.

All data, code, and results are open and reproducible.

Paper 1 (main): https://doi.org/10.5281/zenodo.19457288
Paper 2 (expert analysis): https://doi.org/10.5281/zenodo.19457411
Paper 3 (spectral routing): https://doi.org/10.5281/zenodo.19457473

#AI #MachineLearning #NLP #MixtureOfExperts #NVIDIA #RayTracing #DeepLearning #LLM #Research #OpenScience

---

## LinkedIn — Follow-up (repo público)

**The code is now open source.**

Last week I published 3 papers on repurposing NVIDIA RT Cores for LLM expert routing. Today the full repository is public:

https://github.com/JordiSilvestre/Spectral-AI

What's inside:
- Complete CUDA kernels (BVH router, OptiX 9.0 shaders)
- Python training & evaluation pipeline
- Pre-trained checkpoints for all 16 MoE layers
- 180+ automated tests with full reproduction scripts
- Expert specialization analysis across 3 MoE architectures
- All benchmark data from the papers

Everything runs on a single RTX 5070 Ti. No cluster, no cloud, no API keys.

The papers (open access on Zenodo):
- SpectralAI: https://doi.org/10.5281/zenodo.19457288
- Expert Specialization: https://doi.org/10.5281/zenodo.19457411
- Spectral Routing: https://doi.org/10.5281/zenodo.19457473

If you're working on MoE routing, expert analysis, or hardware-aware inference — feel free to use, fork, or build on it.

#OpenSource #AI #MachineLearning #NVIDIA #RayTracing #MixtureOfExperts #LLM #CUDA #Research

---

## X (Twitter) — Follow-up (repo público)

**Tweet 1/2:**
The SpectralAI repo is now public.

Full CUDA kernels, training pipeline, checkpoints, 180+ tests, and all data from the 3 papers.

Single RTX 5070 Ti. No cluster needed.

https://github.com/JordiSilvestre/Spectral-AI

**Tweet 2/2:**
What's reproducible:
- BVH routing: 218x speedup, 731x VRAM reduction
- Expert specialization analysis on OLMoE, Qwen-MoE, DeepSeek-MoE
- Spectral routing: 98.4% polysemy resolution
- All 16 layers trained + evaluated

Papers:
https://doi.org/10.5281/zenodo.19457288
https://doi.org/10.5281/zenodo.19457411
https://doi.org/10.5281/zenodo.19457473

---

## X (Twitter) — Thread (original)

**Tweet 1/6:**
I used NVIDIA ray tracing cores to route tokens in LLMs.

Result: 218x faster routing, 731x less VRAM, +1.5% perplexity.

RT Cores sit idle during inference. Turns out finding "which expert processes this token" is the same problem as "which object does this ray hit."

3 papers now on Zenodo. Thread:

**Tweet 2/6:**
SpectralAI projects tokens into 3D space, builds a BVH (Bounding Volume Hierarchy), and lets RT Cores do the work.

64 experts searched in ~6 ray-AABB tests instead of a 64-way matrix multiply.

OLMoE-1B-7B, 16 layers, RTX 5070 Ti:
- 95.9% top-8 accuracy
- 19.1 us/batch
- 13.4M queries/sec

**Tweet 3/6:**
Surprising finding: MoE experts DON'T specialize by topic.

I analyzed 3 models (OLMoE, Qwen, DeepSeek). The "best" topic specialist activates only 2.3x above uniform baseline.

What they actually specialize in: syntactic token type. Content words, function words, punctuation. Across all 3 models.

**Tweet 4/6:**
Expert selectivity follows a universal U-shaped curve:

High (early layers) -> Low (middle) -> High (late)

Middle layers = "reorganization zone" where expert roles become fluid. Confirmed in all 3 architectures (7B-16B, top-4 to top-8).

This directly explains why L8 is the hardest layer to route.

**Tweet 5/6:**
Bonus: Spectral Routing uses Snell's law (yes, from optics) to resolve polysemy.

"Bank" in a finance context routes differently than "bank" near a river.

98.4% accuracy on 80 polysemous words (442 context pairs) vs 72.3% baseline. Overhead: <0.12%.

Total internal reflection acts as a natural domain boundary.

**Tweet 6/6:**
Everything is open:

Paper 1 — SpectralAI (BVH routing): https://doi.org/10.5281/zenodo.19457288
Paper 2 — Expert Specialization (3-model study): https://doi.org/10.5281/zenodo.19457411
Paper 3 — Spectral Routing (polysemy via optics): https://doi.org/10.5281/zenodo.19457473

All data + code included. Single consumer GPU. No cluster needed.

---

## Reddit (r/MachineLearning)

**Title:** [R] I repurposed NVIDIA RT Cores (ray tracing) for MoE expert routing — 218x speedup, 731x VRAM reduction

**Body:**

I've been working on a system called SpectralAI that replaces the standard linear routing gate in Mixture-of-Experts models with hardware-accelerated BVH (Bounding Volume Hierarchy) traversal on NVIDIA RT Cores.

**The idea:** Finding which expert should process a token is geometrically equivalent to finding which object a ray intersects. Project tokens into 3D, build a BVH over expert centroids, and let the dedicated RT Core hardware do O(log N) traversal instead of O(N) matrix multiplication.

**Results on OLMoE-1B-7B (64 experts, 16 MoE layers, RTX 5070 Ti):**

| Metric | Value |
|--------|-------|
| Mean top-8 routing accuracy | 95.9% |
| Best pre-filter PPL | 6.79 (+1.5% vs baseline 6.69) |
| RT Core latency | 19.1 us/batch |
| Throughput | 13.4M queries/sec |
| Speedup vs PyTorch gate | 113-218x |
| VRAM reduction | 731x |
| HellaSwag accuracy drop | -1.1 pp (52.0% vs 53.1%) |

**Side discovery — expert specialization is syntactic, not semantic:**

I analyzed expert activation patterns across 3 MoE models (OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, DeepSeek-MoE-16B). Everyone assumes experts become "topic specialists" (the science expert, the code expert, etc.). They don't. The most topic-specialized expert achieves only 2.3x the uniform baseline.

What they actually specialize in: **syntactic token types** — content words, function words, punctuation, capitalized tokens. This was consistent across all 3 architectures despite very different sizes (7B-16B), routing strategies (top-4 to top-8), and expert counts.

Expert selectivity follows a U-shaped curve (high early, low middle, high late) in all models, and co-activation clusters reorganize in middle layers. This is why middle layers are the hardest to route accurately.

**Third paper — Spectral Routing:**

Uses optical principles (Snell's law, chromatic aberration, total internal reflection) for context-dependent routing. Resolves polysemy ("bank" = finance vs river) at 98.4% accuracy vs 72.3% for the standard gate. Total internal reflection acts as a natural domain boundary — when a token's context is incompatible with a semantic node, the "ray" reflects instead of entering. No learned rejection parameters needed.

**Links:**
- Paper 1 (SpectralAI): https://doi.org/10.5281/zenodo.19457288
- Paper 2 (Expert Specialization, 3-model study): https://doi.org/10.5281/zenodo.19457411
- Paper 3 (Spectral Routing): https://doi.org/10.5281/zenodo.19457473

All data and code included. Everything runs on a single RTX 5070 Ti.

Happy to answer questions. This is independent research, no affiliation.

---

## Reddit (r/LocalLLaMA)

**Title:** Used ray tracing cores on my RTX 5070 Ti for LLM routing — 218x speedup, runs entirely on 1 consumer GPU

**Body:**

Quick summary: I found a way to use the RT Cores (normally used for ray tracing in games) to handle expert routing in MoE models. Those cores sit completely idle during LLM inference, so why not put them to work?

**What it does:**
- Takes the routing decision in MoE models (which experts process which tokens)
- Projects tokens into 3D space
- Uses the GPU's dedicated ray tracing hardware to find the right experts
- O(log N) instead of O(N) — hardware-accelerated

**Numbers (OLMoE-1B-7B, RTX 5070 Ti 16GB):**
- 218x faster routing at batch 1024
- 731x less VRAM for routing
- Only +1.5% perplexity hit
- 95.9% routing accuracy

**Unexpected discovery:** I also found that MoE experts don't actually specialize by topic. Tested across 3 different models (OLMoE, Qwen-MoE, DeepSeek-MoE) — they all specialize by syntactic type (content words vs function words vs punctuation). The "science expert" is a myth.

Everything is open access on Zenodo with full data and reproduction instructions.

https://doi.org/10.5281/zenodo.19457288

---

## Hacker News (Show HN)

**Title:** Show HN: I used NVIDIA RT Cores for LLM expert routing – 218x speedup on a single GPU

**URL:** https://github.com/JordiSilvestre/Spectral-AI

**Comment:**

SpectralAI repurposes the ray tracing hardware in NVIDIA GPUs (RT Cores) to handle expert routing in Mixture-of-Experts LLMs. The core insight: finding which expert should process a token is geometrically the same as finding which object a ray intersects.

Instead of a 64-way matrix multiply, we project tokens into 3D, build a BVH, and let RT Cores do O(log N) traversal. On an RTX 5070 Ti with OLMoE-1B-7B (64 experts, 16 layers):

- 113-218x routing speedup
- 731x VRAM reduction
- 95.9% routing accuracy
- +1.5% perplexity vs baseline

Side discovery: MoE experts specialize by syntactic token type (content words, function words, punctuation), not by topic. Validated across 3 architectures (OLMoE, Qwen-MoE, DeepSeek-MoE).

Full code, CUDA kernels, checkpoints, and 180+ tests included. Single consumer GPU, no cloud needed.

Papers (open access):
- https://doi.org/10.5281/zenodo.19457288
- https://doi.org/10.5281/zenodo.19457411
- https://doi.org/10.5281/zenodo.19457473

Independent research, happy to answer questions.

---

## Papers With Code

**Title:** SpectralAI: RT Core BVH Routing for Mixture-of-Experts
**Paper URL:** https://doi.org/10.5281/zenodo.19457288
**Code URL:** https://github.com/JordiSilvestre/Spectral-AI
**Tasks:** Mixture-of-Experts Routing, Expert Selection
**Datasets:** OLMoE-1B-7B, Qwen1.5-MoE-A2.7B, DeepSeek-MoE-16B
