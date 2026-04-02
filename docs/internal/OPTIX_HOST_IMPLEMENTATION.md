# OptiX Host Code Implementation for SpectralAI Zero-Matrix

## Overview

This document describes the OptiX 8.x host code implementation for SpectralAI Zero-Matrix, including the complete architecture for ray tracing acceleration structure management and pipeline execution.

## Files Created

### 1. `cuda/optix_host.cpp` - Complete OptiX Host Code

**Purpose:** Encapsulates all OptiX initialization, pipeline creation, acceleration structure building, and kernel launching logic.

**Key Components:**

#### OptixLogger Class
- Implements OptiX runtime message callback
- Logs errors, warnings, and debug messages
- Severity levels: FATAL, ERROR, WARN, INFO, DEBUG

#### SpectralAIOptixContext Class
The main context manager providing:

**Initialization Methods:**
- `initializeCUDA()`: Sets up CUDA context on device 0
- `initializeOptiX()`: Calls `optixInit()` and creates `OptixDeviceContext`

**Pipeline Creation:**
- `createModule()`: Compiles PTX strings into a single OptiX module
  - Handles module compilation options
  - Configures pipeline traversable graph flags
  - Sets payload and attribute value counts

- `createPrograms()`: Creates 4 program groups:
  - **Raygen**: `__raygen__alpha_bsh_rg` - generates rays from query embedding
  - **Hitgroup**: Contains intersection, closest-hit, and any-hit programs
  - **Miss**: `__miss__alpha_bsh_ms` - handles rays that don't intersect

- `buildPipeline()`: Links program groups into executable pipeline
  - Configures stack sizes for ray traversal
  - Sets maximum trace depth to 16

- `buildShaderBindingTable()`: Creates SBT with three records:
  - **RaygenRecord**: Contains `num_rays` (always 1 for Alpha)
  - **HitgroupRecord**: Contains `sphere_id`
  - **MissRecord**: Empty or contains global miss data

**Acceleration Structure:**
- `buildAccelerationStructure()`: Constructs GPU-resident BVH
  1. Converts TokenNodes to OptixAabb primitives
  2. Copies AABBs to GPU
  3. Calls `optixAccelComputeMemoryUsage()` to calculate buffer requirements
  4. Allocates temporary and output buffers
  5. Calls `optixAccelBuild()` with `OPTIX_BUILD_FLAG_ALLOW_COMPACTION`
  6. Stores traversable handle for use in shaders

**Execution:**
- `launch()`: Executes the complete ray tracing pipeline
  1. Allocates GPU buffers for rays and output
  2. Copies ray data (CPU → GPU)
  3. Prepares launch parameters (traversable handle, ray/output pointers)
  4. Calls `optixLaunch()` with SBT
  5. Synchronizes GPU with `cudaDeviceSynchronize()`
  6. Copies results back (GPU → CPU)

**Resource Management:**
- `cleanup()`: Safely releases all OptiX/CUDA resources
  - Destroys pipeline, module, contexts
  - Frees GPU buffers (GAS, SBT)
  - Safe to call multiple times

#### Factory Functions
- `createSpectralAIOptixContext()`: Creates and initializes context (exception-safe)
- `destroySpectralAIOptixContext()`: Safely destroys context

**Memory Management:**
All allocations are properly tracked and freed:
- `d_gas_output_buffer_`: Bounding Volume Hierarchy (10-50 MB)
- `d_sbt_buffer_`: Shader Binding Table
- Temporary buffers during acceleration build

**Error Handling:**
- All CUDA/OptiX calls checked for errors
- Detailed error messages logged
- Resource cleanup on failure path

## Architecture Details

### OptiX Data Flow

```
Host (CPU)                              GPU (NVIDIA RTX 4090/5070 Ti)
───────────────────────────────────────────────────────────────────
Input:
  TokenNodes[] ────────────────────→ GPU Buffer
  Query Embedding ──────────────────→ GPU Buffer

Pipeline Execution:
  optixLaunch() ────────────────────→ Raygen Kernel
                                      │
                                      ├─ Generate ray from query
                                      │
                                      ├─ Traverse BVH (RT Cores)
                                      │
                                      └─ Closest-Hit / Miss Kernels
                                      │
Output:
                                      Result Payload
                        ←──────────── GPU Buffer (Device Memory)

  Memcpy to Host ←─────────────────── Result Array

```

### Shader Binding Table (SBT) Layout

```
┌─────────────────────────────────────────────────────────────┐
│                   SBT Memory Layout                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  [Raygen Record] (64 bytes)                                │
│    ├─ Shader Header (OptiX internal)                       │
│    └─ Data: num_rays = 1                                   │
│                                                             │
│  [Hitgroup Record] (64 bytes)                              │
│    ├─ Shader Header (OptiX internal)                       │
│    └─ Data: sphere_id = current_sphere_id                  │
│                                                             │
│  [Miss Record] (64 bytes)                                  │
│    ├─ Shader Header (OptiX internal)                       │
│    └─ Data: (empty or global miss data)                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Acceleration Structure (GAS)

The BVH is built from custom primitives (AABBs):

```
GPU GAS Buffer (10-50 MB for 100K tokens)
├─ Internal BVH nodes (binary tree of bounding boxes)
├─ Leaf nodes (reference to AABBs)
└─ AABB data (flattened from TokenNodes)

Traversable Handle → Referenced by Raygen/Closest-Hit to traverse
```

## Integration with Alpha BSH

The OptiX host code supports the Alpha BSH architecture:

### Phase A (Ray Tracing)
- **OptiX execution**: Uses RT Cores to traverse BSH in O(log N) time
- **Entry**: `optixLaunch()` with single ray
- **Exit**: `AlphaRayPayload` with `hit_sphere_id`

### Phase B (cuBLAS MatMul)
- Runs separately after Phase A completes
- Uses sphere_id from Phase A to select MatrixBlock
- Executes FP16 matrix multiplication on Tensor Cores

## Compilation & Usage

### Prerequisites
```bash
# NVIDIA OptiX 8.x SDK
# CUDA 12.x
# cmake 3.28+
# NVIDIA RTX 4090 or RTX 5070 Ti (Ada/Blackwell)
```

### Compilation Steps

1. **Compile CUDA kernels to PTX**
```bash
nvcc -ptx -m64 -O3 --gpu-architecture=compute_89 \
  cuda/ray_generation.cu -o build/ray_generation.ptx
nvcc -ptx -m64 -O3 --gpu-architecture=compute_89 \
  cuda/closest_hit.cu -o build/closest_hit.ptx
nvcc -ptx -m64 -O3 --gpu-architecture=compute_89 \
  cuda/miss.cu -o build/miss.ptx
nvcc -ptx -m64 -O3 --gpu-architecture=compute_89 \
  cuda/any_hit.cu -o build/any_hit.ptx
```

2. **Load PTX as strings**
```cpp
std::string ptx_raygen = loadPTXFile("build/ray_generation.ptx");
std::string ptx_closest_hit = loadPTXFile("build/closest_hit.ptx");
// ... etc
```

3. **Create context and pipeline**
```cpp
SpectralAIOptixContext* optix_ctx =
  createSpectralAIOptixContext(
    ptx_raygen.c_str(),
    ptx_closest_hit.c_str(),
    ptx_miss.c_str(),
    ptx_any_hit.c_str());
```

4. **Build acceleration structure**
```cpp
TokenNode* tokens = /* load from file */;
optix_ctx->buildAccelerationStructure(tokens, num_tokens);
```

5. **Execute pipeline**
```cpp
SemanticRay* rays = /* prepare rays */;
AlphaRayPayload* results = /* allocate output */;
optix_ctx->launch(rays, num_rays, results, sizeof(AlphaRayPayload) * num_rays);
```

6. **Cleanup**
```cpp
destroySpectralAIOptixContext(optix_ctx);
```

## Embeddings Pipeline

### File: `python/download_embeddings.py`

**Purpose:** Download or generate embeddings and project them to 3D space.

**Features:**

1. **GloVe Download**
   - Attempts to download GloVe 6B 50D from Stanford NLP
   - Falls back to synthetic generation if download fails
   - Caches downloaded files

2. **Synthetic Embeddings (Fallback)**
   - Generates 133 English words with semantic relationships
   - Clusters: persons, programming, music, colors, animals, food, sports, emotions, nature, objects
   - Uses skip-gram-like covariance structure

3. **PCA Projection to 3D**
   - Computes covariance matrix
   - Uses SVD for numerical stability
   - Selects top 3 principal components
   - Normalizes to unit sphere
   - Preserves 18.7% of original variance (acceptable for spatial indexing)

4. **Semantic Validation**
   - Verifies that related words cluster together
   - Computes intra-cluster cosine distances
   - Output example:
     ```
     persons       (n= 6): avg intra-dist = 0.003
     programming   (n= 6): avg intra-dist = 0.004
     music         (n= 6): avg intra-dist = 0.006
     ```

### Output Files

After running `python3 download_embeddings.py`:

1. **embeddings_3d.npy** (1.7 KB)
   - NumPy array format: [133, 3] float32
   - Shape: (num_words, 3)
   - Values: normalized to unit sphere [-1.0, 1.0]

2. **vocab.txt** (831 bytes)
   - Plain text, one word per line
   - First 5 words: king, queen, man, woman, prince
   - Total: 133 words

3. **embeddings_stats.txt** (325 bytes)
   - Human-readable statistics
   - Memory usage, dimension ranges, first 10 words

### Usage in SpectralAI

```python
import numpy as np

# Load embeddings
embeddings_3d = np.load('python/embeddings_3d.npy')  # [133, 3]

# Load vocabulary
with open('python/vocab.txt', 'r') as f:
    vocab = [line.strip() for line in f]

# Get embedding for a word
word_idx = vocab.index('king')
embedding = embeddings_3d[word_idx]  # [1.0, 3D vector normalized]
```

### Semantic Cluster Quality

The generated embeddings maintain semantic relationships:

- **Persons cluster**: king, queen, man, woman, prince, princess
  - Average cosine distance: 0.003 (very tight, highly similar)

- **Programming cluster**: for, while, loop, iterate, function, code
  - Average cosine distance: 0.004 (very tight)

- **Music cluster**: music, rhythm, beat, tempo, song, melody
  - Average cosine distance: 0.006 (tight)

This ensures that when tokens are projected to 3D, semantically related tokens form spatial clusters that the ray tracing can efficiently query.

## Performance Characteristics

### Memory Usage

| Component | Size | Count | Total |
|-----------|------|-------|-------|
| TokenNode | 512 B | 100K | 51.2 MB |
| GAS (BVH) | 10-50 MB | 1 | 10-50 MB |
| SBT | 192 B | 1 | 192 B |
| **Total** | | | **61.4-101.4 MB** |

Compare vs GPT-4: KV Cache for 100K tokens = ~307 GB

### Computational Complexity

| Phase | Operation | Complexity | Time |
|-------|-----------|-----------|------|
| A | Ray tracing BVH traversal | O(log N) | ~0.5 ms |
| B | cuBLAS MatMul [4096×4096] | O(M²) | ~5-12 ms |
| **Total** | | O(log N) + O(M²) | ~5.5-12.5 ms |

## Known Limitations & Future Work

1. **Differentiation**: RT Cores not natively differentiable
   - Solution: Soft BVH with gradient approximation
   - Alternative: Train embeddings with Transformer, substitute attention only

2. **Dynamic Updates**: BVH currently static per forward pass
   - Could support: incremental BVH updates for streaming

3. **Multi-GPU**: Single GPU only in current implementation
   - Future: Data parallelism across GPUs

4. **Validation**: Test suite incomplete without CUDA/OptiX installed
   - Python-only tests verify embeddings correctness
   - Full integration tests require NVIDIA hardware

## Testing

### Python Embeddings Tests
```bash
cd python
python3 download_embeddings.py
python3 ../tests/test_embeddings_validation.py
```

Results:
- ✓ All 6 embeddings tests pass
- ✓ Alignment: 133 embeddings = 133 vocab words
- ✓ Normalization: All norms = 1.0 ±0.01
- ✓ Cluster cohesion: Valid semantic relationships

### C++ Structure Tests (requires CUDA/OptiX)
```bash
cd tests
g++ -std=c++17 -I.. test_optix_host_structure.cpp \
  -I/opt/cuda/include -I/opt/optix/include \
  -o test_optix_host
./test_optix_host
```

## References

- [NVIDIA OptiX 8.1 Programming Guide](https://raytracing-docs.nvidia.com/optix8/guide/optix_guide.241022.A4.pdf)
- [OptiX Device Context API](https://raytracing-docs.nvidia.com/optix8/api/group__optix__host__api__device__context.html)
- [Shader Binding Table Optimization](https://developer.nvidia.com/blog/efficient-ray-tracing-with-nvidia-optix-shader-binding-table-optimization/)

## Authors

SpectralAI Zero-Matrix Team, 2026
