/**
 * optix_router_raygen.cu — OptiX RT Core Router: Ray Generation Program
 *
 * Minimal OptiX shader for expert selection via hardware RT Core traversal.
 * Instead of software BVH traversal (bvh_router_kernel.cu, ~10μs),
 * this uses RT Cores for ray-AABB intersection (~0.5-1μs estimated).
 *
 * Architecture:
 *   - 64 experts = 64 AABBs in the scene (one per expert sphere)
 *   - Each query token emits 1 ray from its 3D position
 *   - RT Core finds closest AABB hit = expert_id
 *   - Optional: top-K via multi-ray fan pattern
 *
 * Entry point: __raygen__rt_router
 *
 * Copyright (c) 2026 SpectralAI Studio — Apache 2.0
 */

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>
#include <cstdint>

// ============================================================================
// Launch parameters — set by host before optixLaunch
// ============================================================================

struct RTRouterParams {
    // Acceleration structure handle (GAS with 64 expert AABBs)
    OptixTraversableHandle gas_handle;

    // Input: query positions in 3D semantic space [batch_size x 3]
    const float3* query_positions;

    // Input: query directions (embedding-derived) [batch_size x 3]
    const float3* query_directions;

    // Output: selected expert ID per query [batch_size]
    uint32_t* expert_ids;

    // Output: distance to closest expert [batch_size]
    float* expert_distances;

    // Output: top-K expert IDs [batch_size x top_k] (optional, nullptr if top_k=1)
    uint32_t* topk_expert_ids;

    // Output: top-K distances [batch_size x top_k] (optional)
    float* topk_distances;

    // Config
    uint32_t batch_size;
    uint32_t top_k;          // 1 for single expert, 8 for OLMoE-style top-8
    uint32_t num_experts;    // 64
    float    ray_tmin;       // Minimum ray distance (avoid self-intersection)
    float    ray_tmax;       // Maximum ray distance
};

extern "C" __constant__ RTRouterParams params;

// ============================================================================
// Payload: passed through optixTrace via registers
// ============================================================================
// p0 = hit_expert_id (uint32)
// p1 = hit_distance  (float as uint32)

// ============================================================================
// Ray Generation Program
// ============================================================================

extern "C" __global__ void __raygen__rt_router() {
    const uint32_t idx = optixGetLaunchIndex().x;
    if (idx >= params.batch_size) return;

    // Read query position and direction
    const float3 origin = params.query_positions[idx];
    const float3 direction = params.query_directions[idx];

    if (params.top_k <= 1) {
        // ── Single expert selection (fastest path) ──────────────
        // Cast one ray, RT Core returns closest hit
        uint32_t p0 = 0xFFFFFFFFu;  // MISS sentinel
        uint32_t p1 = __float_as_uint(1e30f);

        optixTrace(
            params.gas_handle,
            origin,
            direction,
            params.ray_tmin,       // tmin
            params.ray_tmax,       // tmax
            0.0f,                  // rayTime
            0xFFu,                 // visibilityMask (all visible)
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // Only closest hit
            0,                     // SBT offset
            1,                     // SBT stride
            0,                     // missSBTIndex
            p0, p1
        );

        params.expert_ids[idx] = p0;
        params.expert_distances[idx] = __uint_as_float(p1);

    } else {
        // ── Top-K expert selection via multi-ray fan ────────────
        // Emit top_k rays in a cone around the main direction
        // Each ray finds one expert; we collect all K hits

        // Build orthonormal basis around direction
        float3 u, v;
        {
            const float3 d = direction;
            // Pick axis least aligned with d for cross product
            float3 tmp;
            if (fabsf(d.x) < 0.9f)
                tmp = make_float3(1.0f, 0.0f, 0.0f);
            else
                tmp = make_float3(0.0f, 1.0f, 0.0f);

            // u = normalize(cross(d, tmp))
            u.x = d.y * tmp.z - d.z * tmp.y;
            u.y = d.z * tmp.x - d.x * tmp.z;
            u.z = d.x * tmp.y - d.y * tmp.x;
            float len = sqrtf(u.x * u.x + u.y * u.y + u.z * u.z);
            if (len > 1e-6f) { u.x /= len; u.y /= len; u.z /= len; }

            // v = cross(d, u)
            v.x = d.y * u.z - d.z * u.y;
            v.y = d.z * u.x - d.x * u.z;
            v.z = d.x * u.y - d.y * u.x;
        }

        const float cone_half_angle = 0.5f;  // radians (~28.6 degrees)
        const uint32_t k = params.top_k;

        for (uint32_t ray_i = 0; ray_i < k; ++ray_i) {
            // Distribute rays in a cone using golden angle spiral
            float t = static_cast<float>(ray_i) / static_cast<float>(k);
            float phi = 2.0f * 3.14159265f * t * 1.6180339887f;  // golden ratio
            float r = cone_half_angle * sqrtf(t);

            float cos_phi = cosf(phi);
            float sin_phi = sinf(phi);

            // Perturbed direction
            float3 d;
            d.x = direction.x + r * (cos_phi * u.x + sin_phi * v.x);
            d.y = direction.y + r * (cos_phi * u.y + sin_phi * v.y);
            d.z = direction.z + r * (cos_phi * u.z + sin_phi * v.z);

            // Normalize
            float len = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
            if (len > 1e-6f) { d.x /= len; d.y /= len; d.z /= len; }

            uint32_t p0 = 0xFFFFFFFFu;
            uint32_t p1 = __float_as_uint(1e30f);

            optixTrace(
                params.gas_handle,
                origin,
                d,
                params.ray_tmin,
                params.ray_tmax,
                0.0f,
                0xFFu,
                OPTIX_RAY_FLAG_DISABLE_ANYHIT,
                0, 1, 0,
                p0, p1
            );

            if (params.topk_expert_ids) {
                params.topk_expert_ids[idx * k + ray_i] = p0;
            }
            if (params.topk_distances) {
                params.topk_distances[idx * k + ray_i] = __uint_as_float(p1);
            }
        }

        // Primary expert = first ray hit (closest to main direction)
        if (params.topk_expert_ids) {
            params.expert_ids[idx] = params.topk_expert_ids[idx * k];
            params.expert_distances[idx] = params.topk_distances
                ? params.topk_distances[idx * k]
                : 0.0f;
        }
    }
}
