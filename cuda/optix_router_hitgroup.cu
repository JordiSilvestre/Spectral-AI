/**
 * optix_router_hitgroup.cu — OptiX RT Core Router: Closest Hit + Miss Programs
 *
 * Closest Hit: When a ray intersects an expert AABB, record the expert ID
 * and distance. The expert_id is stored as the primitive index in the GAS.
 *
 * Miss: When a ray misses all experts, return sentinel values.
 *
 * Entry points:
 *   __intersection__rt_router   (AABB custom primitive intersection)
 *   __closesthit__rt_router
 *   __miss__rt_router
 *
 * Copyright (c) 2026 SpectralAI Studio — Apache 2.0
 */

#include <optix.h>

// ============================================================================
// Intersection Program (REQUIRED for CUSTOM_PRIMITIVES / AABBs)
// ============================================================================
//
// When OptiX traverses the BVH and finds a ray overlapping an AABB,
// it calls this intersection program. We must call optixReportIntersection()
// to confirm the hit; otherwise the primitive is skipped (all rays miss).
//
// For expert routing we treat the AABB itself as the geometry — any ray
// entering the AABB counts as "hitting" that expert.

extern "C" __global__ void __intersection__rt_router() {
    // Accept hit at the AABB entry point.
    // For custom primitives the ray is already clipped to the AABB,
    // so reportintersection at the current tmin is the AABB surface.
    // hitKind = 0 (custom, not triangle).
    const float t_hit = optixGetRayTmin();
    optixReportIntersection(t_hit, 0);
}

// ============================================================================
// Closest Hit Program
// ============================================================================

extern "C" __global__ void __closesthit__rt_router() {
    // For AABBs: primitiveIndex = expert_id directly (1 AABB per expert)
    // For triangles: primitiveIndex / 8 = expert_id (8 triangles per octahedron)
    // The TRIS_PER_EXPERT constant must match buildGAS_triangles()
    constexpr uint32_t TRIS_PER_EXPERT = 8;
    const uint32_t prim_idx = optixGetPrimitiveIndex();
    const uint32_t expert_id = (optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_FRONT_FACE ||
                                 optixGetHitKind() == OPTIX_HIT_KIND_TRIANGLE_BACK_FACE)
                                ? prim_idx / TRIS_PER_EXPERT  // Triangle GAS
                                : prim_idx;                    // AABB GAS
    const float t_hit = optixGetRayTmax();

    // Store in payload registers
    // p0 = expert_id, p1 = distance
    optixSetPayload_0(expert_id);
    optixSetPayload_1(__float_as_uint(t_hit));
}

// ============================================================================
// Miss Program
// ============================================================================

extern "C" __global__ void __miss__rt_router() {
    // No expert hit — return sentinel
    // p0 = 0xFFFFFFFF (invalid), p1 = +inf
    optixSetPayload_0(0xFFFFFFFFu);
    optixSetPayload_1(__float_as_uint(1e30f));
}
