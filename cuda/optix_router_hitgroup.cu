/**
 * optix_router_hitgroup.cu — OptiX RT Core Router: Closest Hit + Miss Programs
 *
 * Closest Hit: When a ray intersects an expert AABB, record the expert ID
 * and distance. The expert_id is stored as the primitive index in the GAS.
 *
 * Miss: When a ray misses all experts, return sentinel values.
 *
 * Entry points:
 *   __closesthit__rt_router
 *   __miss__rt_router
 *
 * Copyright (c) 2026 SpectralAI Studio — Apache 2.0
 */

#include <optix.h>

// ============================================================================
// Closest Hit Program
// ============================================================================

extern "C" __global__ void __closesthit__rt_router() {
    // The primitive index in the GAS = expert_id
    // This is set during GAS build: each AABB corresponds to one expert
    const uint32_t expert_id = optixGetPrimitiveIndex();
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
