/**
 * @file rt_router_params.h
 * @brief Shared launch parameters for OptiX RT Core Router
 *
 * This struct is shared between:
 *   - Device code: optix_router_raygen.cu (read via extern "C" __constant__)
 *   - Host code: optix_router_host.cpp (populated and uploaded to GPU)
 *
 * IMPORTANT: Any changes here must be reflected in BOTH files.
 * The struct layout must match exactly (same types, same order, same padding).
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#pragma once
#ifndef SPECTRAL_RT_ROUTER_PARAMS_H_
#define SPECTRAL_RT_ROUTER_PARAMS_H_

#include <optix.h>
#include <vector_types.h>
#include <cstdint>

/**
 * @struct RTRouterParams
 * @brief Launch parameters for the OptiX RT Core Router pipeline.
 *
 * Uploaded to device constant memory before each optixLaunch().
 * The raygen program reads these to know where inputs/outputs are.
 */
struct RTRouterParams {
    /// Acceleration structure handle (GAS with expert AABBs)
    OptixTraversableHandle gas_handle;

    /// Input: query positions in 3D semantic space [batch_size x 3]
    const float3* query_positions;

    /// Input: query directions (embedding-derived) [batch_size x 3]
    const float3* query_directions;

    /// Output: selected expert ID per query [batch_size]
    uint32_t* expert_ids;

    /// Output: distance to closest expert [batch_size]
    float* expert_distances;

    /// Output: top-K expert IDs [batch_size x top_k] (nullptr if top_k=1)
    uint32_t* topk_expert_ids;

    /// Output: top-K distances [batch_size x top_k] (nullptr if top_k=1)
    float* topk_distances;

    /// Number of queries in the batch
    uint32_t batch_size;

    /// Number of experts per query (1=single, 8=top-8 for OLMoE)
    uint32_t top_k;

    /// Total number of experts in the scene (typically 64)
    uint32_t num_experts;

    /// Minimum ray distance (avoid self-intersection)
    float ray_tmin;

    /// Maximum ray distance
    float ray_tmax;
};

#endif // SPECTRAL_RT_ROUTER_PARAMS_H_
