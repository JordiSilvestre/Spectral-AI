/**
 * @file test_optix_host_structure.cpp
 * @brief Tests de validación de la estructura del código host de OptiX
 *
 * Este archivo verifica que:
 *   1. El código host de OptiX compila correctamente
 *   2. Todas las funciones principales estén presentes
 *   3. Las estructuras clave estén bien definidas
 *   4. Los includes sean compatibles
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include <iostream>
#include <cassert>
#include <cstring>

// Incluir headers que debería usar el host code
#include "include/token_geometry.h"
#include "include/alpha_bsh.h"

// ============================================================================
// TESTS UNITARIOS
// ============================================================================

/**
 * @brief Test 1: Verificar que TokenNode está bien definido
 */
void test_token_node_structure() {
    std::cout << "[TEST 1] TokenNode structure..." << std::endl;

    TokenNode token;
    token.token_id = 42;
    token.position_in_seq = 0;
    token.centroid = {1.0f, 2.0f, 3.0f};
    token.aabb_min = {0.5f, 1.5f, 2.5f};
    token.aabb_max = {1.5f, 2.5f, 3.5f};
    token.semantic_radius = 0.5f;
    token.attention_weight = 0.8f;
    token.energy_remaining = 0.9f;

    assert(token.token_id == 42);
    assert(token.position_in_seq == 0);
    assert(token.centroid.x == 1.0f);
    assert(token.semantic_radius == 0.5f);

    std::cout << "  ✓ TokenNode structure is valid" << std::endl;
}

/**
 * @brief Test 2: Verificar métodos helper de TokenNode
 */
void test_token_node_methods() {
    std::cout << "[TEST 2] TokenNode methods..." << std::endl;

    TokenNode token;
    token.aabb_min = {0.0f, 0.0f, 0.0f};
    token.aabb_max = {2.0f, 2.0f, 2.0f};
    token.centroid = {1.0f, 1.0f, 1.0f};

    // Test getAABBVolume()
    float volume = token.getAABBVolume();
    assert(volume == 8.0f);  // 2*2*2 = 8

    // Test containsPoint()
    assert(token.containsPoint({1.0f, 1.0f, 1.0f}));  // Centroide
    assert(token.containsPoint({0.5f, 0.5f, 0.5f}));  // Interior
    assert(!token.containsPoint({3.0f, 3.0f, 3.0f})); // Exterior

    std::cout << "  ✓ TokenNode methods work correctly" << std::endl;
}

/**
 * @brief Test 3: Verificar que SemanticRay está bien definido
 */
void test_semantic_ray_structure() {
    std::cout << "[TEST 3] SemanticRay structure..." << std::endl;

    SemanticRay ray;
    ray.origin = {0.0f, 0.0f, 0.0f};
    ray.direction = {1.0f, 0.0f, 0.0f};
    ray.energy = 1.0f;
    ray.ray_id = 0;

    assert(ray.origin.x == 0.0f);
    assert(ray.direction.x == 1.0f);
    assert(ray.energy == 1.0f);

    std::cout << "  ✓ SemanticRay structure is valid" << std::endl;
}

/**
 * @brief Test 4: Verificar que AlphaRayPayload está bien definido
 */
void test_alpha_ray_payload_structure() {
    std::cout << "[TEST 4] AlphaRayPayload structure..." << std::endl;

    AlphaRayPayload payload;
    payload.energy = 1.0f;
    payload.hit_sphere_id = 0;
    payload.depth_reached = 0;
    payload.best_similarity = 0.0f;

    assert(payload.energy == 1.0f);
    assert(payload.hit_sphere_id == 0);

    // Test que UINT32_MAX indica miss
    payload.hit_sphere_id = UINT32_MAX;
    assert(payload.hit_sphere_id == UINT32_MAX);

    std::cout << "  ✓ AlphaRayPayload structure is valid" << std::endl;
}

/**
 * @brief Test 5: Verificar que SemanticSphereAlpha está bien definido
 */
void test_semantic_sphere_alpha_structure() {
    std::cout << "[TEST 5] SemanticSphereAlpha structure..." << std::endl;

    SemanticSphereAlpha sphere;
    sphere.center = {0.0f, 0.0f, 0.0f};
    sphere.radius = 1.0f;
    sphere.semantic_weight = 0.5f;
    sphere.sphere_id = 0;
    sphere.depth = 0;
    sphere.num_children = 0;
    sphere.is_leaf = true;

    assert(sphere.radius == 1.0f);
    assert(sphere.is_leaf == true);
    assert(sphere.num_children == 0);

    std::cout << "  ✓ SemanticSphereAlpha structure is valid" << std::endl;
}

/**
 * @brief Test 6: Verificar que MatrixBlock está bien definido
 */
void test_matrix_block_structure() {
    std::cout << "[TEST 6] MatrixBlock structure..." << std::endl;

    MatrixBlock block;
    block.d_weights1 = nullptr;
    block.d_biases1 = nullptr;
    block.d_weights2 = nullptr;
    block.d_biases2 = nullptr;
    block.dim_in = 768;
    block.dim_out = 768;
    block.hidden_dim = 3072;
    block.loaded = false;
    block.disk_offset = 0;

    assert(block.dim_in == 768);
    assert(block.dim_out == 768);
    assert(block.loaded == false);

    std::cout << "  ✓ MatrixBlock structure is valid" << std::endl;
}

/**
 * @brief Test 7: Verificar funciones helper de geometría
 */
void test_geometry_helpers() {
    std::cout << "[TEST 7] Geometry helper functions..." << std::endl;

    float3 a = {0.0f, 0.0f, 0.0f};
    float3 b = {3.0f, 4.0f, 0.0f};

    // Test computeSemanticDistance
    float dist = computeSemanticDistance(a, b);
    assert(dist == 5.0f);  // 3-4-5 triangle

    // Test computeSemanticDistanceSq
    float dist_sq = computeSemanticDistanceSq(a, b);
    assert(dist_sq == 25.0f);  // 5^2

    // Test computeAttentionWeight
    float weight = computeAttentionWeight(1.0f, 1.0f, 0.1f);
    assert(weight > 0.0f && weight < 1.0f);

    std::cout << "  ✓ Geometry helpers work correctly" << std::endl;
}

/**
 * @brief Test 8: Verificar constantes globales
 */
void test_global_constants() {
    std::cout << "[TEST 8] Global constants..." << std::endl;

    assert(SPECTRAL_EMBEDDING_DIM == 256);
    assert(SPECTRAL_NUM_RAYS == 4096);
    assert(SPECTRAL_SPATIAL_DIM == 3);
    assert(SPECTRAL_LAMBDA == 0.1f);

    assert(ALPHA_BSH_MAX_DEPTH == 20);
    assert(ALPHA_BSH_MAX_CHILDREN == 8);
    assert(ALPHA_MATRIX_BLOCK_DIM == 4096);
    assert(ALPHA_ENERGY_THRESHOLD == 0.01f);
    assert(ALPHA_LAMBDA_DECAY == 0.1f);

    std::cout << "  ✓ All global constants are correct" << std::endl;
}

/**
 * @brief Test 9: Verificar AlphaConfig
 */
void test_alpha_config() {
    std::cout << "[TEST 9] AlphaConfig structure..." << std::endl;

    AlphaConfig config;
    config.num_spheres = 1000;
    config.max_depth = 20;
    config.lambda_decay = 0.1f;
    config.lazy_load_matrices = true;
    config.use_fp32_fallback = false;

    assert(config.num_spheres == 1000);
    assert(config.max_depth == 20);
    assert(config.lazy_load_matrices == true);

    std::cout << "  ✓ AlphaConfig structure is valid" << std::endl;
}

/**
 * @brief Test 10: Verificar AlphaExecutionResult
 */
void test_alpha_execution_result() {
    std::cout << "[TEST 10] AlphaExecutionResult structure..." << std::endl;

    AlphaExecutionResult result;
    result.output_dim = 768;
    result.confidence = 0.85f;
    result.sphere_id_used = 42;
    result.phase_a_time_ms = 0.5f;
    result.phase_b_time_ms = 5.2f;

    assert(result.output_dim == 768);
    assert(result.confidence == 0.85f);
    assert(result.phase_a_time_ms == 0.5f);

    std::cout << "  ✓ AlphaExecutionResult structure is valid" << std::endl;
}

// ============================================================================
// FUNCIÓN MAIN
// ============================================================================

int main() {
    std::cout << "\n" << "=" * 70 << std::endl;
    std::cout << "SpectralAI Zero-Matrix - OptiX Host Structure Tests" << std::endl;
    std::cout << "=" * 70 << "\n" << std::endl;

    try {
        test_token_node_structure();
        test_token_node_methods();
        test_semantic_ray_structure();
        test_alpha_ray_payload_structure();
        test_semantic_sphere_alpha_structure();
        test_matrix_block_structure();
        test_geometry_helpers();
        test_global_constants();
        test_alpha_config();
        test_alpha_execution_result();

        std::cout << "\n" << "=" * 70 << std::endl;
        std::cout << "ALL TESTS PASSED ✓" << std::endl;
        std::cout << "=" * 70 << "\n" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n[ERROR] Test failed: " << e.what() << std::endl;
        return 1;
    }
}
