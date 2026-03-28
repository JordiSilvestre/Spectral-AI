/**
 * @file alpha_bsh.cpp
 * @brief Implementación de la clase AlphaBSH
 *
 * Contiene la lógica host-side para orquestar la arquitectura Alpha BSH:
 *   - Construcción del árbol BSH
 *   - Gestión de memoria GPU
 *   - Lanzamiento de kernels CUDA/OptiX
 *   - Profiling y estadísticas
 *   - Carga lazy de MatrixBlocks
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#include "../include/alpha_bsh.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <string>
#include <algorithm>
#include <cmath>

// Declaración forward de kernels CUDA definidos en .cu
extern AlphaRayPayload launch_alpha_phase_a_kernel(
    const SemanticSphereAlpha* d_spheres,
    uint32_t num_spheres,
    float3 query_point,
    float3 ray_direction,
    float ray_energy,
    float lambda_decay);

extern AlphaExecutionResult launch_alpha_phase_b(
    SemanticSphereAlpha* d_spheres,
    uint32_t sphere_id,
    const float* input_activations,
    uint32_t batch_size,
    const AlphaConfig& config);

extern float3 projectEmbeddingTo3D(
    const float* embedding,
    uint32_t embedding_dim);

extern float3 normalizeFloat3(float3 v);

// ============================================================================
// CONSTRUCTOR
// ============================================================================

AlphaBSH::AlphaBSH()
    : d_spheres_(nullptr),
      num_spheres_(0),
      d_payload_result_(nullptr),
      cublas_handle_(nullptr) {

    // Inicializar estadísticas
    stats_.num_executions = 0;
    stats_.avg_phase_a_time_ms = 0.0f;
    stats_.avg_phase_b_time_ms = 0.0f;
    stats_.num_hits = 0;
    stats_.total_depth_reached = 0;

    // Crear handle de cuBLAS
    cublasStatus_t status = cublasCreate(&cublas_handle_);
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "[ERROR] AlphaBSH::ctor - Failed to create cuBLAS handle\n");
        cublas_handle_ = nullptr;
    } else {
        // Configurar para usar Tensor Cores explícitamente
        cublasSetMathMode(cublas_handle_, CUBLAS_TENSOR_OP_MATH);
    }

    printf("[INFO] AlphaBSH constructor completed. cuBLAS handle: %p\n", cublas_handle_);
}

// ============================================================================
// DESTRUCTOR
// ============================================================================

AlphaBSH::~AlphaBSH() {
    printf("[INFO] AlphaBSH destructor: Liberando memoria GPU...\n");

    // Liberar array de esferas
    if (d_spheres_ != nullptr) {
        cudaError_t err = cudaFree(d_spheres_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[WARNING] cudaFree(d_spheres_) failed: %s\n",
                    cudaGetErrorString(err));
        }
        d_spheres_ = nullptr;
    }

    // Liberar payload result
    if (d_payload_result_ != nullptr) {
        cudaError_t err = cudaFree(d_payload_result_);
        if (err != cudaSuccess) {
            fprintf(stderr, "[WARNING] cudaFree(d_payload_result_) failed: %s\n",
                    cudaGetErrorString(err));
        }
        d_payload_result_ = nullptr;
    }

    // Destruir handle de cuBLAS (no libera; solo destruye el handle)
    if (cublas_handle_ != nullptr) {
        cublasStatus_t status = cublasDestroy(cublas_handle_);
        if (status != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "[WARNING] cublasDestroy failed: %d\n", status);
        }
        cublas_handle_ = nullptr;
    }

    printf("[INFO] AlphaBSH destructor completed.\n");
}

// ============================================================================
// MÉTODO: build() - Construir el árbol BSH
// ============================================================================

bool AlphaBSH::build(
    const SemanticSphereAlpha* h_spheres,
    uint32_t num_spheres,
    const AlphaConfig& config) {

    printf("[INFO] AlphaBSH::build - Starting BSH construction with %u spheres\n",
           num_spheres);

    if (h_spheres == nullptr || num_spheres == 0) {
        fprintf(stderr, "[ERROR] AlphaBSH::build - Invalid input: h_spheres=%p, num_spheres=%u\n",
                h_spheres, num_spheres);
        return false;
    }

    num_spheres_ = num_spheres;

    // ====================================================================
    // 1. COPIAR ESFERAS A GPU
    // ====================================================================

    size_t spheres_size = num_spheres * sizeof(SemanticSphereAlpha);

    // Hacer una copia host para modificar (asignar relaciones)
    SemanticSphereAlpha* h_spheres_copy = new SemanticSphereAlpha[num_spheres];
    std::copy(h_spheres, h_spheres + num_spheres, h_spheres_copy);

    // ====================================================================
    // 2. ASIGNAR RELACIONES PADRE-HIJO
    // ====================================================================

    if (!assignParentChildRelationships(h_spheres_copy, num_spheres)) {
        fprintf(stderr, "[ERROR] AlphaBSH::build - Failed to assign parent-child relationships\n");
        delete[] h_spheres_copy;
        return false;
    }

    printf("[INFO] AlphaBSH::build - Parent-child relationships assigned\n");

    // ====================================================================
    // 3. ALOCAR MEMORIA GPU Y COPIAR
    // ====================================================================

    cudaError_t err = cudaMalloc(&d_spheres_, spheres_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] AlphaBSH::build - cudaMalloc failed: %s\n",
                cudaGetErrorString(err));
        delete[] h_spheres_copy;
        return false;
    }

    err = cudaMemcpy(d_spheres_, h_spheres_copy, spheres_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "[ERROR] AlphaBSH::build - cudaMemcpy failed: %s\n",
                cudaGetErrorString(err));
        cudaFree(d_spheres_);
        d_spheres_ = nullptr;
        delete[] h_spheres_copy;
        return false;
    }

    printf("[INFO] AlphaBSH::build - Spheres copied to GPU (%zu bytes)\n", spheres_size);

    // ====================================================================
    // 4. VALIDAR ESTRUCTURA DEL ÁRBOL
    // ====================================================================

    if (!validateTreeStructure()) {
        fprintf(stderr, "[ERROR] AlphaBSH::build - Tree validation failed\n");
        cudaFree(d_spheres_);
        d_spheres_ = nullptr;
        delete[] h_spheres_copy;
        return false;
    }

    printf("[INFO] AlphaBSH::build - Tree structure validated\n");

    // ====================================================================
    // 5. CARGAR MATRIX BLOCKS (si config permite)
    // ====================================================================

    if (!config.lazy_load_matrices) {
        printf("[INFO] AlphaBSH::build - Pre-loading matrix blocks (lazy_load disabled)\n");

        for (uint32_t i = 0; i < num_spheres; ++i) {
            if (h_spheres_copy[i].is_leaf) {
                if (!loadMatrixBlock(i)) {
                    fprintf(stderr, "[WARNING] Failed to load matrix block for sphere %u\n", i);
                    // Continue anyway; will attempt lazy load at execution
                }
            }
        }
    } else {
        printf("[INFO] AlphaBSH::build - Lazy loading enabled (matrices load on demand)\n");
    }

    // ====================================================================
    // 6. LIMPIAR Y RETORNAR
    // ====================================================================

    delete[] h_spheres_copy;

    printf("[INFO] AlphaBSH::build - BSH construction completed successfully\n");
    printf("       Spheres: %u, GPU Memory: %.2f MB\n",
           num_spheres, spheres_size / 1024.0f / 1024.0f);

    return true;
}

// ============================================================================
// MÉTODO PRIVADO: assignParentChildRelationships()
// ============================================================================

bool AlphaBSH::assignParentChildRelationships(
    SemanticSphereAlpha* h_spheres,
    uint32_t num_spheres) {

    if (num_spheres == 0 || h_spheres == nullptr) {
        return false;
    }

    printf("[INFO] Assigning parent-child relationships (greedy nearest-neighbor)\n");

    // Estrategia: para cada esfera, encontrar los ALPHA_BSH_MAX_CHILDREN más cercanos
    // como hijos, y designar la esfera más cercana como padre.

    // La esfera 0 es la raíz (sin padre)
    h_spheres[0].parent_id = 0;
    h_spheres[0].depth = 0;

    // Asignar el resto de esferas
    for (uint32_t i = 1; i < num_spheres; ++i) {
        // Encontrar la esfera más cercana (será el padre potencial)
        uint32_t nearest_parent = 0;
        float min_distance = FLT_MAX;

        for (uint32_t j = 0; j < num_spheres; ++j) {
            if (i == j) continue;

            float3 diff = h_spheres[i].center - h_spheres[j].center;
            float dist = sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);

            if (dist < min_distance) {
                min_distance = dist;
                nearest_parent = j;
            }
        }

        h_spheres[i].parent_id = nearest_parent;
        h_spheres[i].depth = h_spheres[nearest_parent].depth + 1;

        // Limitar profundidad
        if (h_spheres[i].depth > ALPHA_BSH_MAX_DEPTH) {
            h_spheres[i].depth = ALPHA_BSH_MAX_DEPTH;
        }

        // Agregar i como hijo de nearest_parent (si hay espacio)
        if (h_spheres[nearest_parent].num_children < ALPHA_BSH_MAX_CHILDREN) {
            h_spheres[nearest_parent].children_ids[h_spheres[nearest_parent].num_children] = i;
            h_spheres[nearest_parent].num_children++;
        }
    }

    // Marcar hojas (esferas sin hijos o en profundidad máxima)
    for (uint32_t i = 0; i < num_spheres; ++i) {
        if (h_spheres[i].num_children == 0 || h_spheres[i].depth >= ALPHA_BSH_MAX_DEPTH) {
            h_spheres[i].is_leaf = true;
        } else {
            h_spheres[i].is_leaf = false;
        }
    }

    printf("[INFO] Parent-child assignment completed\n");
    return true;
}

// ============================================================================
// MÉTODO PRIVADO: validateTreeStructure()
// ============================================================================

bool AlphaBSH::validateTreeStructure() const {
    if (d_spheres_ == nullptr || num_spheres_ == 0) {
        return false;
    }

    printf("[INFO] Validating BSH tree structure...\n");

    // Copiar esferas a host para validación
    SemanticSphereAlpha* h_spheres = new SemanticSphereAlpha[num_spheres_];
    cudaMemcpy(h_spheres, d_spheres_, num_spheres_ * sizeof(SemanticSphereAlpha),
               cudaMemcpyDeviceToHost);

    bool valid = true;
    uint32_t max_depth_found = 0;

    for (uint32_t i = 0; i < num_spheres_; ++i) {
        const SemanticSphereAlpha& sphere = h_spheres[i];

        // Check: profundidad no excede máximo
        if (sphere.depth > ALPHA_BSH_MAX_DEPTH) {
            fprintf(stderr, "[WARNING] Sphere %u has depth %u > ALPHA_BSH_MAX_DEPTH\n",
                    i, sphere.depth);
            valid = false;
        }
        max_depth_found = std::max(max_depth_found, sphere.depth);

        // Check: número de hijos válido
        if (sphere.num_children > ALPHA_BSH_MAX_CHILDREN) {
            fprintf(stderr, "[WARNING] Sphere %u has %u children > ALPHA_BSH_MAX_CHILDREN\n",
                    i, sphere.num_children);
            valid = false;
        }

        // Check: solo hojas tienen MatrixBlock
        if (!sphere.is_leaf && sphere.matrix_block.d_weights1 != nullptr) {
            fprintf(stderr, "[WARNING] Non-leaf sphere %u has MatrixBlock assigned\n", i);
            // Not necessarily invalid, just unusual
        }

        // Check: IDs de hijos son válidos
        for (uint32_t j = 0; j < sphere.num_children; ++j) {
            uint32_t child_id = sphere.children_ids[j];
            if (child_id >= num_spheres_) {
                fprintf(stderr, "[WARNING] Sphere %u has invalid child ID %u\n", i, child_id);
                valid = false;
            }
        }
    }

    delete[] h_spheres;

    if (valid) {
        printf("[INFO] Tree validation passed. Max depth: %u\n", max_depth_found);
    } else {
        printf("[WARNING] Tree validation found issues (see above)\n");
    }

    return valid;
}

// ============================================================================
// MÉTODO: launchPhaseA()
// ============================================================================

AlphaRayPayload AlphaBSH::launchPhaseA(
    const float* query_embedding,
    uint32_t query_dim,
    const AlphaConfig& config) {

    printf("[INFO] AlphaBSH::launchPhaseA - Starting Phase A (BSH traversal)\n");

    // Proyectar embedding a espacio 3D
    float3 query_point = projectEmbeddingTo3D(query_embedding, query_dim);
    query_point = normalizeFloat3(query_point);

    // Dirección del rayo: normalizar el embedding mismo
    float3 ray_direction = query_point;  // Simplificado: apunta hacia la dirección del embedding

    printf("[DEBUG] Query point: (%.4f, %.4f, %.4f)\n", query_point.x, query_point.y, query_point.z);
    printf("[DEBUG] Ray direction: (%.4f, %.4f, %.4f)\n", ray_direction.x, ray_direction.y, ray_direction.z);

    // Lanzar kernel
    AlphaRayPayload payload = launch_alpha_phase_a_kernel(
        d_spheres_,
        num_spheres_,
        query_point,
        ray_direction,
        1.0f,  // Energía inicial
        config.lambda_decay);

    printf("[INFO] Phase A completed. Hit sphere: %u, Energy: %.4f, Best similarity: %.4f\n",
           payload.hit_sphere_id, payload.energy, payload.best_similarity);

    return payload;
}

// ============================================================================
// MÉTODO: launchPhaseB()
// ============================================================================

AlphaExecutionResult AlphaBSH::launchPhaseB(
    uint32_t sphere_id,
    const float* input_activations,
    uint32_t input_dim,
    const AlphaConfig& config) {

    printf("[INFO] AlphaBSH::launchPhaseB - Starting Phase B (MatMul in sphere %u)\n",
           sphere_id);

    AlphaExecutionResult result = launch_alpha_phase_b(
        d_spheres_,
        sphere_id,
        input_activations,
        1,  // batch_size = 1 para simplificar
        config);

    printf("[INFO] Phase B completed. Output dim: %u, Confidence: %.4f, Time: %.2f ms\n",
           result.output_dim, result.confidence, result.phase_b_time_ms);

    return result;
}

// ============================================================================
// MÉTODO: execute() - Pipeline completo
// ============================================================================

AlphaExecutionResult AlphaBSH::execute(
    const float* query_embedding,
    uint32_t query_dim,
    const float* input_activations,
    uint32_t input_dim,
    const AlphaConfig& config) {

    printf("\n[INFO] ========================================\n");
    printf("[INFO] AlphaBSH::execute - Full pipeline execution\n");
    printf("[INFO] ========================================\n");

    AlphaExecutionResult final_result;
    final_result.sphere_id_used = UINT32_MAX;
    final_result.output_activations = nullptr;
    final_result.output_dim = 0;
    final_result.confidence = 0.0f;

    // ====================================================================
    // FASE A
    // ====================================================================

    cudaEvent_t start_total, end_total;
    cudaEventCreate(&start_total);
    cudaEventCreate(&end_total);
    cudaEventRecord(start_total);

    AlphaRayPayload phase_a_result = launchPhaseA(query_embedding, query_dim, config);

    cudaEvent_t start_phase_b, end_phase_b;
    cudaEventCreate(&start_phase_b);

    if (phase_a_result.hit_sphere_id != UINT32_MAX) {
        // ====================================================================
        // FASE B
        // ====================================================================

        cudaEventRecord(start_phase_b);

        AlphaExecutionResult phase_b_result = launchPhaseB(
            phase_a_result.hit_sphere_id,
            input_activations,
            input_dim,
            config);

        cudaEventRecord(end_phase_b);

        final_result = phase_b_result;
        final_result.phase_a_time_ms = 0.5f;  // Stub
    } else {
        printf("[INFO] Phase A miss: No sphere found\n");
        final_result.phase_a_time_ms = 0.5f;
        final_result.phase_b_time_ms = 0.0f;
    }

    cudaEventRecord(end_total);
    cudaEventSynchronize(end_total);

    float total_ms = 0.0f;
    cudaEventElapsedTime(&total_ms, start_total, end_total);

    // ====================================================================
    // ACTUALIZAR ESTADÍSTICAS
    // ====================================================================

    stats_.num_executions++;
    stats_.avg_phase_a_time_ms = (stats_.avg_phase_a_time_ms * (stats_.num_executions - 1) +
                                   final_result.phase_a_time_ms) / stats_.num_executions;
    stats_.avg_phase_b_time_ms = (stats_.avg_phase_b_time_ms * (stats_.num_executions - 1) +
                                   final_result.phase_b_time_ms) / stats_.num_executions;
    if (phase_a_result.hit_sphere_id != UINT32_MAX) {
        stats_.num_hits++;
    }
    stats_.total_depth_reached += phase_a_result.depth_reached;

    // ====================================================================
    // SUMMARY
    // ====================================================================

    printf("[INFO] ========================================\n");
    printf("[INFO] EXECUTION SUMMARY\n");
    printf("[INFO] ========================================\n");
    printf("[INFO] Sphere found: %u\n", final_result.sphere_id_used);
    printf("[INFO] Confidence: %.4f\n", final_result.confidence);
    printf("[INFO] Phase A time: %.2f ms\n", final_result.phase_a_time_ms);
    printf("[INFO] Phase B time: %.2f ms\n", final_result.phase_b_time_ms);
    printf("[INFO] Total time: %.2f ms\n", total_ms);
    printf("[INFO] Output dimension: %u\n", final_result.output_dim);
    printf("[INFO] ========================================\n\n");

    cudaEventDestroy(start_total);
    cudaEventDestroy(end_total);
    cudaEventDestroy(start_phase_b);
    cudaEventDestroy(end_phase_b);

    return final_result;
}

// ============================================================================
// MÉTODO: loadMatrixBlock()
// ============================================================================

bool AlphaBSH::loadMatrixBlock(uint32_t sphere_id) {
    printf("[INFO] AlphaBSH::loadMatrixBlock - Loading matrix block for sphere %u\n",
           sphere_id);

    if (sphere_id >= num_spheres_) {
        fprintf(stderr, "[ERROR] Invalid sphere_id: %u >= %u\n", sphere_id, num_spheres_);
        return false;
    }

    // Obtener esfera del device
    SemanticSphereAlpha h_sphere;
    cudaMemcpy(&h_sphere, &d_spheres_[sphere_id], sizeof(SemanticSphereAlpha),
               cudaMemcpyDeviceToHost);

    if (!h_sphere.is_leaf) {
        fprintf(stderr, "[ERROR] Sphere %u is not a leaf (no MatrixBlock)\n", sphere_id);
        return false;
    }

    // En prototipo: simular carga desde disco
    // En producción: abrir archivo en h_sphere.matrix_block.disk_offset
    // y copiar a GPU

    printf("[INFO] Matrix block loaded (simulated) for sphere %u\n", sphere_id);

    // Marcar como loaded
    h_sphere.matrix_block.loaded = true;
    cudaMemcpy(&d_spheres_[sphere_id], &h_sphere, sizeof(SemanticSphereAlpha),
               cudaMemcpyHostToDevice);

    return true;
}

// ============================================================================
// MÉTODO: getStats()
// ============================================================================

std::string AlphaBSH::getStats() const {
    char buffer[512];

    snprintf(buffer, sizeof(buffer),
        "===== ALPHA BSH STATISTICS =====\n"
        "Executions: %u\n"
        "Hits: %u (%.2f%% hit rate)\n"
        "Avg Phase A time: %.2f ms\n"
        "Avg Phase B time: %.2f ms\n"
        "Avg total time: %.2f ms\n"
        "Avg depth reached: %.2f\n"
        "Total spheres: %u\n"
        "=================================",
        stats_.num_executions,
        stats_.num_hits,
        (stats_.num_executions > 0) ? (100.0f * stats_.num_hits / stats_.num_executions) : 0.0f,
        stats_.avg_phase_a_time_ms,
        stats_.avg_phase_b_time_ms,
        stats_.avg_phase_a_time_ms + stats_.avg_phase_b_time_ms,
        (stats_.num_executions > 0) ? (float)stats_.total_depth_reached / stats_.num_executions : 0.0f,
        num_spheres_);

    return std::string(buffer);
}

// ============================================================================
// MÉTODO: resetStats()
// ============================================================================

void AlphaBSH::resetStats() {
    printf("[INFO] Resetting statistics\n");

    stats_.num_executions = 0;
    stats_.avg_phase_a_time_ms = 0.0f;
    stats_.avg_phase_b_time_ms = 0.0f;
    stats_.num_hits = 0;
    stats_.total_depth_reached = 0;
}

// ============================================================================
// FUNCIÓN LIBRE: printAlphaStats()
// ============================================================================

void printAlphaStats(const AlphaExecutionResult& result) {
    printf("===== ALPHA BSH EXECUTION RESULT =====\n");
    printf("Sphere Used: %u\n", result.sphere_id_used);
    printf("Confidence: %.4f\n", result.confidence);
    printf("Output Dimension: %u\n", result.output_dim);
    printf("Phase A Time: %.3f ms\n", result.phase_a_time_ms);
    printf("Phase B Time: %.3f ms\n", result.phase_b_time_ms);
    printf("Total Time: %.3f ms\n", result.phase_a_time_ms + result.phase_b_time_ms);
    printf("=======================================\n");
}
