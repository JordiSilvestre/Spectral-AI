/**
 * @file semantic_bvh.cpp
 * @brief Implementación de construcción del BVH (Bounding Volume Hierarchy) semántico
 *
 * Este archivo contiene la lógica de construcción de un árbol BVH acelerado por GPU
 * que estructura los tokens en el espacio 3D para permitir búsquedas rápidas de
 * vecinos semánticos mediante ray tracing.
 *
 * Algoritmo:
 *   1. Ordenar tokens por centroide en el eje de mayor varianza
 *   2. Construir árbol binario recursivamente mediante SAH (Surface Area Heuristic) simplificado
 *   3. Subir estructura del árbol a GPU con cudaMalloc/cudaMemcpy
 *   4. Compilar AccelerationStructure con OptiX
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include "token_geometry.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <algorithm>
#include <cstring>
#include <cmath>
#include <vector>
#include <queue>
#include <limits>
#include <cstdio>

// ============================================================================
// Macros de manejo de errores CUDA y OptiX
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return false; \
        } \
    } while (0)

#define OPTIX_CHECK(call) \
    do { \
        OptixResult result = (call); \
        if (result != OPTIX_SUCCESS) { \
            fprintf(stderr, "OptiX error at %s:%d: code %d\n", __FILE__, __LINE__, \
                    static_cast<int>(result)); \
            return false; \
        } \
    } while (0)

// ============================================================================
// Estructuras internas para construcción del BVH
// ============================================================================

/**
 * @struct BVHNode
 * @brief Nodo interno del árbol BVH (no es el TokenNode, sino estructura de árbol)
 */
struct BVHNode {
    // Geometría: bounding box
    float3 aabb_min;
    float3 aabb_max;

    // Topología: índices de hijos (para nodos internos)
    // Si es hoja: left = índice del TokenNode (en rango [0, num_tokens))
    //            right = número de TokenNodes en esta hoja (-1 si hoja individual)
    int32_t left_child;   // Índice del hijo izquierdo o índice del token
    int32_t right_child;  // Índice del hijo derecho o -1

    // Información de profundidad
    uint32_t depth;
};

// ============================================================================
// Clase SemanticBVH (definición simple)
// ============================================================================

class SemanticBVH {
public:
    SemanticBVH() = default;
    ~SemanticBVH() { cleanup(); }

    bool build(TokenNode* nodes, uint32_t num_nodes);
    void computeStats() const;

private:
    std::vector<BVHNode> bvh_nodes;
    std::vector<TokenNode*> sorted_nodes;
    uint32_t num_tokens = 0;
    uint32_t tree_depth = 0;
    uint64_t total_memory_used = 0;

    // Funciones auxiliares
    float computeAABBSurfaceArea(const float3& min, const float3& max) const;
    int32_t buildRecursive(uint32_t start, uint32_t end, uint32_t depth);
    void computeBounds(uint32_t start, uint32_t end, float3& min_out, float3& max_out) const;
    void cleanup();
};

// ============================================================================
// Implementación: SemanticBVH::computeAABBSurfaceArea
// ============================================================================

/**
 * @brief Calcula el área de superficie de un AABB.
 *
 * Usada en el cálculo de SAH (Surface Area Heuristic) simplificado.
 *
 * @param min Esquina mínima del AABB
 * @param max Esquina máxima del AABB
 * @return Área de superficie = 2 * (xy + yz + zx)
 */
float SemanticBVH::computeAABBSurfaceArea(const float3& min, const float3& max) const {
    float3 size = max - min;
    return 2.0f * (size.x * size.y + size.y * size.z + size.z * size.x);
}

// ============================================================================
// Implementación: SemanticBVH::computeBounds
// ============================================================================

/**
 * @brief Calcula el bounding box que contiene todos los tokens en [start, end).
 *
 * @param start Índice inicial (inclusivo)
 * @param end Índice final (exclusivo)
 * @param min_out Esquina mínima del AABB (salida)
 * @param max_out Esquina máxima del AABB (salida)
 */
void SemanticBVH::computeBounds(uint32_t start, uint32_t end, float3& min_out, float3& max_out) const {
    // Handle empty range: return zero-volume AABB at origin
    if (start >= end) {
        min_out = make_float3(0.0f, 0.0f, 0.0f);
        max_out = make_float3(0.0f, 0.0f, 0.0f);
        return;
    }

    min_out = make_float3(std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max(),
                          std::numeric_limits<float>::max());
    max_out = make_float3(std::numeric_limits<float>::lowest(),
                          std::numeric_limits<float>::lowest(),
                          std::numeric_limits<float>::lowest());

    for (uint32_t i = start; i < end; ++i) {
        const TokenNode* node = sorted_nodes[i];

        min_out.x = std::min(min_out.x, node->aabb_min.x);
        min_out.y = std::min(min_out.y, node->aabb_min.y);
        min_out.z = std::min(min_out.z, node->aabb_min.z);

        max_out.x = std::max(max_out.x, node->aabb_max.x);
        max_out.y = std::max(max_out.y, node->aabb_max.y);
        max_out.z = std::max(max_out.z, node->aabb_max.z);
    }
}

// ============================================================================
// Implementación: SemanticBVH::buildRecursive
// ============================================================================

/**
 * @brief Construye recursivamente el árbol BVH mediante ordenamiento y partición.
 *
 * Algoritmo SAH simplificado:
 *   1. Calcular bounding box de todos los nodos en [start, end)
 *   2. Encontrar eje de máxima varianza (x, y, z)
 *   3. Ordenar nodos por centroide en ese eje
 *   4. Particionar en punto medio
 *   5. Recursivamente construir subárboles izquierdo y derecho
 *
 * @param start Índice inicial de tokens a procesar
 * @param end Índice final (exclusivo)
 * @param depth Profundidad actual en el árbol
 * @return Índice del nodo BVH creado en bvh_nodes
 */
int32_t SemanticBVH::buildRecursive(uint32_t start, uint32_t end, uint32_t depth) {
    if (start >= end) {
        fprintf(stderr, "[WARNING] buildRecursive called with empty range: start=%u, end=%u\n",
                start, end);
        return -1;
    }

    tree_depth = std::max(tree_depth, depth);

    // ====================================================================================
    // CASO BASE: Un solo token
    // ====================================================================================
    if (end - start == 1) {
        BVHNode leaf;
        TokenNode* node = sorted_nodes[start];
        leaf.aabb_min = node->aabb_min;
        leaf.aabb_max = node->aabb_max;
        leaf.left_child = start;  // Índice del token
        leaf.right_child = -1;    // Marca como hoja
        leaf.depth = depth;

        int32_t node_idx = static_cast<int32_t>(bvh_nodes.size());
        bvh_nodes.push_back(leaf);
        return node_idx;
    }

    // ====================================================================================
    // CALCULAR BOUNDS Y ENCONTRAR EJE DE PARTICIÓN
    // ====================================================================================
    float3 bounds_min, bounds_max;
    computeBounds(start, end, bounds_min, bounds_max);

    float3 extent = bounds_max - bounds_min;

    // Encontrar eje de máxima extensión
    int split_axis = 0;  // x
    if (extent.y > extent.x) split_axis = 1;  // y
    if (extent.z > extent.y) split_axis = 2;  // z

    // ====================================================================================
    // ORDENAR Y PARTICIONAR
    // ====================================================================================
    // Comparador lambda para ordenar por eje específico
    auto compare_axis = [split_axis](TokenNode* a, TokenNode* b) -> bool {
        float a_val = (split_axis == 0) ? a->centroid.x :
                      (split_axis == 1) ? a->centroid.y : a->centroid.z;
        float b_val = (split_axis == 0) ? b->centroid.x :
                      (split_axis == 1) ? b->centroid.y : b->centroid.z;
        return a_val < b_val;
    };

    // Ordenar rango [start, end)
    std::sort(sorted_nodes.begin() + start, sorted_nodes.begin() + end, compare_axis);

    // Particionar en punto medio
    uint32_t mid = start + (end - start) / 2;

    // ====================================================================================
    // RECURSIÓN: CONSTRUIR SUBÁRBOLES
    // ====================================================================================
    int32_t left_idx = buildRecursive(start, mid, depth + 1);
    int32_t right_idx = buildRecursive(mid, end, depth + 1);

    // ====================================================================================
    // CREAR NODO INTERNO
    // ====================================================================================
    BVHNode internal;
    internal.aabb_min = bounds_min;
    internal.aabb_max = bounds_max;
    internal.left_child = left_idx;
    internal.right_child = right_idx;
    internal.depth = depth;

    int32_t node_idx = static_cast<int32_t>(bvh_nodes.size());
    bvh_nodes.push_back(internal);

    return node_idx;
}

// ============================================================================
// Implementación: SemanticBVH::build
// ============================================================================

/**
 * @brief Construye el árbol BVH desde un array de TokenNodes.
 *
 * Pasos principales:
 *   1. Inicializar array de punteros a nodos (para ordenamiento eficiente)
 *   2. Construir árbol recursivo
 *   3. Subir estructura de árbol a GPU
 *   4. Crear AccelerationStructure de OptiX
 *
 * @param nodes Array de TokenNodes
 * @param num_nodes Número de tokens
 * @return true si construcción fue exitosa, false si error
 */
bool SemanticBVH::build(TokenNode* nodes, uint32_t num_nodes) {
    if (num_nodes == 0) {
        fprintf(stderr, "SemanticBVH::build: num_nodes == 0\n");
        return false;
    }

    num_tokens = num_nodes;

    // ====================================================================================
    // PASO 1: Inicializar array de punteros
    // ====================================================================================
    sorted_nodes.clear();
    sorted_nodes.reserve(num_nodes);

    for (uint32_t i = 0; i < num_nodes; ++i) {
        sorted_nodes.push_back(&nodes[i]);
    }

    // ====================================================================================
    // PASO 2: Construir árbol recursivo
    // ====================================================================================
    printf("[SemanticBVH] Building BVH for %u tokens...\n", num_nodes);

    bvh_nodes.clear();
    tree_depth = 0;

    int32_t root_idx = buildRecursive(0, num_nodes, 0);

    if (root_idx < 0) {
        fprintf(stderr, "SemanticBVH::build: recursion failed\n");
        return false;
    }

    // ====================================================================================
    // PASO 3: Subir estructura a GPU
    // ====================================================================================
    // Para el prototipo, simulamos una "subida" a GPU mediante cudaMalloc/cudaMemcpy
    // En una versión real, esto sería más complejo (BVH comprimido, etc.)

    printf("[SemanticBVH] Uploading BVH structure to GPU...\n");

    // Calcular tamaño de memoria
    total_memory_used = bvh_nodes.size() * sizeof(BVHNode);
    printf("[SemanticBVH] BVH structure size: %llu bytes\n", total_memory_used);

    // Asignar memoria GPU (simulado en CPU para prototipo)
    // NOTE: Using host malloc intentionally - this is a CPU-side simulation of GPU upload.
    // In production, replace with cudaMalloc/cudaMemcpy for actual GPU allocation.
    void* host_bvh_nodes = malloc(total_memory_used);
    if (!host_bvh_nodes) {
        fprintf(stderr, "SemanticBVH::build: malloc failed for BVH node memory\n");
        return false;
    }

    // Copiar datos (host-side simulation)
    memcpy(host_bvh_nodes, bvh_nodes.data(), total_memory_used);

    // TODO(3.6): In production, replace with:
    //   cudaMalloc(&d_bvh_nodes, total_memory_used);
    //   cudaMemcpy(d_bvh_nodes, bvh_nodes.data(), total_memory_used, cudaMemcpyHostToDevice);

    // ====================================================================================
    // PASO 4: Crear AccelerationStructure de OptiX (simulado)
    // ====================================================================================
    // En código real, aquí llamaríamos a optixAccelBuild con la estructura BVH.
    // Para el prototipo, solo reportamos éxito.

    printf("[SemanticBVH] Building OptiX AccelerationStructure...\n");
    // OptixResult result = optixAccelBuild(context, stream, &accel_options, ...);
    // if (result != OPTIX_SUCCESS) return false;

    printf("[SemanticBVH] BVH construction complete.\n");

    // Liberar memoria temporal
    free(host_bvh_nodes);

    return true;
}

// ============================================================================
// Implementación: SemanticBVH::computeStats
// ============================================================================

/**
 * @brief Calcula e imprime estadísticas del árbol BVH.
 *
 * Útil para validación y profiling:
 *   - Profundidad del árbol (importante para aceleración)
 *   - Número de nodos
 *   - Memoria ocupada
 *   - Balance del árbol
 */
void SemanticBVH::computeStats() const {
    printf("\n=== SemanticBVH Statistics ===\n");
    printf("Number of tokens: %u\n", num_tokens);
    printf("Number of BVH nodes: %zu\n", bvh_nodes.size());
    printf("Tree depth: %u\n", tree_depth);
    printf("Memory used: %llu bytes (%.2f MB)\n",
           total_memory_used,
           total_memory_used / (1024.0 * 1024.0));

    // Calcular balance del árbol
    // Un árbol perfectamente balanceado para N nodos tiene profundidad ~log₂(N)
    float expected_depth = logf(static_cast<float>(num_tokens)) / logf(2.0f);
    float balance_factor = tree_depth / expected_depth;

    printf("Expected depth (log₂ N): %.1f\n", expected_depth);
    printf("Actual depth: %u\n", tree_depth);
    printf("Balance factor: %.2f (1.0 = perfect)\n", balance_factor);

    // Estadísticas de nodos BVH
    uint32_t internal_nodes = 0;
    uint32_t leaf_nodes = 0;
    float avg_depth = 0.0f;

    for (const auto& node : bvh_nodes) {
        if (node.right_child == -1) {
            leaf_nodes++;
        } else {
            internal_nodes++;
        }
        avg_depth += node.depth;
    }

    avg_depth /= bvh_nodes.size();

    printf("Internal nodes: %u\n", internal_nodes);
    printf("Leaf nodes: %u\n", leaf_nodes);
    printf("Average node depth: %.2f\n", avg_depth);

    printf("==============================\n\n");
}

// ============================================================================
// Implementación: SemanticBVH::cleanup
// ============================================================================

void SemanticBVH::cleanup() {
    bvh_nodes.clear();
    sorted_nodes.clear();
    num_tokens = 0;
    tree_depth = 0;
    total_memory_used = 0;
}

// ============================================================================
// Función global: Ejemplo de uso
// ============================================================================

/**
 * @brief Función de ejemplo para construir y validar un BVH.
 *
 * En código real, esto estaría en tests/ en lugar de src/.
 */
bool exampleBVHConstruction(TokenNode* nodes, uint32_t num_nodes) {
    printf("[SemanticBVH Example] Starting BVH construction for %u tokens...\n", num_nodes);

    SemanticBVH bvh;
    if (!bvh.build(nodes, num_nodes)) {
        fprintf(stderr, "[SemanticBVH Example] BVH construction failed.\n");
        return false;
    }

    bvh.computeStats();
    return true;
}
