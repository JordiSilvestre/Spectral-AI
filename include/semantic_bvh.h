/**
 * @file semantic_bvh.h
 * @brief Gestión del árbol BVH (Bounding Volume Hierarchy) semántico para LiquidBit Zero-Matrix
 *
 * Define la estructura del BVH y la interfaz de construcción/consulta.
 * El BVH encapsula todos los TokenNodes en una jerarquía que permite
 * ray tracing eficiente (O(log N)) para encontrar tokens relevantes.
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#pragma once
#ifndef LIQUIDBIT_SEMANTIC_BVH_H_
#define LIQUIDBIT_SEMANTIC_BVH_H_

#include "token_geometry.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>
#include <string>

// ============================================================================
// Enums y constantes para configuración del BVH
// ============================================================================

/**
 * @enum BVHBuildMode
 * @brief Estrategias de construcción del árbol BVH.
 */
enum class BVHBuildMode {
    /// Construcción estándar SAH (Surface Area Heuristic)
    SAH_HEURISTIC = 0,

    /// Construcción rápida para updates frecuentes (binning aproximado)
    FAST_BUILD = 1,

    /// Construcción GPU-acelerada usando thrust::sort
    GPU_ACCELERATED = 2
};

/**
 * @struct BVHBuildConfig
 * @brief Configuración para la construcción del árbol BVH.
 */
struct BVHBuildConfig {
    /// Modo de construcción (SAH, FAST, GPU)
    BVHBuildMode build_mode = BVHBuildMode::SAH_HEURISTIC;

    /// Máximo número de tokens por hoja (típico: 1-4)
    uint32_t max_leaf_size = 1;

    /// Profundidad máxima del árbol (típico: 20-30 para 100K tokens)
    uint32_t max_depth = 30;

    /// Usar full precision (float32) en AABB o compact (float16)
    bool use_full_precision_aabb = true;

    /// Compilar para OptiX (vs CUDA puro)
    bool use_optix = true;
};

// ============================================================================
// Estructura BVHNode: Nodo del árbol
// ============================================================================

/**
 * @struct BVHNode
 * @brief Nodo del árbol BVH que encapsula una región del espacio semántico.
 *
 * Cada nodo contiene un bounding box (AABB) que engloba todos los
 * tokens/subnodos en su subtree.
 *
 * Nodos internos: tienen dos hijos (left_child, right_child).
 * Nodos hoja: almacenan referencias a TokenNodes (token_idx).
 */
struct BVHNode {
    // ========================================================================
    // BOUNDING BOX (AABB)
    // ========================================================================

    /**
     * @brief Esquina mínima del bounding box de este nodo.
     *
     * Envuelve tightamente todos los centros de tokens en el subtree.
     */
    float3 aabb_min;

    /**
     * @brief Esquina máxima del bounding box de este nodo.
     *
     * Junto con aabb_min, define la caja rectangular de la región.
     */
    float3 aabb_max;

    // ========================================================================
    // ESTRUCTURA DEL ÁRBOL
    // ========================================================================

    /**
     * @brief Índice del hijo izquierdo en el array de nodos.
     *
     * Si is_leaf == true, este campo se ignora.
     * Rango: 0 a (num_nodes - 1).
     */
    uint32_t left_child;

    /**
     * @brief Índice del hijo derecho en el array de nodos.
     *
     * Si is_leaf == true, este campo se ignora.
     * Rango: 0 a (num_nodes - 1).
     */
    uint32_t right_child;

    /**
     * @brief Índice del token almacenado en este nodo (solo si es hoja).
     *
     * Para nodos hoja: índice en el array de TokenNodes (0 a num_tokens-1).
     * Para nodos internos: ignorado (típicamente ~0).
     *
     * En una hoja con max_leaf_size > 1, se puede almacenar un rango [token_idx, token_idx + max_leaf_size).
     */
    uint32_t token_idx;

    /**
     * @brief Flag: true si es nodo hoja, false si es interno.
     *
     * Determina cómo interpretar los campos:
     *   - is_leaf = true  → token_idx es válido, left_child/right_child se ignoran
     *   - is_leaf = false → left_child/right_child son válidos, token_idx se ignora
     */
    bool is_leaf;

    // ========================================================================
    // Métodos helper
    // ========================================================================

    /// @brief Calcula el área superficial del AABB (para SAH heuristic).
    __host__ __device__ float getSurfaceArea() const {
        float3 extents = aabb_max - aabb_min;
        return 2.0f * (extents.x * extents.y + extents.y * extents.z + extents.z * extents.x);
    }

    /// @brief Calcula el volumen del AABB.
    __host__ __device__ float getVolume() const {
        float3 extents = aabb_max - aabb_min;
        return extents.x * extents.y * extents.z;
    }

    /// @brief Comprueba si un punto está dentro del AABB.
    __host__ __device__ bool containsPoint(const float3& p) const {
        return (p.x >= aabb_min.x && p.x <= aabb_max.x) &&
               (p.y >= aabb_min.y && p.y <= aabb_max.y) &&
               (p.z >= aabb_min.z && p.z <= aabb_max.z);
    }
};

// ============================================================================
// Clase SemanticBVH: Gestión del árbol completo
// ============================================================================

/**
 * @class SemanticBVH
 * @brief Gestor del árbol BVH semántico para el prototipo LiquidBit.
 *
 * Responsabilidades:
 * 1. Construcción del BVH a partir de un conjunto de TokenNodes
 * 2. Mantenimiento de la estructura en memoria GPU
 * 3. Integración con OptiX para ray tracing acelerado
 * 4. Gestión de ciclo de vida (allocación/liberación de memoria)
 *
 * Uso típico:
 * ```cpp
 * SemanticBVH bvh;
 * bvh.build(tokens, num_tokens);  // Construir desde tokens
 * OpticalAttention attention;
 * attention.launch(rays, num_rays, bvh, results);  // Usar BVH para ray tracing
 * ```
 *
 * @note El BVH se construye una sola vez por secuencia y se reutiliza
 *       para todas las capas del modelo.
 * @note Memoria GPU: ~100-500 bytes por token (según configuración).
 */
class SemanticBVH {
public:
    // ========================================================================
    // Constructores y destructores
    // ========================================================================

    /**
     * @brief Constructor por defecto.
     *
     * No asigna memoria GPU. Debe llamarse build() después de la construcción.
     */
    SemanticBVH();

    /**
     * @brief Destructor.
     *
     * Libera automáticamente la memoria GPU (nodos BVH, buffers OptiX, etc.).
     * Seguro llamar incluso si build() nunca fue invocado.
     */
    ~SemanticBVH();

    // Prohibir copias (la memoria GPU no se puede copiar trivialmente)
    SemanticBVH(const SemanticBVH&) = delete;
    SemanticBVH& operator=(const SemanticBVH&) = delete;

    // Permitir movimiento (C++11)
    SemanticBVH(SemanticBVH&& other) noexcept;
    SemanticBVH& operator=(SemanticBVH&& other) noexcept;

    // ========================================================================
    // Construcción del BVH
    // ========================================================================

    /**
     * @brief Construye el árbol BVH a partir de un array de TokenNodes.
     *
     * @param tokens Puntero a array de TokenNodes (en memoria GPU o host, según implementación)
     * @param count Número de tokens
     * @param config Configuración de construcción (opcional)
     *
     * @return true si la construcción fue exitosa, false en caso de error
     *
     * @note Esta función es bloqueante. En la implementación, puede llamar a
     *       kernels CUDA o APIs OptiX para construir el árbol.
     * @note Coste: O(N log N) tiempo, O(N) memoria adicional temporal.
     * @note Después de build(), los datos en `tokens` ya no son necesarios;
     *       el BVH almacena copias locales.
     *
     * @see getLastError()
     */
    bool build(
        const TokenNode* tokens,
        uint32_t count,
        const BVHBuildConfig& config = BVHBuildConfig()
    );

    // ========================================================================
    // Acceso a estructuras OptiX y GPU
    // ========================================================================

    /**
     * @brief Obtiene el handle OptiX de la estructura de aceleración.
     *
     * @return OptixTraversableHandle válido si use_optix=true en BVHBuildConfig,
     *         0 en caso contrario.
     *
     * @note Este handle es usado por los kernels OptiX (ray_attention.cu)
     *       para ray tracing acelerado.
     * @note Válido solo después de llamar a build() exitosamente.
     *
     * @see LaunchOptix()
     */
    OptixTraversableHandle getOptixAccelStructure() const {
        return optix_trav_handle_;
    }

    /**
     * @brief Obtiene el puntero al buffer GPU de nodos BVH.
     *
     * @return Puntero al array de BVHNode en memoria GPU (device pointer)
     *
     * @note Útil para kernels CUDA que necesiten acceso directo a la topología.
     * @note Válido solo después de llamar a build() exitosamente.
     * @note No modificar directamente; usar build() para actualizar.
     *
     * @see getNodeCount()
     */
    const BVHNode* getDeviceBuffer() const {
        return device_nodes_;
    }

    /**
     * @brief Obtiene el número de nodos en el árbol.
     *
     * @return Número total de nodos (internos + hojas)
     *
     * @note Para N tokens con max_leaf_size=1, típicamente node_count ≈ 2*N - 1.
     */
    uint32_t getNodeCount() const {
        return node_count_;
    }

    /**
     * @brief Obtiene el número de tokens en el árbol.
     *
     * @return Número de TokenNodes (igual al parámetro `count` de build())
     */
    uint32_t getTokenCount() const {
        return token_count_;
    }

    /**
     * @brief Obtiene la profundidad del árbol.
     *
     * @return Profundidad máxima (distancia desde raíz a hoja más lejana)
     *
     * @note Típico: log₂(N) para árboles balanceados.
     */
    uint32_t getDepth() const {
        return tree_depth_;
    }

    /**
     * @brief Obtiene el nodo raíz del árbol.
     *
     * @return Referencia al nodo raíz (índice 0 en getDeviceBuffer())
     *
     * @note Principalmente para debugging y visualización.
     */
    const BVHNode& getRootNode() const {
        return root_node_;
    }

    // ========================================================================
    // Gestión de memoria y estado
    // ========================================================================

    /**
     * @brief Comprueba si el BVH ha sido construido exitosamente.
     *
     * @return true si build() fue llamado y completó sin errores
     */
    bool isBuilt() const {
        return is_built_;
    }

    /**
     * @brief Obtiene el último mensaje de error.
     *
     * @return String con descripción del error, o string vacío si no hay error
     *
     * @note Útil para debugging cuando build() retorna false.
     */
    std::string getLastError() const {
        return last_error_;
    }

    /**
     * @brief Obtiene el tamaño total de memoria GPU utilizada.
     *
     * @return Número de bytes en GPU (nodos, buffers OptiX, etc.)
     *
     * @note Típico: 100-500 bytes por token.
     * @note Para 100K tokens: ~10-50 MB.
     */
    uint64_t getGPUMemoryUsage() const {
        return gpu_memory_usage_;
    }

    // ========================================================================
    // Debugging y estadísticas
    // ========================================================================

    /**
     * @brief Retorna estadísticas del árbol en un string legible.
     *
     * @return String con: token_count, node_count, tree_depth, gpu_memory, etc.
     */
    std::string getStatistics() const;

    /**
     * @brief Valida la integridad estructural del BVH.
     *
     * @return true si el árbol es válido, false si hay inconsistencias
     *
     * @note Operación cara (traversal completo). Solo usar para debugging.
     */
    bool validateStructure() const;

private:
    // ========================================================================
    // Miembros privados
    // ========================================================================

    bool is_built_ = false;

    /// Array de nodos BVH en memoria GPU
    BVHNode* device_nodes_ = nullptr;

    /// Handle OptiX para ray tracing
    OptixTraversableHandle optix_trav_handle_ = 0;

    /// Estructura de aceleración OptiX (internal)
    OptixAccelStruct optix_accel_ = {};

    /// Información del nodo raíz (cached para acceso rápido)
    BVHNode root_node_ = {};

    /// Número de nodos en el árbol
    uint32_t node_count_ = 0;

    /// Número de tokens originales
    uint32_t token_count_ = 0;

    /// Profundidad máxima del árbol
    uint32_t tree_depth_ = 0;

    /// Memoria GPU total asignada
    uint64_t gpu_memory_usage_ = 0;

    /// Último mensaje de error
    std::string last_error_;

    /// Configuración usada en la construcción
    BVHBuildConfig build_config_;

    // ========================================================================
    // Métodos privados de construcción
    // ========================================================================

    /**
     * @brief Implementación privada de construcción SAH heuristic.
     */
    bool buildSAH_(const TokenNode* tokens, uint32_t count);

    /**
     * @brief Implementación privada de construcción rápida.
     */
    bool buildFast_(const TokenNode* tokens, uint32_t count);

    /**
     * @brief Implementación privada de construcción GPU-acelerada.
     */
    bool buildGPUAccelerated_(const TokenNode* tokens, uint32_t count);

    /**
     * @brief Construye la estructura de aceleración OptiX.
     */
    bool buildOptixAccelStructure_();

    /**
     * @brief Libera toda la memoria GPU.
     */
    void freeGPUMemory_();
};

#endif // LIQUIDBIT_SEMANTIC_BVH_H_
