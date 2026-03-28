/**
 * @file token_geometry.h
 * @brief Definiciones de estructuras geométricas para tokens en LiquidBit Zero-Matrix
 *
 * Este header contiene las definiciones fundamentales de cómo los tokens se mapean
 * a objetos geométricos en el espacio 3D para ser procesados por los RT Cores.
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>
#include <array>

// ============================================================================
// CUDA float3 helper (fallback si no está disponible en headers de CUDA)
// ============================================================================

#ifndef __CUDA_INTERNAL_COMPILATION__
    #ifndef float3
    struct float3 {
        float x, y, z;

        __host__ __device__ float3() : x(0.0f), y(0.0f), z(0.0f) {}
        __host__ __device__ float3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}

        __host__ __device__ float3 operator+(const float3& v) const {
            return float3(x + v.x, y + v.y, z + v.z);
        }

        __host__ __device__ float3 operator-(const float3& v) const {
            return float3(x - v.x, y - v.y, z - v.z);
        }

        __host__ __device__ float3 operator*(float s) const {
            return float3(x * s, y * s, z * s);
        }

        __host__ __device__ float dot(const float3& v) const {
            return x * v.x + y * v.y + z * v.z;
        }

        __host__ __device__ float length() const {
            return sqrtf(x * x + y * y + z * z);
        }
    };
    #endif
#endif

// ============================================================================
// Constantes globales de LiquidBit Zero-Matrix
// ============================================================================

/// Dimensión del embedding comprimido en el TokenNode (reducido de 768/4096 a FP16)
constexpr uint32_t LIQUIDBIT_EMBEDDING_DIM = 256;

/// Número máximo de rayos de atención óptica emitidos por token
constexpr uint32_t LIQUIDBIT_NUM_RAYS = 4096;

/// Coeficiente de absorción semántica (decay exponencial): attention = E₀ * exp(-λ * d)
constexpr float LIQUIDBIT_LAMBDA = 0.1f;

/// Dimensión de la compresión PCA proyectada al espacio 3D
constexpr uint32_t LIQUIDBIT_SPATIAL_DIM = 3;

// ============================================================================
// Estructura TokenNode: Representación geométrica de un token
// ============================================================================

/**
 * @struct TokenNode
 * @brief Representa un token como un objeto geométrico en el espacio semántico 3D.
 *
 * Esta estructura es el elemento fundamental del BVH. Cada token se mapea a un
 * bounding box (AABB) en el espacio 3D, cuya posición y tamaño preservan la
 * similitud semántica del embedding original.
 *
 * El embedding se comprime de D dimensiones (768/4096) a 256 half-floats,
 * preservando ~95% de la varianza semántica.
 */
struct TokenNode {
    // ========================================================================
    // IDENTIDAD Y POSICIÓN EN SECUENCIA
    // ========================================================================

    /**
     * @brief ID único del token en el vocabulario.
     *
     * Rango típico: 0 a ~50.000 (vocabulario de ~50K tokens en BERT/GPT).
     * Usado para reconstrucción textual y trazabilidad.
     */
    uint32_t token_id;

    /**
     * @brief Índice de posición del token en la secuencia de entrada.
     *
     * Rango: 0 a (N-1), donde N es la longitud de la secuencia.
     * Crítico para preservar el orden secuencial en atención multi-layer.
     */
    uint32_t position_in_seq;

    // ========================================================================
    // GEOMETRÍA 3D PARA RAY TRACING
    // ========================================================================

    /**
     * @brief Centroide del bounding box en el espacio semántico 3D.
     *
     * Calculado mediante PCA esférica del embedding original:
     *   embedding[D] → centroid ∈ ℝ³
     *
     * Preserva la métrica coseno: tokens semánticamente similares
     * tienen centroides cercanos en el espacio 3D.
     */
    float3 centroid;

    /**
     * @brief Esquina mínima del bounding box (AABB).
     *
     * Define el rango de validez geométrica para ray-triangle tests.
     * Típicamente: aabb_min < centroid < aabb_max (componente a componente).
     */
    float3 aabb_min;

    /**
     * @brief Esquina máxima del bounding box (AABB).
     *
     * Junto con aabb_min, define una caja rectangular en el espacio 3D
     * que actúa como proxy geométrico de la bola semántica del token.
     */
    float3 aabb_max;

    /**
     * @brief Radio semántico del token.
     *
     * Representa la "dispersión" semántica del token en diferentes contextos.
     * Tokens polisémicos (ej. "bank") tienen semantic_radius mayor.
     *
     * Rango típico: 0.01 a 1.0.
     * Usado por el BVH para optimizar la volumetría de AABBs.
     */
    float semantic_radius;

    // ========================================================================
    // EMBEDDING COMPRIMIDO (FP16)
    // ========================================================================

    /**
     * @brief Embedding del token comprimido a FP16 (half-float).
     *
     * Dimensión: 256 valores half-float.
     * Originalmente D dimensiones (768 para BERT-base, 4096 para GPT-4).
     *
     * Reducción: D → 256 mediante PCA esférico.
     * Preservación: ~95% de la varianza de disimilaridad coseno.
     *
     * Almacenamiento comprimido: 256 * 2 bytes = 512 bytes por token.
     * En comparación: 768 floats = 3.072 KB (BERT) o 4096 floats = 16 KB (GPT-4).
     *
     * Beneficio: 6-32x menos memoria en GPU por token,
     * manteniendo la topología semántica necesaria para el BVH.
     */
    half embedding[LIQUIDBIT_EMBEDDING_DIM];

    // ========================================================================
    // ATENCIÓN Y ENERGÍA
    // ========================================================================

    /**
     * @brief Peso de atención acumulado calculado tras raytrace.
     *
     * Resultado de la fórmula de decay semántico:
     *   attention_weight = E₀ · exp(-λ · d_semantic)
     *
     * Donde:
     *   - E₀ = energía inicial del rayo (1.0)
     *   - λ = LIQUIDBIT_LAMBDA (coeficiente de absorción)
     *   - d_semantic = distancia euclídea 3D (proxy de irrelevancia semántica)
     *
     * Rango: [0.0, 1.0], normalizado tras acumulación en todos los rayos.
     */
    float attention_weight;

    /**
     * @brief Energía restante del rayo tras colisión.
     *
     * Modelado físicamente: cuando un rayo golpea un token (ClosestHit),
     * pierde energía inversamente proporcional a la distancia semántica.
     *
     * Rango: [0.0, 1.0].
     * Si energy_remaining < min_energy_threshold, el rayo termina su traversal.
     *
     * Esto implementa el concepto de "atención decayente": tokens lejanos
     * contribuyen menos conforme el rayo pierde energía.
     */
    float energy_remaining;

    // ========================================================================
    // Métodos helper
    // ========================================================================

    /// @brief Calcula el volumen del AABB del token.
    __host__ __device__ float getAABBVolume() const {
        float3 size = aabb_max - aabb_min;
        return size.x * size.y * size.z;
    }

    /// @brief Comprueba si un punto está dentro del AABB.
    __host__ __device__ bool containsPoint(const float3& p) const {
        return (p.x >= aabb_min.x && p.x <= aabb_max.x) &&
               (p.y >= aabb_min.y && p.y <= aabb_max.y) &&
               (p.z >= aabb_min.z && p.z <= aabb_max.z);
    }
};

// ============================================================================
// Estructura SemanticRay: Rayo de atención óptica
// ============================================================================

/**
 * @struct SemanticRay
 * @brief Representa un rayo de atención óptica emitido desde un token query.
 *
 * Cada rayo es análogo a una "head" de atención multi-head, pero representado
 * como un rayo 3D que intersecta el BVH para encontrar tokens relevantes.
 *
 * El rayo mantiene energía (analogía con capacidad de atención) que decae
 * conforme avanza, modelando el fenómeno de "atención dispersa".
 */
struct SemanticRay {
    // ========================================================================
    // GEOMETRÍA DEL RAYO
    // ========================================================================

    /**
     * @brief Punto de origen del rayo en el espacio 3D.
     *
     * Típicamente: centroide del token query + offset semántico basado
     * en el embedding del query.
     */
    float3 origin;

    /**
     * @brief Dirección normalizada del rayo.
     *
     * Vector unitario (||direction|| ≈ 1.0) que apunta desde el token query
     * hacia regiones del espacio semánticamente relevantes.
     *
     * Calculado a partir de:
     *   - Dimensión dominante del embedding del query
     *   - Diversificación pseudo-aleatoria (ray_id)
     */
    float3 direction;

    // ========================================================================
    // ENERGÍA DEL RAYO
    // ========================================================================

    /**
     * @brief Energía inicial del rayo (típicamente 1.0).
     *
     * Cada vez que el rayo golpea un token (ClosestHit), pierde energía.
     * La pérdida sigue la ley de Lambert-Beer en optica:
     *   E(d) = E₀ · exp(-λ · d)
     *
     * Cuando energy < min_energy_threshold, el rayo termina.
     */
    float energy;

    // ========================================================================
    // EMBEDDING DE QUERY (FP16)
    // ========================================================================

    /**
     * @brief Embedding comprimido del token query que emite este rayo.
     *
     * Dimensión: 256 half-floats (igual que TokenNode::embedding).
     *
     * Almacenado en el rayo para permitir que los kernels CUDA
     * (ClosestHit, AnyHit) calculen similitud semántica local
     * sin acceder a memoria global.
     */
    half query_embedding[LIQUIDBIT_EMBEDDING_DIM];

    // ========================================================================
    // IDENTIFICACIÓN
    // ========================================================================

    /**
     * @brief ID único del rayo (para demultiplexado de resultados).
     *
     * Rango: 0 a (num_rays - 1).
     * Usado para indexar en AttentionResult::top_k_tokens y ::attention_weights.
     */
    uint32_t ray_id;
};

// ============================================================================
// Funciones inline helper para geometría
// ============================================================================

/**
 * @brief Calcula la distancia euclídea 3D entre dos puntos.
 *
 * @param a Primer punto
 * @param b Segundo punto
 * @return Distancia euclídea ||a - b||₂
 *
 * @note Inline para uso en kernels CUDA sin overhead de llamada.
 * @note Esta distancia es un proxy de "irrelevancia semántica" en el espacio 3D.
 */
__host__ __device__ inline float computeSemanticDistance(const float3& a, const float3& b) {
    float3 diff = a - b;
    return sqrtf(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
}

/**
 * @brief Calcula la distancia cuadrada euclídea (evita sqrt, más rápido).
 *
 * @param a Primer punto
 * @param b Segundo punto
 * @return Distancia cuadrada ||a - b||₂²
 *
 * @note Útil para comparaciones y búsquedas sin necesidad del valor exacto.
 */
__host__ __device__ inline float computeSemanticDistanceSq(const float3& a, const float3& b) {
    float3 diff = a - b;
    return diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
}

/**
 * @brief Calcula el peso de atención con decay exponencial.
 *
 * @param distance Distancia semántica 3D
 * @param initial_energy Energía inicial del rayo (ej. 1.0)
 * @param lambda Coeficiente de absorción (por defecto LIQUIDBIT_LAMBDA)
 * @return Peso de atención según: E₀ · exp(-λ · d)
 *
 * @note Fórmula clave del mecanismo de atención óptica.
 * @note El exponencial negativo asegura que distancias mayores → pesos menores.
 */
__host__ __device__ inline float computeAttentionWeight(
    float distance,
    float initial_energy = 1.0f,
    float lambda = LIQUIDBIT_LAMBDA) {
    return initial_energy * expf(-lambda * distance);
}

#endif // LIQUIDBIT_TOKEN_GEOMETRY_H_
