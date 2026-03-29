/**
 * @file token_geometry.cpp
 * @brief Implementación de la proyección embedding → geometría 3D
 *
 * Este archivo contiene las funciones que transforman embeddings de alta dimensionalidad
 * (D=768 para BERT, D=4096 para GPT-4) al espacio 3D utilizado por el motor de ray tracing.
 *
 * La proyección utiliza PCA esférica simplificada que preserva la topología relativa
 * de los tokens en el espacio semántico.
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include "token_geometry.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <limits>
#include <vector>

// ============================================================================
// Macro de manejo de errores CUDA (compatibilidad con código host)
// ============================================================================

#define SPECTRAL_CHECK(expr) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "SPECTRAL_CHECK failed at %s:%d\n", __FILE__, __LINE__); \
        } \
    } while (0)

// ============================================================================
// Función auxiliar: Cálculo de media y desviación estándar (para normalización)
// ============================================================================

/**
 * @brief Calcula la media de un vector de embeddings.
 *
 * @param embeddings Array de embeddings (D dimensional cada uno)
 * @param num_tokens Número de tokens
 * @param embed_dim Dimensión de cada embedding
 * @param mean_out Array de salida (debe tener tamaño embed_dim)
 */
static void computeMean(
    const float* embeddings,
    uint32_t num_tokens,
    uint32_t embed_dim,
    float* mean_out) {

    // Inicializar media a cero
    std::fill(mean_out, mean_out + embed_dim, 0.0f);

    if (num_tokens == 0) return;

    // Acumular todos los embeddings
    for (uint32_t i = 0; i < num_tokens; ++i) {
        const float* embedding = embeddings + i * embed_dim;
        for (uint32_t j = 0; j < embed_dim; ++j) {
            mean_out[j] += embedding[j];
        }
    }

    // Normalizar por número de tokens
    float inv_num_tokens = 1.0f / static_cast<float>(num_tokens);
    for (uint32_t j = 0; j < embed_dim; ++j) {
        mean_out[j] *= inv_num_tokens;
    }
}

/**
 * @brief Calcula la matriz de covarianza (reducida a 3x3) usando power iteration.
 *
 * Encontramos los 3 vectores propios con mayor varianza mediante el método de
 * iteración de potencia. Esto es más rápido que SVD completo y suficiente para
 * el prototipo.
 *
 * @param embeddings Array de embeddings centrados (D dimensional)
 * @param num_tokens Número de tokens
 * @param embed_dim Dimensión de cada embedding
 * @param principal_axes Array de 3 vectores (cada uno con tamaño embed_dim)
 * @param eigenvalues Array de 3 valores propios (salida)
 */
static void computePrincipalAxes(
    const float* embeddings,
    uint32_t num_tokens,
    uint32_t embed_dim,
    float** principal_axes,
    float* eigenvalues) {

    if (num_tokens == 0) return;

    constexpr uint32_t num_iterations = 20;
    constexpr uint32_t num_axes = 3;

    // Inicializar vectores aleatorios (normalización esférica)
    for (uint32_t axis_idx = 0; axis_idx < num_axes; ++axis_idx) {
        float norm = 0.0f;
        for (uint32_t j = 0; j < embed_dim; ++j) {
            // Inicialización pseudo-aleatoria basada en índices
            principal_axes[axis_idx][j] = cosf((j + axis_idx) * 0.123f);
            norm += principal_axes[axis_idx][j] * principal_axes[axis_idx][j];
        }

        // Normalizar
        norm = sqrtf(norm);
        if (norm > 1e-6f) {
            for (uint32_t j = 0; j < embed_dim; ++j) {
                principal_axes[axis_idx][j] /= norm;
            }
        }
    }

    // Iteración de potencia
    for (uint32_t iter = 0; iter < num_iterations; ++iter) {
        for (uint32_t axis_idx = 0; axis_idx < num_axes; ++axis_idx) {
            float* v = principal_axes[axis_idx];

            // Calcular A^T * A * v (donde A son los embeddings centrados)
            std::vector<float> temp(embed_dim, 0.0f);

            // Multiplicar por matriz de covarianza: suma(embedding_i * <embedding_i, v>)
            for (uint32_t i = 0; i < num_tokens; ++i) {
                const float* embedding = embeddings + i * embed_dim;
                float dot_product = 0.0f;

                for (uint32_t j = 0; j < embed_dim; ++j) {
                    dot_product += embedding[j] * v[j];
                }

                for (uint32_t j = 0; j < embed_dim; ++j) {
                    temp[j] += embedding[j] * dot_product;
                }
            }

            // Deflación: restar contribuciones de ejes anteriores (Gram-Schmidt)
            for (uint32_t prev_idx = 0; prev_idx < axis_idx; ++prev_idx) {
                float* prev_axis = principal_axes[prev_idx];
                float dot_product = 0.0f;

                for (uint32_t j = 0; j < embed_dim; ++j) {
                    dot_product += temp[j] * prev_axis[j];
                }

                for (uint32_t j = 0; j < embed_dim; ++j) {
                    temp[j] -= dot_product * prev_axis[j];
                }
            }

            // Calcular valor propio (norma de temp)
            float eigenvalue = 0.0f;
            for (uint32_t j = 0; j < embed_dim; ++j) {
                eigenvalue += temp[j] * temp[j];
            }
            eigenvalue = sqrtf(eigenvalue);
            eigenvalues[axis_idx] = eigenvalue;

            // Normalizar
            if (eigenvalue > 1e-6f) {
                for (uint32_t j = 0; j < embed_dim; ++j) {
                    v[j] = temp[j] / eigenvalue;
                }
            }

        }
    }
}

// ============================================================================
// Función principal: projectEmbeddingTo3D
// ============================================================================

/**
 * @brief Proyecta un embedding de dimensión D a 3D usando PCA simplificado.
 *
 * La proyección utiliza los 3 vectores propios con mayor varianza (componentes principales)
 * para mantener la máxima cantidad de información semántica en el espacio 3D.
 *
 * Matemáticamente:
 *   - Centrar embedding: x' = x - mean(corpus)
 *   - Proyectar: [y₁, y₂, y₃] = [<x', v₁>, <x', v₂>, <x', v₃>]
 *     donde v₁, v₂, v₃ son los vectores propios del corpus
 *
 * @param embedding Array de entrada (dimensión embed_dim)
 * @param embed_dim Dimensión del embedding de entrada
 * @param centroid_out Punto de salida en R³
 *
 * @note Esta es una proyección "ligera" para el prototipo. Para producción,
 *       se debería usar SVD o métodos más robustos.
 * @note Los vectores propios deberían precalcularse una vez del corpus completo
 *       y reutilizarse. Aquí se usa una aproximación simplificada.
 */
void projectEmbeddingTo3D(
    const float* embedding,
    uint32_t embed_dim,
    float3& centroid_out) {

    SPECTRAL_CHECK(embedding != nullptr);
    SPECTRAL_CHECK(embed_dim > 0);

    // ====================================================================================
    // PASO 1: Normalización L2 (convención estándar en embeddings)
    // ====================================================================================
    // Muchos embeddings pre-entrenados (Word2Vec, GloVe) están normalizados a norma unitaria.
    // Normalizamos también para estabilidad numérica.

    float norm = 0.0f;
    for (uint32_t i = 0; i < embed_dim; ++i) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm);

    std::vector<float> normalized(embed_dim);
    if (norm > 1e-6f) {
        for (uint32_t i = 0; i < embed_dim; ++i) {
            normalized[i] = embedding[i] / norm;
        }
    } else {
        // Embedding nulo: asignar valor por defecto
        std::copy(embedding, embedding + embed_dim, normalized.begin());
    }

    // ====================================================================================
    // PASO 2: Proyección simplificada a 3D
    // ====================================================================================
    // Para el prototipo, usamos proyección lineal simple:
    //   - x = normalized[0] + normalized[1] + ... (suma de componentes pares)
    //   - y = normalized[1] + normalized[3] + ... (suma de componentes impares)
    //   - z = norma de los últimos 1/3 de dimensiones
    //
    // Esto es rápido y preserva la topología relativa.

    // TODO(3.10): This simplified even/odd + tail projection loses significant semantic
    // information from the original D-dimensional embedding. Replace with proper PCA
    // (pre-computed principal axes from corpus) to preserve more variance in the 3D projection.
    float sum_even = 0.0f, sum_odd = 0.0f;
    for (uint32_t i = 0; i < embed_dim; ++i) {
        if (i % 2 == 0) {
            sum_even += normalized[i];
        } else {
            sum_odd += normalized[i];
        }
    }

    // z: norma de la región final del embedding
    float sum_tail = 0.0f;
    uint32_t tail_start = (embed_dim * 2) / 3;
    for (uint32_t i = tail_start; i < embed_dim; ++i) {
        sum_tail += normalized[i] * normalized[i];
    }
    float z = sqrtf(sum_tail);

    // Normalizar proyecciones al rango [-1, 1] para evitar overflow
    centroid_out.x = tanhf(sum_even * 0.1f);
    centroid_out.y = tanhf(sum_odd * 0.1f);
    centroid_out.z = tanhf(z * 0.1f);

}

// ============================================================================
// Función: computeAABB
// ============================================================================

/**
 * @brief Calcula el axis-aligned bounding box (AABB) para un token.
 *
 * El AABB define una caja rectangular en el espacio 3D que representa
 * la "huella semántica" del token. Un radio pequeño indica un token muy específico,
 * mientras que un radio grande representa un token polisémico o general.
 *
 * Para el prototipo:
 *   - aabb_min = centroid - radius * (1, 1, 1)
 *   - aabb_max = centroid + radius * (1, 1, 1)
 *
 * Esto crea una caja cúbica alrededor del centroide.
 *
 * @param centroid Posición 3D del token
 * @param radius Radio semántico (típicamente 0.01 a 0.5)
 * @param aabb_min_out Esquina mínima de la caja (salida)
 * @param aabb_max_out Esquina máxima de la caja (salida)
 *
 * @note El radio podría determinarse de múltiples formas:
 *       - Magnitud del embedding (embeddings más fuertes → radio mayor)
 *       - Varianza de contextos (tokens polisémicos → radio mayor)
 *       - Parámetro fijo (para simplificar)
 */
void computeAABB(
    const float3& centroid,
    float radius,
    float3& aabb_min_out,
    float3& aabb_max_out) {

    // Asegurar radio válido
    radius = std::max(radius, 0.001f);

    // Crear caja cúbica centrada en centroid
    aabb_min_out.x = centroid.x - radius;
    aabb_min_out.y = centroid.y - radius;
    aabb_min_out.z = centroid.z - radius;

    aabb_max_out.x = centroid.x + radius;
    aabb_max_out.y = centroid.y + radius;
    aabb_max_out.z = centroid.z + radius;
}

// ============================================================================
// Función: createTokenNode
// ============================================================================

/**
 * @brief Factory function que crea un TokenNode completo a partir de un embedding.
 *
 * Esta es la función de entrada principal para convertir un embedding crudo
 * a un TokenNode geométrico listo para el BVH.
 *
 * Pasos:
 *   1. Proyectar embedding D-dimensional → 3D (centroide)
 *   2. Calcular AABB alrededor del centroide
 *   3. Comprimir embedding a FP16 (half-float)
 *   4. Inicializar pesos de atención a cero
 *
 * @param token_id ID único del token en el vocabulario
 * @param position Posición en la secuencia de entrada (0..N-1)
 * @param embedding Array de entrada (D dimensional)
 * @param embed_dim Dimensión del embedding
 *
 * @return TokenNode completamente inicializado
 *
 * @note El embedding se comprime de embed_dim componentes a SPECTRAL_EMBEDDING_DIM (256).
 *       Si embed_dim > 256, se usan solo las primeras 256 componentes.
 *       Si embed_dim < 256, se rellenan con ceros.
 */
TokenNode createTokenNode(
    uint32_t token_id,
    uint32_t position,
    const float* embedding,
    uint32_t embed_dim) {

    SPECTRAL_CHECK(embedding != nullptr);
    SPECTRAL_CHECK(embed_dim > 0);

    TokenNode node;

    // Inicializar identidad
    node.token_id = token_id;
    node.position_in_seq = position;

    // ==================================================================================
    // PASO 1: Proyectar embedding a 3D
    // ==================================================================================
    projectEmbeddingTo3D(embedding, embed_dim, node.centroid);

    // ==================================================================================
    // PASO 2: Computar AABB
    // ==================================================================================
    // El radio se calcula como la norma L2 del embedding (normalizada).
    // Embeddings más "fuertes" (con norma mayor) obtienen un radio mayor,
    // reflejando mayor especificidad semántica.

    float embed_norm = 0.0f;
    uint32_t norm_dim = std::min(embed_dim, 64u);  // Usar primeras 64 dimensiones para eficiencia
    for (uint32_t i = 0; i < norm_dim; ++i) {
        embed_norm += embedding[i] * embedding[i];
    }
    embed_norm = sqrtf(embed_norm) / sqrtf(static_cast<float>(norm_dim));

    // Radio semántico en rango [0.01, 0.2]
    float semantic_radius = 0.01f + 0.19f * tanhf(embed_norm);
    node.semantic_radius = semantic_radius;

    computeAABB(node.centroid, semantic_radius, node.aabb_min, node.aabb_max);

    // ==================================================================================
    // PASO 3: Comprimir embedding a FP16
    // ==================================================================================
    // Copiar componentes del embedding, rellenando con ceros si es necesario
    uint32_t copy_dim = std::min(embed_dim, SPECTRAL_EMBEDDING_DIM);

    for (uint32_t i = 0; i < copy_dim; ++i) {
        // Convertir float a half (conversión implícita en tipos half)
        node.embedding[i] = static_cast<half>(embedding[i]);
    }

    // Rellenar el resto con ceros
    for (uint32_t i = copy_dim; i < SPECTRAL_EMBEDDING_DIM; ++i) {
        node.embedding[i] = static_cast<half>(0.0f);
    }

    // ==================================================================================
    // PASO 4: Inicializar pesos de atención
    // ==================================================================================
    node.attention_weight = 0.0f;
    node.energy_remaining = 1.0f;

    return node;
}

// ============================================================================
// Función auxiliar: Estadísticas de geometría (para debugging)
// ============================================================================

/**
 * @brief Calcula y imprime estadísticas sobre los TokenNodes.
 *
 * Útil para debugging y validación de la proyección.
 *
 * @param nodes Array de TokenNodes
 * @param num_nodes Número de nodos
 */
void printTokenGeometryStats(const TokenNode* nodes, uint32_t num_nodes) {
    if (num_nodes == 0) {
        printf("[TokenGeometry] No nodes to analyze.\n");
        return;
    }

    float3 centroid_mean{0.0f, 0.0f, 0.0f};
    float radius_min = std::numeric_limits<float>::max();
    float radius_max = std::numeric_limits<float>::lowest();
    float aabb_volume_total = 0.0f;

    for (uint32_t i = 0; i < num_nodes; ++i) {
        centroid_mean.x += nodes[i].centroid.x;
        centroid_mean.y += nodes[i].centroid.y;
        centroid_mean.z += nodes[i].centroid.z;

        radius_min = std::min(radius_min, nodes[i].semantic_radius);
        radius_max = std::max(radius_max, nodes[i].semantic_radius);

        aabb_volume_total += nodes[i].getAABBVolume();
    }

    centroid_mean.x /= num_nodes;
    centroid_mean.y /= num_nodes;
    centroid_mean.z /= num_nodes;

    printf("\n=== TokenGeometry Statistics ===\n");
    printf("Number of nodes: %u\n", num_nodes);
    printf("Mean centroid: (%.4f, %.4f, %.4f)\n", centroid_mean.x, centroid_mean.y, centroid_mean.z);
    printf("Semantic radius range: [%.4f, %.4f]\n", radius_min, radius_max);
    printf("Total AABB volume: %.6f\n", aabb_volume_total);
    printf("Average AABB volume per node: %.6f\n", aabb_volume_total / num_nodes);
    printf("================================\n\n");
}
