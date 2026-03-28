/**
 * @file spectral_ray.h
 * @brief Arquitectura Ultra LiquidBit: Codificación Espectral + Refracción Prismática
 *
 * VISIÓN GENERAL:
 * ===============
 * Spectral Ray implementa la "Idea 3" de LiquidBit: La fusión de tres mecanismos:
 *
 *   1. CODIFICACIÓN ESPECTRAL:       Los rayos llevan un "color" f ∈ ℝ^k (vector de
 *                                    contexto conversacional) que modula el mecanismo
 *                                    de atención. Dimensión: k = 64 (SPECTRAL_DIM).
 *
 *   2. REFRACCIÓN PRISMÁTICA:        Las esferas actúan como PRISMAS ÓPTICOS semánticos.
 *                                    El índice de refracción depende del color del rayo:
 *                                    n(esfera, f) = σ(W_dispersion · f)
 *                                    Este índice determina el ángulo de refracción.
 *
 *   3. SELECCIÓN DE MATRICES:        El ángulo de refracción selecciona automáticamente
 *                                    qué sub-bloque de matrices cargar. Resuelve
 *                                    POLISEMIA sin duplicar matrices completas.
 *
 * ANALOGÍA ÓPTICA:
 * ================
 * - El prompt emite rayos coloreados (contexto espectral) desde su posición semántica.
 * - Cada esfera REFRACTA el rayo según su índice de refracción (que depende del color).
 * - El rayo refractado intersecta esferas secundarias, refinando el contexto.
 * - En la hoja final, el ángulo de refracción selecciona el bloque de matrices a usar.
 *
 * COMPLEJIDAD:
 * ============
 * - Traversal del árbol: O(N log N) como en Alpha BSH
 * - Cálculo de refracción por esfera: O(k) donde k = 64 (SPECTRAL_DIM)
 * - Selección de matriz: O(log M) donde M = MAX_DISPERSION_CONTEXTS (≤ 8)
 * - Total: O(N log N · k) pero k es pequeño (64 floats ≈ 256 bytes)
 *
 * PROBLEMA QUE RESUELVE:
 * ======================
 * Polisemia (palabras con múltiples significados). Por ejemplo:
 *   - "bank" puede significar {dinero, orilla de río, inclinar-avión}
 *   - En un Transformer estándar: una ÚNICA representación matricial para todos los casos
 *   - En Spectral Ray: el contexto conversacional ("bank account", "river bank", "rolling bank")
 *     emite rayos con diferentes "colores" espectrales, seleccionando automáticamente
 *     el bloque de matrices correcto. Sin duplicar matrices completas (solo ~8 sub-bloques).
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#pragma once
#ifndef LIQUIDBIT_SPECTRAL_RAY_H_
#define LIQUIDBIT_SPECTRAL_RAY_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <array>
#include <vector>

// ============================================================================
// CONSTANTES DE CONFIGURACIÓN GLOBAL
// ============================================================================

/// Dimensión del vector espectral (color del rayo). Representa k componentes
/// del contexto conversacional. Típicamente suficiente para discriminar 8-16
/// contextos semánticos distintos (bank=money, bank=river, bank=tilt, etc.)
constexpr uint32_t SPECTRAL_DIM = 64;

/// Número máximo de "colores" o contextos distintos que una esfera puede
/// discriminar mediante refracción. Corresponde a MAX_DISPERSION_CONTEXTS
/// sub-bloques de matrices distintos. Típicamente: 4-8.
constexpr uint32_t MAX_DISPERSION_CONTEXTS = 8;

/// Epsilon para evitar división por cero en la ley de Snell.
/// Usado en cos_t^2 = 1 - n_ratio^2 * (1 - cos_i^2) para detectar
/// reflexión total interna (cuando cos_t^2 < SNELL_EPSILON).
constexpr float SNELL_EPSILON = 1e-6f;

/// Profundidad máxima del árbol BSH prismático (análogo a ALPHA_BSH_MAX_DEPTH).
constexpr uint32_t SPECTRAL_MAX_DEPTH = 20;

/// Número máximo de hijos por nodo (árbol octal, análogo a Alpha BSH).
constexpr uint32_t SPECTRAL_MAX_CHILDREN = 8;

/// Coeficiente de absorción/decay espectral. La energía de los rayos decae
/// exponencialmente: E(d) = E₀ * exp(-SPECTRAL_LAMBDA_DECAY * d)
constexpr float SPECTRAL_LAMBDA_DECAY = 0.1f;

/// Umbral mínimo de energía para continuar traversal.
constexpr float SPECTRAL_ENERGY_THRESHOLD = 0.01f;

// ============================================================================
// ESTRUCTURA: SpectralContext
// ============================================================================

/**
 * @struct SpectralContext
 * @brief El "color" del rayo — vector de contexto conversacional.
 *
 * En la analogía óptica, cada rayo lleva una "firma de color" que representa
 * el estado conversacional actual (tema, tono, nivel técnico, etc.).
 * Este vector modula cómo el rayo interactúa con los prismas (esferas).
 *
 * REPRESENTACIÓN:
 * - color_vector[SPECTRAL_DIM] en FP16: compresión del contexto a 64 dims.
 * - color_magnitude: norma del vector (intensidad del contexto).
 * - dominant_context_id: qué contexto semántico es dominante (0=técnico, 1=artístico, etc.)
 * - context_confidence: cuánta certeza hay en el contexto dominante [0.0, 1.0].
 *
 * WORKFLOW:
 *   1. embedding_input (D=768 o 4096) → pasa por matriz W_spectral [D×64]
 *   2. Resultado normalizado → SpectralContext::color_vector
 *   3. dominant_context_id se calcula como argmax de color_vector
 */
struct SpectralContext {
    /// Vector de contexto en FP16 (comprimido a 64 dimensiones).
    /// Normalizado: ||color_vector||₂ = 1.0 (mediante normalize()).
    half color_vector[SPECTRAL_DIM];

    /// Magnitud del vector antes de normalización.
    /// Indica la "intensidad" del contexto. Rango típico: [0.5, 2.0].
    float color_magnitude;

    /// Índice del contexto semántico dominante.
    /// Calculado como: argmax(abs(color_vector[i])) para i = 0..SPECTRAL_DIM-1.
    /// Rango: [0, SPECTRAL_DIM-1]. Usado para logs y debugging.
    uint32_t dominant_context_id;

    /// Confianza en el contexto dominante, en rango [0.0, 1.0].
    /// Calculada como: abs(color_vector[dominant_context_id]) / color_magnitude.
    /// 1.0 = contexto claro, 0.5 = ambiguo (polisemia).
    float context_confidence;

    /// Constructor por defecto.
    __host__ __device__ SpectralContext()
        : color_magnitude(1.0f),
          dominant_context_id(0),
          context_confidence(1.0f) {
        for (uint32_t i = 0; i < SPECTRAL_DIM; ++i) {
            color_vector[i] = __float2half(0.0f);
        }
    }

    /**
     * @brief Normaliza el vector de color en el dispositivo.
     *
     * Operación:
     *   1. Calcula ||color_vector||₂ (L2 norm)
     *   2. Divide cada componente por la norma
     *   3. Actualiza color_magnitude
     *
     * NOTA: Si la norma es muy pequeña (< SNELL_EPSILON), deja el vector sin cambios
     * para evitar NaN.
     *
     * Típicamente se llama tras crear SpectralContext desde embedding.
     *
     * @complexity O(SPECTRAL_DIM) = O(64)
     */
    __device__ inline void normalize() {
        // Calcula L2 norm
        float norm_sq = 0.0f;
        for (uint32_t i = 0; i < SPECTRAL_DIM; ++i) {
            float val = __half2float(color_vector[i]);
            norm_sq += val * val;
        }

        float norm = sqrtf(norm_sq);
        if (norm < SNELL_EPSILON) {
            norm = 1.0f; // Evita división por cero
        }

        // Normaliza
        for (uint32_t i = 0; i < SPECTRAL_DIM; ++i) {
            float val = __half2float(color_vector[i]);
            color_vector[i] = __float2half(val / norm);
        }

        color_magnitude = norm;

        // Actualiza dominant_context_id y context_confidence
        float max_abs_val = 0.0f;
        uint32_t max_idx = 0;
        for (uint32_t i = 0; i < SPECTRAL_DIM; ++i) {
            float val = fabsf(__half2float(color_vector[i]));
            if (val > max_abs_val) {
                max_abs_val = val;
                max_idx = i;
            }
        }
        dominant_context_id = max_idx;
        context_confidence = (norm > SNELL_EPSILON) ? (max_abs_val / norm) : 0.5f;
    }
};

// ============================================================================
// ESTRUCTURA: PrismaticSphere
// ============================================================================

/**
 * @struct PrismaticSphere
 * @brief Nodo del árbol que actúa como PRISMA ÓPTICO semántico.
 *
 * A diferencia de SemanticSphereAlpha (que solo tiene geometría),
 * PrismaticSphere añade:
 *   1. W_dispersion[SPECTRAL_DIM]: vector de pesos para calcular índice de refracción
 *   2. Múltiples matrix_block_ids (uno por contexto/ángulo refractado)
 *   3. refraction_thresholds: ángulos que mapean a cada bloque
 *   4. wormhole_target_id: "atajo óptico" a otra esfera (O(1) traversal jump)
 *
 * FÍSICA INTERPRETADA:
 * - Centro y radio: geometría de la esfera (como en Alpha BSH)
 * - W_dispersion: "material" del prisma. Aprendido en training.
 * - base_refractive_index: índice base (≈ 1.0 en vacío)
 * - Cuando un rayo golpea la esfera:
 *     n(esfera, ray_color) = base_refractive_index + sigmoid(dot(W_dispersion, color_vector))
 *   Este índice determina el ángulo de refracción (ley de Snell).
 * - El ángulo refractado selecciona matrix_block_id dentro de refraction_thresholds[].
 */
struct PrismaticSphere {
    /// Centro 3D de la esfera en el espacio semántico (como Alpha BSH).
    float3 center;

    /// Radio de colisión (como Alpha BSH).
    float radius;

    /// Identificador único de la esfera.
    uint32_t sphere_id;

    /// True si es una hoja del árbol (tiene matrices); false si es nodo interno.
    bool is_leaf;

    /// Vector de pesos de dispersión [SPECTRAL_DIM = 64].
    /// Aprendido en training mediante backprop.
    /// Interpretación: Cómo el vector espectral modula el índice de refracción.
    /// Fórmula: n_delta = sigmoid(dot(W_dispersion, color_vector))
    half W_dispersion[SPECTRAL_DIM];

    /// Índice de refracción base (sin contexto espectral).
    /// Típicamente ≈ 1.0 (vacío) a 1.5 (vidrio).
    /// En esferas de aire (context-insensitive): base_refractive_index ≈ 1.0 (sin dispersión).
    float base_refractive_index;

    /// IDs de los bloques de matrices para cada contexto/ángulo posible.
    /// Tamaño: MAX_DISPERSION_CONTEXTS (8).
    /// Ejemplo: matrix_block_ids[0]=100 (contexto técnico), matrix_block_ids[1]=101 (artístico), etc.
    /// Un bloque contiene sub-matrices W1, b1, W2, b2 de menor tamaño que las completas.
    uint32_t matrix_block_ids[MAX_DISPERSION_CONTEXTS];

    /// Umbrales de ángulo de refracción que mapean a cada bloque.
    /// Tamaño: MAX_DISPERSION_CONTEXTS.
    /// Ejemplo: refraction_thresholds[0]=10° (ángulo bajo = contexto técnico),
    ///          refraction_thresholds[1]=45° (ángulo alto = contexto artístico).
    /// Búsqueda lineal durante selectMatrixBlock(): O(MAX_DISPERSION_CONTEXTS) = O(8).
    float refraction_thresholds[MAX_DISPERSION_CONTEXTS];

    /// Número real de bloques de matrices distintos para esta esfera.
    /// Rango: [0, MAX_DISPERSION_CONTEXTS].
    /// Si num_matrix_blocks == 0: esfera es context-insensitive (no refracta).
    /// Si num_matrix_blocks == 1: esfera es context-insensitive (una única matriz).
    uint32_t num_matrix_blocks;

    /// ID de la esfera destino del wormhole (atajo óptico).
    /// Si == UINT32_MAX: no hay wormhole.
    /// Si != UINT32_MAX: rayo puede "saltar" directamente a esta esfera
    ///   en O(1), saltando la búsqueda geométrica del árbol BSH.
    /// Caso de uso: tokens muy relacionados semánticamente (ej. "bank" → "money").
    uint32_t wormhole_target_id;

    /// Profundidad en el árbol (0 = raíz).
    uint32_t depth;

    /// ID del padre (0 si es raíz).
    uint32_t parent_id;

    /// Array de IDs de los hijos.
    std::array<uint32_t, SPECTRAL_MAX_CHILDREN> children_ids;

    /// Número real de hijos.
    uint32_t num_children;

    /// Constructor por defecto.
    __host__ PrismaticSphere()
        : center({0.0f, 0.0f, 0.0f}),
          radius(0.0f),
          sphere_id(0),
          is_leaf(false),
          base_refractive_index(1.0f),
          num_matrix_blocks(0),
          wormhole_target_id(UINT32_MAX),
          depth(0),
          parent_id(0),
          num_children(0) {
        for (uint32_t i = 0; i < SPECTRAL_DIM; ++i) {
            W_dispersion[i] = __float2half(0.0f);
        }
        for (uint32_t i = 0; i < MAX_DISPERSION_CONTEXTS; ++i) {
            matrix_block_ids[i] = UINT32_MAX;
            refraction_thresholds[i] = 0.0f;
        }
        children_ids.fill(0);
    }

    /**
     * @brief Calcula el índice de refracción de esta esfera dado el contexto espectral.
     *
     * FÓRMULA:
     *   dot_product = Σ(W_dispersion[i] * color_vector[i])  [i=0..SPECTRAL_DIM-1]
     *   delta_n = sigmoid(dot_product) = 1 / (1 + exp(-dot_product))
     *   n_final = base_refractive_index + delta_n
     *
     * INTERPRETACIÓN FÍSICA:
     * - dot_product > 0: espectro "alineado" con el prisma → n aumenta
     * - dot_product < 0: espectro "desalineado" → n disminuye
     * - sigmoid() siempre produce [0.0, 1.0], por eso delta_n ∈ [0, 1]
     * - n_final ∈ [base_refractive_index, base_refractive_index + 1]
     *
     * @param ctx SpectralContext del rayo (debe estar normalizado).
     *
     * @return float: índice de refracción en rango [base_refractive_index, base_refractive_index + 1.0].
     *
     * @complexity O(SPECTRAL_DIM) = O(64)
     */
    __device__ inline float computeRefractiveIndex(const SpectralContext& ctx) const {
        // Calcula dot product
        float dot = 0.0f;
        for (uint32_t i = 0; i < SPECTRAL_DIM; ++i) {
            float w = __half2float(W_dispersion[i]);
            float c = __half2float(ctx.color_vector[i]);
            dot += w * c;
        }

        // Sigmoid: σ(x) = 1 / (1 + exp(-x))
        float sigmoid_val = 1.0f / (1.0f + expf(-dot));

        // Índice final: base + dispersión
        float n = base_refractive_index + sigmoid_val;
        return n;
    }

    /**
     * @brief Selecciona el ID del bloque de matrices basado en el ángulo de refracción.
     *
     * ALGORITMO:
     *   1. Itera refraction_thresholds[0..num_matrix_blocks-1]
     *   2. Busca el threshold más bajo que sea >= refraction_angle
     *   3. Retorna el matrix_block_ids correspondiente
     *   4. Si refraction_angle > threshold máximo: retorna el último bloque
     *
     * INTERPRETACIÓN:
     * - Ángulo bajo (cercano a la normal): bloque para contexto "esperado"
     * - Ángulo alto (alejado de la normal): bloque para contexto "sorpresa" (polisemia)
     *
     * EJEMPLO:
     *   num_matrix_blocks = 3
     *   refraction_thresholds[0] = 15° (contexto #0: dinero/bank)
     *   refraction_thresholds[1] = 35° (contexto #1: rio/bank)
     *   refraction_thresholds[2] = 60° (contexto #2: inclinación/bank)
     *
     *   Si refraction_angle = 20°:
     *     - 20° > 15° y 20° <= 35° → retorna matrix_block_ids[1]
     *
     * @param refraction_angle Ángulo de refracción en grados [0°, 90°].
     *
     * @return uint32_t: matrix_block_id para este ángulo (UINT32_MAX si error).
     *
     * @complexity O(MAX_DISPERSION_CONTEXTS) = O(8) búsqueda lineal.
     */
    __device__ inline uint32_t selectMatrixBlock(float refraction_angle) const {
        // Si no hay bloques, retorna inválido
        if (num_matrix_blocks == 0) {
            return UINT32_MAX;
        }

        // Búsqueda: encuentra el threshold más cercano (o el primero si angle <= threshold[0])
        for (uint32_t i = 0; i < num_matrix_blocks; ++i) {
            if (refraction_angle <= refraction_thresholds[i]) {
                return matrix_block_ids[i];
            }
        }

        // Si refraction_angle supera todos los thresholds, retorna el último bloque
        return matrix_block_ids[num_matrix_blocks - 1];
    }
};

// ============================================================================
// ESTRUCTURA: PrismaticRay
// ============================================================================

/**
 * @struct PrismaticRay
 * @brief Rayo completo con color espectral que traversa el árbol prismático.
 *
 * Análogo a AlphaRayPayload, pero con capacidad de "color" (contexto espectral).
 *
 * WORKFLOW:
 *   1. Rayo se emite desde origin con direction y energy=1.0
 *   2. Ray context es el "color" del rayo (SpectralContext normalizado)
 *   3. Atraviesa árbol de esferas prismáticas
 *   4. En cada golpe: calcula índice de refracción, ley de Snell, nuevas refraction_angle
 *   5. Selecciona matrix_block_id según refraction_angle final
 *   6. Almacena resultado en selected_matrix_block_id y final_refraction_angle
 */
struct PrismaticRay {
    /// Origen del rayo en el espacio 3D.
    float3 origin;

    /// Dirección del rayo (debe estar normalizada, ||direction||₂ = 1.0).
    float3 direction;

    /// Energía restante del rayo (1.0 inicial).
    /// Decae conforme golpea esferas: E → E * exp(-lambda_decay * distancia).
    /// Si energy < SPECTRAL_ENERGY_THRESHOLD: traversal termina.
    float energy;

    /// El "color" del rayo — contexto espectral conversacional.
    /// Permanece constante durante traversal (no cambia de color entre esferas).
    SpectralContext context;

    /// ID de la esfera actual (para debugging y traversal state).
    /// Actualizado conforme el rayo se mueve.
    uint32_t current_sphere_id;

    /// ID del bloque de matrices seleccionado por refracción final.
    /// Determinado en la esfera hoja basándose en final_refraction_angle.
    uint32_t selected_matrix_block_id;

    /// Ángulo de refracción en la esfera hoja (en grados, [0°, 90°]).
    /// Calculado por ley de Snell. Determina qué matrix_block_ids usar.
    float final_refraction_angle;

    /// Profundidad alcanzada en el árbol (0 = raíz).
    /// Métrica de rendimiento.
    uint32_t traversal_depth;

    /// Constructor por defecto.
    __host__ __device__ PrismaticRay()
        : origin({0.0f, 0.0f, 0.0f}),
          direction({0.0f, 0.0f, 1.0f}),
          energy(1.0f),
          current_sphere_id(0),
          selected_matrix_block_id(UINT32_MAX),
          final_refraction_angle(0.0f),
          traversal_depth(0) {}

    /**
     * @brief Implementa la ley de Snell vectorial 3D.
     *
     * FÓRMULA (refracción vectorial):
     *   cos_i = -dot(d_in, normal)                                    [ángulo incidente]
     *   discriminant = 1 - n_ratio² * (1 - cos_i²)
     *
     *   Si discriminant < 0: reflexión total interna → rebota (no refracta)
     *
     *   Si discriminant >= 0: refracta
     *     cos_t = sqrt(discriminant)
     *     d_out = n_ratio * d_in + (n_ratio * cos_i - cos_t) * normal
     *
     * PARÁMETROS:
     *   n_ratio = n_in / n_out (ratio de índices)
     *   Si n_ratio > 1 y cos_t² < 0: reflexión total interna
     *
     * CASO ESPECIAL:
     *   Si reflexión total interna: rayo "rebota" en la normal
     *   (usado para esferas con altos índices de refracción)
     *
     * @param normal Vector normal a la superficie (normalizado).
     * @param n_ratio Ratio de índices refractivos (n_in / n_out).
     *
     * @return float3: dirección refractada (normalizada).
     *
     * @complexity O(1)
     */
    __device__ inline float3 refract(const float3& normal, float n_ratio) const {
        // Ángulo incidente: cos(θ_i) = -dot(d_in, normal)
        float cos_i = -dot(direction, normal);

        // Evita acotamiento numérico
        cos_i = fmaxf(-1.0f, fminf(1.0f, cos_i));

        // Discriminante: cos(θ_t)² = 1 - n_ratio² * (1 - cos_i²)
        float discriminant = 1.0f - n_ratio * n_ratio * (1.0f - cos_i * cos_i);

        // Reflexión total interna: cos_t² < 0
        if (discriminant < -SNELL_EPSILON) {
            // Rebota: d_reflected = d_in - 2*(dot(d_in, normal))*normal
            float3 reflected = direction - 2.0f * cos_i * normal;
            return normalize(reflected);
        }

        // Refracción normal
        float cos_t = sqrtf(fmaxf(0.0f, discriminant));
        float3 refracted = n_ratio * direction + (n_ratio * cos_i - cos_t) * normal;
        return normalize(refracted);
    }
};

// ============================================================================
// ESTRUCTURA: SpectralAttentionResult
// ============================================================================

/**
 * @struct SpectralAttentionResult
 * @brief Resultado completo del mecanismo de atención espectral.
 *
 * Contiene:
 *   1. Esfera hoja encontrada
 *   2. Bloque de matrices seleccionado por refracción
 *   3. Índice de refracción en la hoja
 *   4. Ángulo de refracción (determinante para polisemia)
 *   5. Profundidad del traversal
 *   6. Si se usó wormhole (atajo)
 *   7. Energía final (para weight de atención)
 */
struct SpectralAttentionResult {
    /// ID de la esfera hoja encontrada (terminal node).
    /// UINT32_MAX indica "no se encontró esfera" (miss).
    uint32_t leaf_sphere_id;

    /// ID del bloque de matrices seleccionado.
    /// Determinado por selectMatrixBlock(refraction_angle_deg).
    /// Este bloque contiene sub-matrices W1, b1, W2, b2 para
    /// transformación de activaciones.
    uint32_t matrix_block_id;

    /// Índice de refracción calculado en la esfera hoja.
    /// n = base_refractive_index + sigmoid(dot(W_dispersion, color_vector))
    /// Rango: [base_refractive_index, base_refractive_index + 1.0]
    float refractive_index_at_leaf;

    /// Ángulo de refracción final en la esfera hoja (en grados).
    /// Rango: [0°, 90°].
    /// Determina qué matriz usar (via selectMatrixBlock).
    /// Valor alto (> 45°): polisemia / contexto sorpresa.
    /// Valor bajo (< 15°): contexto esperado.
    float refraction_angle_deg;

    /// Profundidad alcanzada en el árbol.
    /// Métrica de rendimiento: debe ser ~ log₂(num_spheres).
    uint32_t traversal_depth;

    /// True si se usó un wormhole para llegar a la hoja.
    /// Indica si fue un atajo directo vs traversal normal.
    bool used_wormhole;

    /// Energía total restante en la hoja.
    /// Indica confianza del resultado.
    /// Rango: [0.0, 1.0], donde 1.0 = máxima confianza.
    float total_energy_at_leaf;

    /// Constructor por defecto.
    SpectralAttentionResult()
        : leaf_sphere_id(UINT32_MAX),
          matrix_block_id(UINT32_MAX),
          refractive_index_at_leaf(1.0f),
          refraction_angle_deg(0.0f),
          traversal_depth(0),
          used_wormhole(false),
          total_energy_at_leaf(0.0f) {}
};

// ============================================================================
// FUNCIONES LIBRES (KERNELS Y HELPERS)
// ============================================================================

/**
 * @brief Implementa la ley de Snell vectorial 3D completa, con manejo
 * de reflexión total interna.
 *
 * CASO DE USO:
 *   Refracción de un rayo en una interfaz entre dos medios.
 *   Típicamente se llama desde snell_refract(d_in, normal, n_in, n_out).
 *
 * FÓRMULA:
 *   Véase PrismaticRay::refract() — lógica idéntica pero función libre.
 *
 * @param d_in Dirección incidente (normalizada).
 * @param normal Vector normal a la superficie (normalizado).
 * @param n_in Índice de refracción del medio incidente.
 * @param n_out Índice de refracción del medio saliente.
 *
 * @return float3: dirección refractada (o reflejada si reflexión total).
 *
 * @complexity O(1)
 */
__device__ inline float3 snell_refract(float3 d_in, float3 normal, float n_in, float n_out) {
    if (n_out < SNELL_EPSILON) {
        n_out = SNELL_EPSILON; // Evita división por cero
    }

    float n_ratio = n_in / n_out;

    float cos_i = -dot(d_in, normal);
    cos_i = fmaxf(-1.0f, fminf(1.0f, cos_i));

    float discriminant = 1.0f - n_ratio * n_ratio * (1.0f - cos_i * cos_i);

    if (discriminant < -SNELL_EPSILON) {
        // Reflexión total interna
        float3 reflected = d_in - 2.0f * cos_i * normal;
        return normalize(reflected);
    }

    float cos_t = sqrtf(fmaxf(0.0f, discriminant));
    float3 refracted = n_ratio * d_in + (n_ratio * cos_i - cos_t) * normal;
    return normalize(refracted);
}

/**
 * @brief Proyecta un embedding de entrada al espacio espectral (64 dimensiones).
 *
 * ALGORITMO (Host):
 *   1. Toma embedding [D] (ej. D=768 o 4096)
 *   2. Multiplica por matriz W_spectral [D × SPECTRAL_DIM]
 *   3. Resultado [SPECTRAL_DIM = 64]
 *   4. Normaliza (norm L2 = 1.0)
 *   5. Actualiza SpectralContext (dominant_context_id, confidence)
 *
 * TIPO DE DATOS:
 *   - embedding: puede ser FP32 o FP16
 *   - W_spectral: FP16 (para velocidad)
 *   - resultado: FP16 (SpectralContext::color_vector)
 *
 * @param embedding Embedding de entrada [D dimensiones], en host o device.
 * @param D Dimensión del embedding (768, 4096, etc.).
 * @param W_spectral Matriz de proyección [D × SPECTRAL_DIM] en FP16, en device.
 * @param out_ctx SpectralContext resultado (en device).
 *
 * @complexity O(D * SPECTRAL_DIM) = O(D * 64)
 *
 * @note Esta es una función HOST que copia embedding a GPU, lanza kernel,
 *       y copia resultado de vuelta. Para batch processing, usar kernel directo.
 */
__host__ void compute_spectral_color(
    const half* embedding,
    uint32_t D,
    const half* W_spectral,
    SpectralContext* out_ctx);

// ============================================================================
// CLASE: SpectralBSH
// ============================================================================

/**
 * @class SpectralBSH
 * @brief Árbol BSH prismático completo con refracción espectral.
 *
 * Responsabilidades:
 *   1. Construcción del árbol BSH de esferas prismáticas
 *   2. Gestión de la matriz W_spectral [D × 64] para proyección de embeddings
 *   3. Lanzamiento de rayos espectrales a través del árbol
 *   4. Cálculo de refracción y selección de matrices
 *   5. Gestión de wormholes (atajos ópticos)
 *
 * WORKFLOW TÍPICO:
 *   1. Constructor: aloca memoria GPU
 *   2. build(spheres, num_spheres): construye árbol y carga W_spectral
 *   3. encodeContext(embedding, D): proyecta embedding → SpectralContext
 *   4. trace(ray): lanza rayo espectral, retorna resultado
 *   5. Destructor: libera memoria GPU
 *
 * NOTA: Prototipo. Asume:
 *   - NVIDIA RTX 4090 / 5070 Ti (RT Cores)
 *   - CUDA Compute Capability >= sm_89
 */
class SpectralBSH {
public:
    // ========================================================================
    // CONSTRUCTOR / DESTRUCTOR
    // ========================================================================

    /**
     * @brief Constructor. Inicializa SpectralBSH.
     *
     * Alocaciones iniciales:
     *   - Buffers auxiliares para rayos y contextos
     *   - Estructuras de gestión interna
     *
     * El árbol real se construye en build().
     */
    __host__ SpectralBSH();

    /**
     * @brief Destructor. Libera toda memoria GPU.
     *
     * Responsable de:
     *   - cudaFree(d_W_spectral_)
     *   - cudaFree(d_spheres_)
     *   - cudaFree(d_results_buffer_)
     *   - Otros buffers auxiliares
     */
    __host__ ~SpectralBSH();

    // ========================================================================
    // CONSTRUCCIÓN DEL ÁRBOL
    // ========================================================================

    /**
     * @brief Construye el árbol BSH prismático a partir de un array de esferas.
     *
     * Algoritmo:
     *   1. Copia esferas a GPU (d_spheres_)
     *   2. Asigna relaciones padre-hijo por proximidad geométrica
     *   3. Valida conectividad y profundidad
     *   4. Carga matriz W_spectral desde host (si se proporciona)
     *   5. Valida que cada esfera hoja tenga matrix_block_ids válidos
     *
     * @param h_spheres Array host-side de PrismaticSphere.
     * @param num_spheres Número de esferas.
     * @param h_W_spectral Matriz de proyección [D × SPECTRAL_DIM] en FP16 (host).
     * @param embedding_dim Dimensión D del embedding original (768, 4096, etc.).
     *
     * @return true si construcción exitosa, false si error.
     *
     * @complexity O(N log N) amortizado (construcción de árbol).
     */
    __host__ bool build(
        const PrismaticSphere* h_spheres,
        uint32_t num_spheres,
        const half* h_W_spectral,
        uint32_t embedding_dim);

    // ========================================================================
    // PROYECCIÓN ESPECTRAL
    // ========================================================================

    /**
     * @brief Proyecta un embedding de contexto al espacio espectral.
     *
     * OPERACIÓN:
     *   embedding [D] → (W_spectral [D×64]) → color_vector [SPECTRAL_DIM]
     *   Normaliza color_vector: ||color_vector||₂ = 1.0
     *   Calcula dominant_context_id y context_confidence
     *
     * TÍPICO:
     *   - embedding: representación actual del prompt/token
     *   - Resultado: contexto que modula refracción en toda la traversal
     *
     * @param context_embedding Embedding de contexto (host-side FP32 o FP16) [D].
     * @param embedding_dim Dimensión D.
     * @param out_context SpectralContext resultado (device).
     *
     * @return true si proyección exitosa, false si error.
     *
     * @complexity O(D * SPECTRAL_DIM) = O(D * 64)
     */
    __host__ bool encodeContext(
        const float* context_embedding,
        uint32_t embedding_dim,
        SpectralContext* out_context);

    // ========================================================================
    // TRACING DE RAYOS
    // ========================================================================

    /**
     * @brief Lanza un rayo espectral a través del árbol prismático.
     *
     * ALGORITMO:
     *   1. Inicia desde ray.origin con ray.direction y ray.context
     *   2. Traversa árbol de esferas (BVH)
     *   3. En cada golpe: calcula índice de refracción, ley de Snell
     *   4. Sigue rayo refractado a esferas secundarias
     *   5. Decae energía conforme avanza
     *   6. En hoja terminal: calcula final_refraction_angle
     *   7. selectMatrixBlock(angle) → matrix_block_id
     *   8. Retorna SpectralAttentionResult
     *
     * COMPLEJIDAD:
     *   - Traversal del árbol: O(N log N) donde N = num_spheres
     *   - Cálculo de refracción por golpe: O(SPECTRAL_DIM) = O(64)
     *   - Total: O(log N * 64) típicamente
     *
     * @param ray PrismaticRay entrada (origin, direction, context, energy).
     * @param result SpectralAttentionResult salida (device).
     *
     * @return true si trace exitoso (ray hit), false si miss.
     */
    __host__ bool trace(
        const PrismaticRay& ray,
        SpectralAttentionResult* result);

    // ========================================================================
    // GESTIÓN DE WORMHOLES
    // ========================================================================

    /**
     * @brief Obtiene el target de un wormhole si existe.
     *
     * Caso de uso: tokens altamente relacionados pueden tener wormholes
     * que permiten saltos O(1) en lugar de traversal normal.
     *
     * Ejemplo: "bank" puede tener wormholes a {money, river, tilt}
     * dependiendo del contexto espectral.
     *
     * @param sphere_id ID de la esfera.
     *
     * @return ID de la esfera destino (UINT32_MAX si no hay wormhole).
     *
     * @complexity O(1)
     */
    __host__ uint32_t getWormhole(uint32_t sphere_id) const;

    // ========================================================================
    // PROFILING Y DEBUGGING
    // ========================================================================

    /**
     * @brief Obtiene estadísticas del árbol.
     *
     * Retorna string con:
     *   - Número de esferas
     *   - Profundidad máxima
     *   - Número de wormholes
     *   - Memoria GPU usada
     *
     * @return std::string formateado.
     */
    __host__ std::string getStats() const;

private:
    // ========================================================================
    // MIEMBROS PRIVADOS
    // ========================================================================

    /// Matriz de proyección embeddings → espacio espectral [D × SPECTRAL_DIM].
    /// En FP16 (device memory).
    /// Tamaño: embedding_dim * SPECTRAL_DIM * sizeof(half)
    half* d_W_spectral_;

    /// Dimensión del embedding original (768, 4096, etc.).
    uint32_t embedding_dim_;

    /// Array GPU de esferas prismáticas (PrismaticSphere*).
    PrismaticSphere* d_spheres_;

    /// Número de esferas construidas.
    uint32_t num_spheres_;

    /// Buffer auxiliar para resultados de tracing (device).
    SpectralAttentionResult* d_results_buffer_;

    /// Tabla de wormholes: sphere_id → target_sphere_id.
    /// std::vector<uint32_t> wormhole_targets_;
    /// (Puntero GPU)
    uint32_t* d_wormhole_targets_;

    /// Estadísticas
    struct {
        uint32_t num_traces;
        uint32_t num_hits;
        uint32_t total_depth;
        uint32_t wormhole_uses;
    } stats_;

    // ========================================================================
    // MÉTODOS PRIVADOS HELPER
    // ========================================================================

    /**
     * @brief Asigna relaciones padre-hijo en el árbol.
     *
     * Algoritmo: Para cada esfera, busca las SPECTRAL_MAX_CHILDREN
     * esferas más cercanas y las designa como hijos.
     *
     * @param h_spheres Array host-side.
     * @param num_spheres Número de esferas.
     *
     * @return true si exitoso.
     *
     * @complexity O(N² log N)
     */
    __host__ bool assignParentChildRelationships(
        PrismaticSphere* h_spheres,
        uint32_t num_spheres);

    /**
     * @brief Valida la estructura del árbol.
     *
     * Checks:
     *   - Profundidad <= SPECTRAL_MAX_DEPTH
     *   - Cada nodo <= SPECTRAL_MAX_CHILDREN hijos
     *   - parent_id/children_ids consistentes
     *   - Solo hojas tienen matrix_block_ids válidos
     *   - W_dispersion no es NaN
     *
     * @return true si válido.
     */
    __host__ bool validateTreeStructure() const;
};

// ============================================================================
// FUNCIONES LIBRES (HELPERS Y LOGGING)
// ============================================================================

/**
 * @brief Imprime resultado de atención espectral de forma legible.
 *
 * Formato:
 *   ===== SPECTRAL ATTENTION RESULT =====
 *   Leaf Sphere ID: 42
 *   Matrix Block ID: 101
 *   Refraction Index: 1.45
 *   Refraction Angle: 22.5°
 *   Traversal Depth: 5
 *   Used Wormhole: false
 *   Energy at Leaf: 0.87
 *   =====================================
 *
 * @param result SpectralAttentionResult a imprimir.
 */
void printSpectralResult(const SpectralAttentionResult& result);

/**
 * @brief Convierte ángulo de refracción a "confianza de polisemia".
 *
 * IDEA:
 *   Ángulo bajo (< 15°) = contexto claro, sin ambigüedad
 *   Ángulo alto (> 45°) = contexto ambiguo, posible polisemia
 *
 * FÓRMULA:
 *   ambiguity = (refraction_angle / 90°) clamped a [0, 1]
 *   clarity = 1.0 - ambiguity
 *
 * @param refraction_angle Ángulo en grados [0, 90].
 *
 * @return float: clarity score [0.0, 1.0].
 */
__host__ __device__ inline float refraction_angle_to_clarity(float refraction_angle) {
    float ambiguity = fminf(1.0f, refraction_angle / 90.0f);
    return 1.0f - ambiguity;
}

#endif // LIQUIDBIT_SPECTRAL_RAY_H_
