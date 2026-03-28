/**
 * @file optical_attention.h
 * @brief Interfaz de mecanismo de atención óptica acelerada por ray tracing
 *
 * Define la configuración y los kernels OptiX para ejecutar el mecanismo
 * de atención óptica O(N log N) basado en ray tracing contra el BVH semántico.
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#pragma once

#include "token_geometry.h"
#include "semantic_bvh.h"
#include <cuda_runtime.h>
#include <optix.h>
#include <cstdint>
#include <string>

// ============================================================================
// Configuración de atención óptica
// ============================================================================

/**
 * @struct AttentionConfig
 * @brief Hiperparámetros del mecanismo de atención óptica.
 *
 * Controla el comportamiento del ray tracing y el cálculo de pesos
 * de atención en los kernels OptiX.
 */
struct AttentionConfig {
    /**
     * @brief Número de rayos a emitir por token query.
     *
     * Análogo a heads en Multi-Head Attention.
     * Rango típico: 64-4096.
     * Mayor num_rays → mayor cobertura del espacio semántico.
     * Coste: O(num_rays * log N) intersecciones de rayos.
     */
    uint32_t num_rays = 256;

    /**
     * @brief Coeficiente de absorción semántica (λ en la fórmula de decay).
     *
     * Controla qué tan rápido decae el peso de atención con la distancia:
     *   attention = E₀ · exp(-λ_decay · d_semantic)
     *
     * Valores típicos:
     *   - 0.05f: decay lento, tokens lejanos aún contribuyen
     *   - 0.1f: decay moderado (default, balanceado)
     *   - 0.5f: decay rápido, solo tokens cercanos importan
     *
     * Rango: (0.0, ∞), pero prácticamente [0.01f, 1.0f].
     */
    float lambda_decay = 0.1f;

    /**
     * @brief Número máximo de rebotes del rayo.
     *
     * Un rayo puede rebotar en múltiples tokens antes de terminar
     * (similar a multi-bounce ray tracing en gráficos).
     *
     * max_bounces = 1: cada rayo golpea 1 token máximo (default).
     * max_bounces > 1: el rayo continúa si energy_remaining > threshold.
     *
     * Rango típico: 1-8.
     * Nota: El mayor factor de coste, afecta linealmente a #intersecciones.
     */
    uint32_t max_bounces = 1;

    /**
     * @brief Umbral de energía mínima para terminar el rayo.
     *
     * Cuando energy_remaining < min_energy_threshold, el rayo termina
     * su traversal aunque queden bounces disponibles.
     *
     * Rango: [0.0, 1.0].
     * Típico: 0.01f (termina cuando energía es <1% de la inicial).
     */
    float min_energy_threshold = 0.01f;

    /**
     * @brief Usar precisión completa (float32) en cálculos de atención.
     *
     * Si false: usar float16 (más rápido, menos memoria).
     * Si true: usar float32 (más preciso, más memoria).
     */
    bool use_full_precision = true;

    /**
     * @brief Normalizar pesos de atención tras acumulación.
     *
     * Si true: sum(attention_weights) = 1.0 para cada query.
     * Si false: pesos crudos (suma puede variar).
     */
    bool normalize_weights = true;

    /**
     * @brief Usar OptiX para ray tracing (vs CUDA puro).
     *
     * Si true: usar OptiX 8.x (más rápido, requiere hardware OptiX).
     * Si false: implementación CUDA pura (más lenta, más portable).
     */
    bool use_optix = true;
};

// ============================================================================
// Resultados de atención óptica
// ============================================================================

/**
 * @struct AttentionResult
 * @brief Resultado de una consulta de atención óptica.
 *
 * Contiene los top-K tokens más relevantes según ray tracing,
 * junto con sus pesos de atención normalizados.
 */
struct AttentionResult {
    /// Número máximo de resultados por consulta
    static constexpr uint32_t MAX_TOP_K = 64;

    /**
     * @brief Array de índices de tokens en el top-K.
     *
     * Contiene los token_ids de los tokens más relevantes encontrados
     * por los rayos. Los índices están ordenados por weight decreciente.
     *
     * Rango válido: [0, num_hits - 1].
     * Índices no utilizados: garbage (ignorar si idx >= num_hits).
     */
    uint32_t top_k_tokens[MAX_TOP_K];

    /**
     * @brief Pesos de atención normalizados para cada top-K token.
     *
     * attention_weights[i] = peso normalizado para top_k_tokens[i].
     * Suma: sum(attention_weights[0..num_hits-1]) ≈ 1.0 (si normalize=true).
     *
     * Rango: [0.0, 1.0] (tipicamente).
     */
    float attention_weights[MAX_TOP_K];

    /**
     * @brief Número real de hits (tokens encontrados).
     *
     * Puede ser < MAX_TOP_K si hay pocos tokens relevantes en la secuencia.
     * Rango: [0, MAX_TOP_K].
     */
    uint32_t num_hits;

    /**
     * @brief ID del rayo/query que generó este resultado.
     *
     * Usado para demultiplexar resultados cuando se procesan múltiples
     * rayos en paralelo. Rango: [0, num_rays - 1].
     */
    uint32_t ray_id;

    /**
     * @brief Energía restante del rayo tras último bounce (debug).
     *
     * Información de trazabilidad para debugging.
     * Rango: [0.0, 1.0].
     */
    float final_energy;
};

// ============================================================================
// Clase OpticalAttention: Gestor de atención óptica
// ============================================================================

/**
 * @class OpticalAttention
 * @brief Ejecutor de consultas de atención óptica usando ray tracing.
 *
 * Responsabilidades:
 * 1. Lanzar kernels OptiX para ray tracing contra el BVH
 * 2. Compilar programas OptiX (ray_generation, closest_hit, any_hit, miss)
 * 3. Gestionar buffers GPU para rayos y resultados
 * 4. Computar rayos de query desde tokens
 *
 * Workflow típico:
 * ```cpp
 * OpticalAttention attention;
 * SemanticRay* d_rays;
 * computeQueryRays(query_token, num_rays, d_rays);  // Host-side
 * AttentionResult* d_results;
 * attention.launch(d_rays, num_rays, bvh, d_results);  // Device
 * // Copiar d_results a host y procesar
 * ```
 *
 * @note La compilación del pipeline OptiX ocurre una sola vez en el constructor.
 * @note Los rayos y resultados deben estar en memoria GPU.
 * @note Seguro para múltiples llamadas a launch() con distintos rayos/BVHs.
 */
class OpticalAttention {
public:
    // ========================================================================
    // Constructores y destructores
    // ========================================================================

    /**
     * @brief Constructor.
     *
     * Inicializa OptiX (si use_optix=true) y compila el pipeline de shaders.
     * Puede ser costoso (~1-2 segundos si hay recompilación JIT).
     *
     * @param config Configuración de atención (hiperparámetros)
     *
     * @note Lanza una excepción o seta last_error_ si la inicialización falla.
     */
    explicit OpticalAttention(const AttentionConfig& config = AttentionConfig());

    /**
     * @brief Destructor.
     *
     * Libera recursos OptiX y memoria GPU interna.
     */
    ~OpticalAttention();

    // Prohibir copias
    OpticalAttention(const OpticalAttention&) = delete;
    OpticalAttention& operator=(const OpticalAttention&) = delete;

    // Permitir movimiento
    OpticalAttention(OpticalAttention&& other) noexcept;
    OpticalAttention& operator=(OpticalAttention&& other) noexcept;

    // ========================================================================
    // Lanzamiento de kernel de atención
    // ========================================================================

    /**
     * @brief Lanza el kernel OptiX de atención óptica.
     *
     * Ejecuta ray tracing para los rayos dados contra el BVH,
     * calculando pesos de atención para tokens relevantes.
     *
     * @param rays Array de SemanticRay en GPU (device pointer)
     * @param num_rays Número de rayos a procesar
     * @param bvh BVH semántico construido (debe ser válido)
     * @param results Array de AttentionResult en GPU (device pointer)
     *                Debe tener capacidad para num_rays resultados.
     *
     * @return true si el lanzamiento fue exitoso, false si hubo error.
     *         Consultar getLastError() para más detalles.
     *
     * @note Operación asincrónica (retorna después de enqueuing del kernel).
     *       Llamar a cudaDeviceSynchronize() o cudaStreamSynchronize()
     *       en el stream usado para asegurar completitud.
     *
     * @note El BVH debe haber sido construido exitosamente (bvh.isBuilt() == true).
     *
     * @note Coste: O(num_rays * log N) intersecciones ray-AABB.
     */
    bool launch(
        const SemanticRay* rays,
        uint32_t num_rays,
        const SemanticBVH& bvh,
        AttentionResult* results
    );

    // ========================================================================
    // Configuración y estado
    // ========================================================================

    /**
     * @brief Obtiene el último mensaje de error.
     *
     * @return String con descripción del error, o vacío si no hay error.
     *
     * @note Llamar después de launch() si retorna false.
     */
    std::string getLastError() const {
        return last_error_;
    }

    /**
     * @brief Actualiza la configuración de atención.
     *
     * @param config Nueva configuración
     *
     * @return true si la actualización fue exitosa.
     *
     * @note Algunos cambios pueden requerir recompilación del pipeline OptiX.
     */
    bool updateConfig(const AttentionConfig& config);

    /**
     * @brief Obtiene la configuración actual.
     *
     * @return Configuración de atención activa
     */
    const AttentionConfig& getConfig() const {
        return config_;
    }

    /**
     * @brief Comprueba si OptiX está disponible y compilado.
     *
     * @return true si el mecanismo está listo para launch()
     */
    bool isReady() const {
        return is_ready_;
    }

private:
    // ========================================================================
    // Miembros privados: configuración y estado OptiX
    // ========================================================================

    bool is_ready_ = false;

    AttentionConfig config_;

    // Pipeline OptiX
    OptixPipeline optix_pipeline_ = nullptr;
    OptixPipelineCompileOptions optix_compile_options_ = {};
    OptixProgramGroup optix_raygen_group_ = nullptr;
    OptixProgramGroup optix_hit_group_ = nullptr;
    OptixProgramGroup optix_miss_group_ = nullptr;

    // SBT (Shader Binding Table)
    CUdeviceptr d_sbt_buffer_ = 0;
    size_t sbt_size_ = 0;

    // Contexto OptiX
    OptixDeviceContext optix_context_ = nullptr;

    // Último error
    std::string last_error_;

    // ========================================================================
    // Métodos privados de inicialización
    // ========================================================================

    /**
     * @brief Inicializa OptiX si use_optix=true.
     *
     * @return true si exitoso
     */
    bool initializeOptix_();

    /**
     * @brief Compila los programas OptiX (raygen, closest_hit, any_hit, miss).
     *
     * @return true si exitoso
     */
    bool compileOptixPrograms_();

    /**
     * @brief Construye la Shader Binding Table (SBT).
     *
     * @return true si exitoso
     */
    bool buildSBT_();

    /**
     * @brief Libera recursos OptiX.
     */
    void cleanupOptix_();
};

// ============================================================================
// Funciones host para generación de rayos
// ============================================================================

/**
 * @brief Genera rayos de atención óptica desde un token query.
 *
 * @param query_token Token que actúa como fuente de rayos
 * @param num_rays Número de rayos a generar
 * @param output_rays Array de SemanticRay donde escribir los rayos generados
 *                    Debe tener capacidad para >= num_rays elementos.
 *
 * @note Función HOST (CPU-side). No requiere CUDA.
 * @note Los rayos están en space coordinates (float3) pero sin estar
 *       en GPU. Usar cudaMemcpy para transferirlos.
 *
 * Algoritmo:
 * 1. Inicializar origen = query_token.centroid
 * 2. Para cada rayo i:
 *    a. Calcular dirección pseudo-aleatoria basada en:
 *       - Dimensión dominante del embedding
 *       - Hash determinístico de (token_id, ray_id)
 *    b. Normalizar dirección
 *    c. Copiar query_token.embedding a ray.query_embedding
 *    d. Inicializar ray.energy = 1.0f
 *    e. Asignar ray.ray_id = i
 *
 * @see OpticalAttention::launch()
 */
__host__ void computeQueryRays(
    const TokenNode& query_token,
    uint32_t num_rays,
    SemanticRay* output_rays
);

/**
 * @brief Versión batched de computeQueryRays.
 *
 * @param query_tokens Array de tokens query
 * @param num_query_tokens Número de tokens query
 * @param rays_per_query Rayos por cada token query
 * @param output_rays Array de SemanticRay (tamaño: num_query_tokens * rays_per_query)
 *
 * @return Número total de rayos generados
 *
 * @note Útil para procesamiento de múltiples queries simultáneamente.
 */
__host__ uint32_t computeQueryRaysBatch(
    const TokenNode* query_tokens,
    uint32_t num_query_tokens,
    uint32_t rays_per_query,
    SemanticRay* output_rays
);

/**
 * @brief Copia resultados de atención desde GPU a CPU.
 *
 * @param d_results Array de AttentionResult en GPU (device pointer)
 * @param num_results Número de resultados a copiar
 * @param h_results Array de AttentionResult en CPU para destino
 *
 * @return true si exitoso
 *
 * @note Operación sincrónica (bloquea hasta completar).
 */
__host__ bool copyResultsToHost(
    const AttentionResult* d_results,
    uint32_t num_results,
    AttentionResult* h_results
);

// ============================================================================
// Utilidades de debugging
// ============================================================================

/**
 * @brief Imprime un resumen de un resultado de atención.
 *
 * @param result Resultado a imprimir
 * @param token_vocab Vocabulario de tokens para traducir IDs a strings (opcional)
 *
 * @note Función HOST (CPU-side). Útil para debugging.
 */
__host__ void printAttentionResult(
    const AttentionResult& result,
    const char** token_vocab = nullptr
);

#endif // LIQUIDBIT_OPTICAL_ATTENTION_H_
