/**
 * @file alpha_bsh.h
 * @brief Arquitectura Alpha BSH: Bounding Sphere Hierarchy + cuBLAS MatMul Selectivo
 *
 * VISIÓN GENERAL:
 * ===============
 * Alpha BSH es una evolución del modelo LiquidBit Zero-Matrix que introduce un
 * mecanismo de dos fases:
 *
 *   FASE A (Ray Tracing OptiX):     O(N log N) - Traversal del árbol BSH para
 *                                    encontrar la ESFERA SEMÁNTICA más relevante
 *                                    basándose en geometría 3D pura.
 *
 *   FASE B (cuBLAS MatMul):         O(M²) - Multiplicación de matrices FP16 de
 *                                    alta precisión, donde M es la dimensión de la
 *                                    esfera encontrada (típicamente 4096).
 *
 * La innovación clave es que la FASE B solo se ejecuta en UNA esfera (el contexto
 * más relevante), no en todas las esferas del árbol. Esto mantiene la complejidad
 * global O(N log N) + O(M²), donde M << N.
 *
 * ANALOGÍA CONCEPTUAL:
 * ====================
 * - FASE A: "¿Cuál es el área temática más relevante?" (busca por geometría)
 * - FASE B: "Ahora que sé qué área, realiza transformaciones de precisión alta" (MatMul)
 *
 * Este diseño aprovecha:
 *   1. RT Cores (NVIDIA OptiX) para búsqueda rápida O(log N)
 *   2. Tensor Cores (cuBLAS) para cálculo denso y preciso en la esfera activada
 *   3. Memoria GPU: El BSH pesa ~10-50 MB, mucho más pequeño que el KV Cache
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#pragma once

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cmath>
#include <array>
#include <vector>

// ============================================================================
// CONSTANTES DE CONFIGURACIÓN GLOBAL
// ============================================================================

/// Profundidad máxima del árbol BSH (para asegurar O(log N))
constexpr uint32_t ALPHA_BSH_MAX_DEPTH = 20;

/// Número máximo de hijos por nodo en el árbol octal (8 niños = árbol octal)
constexpr uint32_t ALPHA_BSH_MAX_CHILDREN = 8;

/// Dimensión máxima de las matrices FP16 en la esfera (4096x4096)
constexpr uint32_t ALPHA_MATRIX_BLOCK_DIM = 4096;

/// Umbral mínimo de energía del rayo para continuar traversal
constexpr float ALPHA_ENERGY_THRESHOLD = 0.01f;

/// Coeficiente de absorción semántica (decay exponencial de energía)
constexpr float ALPHA_LAMBDA_DECAY = 0.1f;

// ============================================================================
// ESTRUTURA: MatrixBlock
// ============================================================================

/**
 * @struct MatrixBlock
 * @brief Bloque de matrices FP16 para transformación de activaciones en la esfera.
 *
 * Representa dos capas densas transformadoras:
 *   - Layer 1: W1 [dim_in × hidden_dim], b1 [hidden_dim] + activación GELU
 *   - Layer 2: W2 [hidden_dim × dim_out], b2 [dim_out]
 *
 * Este bloque se carga lazy desde disco/host-pinned memory a VRAM solo cuando
 * la esfera correspondiente es seleccionada en la FASE A.
 */
struct MatrixBlock {
    /// @brief Puntero GPU a los pesos de la primera capa (W1).
    /// Tamaño: [dim_in × hidden_dim] en FP16 (half).
    /// Almacenamiento: device global memory, inicialmente nullptr hasta loadMatrixBlock().
    half* d_weights1;

    /// @brief Puntero GPU a los biases de la primera capa (b1).
    /// Tamaño: [hidden_dim] en FP16.
    half* d_biases1;

    /// @brief Puntero GPU a los pesos de la segunda capa (W2).
    /// Tamaño: [hidden_dim × dim_out] en FP16.
    half* d_weights2;

    /// @brief Puntero GPU a los biases de la segunda capa (b2).
    /// Tamaño: [dim_out] en FP16.
    half* d_biases2;

    /// @brief Dimensión de entrada del bloque (número de features de entrada).
    /// Típicamente: 768 (BERT-base) o 4096 (GPT-4 scale).
    uint32_t dim_in;

    /// @brief Dimensión de salida del bloque (número de features de salida).
    /// Típicamente igual a dim_in para transformadores.
    uint32_t dim_out;

    /// @brief Dimensión oculta de la capa intermedia.
    /// Típicamente: 4 * dim_in (e.g., 3072 para BERT).
    uint32_t hidden_dim;

    /// @brief Flag indicando si el bloque está actualmente cargado en VRAM.
    bool loaded;

    /// @brief Offset en el archivo de disco para carga lazy.
    /// Permite recuperar el bloque desde almacenamiento externo si es necesario.
    uint64_t disk_offset;

    /// @brief Constructor por defecto.
    __host__ MatrixBlock()
        : d_weights1(nullptr),
          d_biases1(nullptr),
          d_weights2(nullptr),
          d_biases2(nullptr),
          dim_in(0),
          dim_out(0),
          hidden_dim(0),
          loaded(false),
          disk_offset(0) {}
};

// ============================================================================
// ESTRUCTURA: SemanticSphereAlpha
// ============================================================================

/**
 * @struct SemanticSphereAlpha
 * @brief Nodo del árbol BSH representando una esfera semántica.
 *
 * Cada esfera es un nodo en el árbol jerárquico que:
 *   1. Ocupa una región del espacio semántico 3D (definida por centro y radio)
 *   2. Tiene hijos (otras esferas de menor escala, más específicas)
 *   3. Almacena un MatrixBlock (solo en hojas) para transformación de activaciones
 *   4. Lleva etiqueta semántica descriptiva (e.g., "QuantumPhysics", "Sentimentos")
 *
 * La traversal del BSH descarta rápidamente mitades del árbol gracias a pruebas
 * geométricas (rayo-esfera), logrando O(log N) complejidad.
 */
struct SemanticSphereAlpha {
    /// @brief Centroide 3D de la esfera en el espacio semántico.
    /// Preserva clústeres de embeddings similares.
    float3 center;

    /// @brief Radio de la esfera en el espacio 3D.
    /// Define el volumen de influencia semántica de este nodo.
    /// Cuanto más pequeño el radio, más específico el concepto.
    float radius;

    /// @brief Peso semántico de esta esfera (importancia/frecuencia).
    /// Rango: [0.0, 1.0].
    /// Usada para determinar sesgo hacia esferas comunes (ej., "The" vs raro).
    float semantic_weight;

    /// @brief Identificador único de la esfera en toda la estructura.
    /// Usada para indexación, debugging y trazabilidad.
    uint32_t sphere_id;

    /// @brief Profundidad de este nodo en el árbol BSH (raíz = 0).
    /// Limitada a ALPHA_BSH_MAX_DEPTH para garantizar O(log N).
    uint32_t depth;

    /// @brief ID del nodo padre (0 si es raíz).
    /// Permite reconstructión de paths desde hoja a raíz.
    uint32_t parent_id;

    /// @brief Array de IDs de los hijos de este nodo.
    /// Tamaño: ALPHA_BSH_MAX_CHILDREN.
    /// Los elementos sin usar (sin hijo) almacenan 0.
    /// Ejemplo: children_ids[0]=101, children_ids[1]=102, children_ids[2]=0 (vacío).
    std::array<uint32_t, ALPHA_BSH_MAX_CHILDREN> children_ids;

    /// @brief Número real de hijos (0..ALPHA_BSH_MAX_CHILDREN).
    /// Evita iterar sobre el array completo en traversal.
    uint32_t num_children;

    /// @brief True si es una hoja (terminal node).
    /// Solo los nodos hoja tienen MatrixBlock válido.
    bool is_leaf;

    /// @brief Etiqueta semántica legible del concepto (e.g., "QuantumPhysics").
    /// Máximo 64 caracteres para debugging y trazabilidad.
    /// Nullable: puede ser "" para nodos internos.
    char label[64];

    /// @brief Bloque de matrices FP16 para esta esfera.
    /// Válido solo si is_leaf == true.
    /// Contiene W1, b1, W2, b2 para la transformación de activaciones.
    MatrixBlock matrix_block;

    /// @brief Constructor por defecto.
    __host__ SemanticSphereAlpha()
        : center({0.0f, 0.0f, 0.0f}),
          radius(0.0f),
          semantic_weight(0.5f),
          sphere_id(0),
          depth(0),
          parent_id(0),
          num_children(0),
          is_leaf(false) {
        children_ids.fill(0);
        label[0] = '\0';
    }
};

// ============================================================================
// ESTRUCTURA: AlphaRayPayload
// ============================================================================

/**
 * @struct AlphaRayPayload
 * @brief Datos que porta el rayo durante el traversal del BSH.
 *
 * En OptiX, el payload es la estructura que se pasa entre programas de shader
 * (raygen, intersection, closest-hit, miss). Almacena el "estado" del rayo
 * conforme traversa el árbol.
 */
struct AlphaRayPayload {
    /// @brief Energía restante del rayo (empieza en 1.0).
    /// Decae exponencialmente conforme golpea esferas.
    /// Si energy < ALPHA_ENERGY_THRESHOLD, el rayo termina.
    float energy;

    /// @brief ID de la esfera hoja encontrada tras el traversal.
    /// UINT32_MAX indica "no se encontró esfera" (miss).
    /// Este es el valor más importante: determina qué MatrixBlock se usa en Fase B.
    uint32_t hit_sphere_id;

    /// @brief Punto exacto de impacto en la superficie de la esfera.
    /// Usado para cálculos de similitud local y debug.
    float3 hit_point;

    /// @brief Profundidad alcanzada en el árbol antes de llegar a la hoja.
    /// Métrica de rendimiento: profundidad < log₂(N) es lo esperado.
    uint32_t depth_reached;

    /// @brief Similitud semántica más alta encontrada en el traversal.
    /// Rango: [0.0, 1.0].
    /// Usada para medir confianza del resultado.
    float best_similarity;

    /// @brief Constructor por defecto.
    __host__ __device__ AlphaRayPayload()
        : energy(1.0f),
          hit_sphere_id(UINT32_MAX),
          depth_reached(0),
          best_similarity(0.0f) {}
};

// ============================================================================
// ESTRUCTURA: AlphaExecutionResult
// ============================================================================

/**
 * @struct AlphaExecutionResult
 * @brief Resultado completo de la ejecución de ambas fases (A + B).
 *
 * Contiene:
 *   1. Activaciones de salida (FP16)
 *   2. Confianza del resultado
 *   3. Información de qué esfera se usó
 *   4. Métricas de timing para profiling
 */
struct AlphaExecutionResult {
    /// @brief Puntero GPU a las activaciones de salida en FP16.
    /// Tamaño: [output_dim] half values.
    /// Propiedad: AlphaBSH (será liberado en destructor).
    half* output_activations;

    /// @brief Dimensión de las activaciones de salida.
    /// Típicamente igual a dim_out del MatrixBlock de la esfera.
    uint32_t output_dim;

    /// @brief Confianza del resultado (0.0 = sin confianza, 1.0 = máxima).
    /// Basada en:
    ///   - Energía final del rayo
    ///   - Similitud semántica encontrada
    ///   - Profundidad alcanzada
    float confidence;

    /// @brief ID de la esfera utilizada para la Fase B.
    /// Permite auditoría y debugging: "¿Qué esfera se escogió?"
    uint32_t sphere_id_used;

    /// @brief Tiempo en millisegundos para la Fase A (traversal BSH).
    /// Incluye transferencia de datos CPU-GPU y ejecución del kernel.
    float phase_a_time_ms;

    /// @brief Tiempo en millisegundos para la Fase B (cuBLAS MatMul).
    /// Incluye carga de matrices si es necesario, MatMul, y carga lazy.
    float phase_b_time_ms;

    /// @brief Constructor por defecto.
    AlphaExecutionResult()
        : output_activations(nullptr),
          output_dim(0),
          confidence(0.0f),
          sphere_id_used(UINT32_MAX),
          phase_a_time_ms(0.0f),
          phase_b_time_ms(0.0f) {}
};

// ============================================================================
// ESTRUCTURA: AlphaConfig
// ============================================================================

/**
 * @struct AlphaConfig
 * @brief Configuración global para la ejecución de Alpha BSH.
 *
 * Agrupa todos los hiperparámetros y handles de librerías necesarias
 * para ejecutar las dos fases.
 */
struct AlphaConfig {
    /// @brief Número total de esferas en el árbol BSH.
    uint32_t num_spheres;

    /// @brief Profundidad máxima del árbol (para truncado de búsqueda).
    uint32_t max_depth;

    /// @brief Coeficiente de absorción del rayo (decay exponencial).
    /// Fórmula: energy(d) = E₀ * exp(-lambda_decay * d)
    /// Valores típicos: 0.05 a 0.2.
    float lambda_decay;

    /// @brief Enable lazy loading de MatrixBlocks desde disco.
    /// Si true: carga desde almacenamiento cuando sea necesario.
    /// Si false: todas las matrices deben estar precargadas en VRAM.
    bool lazy_load_matrices;

    /// @brief Enable fallback a FP32 si FP16 produce NaN/Inf.
    /// Útil para debugging y robustez numérica.
    bool use_fp32_fallback;

    /// @brief Handle de cuBLAS para operaciones MatMul.
    /// Requiere inicialización: cublasCreate(&cublas_handle).
    /// AlphaBSH no toma posesión; caller debe liberar.
    cublasHandle_t cublas_handle;

    /// @brief Constructor por defecto.
    AlphaConfig()
        : num_spheres(0),
          max_depth(ALPHA_BSH_MAX_DEPTH),
          lambda_decay(ALPHA_LAMBDA_DECAY),
          lazy_load_matrices(true),
          use_fp32_fallback(false),
          cublas_handle(nullptr) {}
};

// ============================================================================
// CLASE: AlphaBSH
// ============================================================================

/**
 * @class AlphaBSH
 * @brief Orquestador principal de la arquitectura Alpha BSH.
 *
 * Responsabilidades:
 *   1. Construcción y gestión del árbol BSH en GPU
 *   2. Lanzamiento de la Fase A (OptiX traversal)
 *   3. Lanzamiento de la Fase B (cuBLAS MatMul)
 *   4. Carga lazy de MatrixBlocks
 *   5. Profiling y estadísticas
 *
 * Típicamente:
 *   - Constructor: aloca memoria GPU, inicializa estruturas
 *   - build(): construye el árbol desde esferas host-side
 *   - execute(): ejecuta ambas fases, devuelve resultado
 *   - Destructor: libera toda memoria GPU
 *
 * NOTA: Esta clase es un prototipo. El código asume:
 *   - Hardware NVIDIA RTX 4090 / RTX 5070 Ti (Ada/Blackwell)
 *   - CUDA Compute Capability >= sm_89 (para FP16 y RT Cores)
 *   - OptiX 8.x instalado
 *   - cuBLAS disponible
 */
class AlphaBSH {
public:
    // ========================================================================
    // CONSTRUCTOR / DESTRUCTOR
    // ========================================================================

    /**
     * @brief Constructor. Inicializa AlphaBSH (sin construir el árbol aún).
     *
     * Alocaciones iniciales:
     *   - Estructuras internas para manager de esferas
     *   - Buffers auxiliares para Fase A y B
     *
     * El árbol real se construye en build().
     */
    __host__ AlphaBSH();

    /**
     * @brief Destructor. Libera toda memoria GPU de forma segura.
     *
     * Responsable de:
     *   - cudaFree(d_spheres_)
     *   - cudaFree(d_payload_result_)
     *   - Liberar handles OptiX si aplicable
     *   - NO libera cublas_handle (responsabilidad del caller)
     *
     * GARANTÍA: No falla incluso si la memoria fue previamente liberada.
     */
    __host__ ~AlphaBSH();

    // ========================================================================
    // CONSTRUCCIÓN DEL ÁRBOL
    // ========================================================================

    /**
     * @brief Construye el árbol BSH a partir de un array de esferas host-side.
     *
     * Algoritmo:
     *   1. Copia esferas a GPU (d_spheres_)
     *   2. Asigna relaciones padre-hijo por proximidad geométrica (espacio 3D)
     *   3. Valida el árbol (profundidad <= ALPHA_BSH_MAX_DEPTH, conectividad)
     *   4. Carga MetrixBlocks en GPU (totalmente o lazy, según config)
     *
     * @param h_spheres Array host-side de SemanticSphereAlpha
     * @param num_spheres Número de esferas
     * @param config Configuración global (incluye flags de lazy loading)
     *
     * @return true si la construcción fue exitosa, false si hay error.
     *
     * @note Una vez llamado build(), el árbol es inmutable hasta next build().
     * @note Costoso en tiempo. Típicamente O(N log N) amortizado.
     */
    __host__ bool build(
        const SemanticSphereAlpha* h_spheres,
        uint32_t num_spheres,
        const AlphaConfig& config);

    // ========================================================================
    // EJECUCIÓN DE FASES
    // ========================================================================

    /**
     * @brief Lanza la FASE A: Traversal OptiX del BSH para encontrar esfera relevante.
     *
     * Internamente:
     *   1. Prepara query_embedding en GPU
     *   2. Lanza kernel/programa OptiX raygen para generar UN rayo desde query
     *   3. Intersecta rayo con esferas, sigue traversal hacia hoja
     *   4. Devuelve ID de la esfera más relevante (hit_sphere_id)
     *
     * @param query_embedding Embedding del token de query (host-side, FP32 o FP16)
     * @param query_dim Dimensión del embedding (768, 4096, etc.)
     * @param config Configuración (lambda_decay, thresholds)
     *
     * @return AlphaRayPayload con sphere_id_found (UINT32_MAX si miss)
     *
     * @complexity O(log N) donde N = num_spheres
     */
    __host__ AlphaRayPayload launchPhaseA(
        const float* query_embedding,
        uint32_t query_dim,
        const AlphaConfig& config);

    /**
     * @brief Lanza la FASE B: cuBLAS MatMul selectivo en la esfera encontrada.
     *
     * Internamente:
     *   1. Obtiene MatrixBlock de sphere_id
     *   2. Si !block.loaded: carga lazy desde disco (simulado)
     *   3. Copia input_activations a GPU si es necesario
     *   4. Ejecuta:
     *        hidden = GELU(W1 · input + b1)         [cublasHgemm]
     *        output = W2 · hidden + b2              [cublasHgemm]
     *   5. Devuelve activaciones de salida + métricas
     *
     * @param sphere_id ID de la esfera (típicamente de launchPhaseA)
     * @param input_activations Activaciones de entrada (host-side)
     * @param input_dim Dimensión de entrada (rows)
     * @param config Configuración (cublas_handle, fallback settings)
     *
     * @return AlphaExecutionResult con output_activations, timing, confianza
     *
     * @complexity O(M²) donde M = dim_in/dim_out del MatrixBlock
     */
    __host__ AlphaExecutionResult launchPhaseB(
        uint32_t sphere_id,
        const float* input_activations,
        uint32_t input_dim,
        const AlphaConfig& config);

    /**
     * @brief Ejecuta ambas fases en secuencia: Fase A + Fase B.
     *
     * Es un wrapper conveniente que:
     *   1. Llama launchPhaseA(query_embedding, ...)
     *   2. Si hit_sphere_id != UINT32_MAX:
     *        Llama launchPhaseB(hit_sphere_id, input_activations, ...)
     *   3. Devuelve resultado final con ambos tiempos
     *
     * @param query_embedding Embedding del query
     * @param query_dim Dimensión del embedding
     * @param input_activations Activaciones de entrada para Fase B
     * @param input_dim Dimensión de entrada para Fase B
     * @param config Configuración global
     *
     * @return AlphaExecutionResult completo
     *
     * @complexity O(N log N) + O(M²)
     * @note Este es el punto de entrada típico para usuarios.
     */
    __host__ AlphaExecutionResult execute(
        const float* query_embedding,
        uint32_t query_dim,
        const float* input_activations,
        uint32_t input_dim,
        const AlphaConfig& config);

    // ========================================================================
    // GESTIÓN DE MEMORIA Y CARGA LAZY
    // ========================================================================

    /**
     * @brief Carga el MatrixBlock de una esfera desde disco a VRAM.
     *
     * Asumciones:
     *   - MatrixBlock::disk_offset es válido
     *   - El archivo está disponible en ruta configurada
     *   - Hay espacio libre en VRAM
     *
     * Internamente:
     *   1. Abre archivo en disk_offset
     *   2. Lee W1, b1, W2, b2 a host-pinned memory
     *   3. Transfiere a device memory (cudaMemcpy)
     *   4. Marca block.loaded = true
     *
     * @param sphere_id ID de la esfera cuyo MatrixBlock se cargará
     *
     * @return true si la carga fue exitosa, false si error (archivo no encontrado, etc.)
     *
     * @note Esta llamada es bloqueante (sincrónica).
     * @note Para producción, considerar overlapping de carga con ejecución.
     */
    __host__ bool loadMatrixBlock(uint32_t sphere_id);

    // ========================================================================
    // PROFILING Y ESTADÍSTICAS
    // ========================================================================

    /**
     * @brief Obtiene estadísticas acumuladas de ejecuciones previas.
     *
     * Retorna un string formateado con:
     *   - Número de ejecuciones
     *   - Tiempo promedio de Fase A y B
     *   - Tasa de hits (cuántas veces encontró esfera)
     *   - Profundidad promedio alcanzada
     *   - Memoria GPU usada
     *
     * @return std::string con estadísticas formateadas
     */
    __host__ std::string getStats() const;

    /**
     * @brief Reinicia las estadísticas acumuladas.
     *
     * Útil entre runs para medir performance de un subconjunto.
     */
    __host__ void resetStats();

private:
    // ========================================================================
    // MIEMBROS PRIVADOS
    // ========================================================================

    /// Array GPU de todas las esferas (SemanticSphereAlpha*)
    SemanticSphereAlpha* d_spheres_;

    /// Número de esferas construidas
    uint32_t num_spheres_;

    /// Buffer GPU para resultado del payload (AlphaRayPayload*)
    AlphaRayPayload* d_payload_result_;

    /// Handle de cuBLAS (almacenado localmente, creado en build())
    cublasHandle_t cublas_handle_;

    /// Estadísticas de ejecución
    struct {
        uint32_t num_executions;
        float avg_phase_a_time_ms;
        float avg_phase_b_time_ms;
        uint32_t num_hits;
        uint32_t total_depth_reached;
    } stats_;

    // ========================================================================
    // MÉTODOS PRIVADOS HELPER
    // ========================================================================

    /**
     * @brief Asigna relaciones padre-hijo en el árbol BSH.
     *
     * Algoritmo: Para cada esfera, busca los ALPHA_BSH_MAX_CHILDREN esferas
     * más cercanas y las designa como hijos.
     *
     * @param h_spheres Array host-side de esferas
     * @param num_spheres Número de esferas
     *
     * @return true si la asignación fue exitosa
     *
     * @complexity O(N² log N) para construcción (puede optimizarse)
     */
    __host__ bool assignParentChildRelationships(
        SemanticSphereAlpha* h_spheres,
        uint32_t num_spheres);

    /**
     * @brief Valida la estructura del árbol (profundidad, conectividad, etc.).
     *
     * Checks:
     *   - Profundidad máxima <= ALPHA_BSH_MAX_DEPTH
     *   - Cada nodo tiene <= ALPHA_BSH_MAX_CHILDREN hijos
     *   - parent_id/children_ids son consistentes
     *   - Solo hojas tienen MatrixBlock válido
     *
     * @return true si el árbol es válido, false si hay inconsistencia
     */
    __host__ bool validateTreeStructure() const;
};

// ============================================================================
// FUNCIONES LIBRES (HELPERS)
// ============================================================================

/**
 * @brief Imprime estadísticas de ejecución de forma legible.
 *
 * Formato típico:
 *   ===== ALPHA BSH EXECUTION RESULT =====
 *   Sphere Used: 42
 *   Confidence: 0.85
 *   Phase A Time: 0.23 ms
 *   Phase B Time: 12.45 ms
 *   Total Time: 12.68 ms
 *   =======================================
 *
 * @param result AlphaExecutionResult a imprimir
 */
void printAlphaStats(const AlphaExecutionResult& result);

/**
 * @brief Convierte una distancia 3D a similitud coseno aproximada.
 *
 * Fórmula: similarity = exp(-lambda * distance²)
 * Rango: [0.0, 1.0], donde 1.0 = idéntico, 0.0 = lejano.
 *
 * @param distance Distancia euclídea en espacio 3D
 * @param lambda Coeficiente de absorción (típ. 0.1)
 *
 * @return Similitud aproximada [0.0, 1.0]
 */
__host__ __device__ inline float distanceToSimilarity(float distance, float lambda) {
    return expf(-lambda * distance * distance);
}

#endif // LIQUIDBIT_ALPHA_BSH_H_
