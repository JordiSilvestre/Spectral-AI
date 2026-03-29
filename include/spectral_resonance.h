/**
 * @file spectral_resonance.h
 * @brief String-Inception Engine — Arquitectura de IAS anidados con resonancia de Fourier
 *
 * CONCEPTO (del documento de invención):
 * =======================================
 * En lugar de almacenar matrices de pesos estáticas, cada nodo hoja del árbol de
 * aceleración almacena una "SemanticString": una serie de Fourier cuyos coeficientes
 * codifican el comportamiento semántico del nodo. La frecuencia de contexto ω del
 * rayo determina el estado de resonancia, resolviendo la polisemia sin duplicar matrices.
 *
 * ESTRUCTURA DE IAS ANIDADOS (4 niveles):
 * ========================================
 *   Nivel 0 (raíz):  IAS global — categorías semánticas grandes (e.g. Código, Música, Ciencia)
 *   Nivel 1:         IAS de dominio — subdominios (e.g. Python, Jazz, Física)
 *   Nivel 2:         IAS de concepto — conceptos específicos (e.g. Bucles, Acordes, Gravedad)
 *   Nivel 3 (hoja):  SemanticString — resonancia Fourier ON-THE-FLY
 *
 * VENTAJA vs MATRICES ESTÁTICAS:
 * ================================
 *   - Polisemia: misma esfera, distintas ω → distintas salidas semánticas
 *   - Memoria: 8 coeficientes (64 bytes) vs bloque de matriz (kB a MB)
 *   - Cómputo: Fourier con 8 modos = 16 mul + 15 add (CUDA cores) vs MatMul (Tensor Cores)
 *   - Hardware: solo RT Cores + CUDA cores — Tensor Cores a 0%
 *
 * FÓRMULA DE RESONANCIA:
 * =======================
 *   W_salida = T( Σ_{k=1}^{M} (a_k·sin(kω) + b_k·cos(kω)), ω_rayo )
 *   donde T es una transformación de activación suave (tanh)
 *   y M = numModes (por defecto 8)
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

// OptixTraversableHandle y tipos OptiX necesarios para InceptionLaunchParams
#include <optix.h>

// ============================================================================
// CONSTANTES
// ============================================================================

/// Número de modos Fourier por SemanticString
#define RESONANCE_NUM_MODES 8

// Incluir math para CUDA device code (sinf/cosf/tanhf)
#ifdef __CUDACC__
#include <math.h>
#endif

/// Número máximo de niveles de IAS anidados
#define INCEPTION_MAX_DEPTH 4

/// Dimensión del color espectral del rayo (frecuencia de contexto)
#define INCEPTION_SPECTRAL_DIM 64

// ============================================================================
// STRUCTS FUNDAMENTALES
// ============================================================================

/**
 * @brief Esfera semántica en el árbol de IAS anidados.
 *
 * Cada instancia en el IAS es una SemanticSphere. A depth < INCEPTION_MAX_DEPTH-1,
 * contiene un puntero a un IAS hijo (instancia anidada). En el nivel hoja
 * (depth == INCEPTION_MAX_DEPTH-1), activa la resonancia de Fourier.
 *
 * El campo `frequencyBias` desplaza la frecuencia de contexto ω del rayo
 * al entrar en la esfera: ω_local = ω_rayo + frequencyBias.
 * Esto permite que esferas hermanas respondan de forma distinta al mismo rayo.
 */
struct alignas(16) SemanticSphere {
    float3   center;           ///< Posición 3D en el espacio semántico
    float    radius;           ///< Radio semántico (influencia contextual)
    uint32_t instanceId;       ///< ID único en el IAS padre
    uint32_t childIAS;         ///< Handle del IAS hijo (0 = nodo hoja)
    uint32_t depth;            ///< Nivel en la jerarquía (0 = raíz)
    float    frequencyBias;    ///< Sesgo de frecuencia Δω al entrar en esta esfera
};

/**
 * @brief Portal afín para saltos dimensionales entre niveles de IAS.
 *
 * Cuando el rayo atraviesa de un nivel IAS al siguiente, su vector de contexto
 * se transforma por este portal: f_nuevo = rows * [f_actual; 1].
 *
 * Permite que conceptos abstractos "remapeen" el espacio semántico del hijo.
 * Ejemplo: la esfera "Programación" en el nivel 0 transforma f de forma que
 * el nivel 1 recibe el contexto en coordenadas de "dominio técnico".
 */
struct alignas(64) AffinePortal {
    float4 rows[4];    ///< Matriz 4×4 de transformación afín del contexto
};

/**
 * @brief Parámetros de resonancia Fourier almacenados en el nodo hoja.
 *
 * Estos coeficientes NO son matrices de pesos fijas — son parámetros de una
 * función continua que produce el peso de salida en función de la frecuencia ω.
 * Se aprenden durante el entrenamiento minimizando L_total (ver CLAUDE.md).
 */
struct alignas(32) ResonanceParams {
    float    a[RESONANCE_NUM_MODES];    ///< Coeficientes de seno (a_1 .. a_M)
    float    b[RESONANCE_NUM_MODES];    ///< Coeficientes de coseno (b_1 .. b_M)
    uint32_t numModes;                  ///< Número de modos activos (<= RESONANCE_NUM_MODES)
    float    outputScale;               ///< Factor de escala de salida (aprendido)
    uint32_t semanticTag;               ///< Etiqueta semántica para debug/interpretabilidad
    uint32_t _pad;                      ///< Padding a múltiplo de 32 bytes
};

/**
 * @brief Nodo hoja del String-Inception Engine.
 *
 * Reemplaza el bloque de matriz de weights por una SemanticString.
 * La salida se genera ON-THE-FLY según la ω del rayo que la golpea,
 * sin cargar ninguna matriz desde memoria.
 */
struct alignas(32) SemanticString {
    ResonanceParams resonance;    ///< Coeficientes Fourier (función semántica del nodo)
    float3          position;     ///< Posición en espacio semántico del nodo hoja
    uint32_t        stringId;     ///< ID único de esta SemanticString
};

// ============================================================================
// STRUCTS DE ESTADO DEL RAYO (payload de OptiX)
// ============================================================================

/**
 * @brief Payload del rayo en el pipeline String-Inception.
 *
 * Se pasa entre raygen → closesthit → miss usando optixGetPayload/optixSetPayload.
 * Los 4 float32 de payload caben en 4 registros de 32 bits del SM.
 *
 * Nota: OptiX permite hasta 8 registros de payload de 32 bits.
 */
struct InceptionPayload {
    float    omega;            ///< Frecuencia de contexto actual del rayo
    float    accumulated;      ///< Peso acumulado de resonancia
    uint32_t depth;            ///< Profundidad actual en el árbol de IAS
    uint32_t hitCount;         ///< Número de colisiones en esta traversal
};

/**
 * @brief Resultado de una traversal completa del String-Inception Engine.
 *
 * Devuelto al host tras optixLaunch. Contiene el peso de atención final
 * calculado por la resonancia Fourier de todos los nodos golpeados.
 */
struct SpectralAttentionResult {
    float    attentionWeight;     ///< Peso de atención final (salida del pipeline)
    float    finalOmega;          ///< Frecuencia ω tras atravesar la jerarquía completa
    uint32_t dominantStringId;    ///< ID de la SemanticString más resonante
    uint32_t traversalDepth;      ///< Profundidad máxima alcanzada en el IAS anidado
    float3   exitDirection;       ///< Dirección del rayo al salir del último IAS
    float    energyRemaining;     ///< Energía residual (para debugging)
};

// ============================================================================
// PARÁMETROS DE LANZAMIENTO
// ============================================================================

/**
 * @brief Parámetros globales del pipeline String-Inception.
 *
 * Transferidos al device via cudaMemcpyToSymbol o como launch params.
 * El pipeline los lee desde un buffer constante en GPU.
 */
struct InceptionLaunchParams {
    OptixTraversableHandle topLevelIAS;       ///< Handle del IAS raíz (nivel 0)
    const SemanticSphere*  spheres;           ///< Array de esferas (todos los niveles)
    const AffinePortal*    portals;           ///< Portal afín por nivel (INCEPTION_MAX_DEPTH)
    const SemanticString*  strings;           ///< Array de SemanticStrings (hojas)
    SpectralAttentionResult* results;         ///< Buffer de salida (uno por rayo)
    float                  baseOmega;         ///< Frecuencia de contexto base del prompt
    uint32_t               numRays;           ///< Número de rayos a lanzar
    uint32_t               numStrings;        ///< Total de SemanticStrings en la escena
};

// ============================================================================
// FUNCIÓN INLINE: Resonancia Fourier (compartida entre todos los kernels)
//
// Definida aquí como __device__ __forceinline__ para que cualquier .cu que
// incluya este header la use sin depender de spectral_resonance.cu como
// unidad de compilación separada. Esto evita el error de linker:
//   "undefined reference to semanticStringResonance"
// ============================================================================

#ifdef __CUDACC__
/**
 * @brief Calcula la resonancia Fourier de una SemanticString para una frecuencia ω.
 *
 *   W(ω) = outputScale · tanh( Σ_{k=1}^{numModes} (a_k·sin(kω) + b_k·cos(kω)) )
 *
 * @param params  Coeficientes Fourier del nodo hoja
 * @param omega   Frecuencia de contexto del rayo [0, 2π]
 * @return        Peso de resonancia en [-outputScale, +outputScale]
 */
__device__ __forceinline__
float semanticStringResonance(const ResonanceParams& params, float omega) {
    float sum = 0.0f;
    const uint32_t modes = min(params.numModes, (uint32_t)RESONANCE_NUM_MODES);

    #pragma unroll
    for (uint32_t k = 1; k <= RESONANCE_NUM_MODES; ++k) {
        if (k <= modes) {
            const float kw = (float)k * omega;
            sum += params.a[k - 1] * sinf(kw) + params.b[k - 1] * cosf(kw);
        }
    }
    return params.outputScale * tanhf(sum);
}
#endif // __CUDACC__
