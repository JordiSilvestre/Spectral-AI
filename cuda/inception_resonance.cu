/**
 * @file inception_resonance.cu
 * @brief Inception Engine — Resonancia Fourier con gradientes + evaluación batch
 *
 * EXTENSIONES SOBRE spectral_resonance.cu:
 * ==========================================
 * 1. Gradientes analíticos de W(ω) respecto a coeficientes a[], b[] y a ω
 *    → Necesario para training end-to-end de los SemanticStrings
 *
 * 2. resonanceInceptionBatch: evaluación batch respetando los portales
 *    → La ω se transforma por los portales de cada nivel antes de la resonancia
 *    → Simula la propagación completa de contexto en el Inception Engine
 *
 * 3. resonanceGradKernel: kernel de entrenamiento
 *    → Calcula dL/d(a_k), dL/d(b_k) para backprop desde Python/PyTorch
 *
 * FÓRMULAS:
 * ==========
 *   W(ω) = s · tanh( Σ_{k=1}^{M} (a_k·sin(kω) + b_k·cos(kω)) )
 *
 *   Sea sum = Σ a_k·sin(kω) + b_k·cos(kω)
 *   Sea T = tanh(sum)
 *   Sea dL/dW el gradiente entrante
 *
 *   dL/da_k = dL/dW · s · (1 - T²) · sin(k·ω)
 *   dL/db_k = dL/dW · s · (1 - T²) · cos(k·ω)
 *   dL/dω   = dL/dW · s · (1 - T²) · Σ_k k·(a_k·cos(kω) - b_k·sin(kω))
 *   dL/ds   = dL/dW · T
 *
 * CONEXIÓN CON EL PIPELINE PYTHON:
 * ==================================
 * Los kernels de este archivo se llaman desde `python/train_spectral.py`
 * via ctypes o via PyTorch custom CUDA extensions para hacer backprop real
 * a través de los coeficientes Fourier de los SemanticStrings.
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include <cuda_runtime.h>
#include <math.h>
#include <math_constants.h>
#include <stdint.h>
#include <stdio.h>

#include "../include/spectral_resonance.h"
#include "../include/inception_engine.h"

// ============================================================================
// DEVICE: Función de resonancia + gradientes internos
// ============================================================================

/**
 * @brief Calcula W(ω) y los valores intermedios necesarios para backprop.
 *
 * @param params     Coeficientes Fourier del nodo
 * @param omega      Frecuencia de contexto
 * @param outW       Valor de resonancia W(ω)
 * @param outSum     Suma interior (pre-tanh) — para backward
 * @param outTanh    Valor tanh(sum) — para backward
 */
__device__ __forceinline__
void resonanceForwardFull(
    const ResonanceParams& params,
    float omega,
    float& outW, float& outSum, float& outTanh
) {
    float sum = 0.0f;
    const uint32_t M = min(params.numModes, (uint32_t)RESONANCE_NUM_MODES);

    #pragma unroll
    for (uint32_t k = 1; k <= RESONANCE_NUM_MODES; ++k) {
        if (k <= M) {
            const float kw = (float)k * omega;
            sum += params.a[k - 1] * sinf(kw) + params.b[k - 1] * cosf(kw);
        }
    }

    outSum  = sum;
    outTanh = tanhf(sum);
    outW    = params.outputScale * outTanh;
}

/**
 * @brief Calcula los gradientes de W(ω) respecto a todos los parámetros.
 *
 * @param params       Coeficientes del nodo
 * @param omega        Frecuencia de contexto
 * @param dLdW         Gradiente de la loss respecto a W(ω) (entrante)
 * @param gradA        Salida: dL/d(a_k) para k=1..M   [tamaño RESONANCE_NUM_MODES]
 * @param gradB        Salida: dL/d(b_k) para k=1..M   [tamaño RESONANCE_NUM_MODES]
 * @param gradOmega    Salida: dL/dω (escalar)
 * @param gradScale    Salida: dL/d(outputScale)
 */
__device__ __forceinline__
void resonanceBackward(
    const ResonanceParams& params,
    float omega,
    float dLdW,
    float* __restrict__ gradA,
    float* __restrict__ gradB,
    float& gradOmega,
    float& gradScale
) {
    // Forward para obtener valores intermedios
    float sumW, T, W;
    resonanceForwardFull(params, omega, W, sumW, T);

    // d(tanh)/d(sum) = 1 - tanh²
    const float dtanh = 1.0f - T * T;
    // Factor común: dL/dW · s · (1 - T²)
    const float factor = dLdW * params.outputScale * dtanh;

    const uint32_t M = min(params.numModes, (uint32_t)RESONANCE_NUM_MODES);

    gradOmega = 0.0f;

    #pragma unroll
    for (uint32_t k = 1; k <= RESONANCE_NUM_MODES; ++k) {
        if (k <= M) {
            const float kw   = (float)k * omega;
            const float s_kw = sinf(kw);
            const float c_kw = cosf(kw);

            // dL/da_k = factor · sin(kω)
            gradA[k - 1] = factor * s_kw;
            // dL/db_k = factor · cos(kω)
            gradB[k - 1] = factor * c_kw;
            // dL/dω += factor · k · (a_k·cos(kω) - b_k·sin(kω))
            gradOmega += factor * (float)k
                       * (params.a[k - 1] * c_kw - params.b[k - 1] * s_kw);
        } else {
            gradA[k - 1] = 0.0f;
            gradB[k - 1] = 0.0f;
        }
    }

    // dL/ds = dL/dW · T
    gradScale = dLdW * T;
}

// ============================================================================
// KERNEL: Evaluación batch con propagación de portales (Inception-aware)
// ============================================================================

/**
 * @brief Evalúa resonancia para todas las (string, omega) en un batch.
 *
 * La ω de cada par (nivel, string) se transforma por el portal del nivel
 * antes de calcular la resonancia final. Esto simula el flujo completo
 * del Inception Engine sin lanzar el pipeline OptiX completo.
 *
 * Útil para:
 *   - Verificar que los portales propagan el contexto correctamente
 *   - Pre-computar resonancias para entrenamiento (sin overhead OptiX)
 *   - Test de gradientes con torch.autograd.gradcheck
 *
 * Grid: <<< ceil(numStrings*numOmegas/256), 256 >>>
 *
 * @param strings        Array de SemanticStrings [numStrings]
 * @param omegas         Array de frecuencias base [numOmegas]
 * @param portals        Array de portales por nivel [INCEPTION_MAX_DEPTH]
 * @param levels         Nivel de cada string (0-3) [numStrings]
 * @param results        Buffer de salida: results[i*numOmegas + j] = W(portal(level_i, ω_j))
 * @param numStrings     Número de SemanticStrings
 * @param numOmegas      Número de valores ω base
 */
extern "C" __global__
void resonanceInceptionBatch(
    const SemanticString* __restrict__ strings,
    const float*          __restrict__ omegas,
    const AffinePortal*   __restrict__ portals,
    const uint32_t*       __restrict__ levels,
    float*                             results,
    uint32_t numStrings,
    uint32_t numOmegas
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = numStrings * numOmegas;
    if (tid >= total) return;

    const uint32_t strIdx   = tid / numOmegas;
    const uint32_t omegaIdx = tid % numOmegas;

    if (strIdx >= numStrings || omegaIdx >= numOmegas) return;

    float omega = omegas[omegaIdx];

    // Aplicar portal del nivel de la string (transforma ω al espacio local)
    const uint32_t level = levels[strIdx];
    if (level < INCEPTION_MAX_DEPTH && portals != nullptr) {
        const AffinePortal& portal = portals[level];
        const float4 ctx = make_float4(omega, omega * omega, sinf(omega), 1.0f);
        const float4 row = portal.rows[3];
        const float new_omega = row.x * ctx.x + row.y * ctx.y
                              + row.z * ctx.z + row.w * ctx.w;
        // Bug 2.10 fix: Use CUDART_PI_F for consistency with other files
        // (e.g., spectral_kernels.cu:99) instead of truncated literal
        omega = fmodf(fabsf(new_omega), 2.0f * CUDART_PI_F);
    }

    // Añadir sesgo de frecuencia de la esfera (si está disponible)
    // En este kernel no tenemos acceso a SemanticSphere, así que usamos
    // solo el portal. En el pipeline completo se añade sphere.frequencyBias.

    const float w = semanticStringResonance(strings[strIdx].resonance, omega);
    results[tid] = w;
}

// ============================================================================
// KERNEL: Gradientes para training
// ============================================================================

/**
 * @brief Calcula dL/d(a_k) y dL/d(b_k) para un batch de strings.
 *
 * Llamado desde el training loop (python/train_spectral.py) para actualizar
 * los coeficientes Fourier de los SemanticStrings via AdamW.
 *
 * NOTA: Este kernel asume que los gradientes se acumulan (+=) en gradA/gradB
 * si el mismo string aparece múltiples veces en el batch.
 *
 * Grid: <<< ceil(batchSize/256), 256 >>>
 *
 * @param strings        Array de SemanticStrings (device)
 * @param omegas         Array de ω para cada elemento del batch [batchSize]
 * @param dLdW           Array de gradientes entrantes [batchSize]
 * @param stringIndices  Índice del string para cada elemento del batch [batchSize]
 * @param gradA          Acumulador de gradientes dL/da_k  [numStrings × RESONANCE_NUM_MODES]
 * @param gradB          Acumulador de gradientes dL/db_k  [numStrings × RESONANCE_NUM_MODES]
 * @param gradOmega      Acumulador de dL/dω por elemento [batchSize]
 * @param batchSize      Número de elementos en el batch
 * @param numStrings     Número total de SemanticStrings
 */
extern "C" __global__
void resonanceGradKernel(
    const SemanticString* __restrict__ strings,
    const float*          __restrict__ omegas,
    const float*          __restrict__ dLdW,
    const uint32_t*       __restrict__ stringIndices,
    float*                             gradA,
    float*                             gradB,
    float*                             gradOmega,
    uint32_t batchSize,
    uint32_t numStrings
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batchSize) return;

    const uint32_t strIdx = stringIndices[tid];
    if (strIdx >= numStrings) return;

    const SemanticString& str = strings[strIdx];
    const float omega  = omegas[tid];
    const float dloss  = dLdW[tid];

    // Calcular gradientes analíticos
    float gA[RESONANCE_NUM_MODES];
    float gB[RESONANCE_NUM_MODES];
    float gOmega, gScale;

    resonanceBackward(str.resonance, omega, dloss, gA, gB, gOmega, gScale);

    // Acumular gradientes en el buffer global (atomic para evitar race conditions)
    const uint32_t base = strIdx * RESONANCE_NUM_MODES;
    for (uint32_t k = 0; k < RESONANCE_NUM_MODES; ++k) {
        atomicAdd(&gradA[base + k], gA[k]);
        atomicAdd(&gradB[base + k], gB[k]);
    }
    gradOmega[tid] = gOmega;
}

// ============================================================================
// KERNEL: Curva de resonancia para visualización
// ============================================================================

/**
 * @brief Evalúa W(ω) en numPoints puntos equiespaciados en [0, 2π].
 *
 * Útil para generar curvas de resonancia durante el training:
 * visualizar qué frecuencias activa cada SemanticString.
 *
 * Grid: <<< ceil(numStrings * numPoints / 256), 256 >>>
 *
 * @param strings        Array de SemanticStrings
 * @param results        Buffer de salida [numStrings × numPoints]
 * @param numStrings     Número de strings a evaluar
 * @param numPoints      Número de puntos de muestreo
 */
extern "C" __global__
void resonanceCurveEval(
    const SemanticString* __restrict__ strings,
    float*                             results,
    uint32_t numStrings,
    uint32_t numPoints
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStrings * numPoints) return;

    const uint32_t strIdx   = tid / numPoints;
    const uint32_t pointIdx = tid % numPoints;

    // Bug 2.10 fix: Use CUDART_PI_F for consistency
    const float omega = (float)pointIdx * (2.0f * CUDART_PI_F) / (float)numPoints;
    results[tid] = semanticStringResonance(strings[strIdx].resonance, omega);
}

// ============================================================================
// KERNEL: Actualizar coeficientes Fourier (paso de AdamW)
// ============================================================================

/**
 * @brief Actualiza los coeficientes a[], b[] de todas las SemanticStrings.
 *
 * Implementa un paso de AdamW simplificado directamente en GPU:
 *   m_t = β1·m_{t-1} + (1-β1)·g
 *   v_t = β2·v_{t-1} + (1-β2)·g²
 *   param -= lr · m̂_t / (√v̂_t + ε) + wd · param
 *
 * Llamado desde train_spectral.py después de resonanceGradKernel.
 *
 * Grid: <<< ceil(numStrings * RESONANCE_NUM_MODES / 256), 256 >>>
 *
 * @param strings     SemanticStrings a actualizar (device, modificadas in-place)
 * @param gradA       Gradientes acumulados dL/da_k [numStrings × M]
 * @param gradB       Gradientes acumulados dL/db_k [numStrings × M]
 * @param mA, mB      Primer momento (Adam m) [numStrings × M] — actualizado in-place
 * @param vA, vB      Segundo momento (Adam v) [numStrings × M] — actualizado in-place
 * @param lr          Learning rate
 * @param beta1       Decay del primer momento (típico: 0.9)
 * @param beta2       Decay del segundo momento (típico: 0.999)
 * @param eps         Epsilon de Adam (típico: 1e-8)
 * @param wd          Weight decay (típico: 0.01)
 * @param t           Paso actual de entrenamiento (para corrección de bias)
 * @param numStrings  Número de SemanticStrings
 */
extern "C" __global__
void resonanceAdamWStep(
    SemanticString* strings,
    const float* __restrict__ gradA,
    const float* __restrict__ gradB,
    float*  mA, float*  mB,
    float*  vA, float*  vB,
    float   lr,
    float   beta1,
    float   beta2,
    float   eps,
    float   wd,
    uint32_t t,
    uint32_t numStrings
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = numStrings * RESONANCE_NUM_MODES;
    if (tid >= total) return;

    const uint32_t strIdx = tid / RESONANCE_NUM_MODES;
    const uint32_t kIdx   = tid % RESONANCE_NUM_MODES;

    if (strIdx >= numStrings) return;

    ResonanceParams& p = strings[strIdx].resonance;
    const float ga = gradA[tid];
    const float gb = gradB[tid];

    // Actualizar momentos
    mA[tid] = beta1 * mA[tid] + (1.0f - beta1) * ga;
    mB[tid] = beta1 * mB[tid] + (1.0f - beta1) * gb;
    vA[tid] = beta2 * vA[tid] + (1.0f - beta2) * ga * ga;
    vB[tid] = beta2 * vB[tid] + (1.0f - beta2) * gb * gb;

    // Corrección de bias
    const float inv_beta1_t = 1.0f / (1.0f - powf(beta1, (float)t));
    const float inv_beta2_t = 1.0f / (1.0f - powf(beta2, (float)t));
    const float m_hat_a = mA[tid] * inv_beta1_t;
    const float m_hat_b = mB[tid] * inv_beta1_t;
    const float v_hat_a = vA[tid] * inv_beta2_t;
    const float v_hat_b = vB[tid] * inv_beta2_t;

    // Paso de AdamW
    p.a[kIdx] -= lr * (m_hat_a / (sqrtf(v_hat_a) + eps) + wd * p.a[kIdx]);
    p.b[kIdx] -= lr * (m_hat_b / (sqrtf(v_hat_b) + eps) + wd * p.b[kIdx]);
}

// ============================================================================
// DEBUG: Impresión de coeficientes
// ============================================================================

/**
 * @brief Imprime los coeficientes Fourier de un rango de SemanticStrings.
 *
 * Solo para debug — un solo thread imprime toda la info.
 */
extern "C" __global__
void resonancePrintCoefficients(
    const SemanticString* strings,
    uint32_t startIdx,
    uint32_t count
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (uint32_t i = startIdx; i < startIdx + count; ++i) {
        const ResonanceParams& p = strings[i].resonance;
        printf("[String #%u] modes=%u scale=%.4f tag=%u\n",
               i, p.numModes, p.outputScale, p.semanticTag);
        printf("  a: ");
        for (uint32_t k = 0; k < p.numModes; ++k) printf("%.4f ", p.a[k]);
        printf("\n  b: ");
        for (uint32_t k = 0; k < p.numModes; ++k) printf("%.4f ", p.b[k]);
        printf("\n");
    }
}
