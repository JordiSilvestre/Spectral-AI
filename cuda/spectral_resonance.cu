/**
 * @file spectral_resonance.cu
 * @brief Kernel CUDA para el cálculo de resonancia Fourier ON-THE-FLY
 *
 * CONCEPTO:
 * ==========
 * La función de resonancia de Fourier calcula el peso de salida de una
 * SemanticString dado su contexto de frecuencia ω:
 *
 *   W(ω) = outputScale · tanh( Σ_{k=1}^{M} (a_k·sin(kω) + b_k·cos(kω)) )
 *
 * Esta función es:
 *   - Continua y diferenciable (entrenable con backprop)
 *   - Periódica con periodo 2π/k → responde diferente a distintos contextos ω
 *   - Computacionalmente trivial: 16 mul + 15 add + 1 tanh por nodo hoja
 *
 * VENTAJA vs MATMUL:
 * ===================
 *   MatMul (D=768, bloque 64×64):  64×64×2 = 8.192 FLOPs
 *   Resonancia Fourier (M=8):      16 mul + 15 add = 31 FLOPs
 *   Factor de ahorro: ~264x por nodo hoja
 *
 * USO:
 * =====
 *   Este archivo define tanto una función __device__ inline (usable desde
 *   otros kernels CUDA/OptiX) como un kernel independiente para pruebas.
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "../include/spectral_resonance.h"

// semanticStringResonance() está definida como __device__ __forceinline__ en
// include/spectral_resonance.h para evitar problemas de linkage entre unidades
// de compilación CUDA (ODR con --relocatable-device-code).

/**
 * @brief Variante escalar sin struct — útil para llamadas desde OptiX device programs.
 *
 * @param a         Array de coeficientes de seno (longitud numModes)
 * @param b         Array de coeficientes de coseno (longitud numModes)
 * @param numModes  Número de modos activos
 * @param omega     Frecuencia de contexto
 * @param scale     Factor de escala de salida
 * @return          Peso de resonancia en [-scale, +scale]
 */
__device__ __forceinline__
float semanticStringResonanceRaw(
    const float* __restrict__ a,
    const float* __restrict__ b,
    uint32_t numModes,
    float omega,
    float scale)
{
    float sum = 0.0f;
    const uint32_t modes = min(numModes, (uint32_t)RESONANCE_NUM_MODES);
    for (uint32_t k = 1; k <= modes; ++k) {
        const float kw = (float)k * omega;
        sum += a[k - 1] * sinf(kw) + b[k - 1] * cosf(kw);
    }
    return scale * tanhf(sum);
}

// ============================================================================
// KERNEL STANDALONE: Para pruebas unitarias de la resonancia
// ============================================================================

/**
 * @brief Kernel de prueba: evalúa resonancia para un array de ω values.
 *
 * Útil para verificar que los coeficientes producen la función semántica
 * esperada antes de integrar en el pipeline OptiX completo.
 *
 * @param strings       Array de SemanticStrings a evaluar
 * @param omegas        Array de frecuencias de contexto (una por thread)
 * @param results       Buffer de salida (un float por thread)
 * @param numStrings    Número de SemanticStrings
 * @param numOmegas     Número de valores ω a evaluar
 */
extern "C" __global__
void resonanceBatchEval(
    const SemanticString* __restrict__ strings,
    const float*          __restrict__ omegas,
    float*                             results,
    uint32_t numStrings,
    uint32_t numOmegas)
{
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStrings * numOmegas) return;

    const uint32_t stringIdx = tid / numOmegas;
    const uint32_t omegaIdx  = tid % numOmegas;

    if (stringIdx >= numStrings) return;

    const float w = semanticStringResonance(strings[stringIdx].resonance, omegas[omegaIdx]);
    results[tid] = w;
}

/**
 * @brief Kernel de debugging: imprime la función de resonancia de una string
 *        para 16 valores equiespaciados de ω en [0, 2π].
 *
 * Solo útil en builds de debug. No usar en el pipeline de producción.
 *
 * @param strings   Array de SemanticStrings
 * @param stringIdx Índice de la SemanticString a inspeccionar
 */
extern "C" __global__
void resonanceDebugPrint(const SemanticString* strings, uint32_t stringIdx) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const SemanticString& s = strings[stringIdx];
    printf("[Resonance] SemanticString #%u (tag=%u, modes=%u, scale=%.4f)\n",
           s.stringId,
           s.resonance.semanticTag,
           s.resonance.numModes,
           s.resonance.outputScale);

    for (int i = 0; i < 16; ++i) {
        const float omega = (float)i * (2.0f * 3.14159265f / 16.0f);
        const float w = semanticStringResonance(s.resonance, omega);
        printf("  omega=%.4f -> resonance=%.6f\n", omega, w);
    }
}
