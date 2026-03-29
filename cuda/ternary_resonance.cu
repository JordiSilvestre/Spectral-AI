/**
 * @file ternary_resonance.cu
 * @brief SpectralAI v4.0 — Phase 2: Ternary Accumulator Units (rayos verdes)
 *
 * CONCEPTO (Idea 5 de SpectralAI v4.0 "Inception Engine"):
 * ========================================================
 *
 * Phase 1 (FP16 — rayos rojos):
 *   out = Σ W[i] · x[i]           ← multiplicaciones FP16 (Tensor Cores)
 *   RAM: alta (FP16 = 2 bytes por parámetro)
 *   Calor: alto
 *
 * Phase 2 (Ternario — rayos verdes):
 *   out = Σ { +x[i] si W[i]=+1    ← solo suma (no multiplicación)
 *           {  0    si W[i]= 0
 *           { -x[i] si W[i]=-1    ← solo resta
 *   RAM: ~10x menor (1.58 bits efectivos vs 32 bits)
 *   Calor: mínimo — no usa Tensor Cores
 *
 * APLICADO A SEMANSTRINGS:
 * ========================
 *   FP32:    W(ω) = s · tanh(Σ a_k·sin(kω) + b_k·cos(kω))     ← FMAs
 *   Ternario: W(ω) = s · tanh(Σ ternary_accum(a_t, b_t, ω))    ← adiciones
 *
 * KERNELS INCLUIDOS:
 * ==================
 *   1. ternaryStringResonance()        — función device inline (para OptiX)
 *   2. ternaryBatchEval()              — kernel batch para benchmarking
 *   3. ternaryVsFP32Benchmark()        — compara latencia Ternario vs FP32
 *   4. ternaryResonanceOptiX()         — versión para integrar en inception_kernels.cu
 *
 * COMPARATIVA DE OPERACIONES (M=8 modos):
 * =========================================
 *   FP32 resonance:    16 mul + 15 add + 1 tanh = 32 FLOPs
 *   Ternary resonance: 16 sin/cos + ~8 add/sub + 1 tanh = ~25 ops
 *                      (solo ~50% de los add son no-cero con sparsidad=50%)
 *
 * VENTAJA REAL:
 *   No es la reducción en operaciones — es la eliminación de la latencia
 *   de las multiplicaciones FP (que van por unidades de punto flotante).
 *   Las Ternary Accumulator Units son contadores de enteros + 1 tanh final.
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_types.h>
#include <stdint.h>
#include <stdio.h>

#include "../include/spectral_resonance.h"

// ============================================================================
// CONSTANTES TERNARIAS
// ============================================================================

/// Codificación ternaria: {-1, 0, +1} como int8_t
#define TERNARY_POS   ((int8_t) +1)
#define TERNARY_ZERO  ((int8_t)  0)
#define TERNARY_NEG   ((int8_t) -1)

// ============================================================================
// STRUCT: TernaryResonanceParams
// ============================================================================

/**
 * @brief Coeficientes Fourier cuantizados a ternario {-1, 0, +1}.
 *
 * Reemplaza ResonanceParams (FP32) en Phase 2.
 * Memoria: 16 bytes vs 68 bytes de ResonanceParams FP32.
 * (8 + 8 int8_t + 1 float scale = 17 bytes, con padding = 20 bytes)
 */
struct alignas(32) TernaryResonanceParams {
    int8_t   a[RESONANCE_NUM_MODES];    ///< Coef. seno ternarios {-1, 0, +1}
    int8_t   b[RESONANCE_NUM_MODES];    ///< Coef. coseno ternarios {-1, 0, +1}
    float    scale;                     ///< Factor de escala FP32 (único FP restante)
    float    outputScale;               ///< Escala de salida (outputScale del nodo)
    uint32_t semanticTag;               ///< Etiqueta semántica para debug
    uint32_t _pad;                      ///< Padding a 32 bytes
};

// ============================================================================
// DEVICE FUNCTION: Ternary Accumulator (Phase 2 core)
// ============================================================================

/**
 * @brief Calcula la resonancia Fourier usando acumulación ternaria.
 *
 * W(ω) = outputScale · tanh( scale · Σ_{k=1}^{M}
 *            {+sin(kω) si a_t[k]=+1 | +cos(kω) si b_t[k]=+1
 *             -sin(kω) si a_t[k]=-1 | -cos(kω) si b_t[k]=-1
 *              0       si a_t[k]=0  |  0        si b_t[k]=0}  )
 *
 * Las ramas if/else se convierten en instrucciones de selección de hardware
 * (SETP + SELP) — sin bifurcación real. El compilador nvcc las vectoriza.
 *
 * @param params     Coeficientes ternarios + escala
 * @param omega      Frecuencia de contexto ω [0, 2π]
 * @return           Peso de resonancia ternaria
 */
__device__ __forceinline__
float ternaryStringResonance(const TernaryResonanceParams& params, float omega) {
    float accum = 0.0f;

    // Bug 2.8 fix: The unrolled loop accesses all RESONANCE_NUM_MODES entries.
    // If params.a[]/b[] are not fully initialized (e.g., fewer active modes),
    // uninitialized values could cause incorrect accumulation.
    // Guard: TERNARY_ZERO entries (0) are skipped, so zero-initialized padding is safe.
    // Ensure callers always zero-initialize TernaryResonanceParams (see fp32ToTernary).
    #pragma unroll
    for (uint32_t k = 1; k <= RESONANCE_NUM_MODES; ++k) {
        const int8_t ak = params.a[k - 1];
        const int8_t bk = params.b[k - 1];

        // Skip entirely if both coefficients are zero (common with sparse ternary)
        if (ak == TERNARY_ZERO && bk == TERNARY_ZERO) continue;

        const float kw    = (float)k * omega;
        const float s_kw  = __sinf(kw);
        const float c_kw  = __cosf(kw);

        // Ternary accumulation: solo suma/resta según {-1, 0, +1}
        // El compilador convierte estas ramas en SELP (select predicate) sin divergencia
        if (ak == TERNARY_POS)       accum += s_kw;
        else if (ak == TERNARY_NEG)  accum -= s_kw;
        // a=0: no operación

        if (bk == TERNARY_POS)       accum += c_kw;
        else if (bk == TERNARY_NEG)  accum -= c_kw;
        // b=0: no operación
    }

    return params.outputScale * tanhf(params.scale * accum);
}

/**
 * @brief Variante sin struct — para compatibilidad con arrays planos desde Python.
 *
 * Los arrays ternary_a y ternary_b contienen los coeficientes de UNA string.
 */
__device__ __forceinline__
float ternaryStringResonanceRaw(
    const int8_t* __restrict__ ternary_a,
    const int8_t* __restrict__ ternary_b,
    uint32_t num_modes,
    float omega,
    float scale,
    float output_scale
) {
    float accum = 0.0f;
    const uint32_t M = min(num_modes, (uint32_t)RESONANCE_NUM_MODES);

    #pragma unroll
    for (uint32_t k = 1; k <= RESONANCE_NUM_MODES; ++k) {
        if (k <= M) {
            const float kw   = (float)k * omega;
            const float s_kw = __sinf(kw);
            const float c_kw = __cosf(kw);

            if (ternary_a[k - 1] == TERNARY_POS)       accum += s_kw;
            else if (ternary_a[k - 1] == TERNARY_NEG)  accum -= s_kw;

            if (ternary_b[k - 1] == TERNARY_POS)       accum += c_kw;
            else if (ternary_b[k - 1] == TERNARY_NEG)  accum -= c_kw;
        }
    }
    return output_scale * tanhf(scale * accum);
}

// ============================================================================
// KERNEL: Evaluación batch ternaria
// ============================================================================

/**
 * @brief Evalúa resonancia ternaria para todos los pares (string, omega) en un batch.
 *
 * Análogo a resonanceBatchEval() en inception_resonance.cu pero usando
 * acumulación ternaria en lugar de multiplicaciones FP32.
 *
 * Grid: <<< ceil(numStrings * numOmegas / 256), 256 >>>
 *
 * @param params      Array de TernaryResonanceParams [numStrings]
 * @param omegas      Array de frecuencias [numOmegas]
 * @param results     Buffer de salida [numStrings × numOmegas]
 * @param numStrings  Número de SemanticStrings
 * @param numOmegas   Número de valores ω
 */
extern "C" __global__
void ternaryBatchEval(
    const TernaryResonanceParams* __restrict__ params,
    const float*                  __restrict__ omegas,
    float*                                     results,
    uint32_t numStrings,
    uint32_t numOmegas
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStrings * numOmegas) return;

    const uint32_t strIdx   = tid / numOmegas;
    const uint32_t omegaIdx = tid % numOmegas;

    if (strIdx >= numStrings) return;

    results[tid] = ternaryStringResonance(params[strIdx], omegas[omegaIdx]);
}

// ============================================================================
// KERNEL: Benchmark Ternario vs FP32
// ============================================================================

/**
 * @brief Mide el tiempo de ejecución de resonancia ternaria vs FP32.
 *
 * Ejecuta N iteraciones de cada versión en paralelo (un warp para FP32,
 * un warp para ternario) y escribe los resultados para timing externo.
 *
 * Grid: <<< 1, 256 >>>
 */
extern "C" __global__
void ternaryVsFP32Benchmark(
    const TernaryResonanceParams* __restrict__ ternary_params,
    const SemanticString*         __restrict__ fp32_strings,
    const float*                  __restrict__ omegas,
    float*                                     ternary_results,
    float*                                     fp32_results,
    uint32_t numStrings,
    uint32_t numOmegas,
    uint32_t iterations
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStrings || tid >= numOmegas) return;

    // Cada thread procesa su propio string y omega
    float t_accum = 0.0f;
    float f_accum = 0.0f;

    for (uint32_t iter = 0; iter < iterations; ++iter) {
        const float omega = omegas[tid % numOmegas];

        // Resonancia ternaria
        if (tid < numStrings) {
            t_accum += ternaryStringResonance(ternary_params[tid], omega);
        }

        // Resonancia FP32 (para comparación)
        if (tid < numStrings) {
            f_accum += semanticStringResonance(fp32_strings[tid].resonance, omega);
        }
    }

    if (ternary_results) ternary_results[tid] = t_accum / (float)iterations;
    if (fp32_results)    fp32_results[tid]    = f_accum / (float)iterations;
}

// ============================================================================
// KERNEL: Conversión FP32 → Ternario (para ejecutar en GPU)
// ============================================================================

/**
 * @brief Convierte un array de ResonanceParams FP32 a TernaryResonanceParams.
 *
 * Equivalente al ternary_quantize.py pero en GPU.
 * Útil para conversión on-the-fly sin volver a CPU.
 *
 * @param fp32_strings    Strings FP32 de entrada [numStrings]
 * @param ternary_out     Params ternarios de salida [numStrings]
 * @param threshold       Umbral de cuantización τ
 * @param numStrings      Número de SemanticStrings
 */
extern "C" __global__
void fp32ToTernary(
    const SemanticString*  __restrict__ fp32_strings,
    TernaryResonanceParams*             ternary_out,
    float                               threshold,
    uint32_t                            numStrings
) {
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= numStrings) return;

    const ResonanceParams& src = fp32_strings[tid].resonance;
    TernaryResonanceParams& dst = ternary_out[tid];

    // Cuantizar coeficientes
    #pragma unroll
    for (uint32_t k = 0; k < RESONANCE_NUM_MODES; ++k) {
        dst.a[k] = (src.a[k] >  threshold) ? TERNARY_POS :
                   (src.a[k] < -threshold) ? TERNARY_NEG : TERNARY_ZERO;
        dst.b[k] = (src.b[k] >  threshold) ? TERNARY_POS :
                   (src.b[k] < -threshold) ? TERNARY_NEG : TERNARY_ZERO;
    }

    // Calcular escala óptima (promedio de coeficientes no-cero)
    float sum_nonzero = 0.0f;
    int   count = 0;
    for (uint32_t k = 0; k < RESONANCE_NUM_MODES; ++k) {
        if (dst.a[k] != TERNARY_ZERO) { sum_nonzero += fabsf(src.a[k]); ++count; }
        if (dst.b[k] != TERNARY_ZERO) { sum_nonzero += fabsf(src.b[k]); ++count; }
    }
    dst.scale       = (count > 0) ? (sum_nonzero / (float)count) : 1.0f;
    dst.outputScale = src.outputScale;
    dst.semanticTag = src.semanticTag;
    dst._pad        = 0;
}

// ============================================================================
// DEBUG: Imprimir coeficientes ternarios
// ============================================================================

extern "C" __global__
void ternaryPrintParams(
    const TernaryResonanceParams* params,
    uint32_t stringIdx
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    const TernaryResonanceParams& p = params[stringIdx];
    printf("[Ternary #%u] scale=%.4f output_scale=%.4f tag=%u\n",
           stringIdx, p.scale, p.outputScale, p.semanticTag);
    printf("  a: ");
    for (int k = 0; k < RESONANCE_NUM_MODES; ++k)
        printf("%+d ", (int)p.a[k]);
    printf("\n  b: ");
    for (int k = 0; k < RESONANCE_NUM_MODES; ++k)
        printf("%+d ", (int)p.b[k]);
    printf("\n");

    // Evaluar en 4 puntos de muestra
    float test_omegas[4] = { 0.0f, 0.785f, 1.571f, 3.142f };
    printf("  W(omega): ");
    for (int i = 0; i < 4; ++i) {
        printf("%.4f ", ternaryStringResonance(p, test_omegas[i]));
    }
    printf("\n");
}

// ============================================================================
// HOST-CALLABLE WRAPPER (para tests C++)
// ============================================================================

/**
 * @brief Wrapper de host para lanzar ternaryBatchEval con manejo de errores.
 *
 * @param params          Device pointer a TernaryResonanceParams
 * @param omegas          Device pointer a omega values
 * @param results         Device pointer a output buffer
 * @param numStrings      Número de strings
 * @param numOmegas       Número de omegas
 * @param stream          CUDA stream
 */
#ifdef __cplusplus
extern "C"
cudaError_t launchTernaryBatchEval(
    const TernaryResonanceParams* params,
    const float* omegas,
    float* results,
    uint32_t numStrings,
    uint32_t numOmegas,
    cudaStream_t stream
) {
    const uint32_t total = numStrings * numOmegas;
    const uint32_t blockSz = 256;
    const uint32_t gridSz  = (total + blockSz - 1) / blockSz;
    ternaryBatchEval<<<gridSz, blockSz, 0, stream>>>(
        params, omegas, results, numStrings, numOmegas
    );
    return cudaGetLastError();
}
#endif // __cplusplus
