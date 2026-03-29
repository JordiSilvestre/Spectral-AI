/**
 * ray_attention.cu
 *
 * Kernel principal del mecanismo de atención óptica LiquidBit Zero-Matrix.
 *
 * Este kernel implementa la traversal de rayos en un BVH acelerada por OptiX.
 * Cada rayo representa una "dimensión de pensamiento" que se lanza desde el token
 * query y acumula pesos de atención de los tokens relevantes golpeados.
 *
 * Complejidad: O(N log N) para N tokens en la secuencia
 * (vs O(N²) de la atención tradicional)
 */

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "../include/optical_attention.h"
#include "../include/token_geometry.h"

/* ============================================================================
 * CONSTANTES Y TIPOS GLOBALES
 * ============================================================================
 */

// Buffer device para los TokenNodes del BVH
extern "C" __constant__ TokenNode* c_token_nodes;

// Buffer device para los resultados de atención
extern "C" __constant__ AttentionResult* c_attention_results;

// Buffer device para los parámetros del BVH (acceleration structure)
extern "C" __constant__ OptixTraversableHandle c_bvh_handle;

// Número total de tokens en la secuencia
extern "C" __constant__ uint32_t c_num_tokens;

// Número de rayos generados por query token
extern "C" __constant__ uint32_t c_rays_per_query;

#if LIQUIDBIT_SPECTRAL_ENABLED
// W_spectral projection matrix: [LIQUIDBIT_CUDA_SPECTRAL_DIM x LIQUIDBIT_EMBEDDING_DIM]
// Used by the kernel to compute spectral color from query embeddings.
extern "C" __constant__ float c_W_spectral_kernel[LIQUIDBIT_CUDA_SPECTRAL_DIM * LIQUIDBIT_EMBEDDING_DIM];

/**
 * Compute spectral color from a token embedding using the W_spectral matrix.
 * Identical to the version in ray_generation.cu but uses c_W_spectral_kernel.
 */
__device__ static void kernel_compute_spectral_color(
    const half* embedding,
    float* out_color
) {
    float norm_sq = 0.0f;
    for (uint32_t j = 0; j < LIQUIDBIT_CUDA_SPECTRAL_DIM; ++j) {
        float acc = 0.0f;
        const uint32_t row_offset = j * LIQUIDBIT_EMBEDDING_DIM;
        for (uint32_t i = 0; i < LIQUIDBIT_EMBEDDING_DIM; ++i) {
            acc += c_W_spectral_kernel[row_offset + i] * __half2float(embedding[i]);
        }
        out_color[j] = acc;
        norm_sq += acc * acc;
    }
    float inv_norm = (norm_sq > 1e-6f) ? rsqrtf(norm_sq) : 1.0f;
    for (uint32_t j = 0; j < LIQUIDBIT_CUDA_SPECTRAL_DIM; ++j) {
        out_color[j] *= inv_norm;
    }
}
#endif // LIQUIDBIT_SPECTRAL_ENABLED

/* ============================================================================
 * FUNCIÓN HELPER: insert_top_token
 *
 * Inserta un token en la lista top-K mantenida en orden descendente de peso.
 * Si la lista está llena, descarta el token con menor peso si el nuevo es mayor.
 * ============================================================================
 */
__device__ static void insert_top_token(
    uint32_t* top_tokens,
    float*    top_weights,
    uint32_t& count,
    uint32_t  token_id,
    float     weight
) {
    // Buscar posición de inserción (lista ordenada descendentemente)
    uint32_t capacity = LIQUIDBIT_MAX_TOP_TOKENS;

    if (count < capacity) {
        // Lista no llena: insertar en la posición correcta
        uint32_t pos = count;
        while (pos > 0 && top_weights[pos - 1] < weight) {
            top_tokens[pos]  = top_tokens[pos - 1];
            top_weights[pos] = top_weights[pos - 1];
            pos--;
        }
        top_tokens[pos]  = token_id;
        top_weights[pos] = weight;
        count++;
    } else if (weight > top_weights[capacity - 1]) {
        // Lista llena pero el nuevo peso supera al mínimo: reemplazar
        uint32_t pos = capacity - 1;
        while (pos > 0 && top_weights[pos - 1] < weight) {
            top_tokens[pos]  = top_tokens[pos - 1];
            top_weights[pos] = top_weights[pos - 1];
            pos--;
        }
        top_tokens[pos]  = token_id;
        top_weights[pos] = weight;
    }
}

/* ============================================================================
 * KERNEL PRINCIPAL: ray_traced_attention_kernel
 *
 * Entrada:
 *   query_tokens: IDs de los tokens query (uno por thread)
 *   query_positions: posiciones 3D de los tokens query
 *   num_queries: número de tokens query
 *   ray_directions: direcciones normalizadas de los rayos (num_queries × rays_per_query)
 *
 * Salida:
 *   attention_results: array de AttentionResult (uno por query)
 *
 * Flujo:
 * 1. Cada thread lee un token query
 * 2. Genera rays_per_query rayos desde la posición del query
 * 3. Para cada rayo, lanza optixTrace() y acumula los resultados
 * 4. Normaliza los pesos de atención
 * 5. Escribe los resultados en el buffer de salida
 *
 * Nota matemática: La fórmula de attention decay es:
 *
 *     attention_weight = E₀ · exp(-λ · d_semantic)
 *
 * donde:
 *   - E₀: energía inicial del rayo (siempre 1.0)
 *   - λ: coeficiente de absorción semántica (LIQUIDBIT_LAMBDA ≈ 0.1)
 *   - d_semantic: distancia euclídea 3D entre los centroides del rayo y el token golpeado
 *
 * Interpretación física:
 * - Similar a la atenuación de luz en un medio absorbente (Beer-Lambert Law)
 * - Tokens "semánticamente distantes" (baja similitud coseno) reciben menos peso
 * - El rayo pierde energía exponencialmente a medida que se propaga
 * - La complejidad O(N log N) surge del árbol BVH que descarta mitades del espacio
 *   en cada nivel de traversal
 *
 * ============================================================================
 */
__global__ void ray_traced_attention_kernel(
    const uint32_t* query_tokens,
    const float3* query_positions,
    const uint32_t num_queries,
    const float3* ray_directions,  // [num_queries * rays_per_query]
    AttentionResult* output_results
) {
    // Identificador global del thread
    uint32_t thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Salir si estamos fuera del rango de queries
    if (thread_idx >= num_queries) {
        return;
    }

    // Leer el token query asignado a este thread
    uint32_t query_token_id = query_tokens[thread_idx];
    float3 query_position = query_positions[thread_idx];

    // Inicializar el resultado de salida para esta query
    AttentionResult result;
    result.query_token_id = query_token_id;
    result.hit_count = 0;
    result.total_attention = 0.0f;

    // Memoria compartida para reducción local de hit count
    // +32 padding to avoid shared memory bank conflicts (Bug 2.12)
    __shared__ uint32_t shared_hit_count[256 + 32];

    // Inicializar shared memory
    if (threadIdx.x < num_queries) {
        shared_hit_count[threadIdx.x] = 0;
    }
    __syncthreads();

    // Generar y lanzar rayos desde esta query
    // Cada thread lanza rays_per_query rayos en paralelo
    uint32_t rays_base_idx = thread_idx * c_rays_per_query;

    // Variables para acumular resultados de todos los rayos de esta query
    uint32_t total_hit_count = 0;
    float total_attention_weight = 0.0f;
    uint32_t accumulated_top_tokens[LIQUIDBIT_MAX_TOP_TOKENS];
    float accumulated_top_weights[LIQUIDBIT_MAX_TOP_TOKENS];
    // Bug 2.1 fix: separate counter for top-K accumulation to prevent buffer overflow.
    // total_hit_count grows unbounded across rays, but the top-K array has fixed capacity.
    uint32_t accumulated_top_count = 0;

#if LIQUIDBIT_SPECTRAL_ENABLED
    // Compute spectral color once per query (shared across all rays of this query).
    // The color encodes the conversational context and remains constant per query.
    float query_spectral_color[LIQUIDBIT_CUDA_SPECTRAL_DIM];
    {
        // Use the query token's embedding from the global TokenNode buffer
        const TokenNode& qt = c_token_nodes[thread_idx];
        kernel_compute_spectral_color(qt.embedding, query_spectral_color);
    }
#endif

    for (uint32_t ray_idx = 0; ray_idx < c_rays_per_query; ray_idx++) {
        // Obtener la dirección normalizada del rayo actual
        uint32_t ray_dir_idx = rays_base_idx + ray_idx;
        float3 ray_direction = ray_directions[ray_dir_idx];

        // Normalizar dirección (debería ya estar normalizada, pero por seguridad)
        float dir_len = sqrtf(ray_direction.x * ray_direction.x +
                              ray_direction.y * ray_direction.y +
                              ray_direction.z * ray_direction.z);
        if (dir_len > 1e-6f) {
            ray_direction.x /= dir_len;
            ray_direction.y /= dir_len;
            ray_direction.z /= dir_len;
        }

        // ====================================================================
        // LANZAR RAYO EN EL BVH
        //
        // OptixTrace realiza la traversal del árbol BVH acelerada por hardware.
        // Los programas OptiX (ClosestHit, Miss) se invocan automáticamente.
        // ====================================================================

        // Preparar el payload del rayo
        RayPayload ray_payload;
        ray_payload.accumulated_attention = 0.0f;
        ray_payload.energy_remaining = 1.0f;
        ray_payload.hit_count = 0;
        ray_payload.ray_origin_x = __float_as_uint(query_position.x);
        ray_payload.ray_origin_y = __float_as_uint(query_position.y);
        ray_payload.ray_origin_z = __float_as_uint(query_position.z);

#if LIQUIDBIT_SPECTRAL_ENABLED
        // Copy the pre-computed spectral color into the ray payload
        for (uint32_t s = 0; s < LIQUIDBIT_CUDA_SPECTRAL_DIM; ++s) {
            ray_payload.spectral_color[s] = query_spectral_color[s];
        }
        ray_payload.selected_matrix_block_id = UINT32_MAX;
        ray_payload.refraction_angle_deg = 0.0f;
#endif

        // Parámetros de optixTrace
        // t_min: distancia mínima (cerca del rayo)
        // t_max: distancia máxima (lejos del rayo)
        float t_min = 0.001f;      // Offset pequeño para evitar auto-intersecciones
        float t_max = 1e16f;       // Distancia "infinita"

        // Convertir payload a uint32_t para optixTrace (requiere variables lvalue)
        uint32_t p0 = __float_as_uint(ray_payload.accumulated_attention);
        uint32_t p1 = __float_as_uint(ray_payload.energy_remaining);
        uint32_t p2 = ray_payload.hit_count;

#if LIQUIDBIT_SPECTRAL_ENABLED
        // Pack spectral color into payload words p3..p18 (16 floats as uint32)
        uint32_t ps[LIQUIDBIT_CUDA_SPECTRAL_DIM];
        for (uint32_t s = 0; s < LIQUIDBIT_CUDA_SPECTRAL_DIM; ++s) {
            ps[s] = __float_as_uint(ray_payload.spectral_color[s]);
        }
        uint32_t p_blk = ray_payload.selected_matrix_block_id;
        uint32_t p_ang = __float_as_uint(ray_payload.refraction_angle_deg);

        // Lanzar el rayo with spectral payload (21 words total)
        optixTrace(
            c_bvh_handle,
            query_position,
            ray_direction,
            t_min, t_max, 0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            0, 0, 0,
            p0, p1, p2,
            ps[0],  ps[1],  ps[2],  ps[3],
            ps[4],  ps[5],  ps[6],  ps[7],
            ps[8],  ps[9],  ps[10], ps[11],
            ps[12], ps[13], ps[14], ps[15],
            p_blk, p_ang
        );

        // Read back spectral results
        ray_payload.selected_matrix_block_id = p_blk;
        ray_payload.refraction_angle_deg = __uint_as_float(p_ang);
#else
        // Lanzar el rayo (monochrome path)
        optixTrace(
            c_bvh_handle,              // Acceleration structure (BVH)
            query_position,            // Ray origin (posición del query)
            ray_direction,             // Ray direction (normalizada)
            t_min,                     // t_min
            t_max,                     // t_max
            0.0f,                      // ray_time (no usamos temporal)
            OptixVisibilityMask(255),  // Visibility mask (todos los objetos visibles)
            OPTIX_RAY_FLAG_NONE,       // Ray flags
            0,                         // SBT offset
            0,                         // SBT stride
            0,                         // Miss SBT index
            p0, p1, p2                 // Payloads (uint32_t por referencia)
        );
#endif // LIQUIDBIT_SPECTRAL_ENABLED

        // Después de optixTrace, p0/p1/p2 han sido actualizados por ClosestHit o Miss
        ray_payload.accumulated_attention = __uint_as_float(p0);
        ray_payload.energy_remaining      = __uint_as_float(p1);
        ray_payload.hit_count             = p2;

        // Acumular resultados de este rayo a los totales de la query
        total_attention_weight += ray_payload.accumulated_attention;
        total_hit_count += ray_payload.hit_count;

        // Fusionar los top-K tokens de este rayo con los acumulados
        for (uint32_t i = 0; i < ray_payload.hit_count && i < LIQUIDBIT_MAX_TOP_TOKENS; i++) {
            uint32_t token_id = ray_payload.top_tokens[i];
            float weight = ray_payload.top_weights[i];

            // Insertar en la lista acumulada (mantener ordenado)
            // Bug 2.1 fix: use accumulated_top_count (bounded by capacity)
            // instead of total_hit_count (unbounded across rays)
            insert_top_token(
                accumulated_top_tokens,
                accumulated_top_weights,
                accumulated_top_count,
                token_id,
                weight
            );
        }
    }

    // ========================================================================
    // NORMALIZACIÓN Y ESCRITURA DE RESULTADOS
    // ========================================================================

    // Normalizar pesos de atención (sum-to-1)
    if (total_attention_weight > 1e-6f) {
        for (uint32_t i = 0; i < accumulated_top_count; i++) {
            accumulated_top_weights[i] /= total_attention_weight;
        }
    }

    // Escribir resultados al buffer global
    result.total_attention = total_attention_weight;
    result.hit_count = accumulated_top_count;

    for (uint32_t i = 0; i < result.hit_count; i++) {
        result.top_token_ids[i] = accumulated_top_tokens[i];
        result.top_attention_weights[i] = accumulated_top_weights[i];
    }

    output_results[thread_idx] = result;
}

/* ============================================================================
 * KERNEL HELPER: reduce_and_normalize_attention
 *
 * Ejecuta una reducción multi-block para normalizar los pesos de atención
 * y calcular las estadísticas finales.
 *
 * (Usado si se requiere reducción multi-GPU o post-procesamiento)
 * ============================================================================
 */
__global__ void reduce_and_normalize_attention(
    AttentionResult* results,
    uint32_t num_results
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= num_results) {
        return;
    }

    AttentionResult& result = results[idx];

    // Normalizar pesos de atención
    float total_weight = 0.0f;
    for (uint32_t i = 0; i < result.hit_count; i++) {
        total_weight += result.top_attention_weights[i];
    }

    if (total_weight > 1e-6f) {
        for (uint32_t i = 0; i < result.hit_count; i++) {
            result.top_attention_weights[i] /= total_weight;
        }
    }
}

/* ============================================================================
 * PUNTO DE ENTRADA: Función wrapper para host
 *
 * Llamada desde C++ host code para lanzar el kernel
 * ============================================================================
 */
extern "C" cudaError_t launch_ray_traced_attention_kernel(
    uint32_t grid_size,
    uint32_t block_size,
    const uint32_t* d_query_tokens,
    const float3* d_query_positions,
    uint32_t num_queries,
    const float3* d_ray_directions,
    AttentionResult* d_output_results
) {
    // Bug 2.16 fix: validate inputs before kernel launch
    if (!d_query_tokens || !d_query_positions || !d_ray_directions || !d_output_results) {
        return cudaErrorInvalidValue;
    }
    if (num_queries == 0 || grid_size == 0 || block_size == 0) {
        return cudaErrorInvalidConfiguration;
    }

    ray_traced_attention_kernel<<<grid_size, block_size>>>(
        d_query_tokens,
        d_query_positions,
        num_queries,
        d_ray_directions,
        d_output_results
    );

    return cudaGetLastError();
}

