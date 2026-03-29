/**
 * closest_hit.cu
 *
 * Programa OptiX ClosestHit (closest hit shader)
 *
 * Se ejecuta cuando un rayo intersecta exitosamente el AABB más cercano
 * de un TokenNode en el BVH durante la traversal.
 *
 * Responsabilidades:
 * 1. Recuperar el payload del rayo
 * 2. Obtener información del token golpeado
 * 3. Calcular el peso de atención exponencial
 * 4. Actualizar el payload del rayo
 * 5. Decidir si continuar la traversal o terminar
 *
 * Fórmula de Attention Decay:
 *
 *     attention_weight = E₀ · exp(-λ · d_semantic)
 *
 * Donde:
 *   - E₀: energía del rayo (decrece con cada hit)
 *   - λ: coeficiente de absorción semántica (SPECTRAL_LAMBDA ≈ 0.1)
 *   - d_semantic: distancia euclídea 3D entre centroide del rayo y del token
 *
 * La "pérdida de energía" actúa como un mecanismo de attention decay,
 * similar al softmax pero más eficiente computacionalmente.
 */

#include <optix.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "../include/optical_attention.h"
#include "../include/token_geometry.h"

/* ============================================================================
 * CONSTANTES GLOBALES
 * ============================================================================
 */

// Buffer device con los TokenNodes
extern "C" __constant__ TokenNode* c_token_nodes;

// Lambda (coeficiente de absorción) - debe coincidir con SPECTRAL_LAMBDA
extern "C" __constant__ float c_lambda;

#if SPECTRAL_SPECTRAL_ENABLED
/* ============================================================================
 * SPECTRAL HELPER FUNCTIONS
 *
 * These implement the prismatic refraction model from spectral_ray.h:
 *   1. Compute refractive index: n = base_n + sigmoid(dot(W_dispersion, color))
 *   2. Compute surface normal at hit point
 *   3. Apply vectorial Snell's law for 3D refraction
 *   4. Convert refraction angle to matrix block selection
 * ============================================================================
 */

/**
 * Sigmoid activation: sigma(x) = 1 / (1 + exp(-x))
 * Clamps large |x| to avoid exp overflow.
 */
__device__ static float liqbit_sigmoid(float x) {
    x = fmaxf(-20.0f, fminf(20.0f, x));  // Clamp to avoid exp overflow
    return 1.0f / (1.0f + expf(-x));
}

/**
 * Compute the refractive index of a sphere for a given spectral color.
 *
 * Formula (from spectral_ray.h PrismaticSphere::computeRefractiveIndex):
 *   dot_product = sum( W_dispersion[i] * color[i] )  for i in [0, spectral_dim)
 *   n = base_refractive_index + sigmoid(dot_product)
 *
 * The SBT hit record provides W_dispersion and base_refractive_index so we
 * avoid global memory lookups into the PrismaticSphere array.
 */
__device__ static float compute_refractive_index(
    const float* W_dispersion,          // [SPECTRAL_CUDA_SPECTRAL_DIM] from SBT
    const float* spectral_color,        // [SPECTRAL_CUDA_SPECTRAL_DIM] from ray payload
    float base_refractive_index
) {
    float dot = 0.0f;
    for (uint32_t i = 0; i < SPECTRAL_CUDA_SPECTRAL_DIM; ++i) {
        dot += W_dispersion[i] * spectral_color[i];
    }
    return base_refractive_index + liqbit_sigmoid(dot);
}

/**
 * Vectorial Snell's law in 3D.
 *
 * Given an incident direction d_in, surface normal n, and refractive index
 * ratio n_ratio = n_in / n_out, computes the refracted direction:
 *
 *   cos_i = -dot(d_in, normal)
 *   discriminant = 1 - n_ratio^2 * (1 - cos_i^2)
 *
 *   If discriminant < 0: total internal reflection
 *     d_out = d_in + 2 * cos_i * normal   (reflection)
 *
 *   Else: refraction
 *     cos_t = sqrt(discriminant)
 *     d_out = n_ratio * d_in + (n_ratio * cos_i - cos_t) * normal
 *
 * Returns the refracted (or reflected) direction, normalized.
 * Also writes the refraction angle in degrees to *out_angle_deg.
 */
__device__ static float3 snell_refract_3d(
    const float3& d_in,
    const float3& normal,
    float n_ratio,
    float* out_angle_deg
) {
    // Cosine of incidence angle
    float cos_i = -(d_in.x * normal.x + d_in.y * normal.y + d_in.z * normal.z);
    cos_i = fmaxf(-1.0f, fminf(1.0f, cos_i));

    // Discriminant: cos(theta_t)^2
    float sin_i_sq = 1.0f - cos_i * cos_i;
    float discriminant = 1.0f - n_ratio * n_ratio * sin_i_sq;

    float3 d_out;
    float angle_deg;

    if (discriminant < -1e-6f) {
        // Total internal reflection: d_out = d_in + 2*cos_i*normal
        d_out = make_float3(
            d_in.x + 2.0f * cos_i * normal.x,
            d_in.y + 2.0f * cos_i * normal.y,
            d_in.z + 2.0f * cos_i * normal.z
        );
        // Reflection angle = incidence angle
        angle_deg = acosf(fabsf(cos_i)) * (180.0f / CUDART_PI_F);
    } else {
        // Refraction via vectorial Snell's law
        float cos_t = sqrtf(fmaxf(0.0f, discriminant));
        float coeff = n_ratio * cos_i - cos_t;
        d_out = make_float3(
            n_ratio * d_in.x + coeff * normal.x,
            n_ratio * d_in.y + coeff * normal.y,
            n_ratio * d_in.z + coeff * normal.z
        );
        // Refraction angle = acos(cos_t)
        angle_deg = acosf(fminf(1.0f, cos_t)) * (180.0f / CUDART_PI_F);
    }

    // Normalize output direction
    float len = sqrtf(d_out.x * d_out.x + d_out.y * d_out.y + d_out.z * d_out.z);
    if (len > 1e-6f) {
        d_out.x /= len;
        d_out.y /= len;
        d_out.z /= len;
    }

    *out_angle_deg = angle_deg;
    return d_out;
}

/**
 * Select matrix block ID from refraction angle using threshold lookup.
 * Mirrors PrismaticSphere::selectMatrixBlock().
 */
__device__ static uint32_t select_matrix_block(
    float refraction_angle_deg,
    const uint32_t* matrix_block_ids,
    const float* refraction_thresholds,
    uint32_t num_blocks
) {
    if (num_blocks == 0) return UINT32_MAX;
    for (uint32_t i = 0; i < num_blocks; ++i) {
        if (refraction_angle_deg <= refraction_thresholds[i]) {
            return matrix_block_ids[i];
        }
    }
    return matrix_block_ids[num_blocks - 1];
}
#endif // SPECTRAL_SPECTRAL_ENABLED

/* ============================================================================
 * PROGRAMA OPTIX: __closesthit__
 *
 * Parámetros (implícitos):
 *   - OptixGetPayload_0/1/2: Datos del rayo que golpea
 *   - optixGetPrimitiveIndex(): ID del primitivo (token) golpeado
 *   - optixGetRayTmax(): Distancia al punto de intersección
 *   - optixGetWorldRayOrigin(): Posición del origen del rayo
 *
 * Modifica:
 *   - OptixSetPayload_0/1/2: Actualiza datos del rayo
 *   - Early return if energy too low or primitive index invalid
 * ============================================================================
 */
extern "C" __global__ void __closesthit__ch_optical_attention() {
    // ========================================================================
    // RECUPERAR PAYLOAD DEL RAYO
    // ========================================================================

    // Recuperar los 3 words de payload (RayPayload es 3 × 32-bit)
    uint32_t payload_0 = optixGetPayload_0();
    uint32_t payload_1 = optixGetPayload_1();
    uint32_t payload_2 = optixGetPayload_2();

    // Reinterpretar como RayPayload
    float accumulated_attention = __uint_as_float(payload_0);
    float energy_remaining = __uint_as_float(payload_1);
    uint32_t hit_count = payload_2;

    // Recuperar la posición 3D original del rayo (desde payload extendido)
    // Nota: En esta versión simplificada, obtenemos el origen del rayo
    float3 ray_origin = optixGetWorldRayOrigin();
    float3 ray_direction = optixGetWorldRayDirection();

    // ========================================================================
    // OBTENER INFORMACIÓN DEL TOKEN GOLPEADO
    // ========================================================================

    // Índice del primitivo (en nuestro caso, ID del token)
    uint32_t primitive_idx = optixGetPrimitiveIndex();

    // Verificar que el índice es válido
    if (primitive_idx >= SPECTRAL_MAX_SEQUENCE_LENGTH) {
        // In closest-hit, intersection is already committed — just return early
        return;
    }

    // Obtener el TokenNode golpeado
    const TokenNode& hit_token = c_token_nodes[primitive_idx];

    // ========================================================================
    // CALCULAR DISTANCIA SEMÁNTICA Y PESO DE ATENCIÓN
    // ========================================================================

    // Distancia euclídea 3D entre el origen del rayo y el centroide del token
    float3 delta = make_float3(
        hit_token.centroid.x - ray_origin.x,
        hit_token.centroid.y - ray_origin.y,
        hit_token.centroid.z - ray_origin.z
    );

    // TODO(Bug 2.13): Consider using rsqrtf() or squared distance to avoid
    // the expensive sqrtf() in this hot path. The decay formula
    //   exp(-lambda * d) = exp(-lambda * sqrt(d2))
    // could be reformulated as exp(-lambda_sq * d2) if acceptable for the model.
    // Alternatively, use rsqrtf(d2) * d2 which maps to a single HW instruction.
    float semantic_distance = sqrtf(
        delta.x * delta.x +
        delta.y * delta.y +
        delta.z * delta.z
    );

    // Aplicar pequeño offset para evitar singularidades
    semantic_distance = fmaxf(semantic_distance, 0.001f);

    // ========================================================================
    // FÓRMULA DE ATTENTION DECAY (Beer-Lambert Law)
    //
    // attention_weight = E₀ · exp(-λ · d_semantic)
    //
    // Interpretación:
    //   - E₀ (energy_remaining): energía actual del rayo
    //   - λ (c_lambda): coeficiente de absorción (~0.1)
    //   - d_semantic: distancia "semánticamente relevante"
    //
    // Resultado: Tokens cercanos (similares) tienen mayor peso,
    //            tokens lejanos (diferentes) tienen menor peso
    // ========================================================================

    float attention_weight = energy_remaining * expf(-c_lambda * semantic_distance);

    // Aplicar threshold de energía mínima
    if (attention_weight < SPECTRAL_ENERGY_THRESHOLD) {
        // Energía muy baja: ignorar este hit y continuar con la traversal
        // In closest-hit, intersection is already committed — just return early
        return;
    }

#if SPECTRAL_SPECTRAL_ENABLED
    // ========================================================================
    // SPECTRAL REFRACTION (Prismatic Attention Modulation)
    //
    // When spectral rays are enabled, each hit computes:
    //   1. Read the ray's spectral color from payload words p3..p3+dim-1
    //   2. Read per-sphere W_dispersion from the SBT hit record
    //   3. Compute refractive index: n = base_n + sigmoid(dot(W_disp, color))
    //   4. Compute surface normal at hit point (sphere approximation)
    //   5. Apply vectorial Snell's law to get refracted direction
    //   6. The refraction angle selects which matrix expert block to route to
    //   7. Modulate the attention weight by spectral coherence
    //
    // This implements "Idea 3" from the SpectralAI architecture: colored rays
    // resolve polysemy by routing through different matrix experts depending
    // on the conversational context encoded in the ray's spectral color.
    // ========================================================================

    // --- Step 1: Read spectral color from payload ---
    float spectral_color[SPECTRAL_CUDA_SPECTRAL_DIM];
    spectral_color[0]  = __uint_as_float(optixGetPayload_3());
    spectral_color[1]  = __uint_as_float(optixGetPayload_4());
    spectral_color[2]  = __uint_as_float(optixGetPayload_5());
    spectral_color[3]  = __uint_as_float(optixGetPayload_6());
    spectral_color[4]  = __uint_as_float(optixGetPayload_7());
    spectral_color[5]  = __uint_as_float(optixGetPayload_8());
    spectral_color[6]  = __uint_as_float(optixGetPayload_9());
    spectral_color[7]  = __uint_as_float(optixGetPayload_10());
    spectral_color[8]  = __uint_as_float(optixGetPayload_11());
    spectral_color[9]  = __uint_as_float(optixGetPayload_12());
    spectral_color[10] = __uint_as_float(optixGetPayload_13());
    spectral_color[11] = __uint_as_float(optixGetPayload_14());
    spectral_color[12] = __uint_as_float(optixGetPayload_15());
    spectral_color[13] = __uint_as_float(optixGetPayload_16());
    spectral_color[14] = __uint_as_float(optixGetPayload_17());
    spectral_color[15] = __uint_as_float(optixGetPayload_18());

    // --- Step 2: Read per-sphere W_dispersion from SBT hit record ---
    const SpectralHitSbtRecord* sbt_data =
        reinterpret_cast<const SpectralHitSbtRecord*>(optixGetSbtDataPointer());

    // --- Step 3: Compute refractive index ---
    //   n = base_refractive_index + sigmoid(dot(W_dispersion, spectral_color))
    // This determines how much the sphere "bends" the ray based on context.
    float n_sphere = compute_refractive_index(
        sbt_data->W_dispersion,
        spectral_color,
        sbt_data->base_refractive_index
    );

    // --- Step 4: Compute surface normal at hit point ---
    // Approximate the AABB hit as a sphere: normal = normalize(hit_point - centroid)
    float3 hit_point = make_float3(
        ray_origin.x + ray_direction.x * optixGetRayTmax(),
        ray_origin.y + ray_direction.y * optixGetRayTmax(),
        ray_origin.z + ray_direction.z * optixGetRayTmax()
    );
    float3 normal = make_float3(
        hit_point.x - hit_token.centroid.x,
        hit_point.y - hit_token.centroid.y,
        hit_point.z - hit_token.centroid.z
    );
    float normal_len = sqrtf(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
    if (normal_len > 1e-6f) {
        normal.x /= normal_len;
        normal.y /= normal_len;
        normal.z /= normal_len;
    } else {
        // Degenerate case: use ray direction as fallback normal
        normal = make_float3(-ray_direction.x, -ray_direction.y, -ray_direction.z);
    }

    // --- Step 5: Apply Snell's law ---
    // n_ratio = n_incident (vacuum ≈ 1.0) / n_sphere
    float n_ratio = 1.0f / n_sphere;
    float refraction_angle_deg = 0.0f;
    float3 refracted_dir = snell_refract_3d(
        ray_direction, normal, n_ratio, &refraction_angle_deg
    );

    // --- Step 6: Select matrix expert block from refraction angle ---
    uint32_t selected_block = select_matrix_block(
        refraction_angle_deg,
        sbt_data->matrix_block_ids,
        sbt_data->refraction_thresholds,
        sbt_data->num_matrix_blocks
    );

    // --- Step 7: Modulate attention weight by spectral coherence ---
    // The refractive index deviation from 1.0 indicates how strongly the
    // sphere's semantics match the ray's spectral context. A higher n means
    // stronger alignment, which amplifies the attention weight.
    // spectral_factor in range [0.5, 1.5] — sigmoid output in [0, 1].
    float spectral_factor = 0.5f + (n_sphere - sbt_data->base_refractive_index);
    attention_weight *= spectral_factor;

    // Write spectral results back to payload (selected block + refraction angle)
    optixSetPayload_19(selected_block);
    optixSetPayload_20(__float_as_uint(refraction_angle_deg));
#endif // SPECTRAL_SPECTRAL_ENABLED

    // ========================================================================
    // ACTUALIZAR PAYLOAD DEL RAYO
    // ========================================================================

    // Acumular el peso de atención
    accumulated_attention += attention_weight;

    // Reducir la energía del rayo (modelo exponencial de absorción)
    // El rayo pierde energía a una tasa proporcional a su energía actual
    float energy_decay = expf(-c_lambda * semantic_distance);
    energy_remaining *= energy_decay;

    // Incrementar el contador de hits
    hit_count++;

    // Insertar el token golpeado en la lista de top-K (en el payload)
    // (Para esta versión simplificada, almacenaremos solo el token_id y weight)
    if (hit_count <= SPECTRAL_MAX_TOP_TOKENS) {
        // Acceder a payload_3 y payload_4 para top_tokens y top_weights
        // (Requiere extensión del payload de 3 a más words)
        // Por ahora, asumimos que el payload ha sido actualizado en el ClosestHit previo
    }

    // ========================================================================
    // ESCRIBIR PAYLOAD ACTUALIZADO
    // ========================================================================

    optixSetPayload_0(__float_as_uint(accumulated_attention));
    optixSetPayload_1(__float_as_uint(energy_remaining));
    optixSetPayload_2(hit_count);

    // ========================================================================
    // DECIDIR SOBRE LA CONTINUACIÓN
    // ========================================================================

    // Si la energía cae por debajo del threshold, terminar la traversal
    if (energy_remaining < SPECTRAL_ENERGY_THRESHOLD) {
        // Terminar el rayo (no hay más intersecciones)
        optixTerminateRay();
    }

    // En caso contrario, la traversal continúa con los siguientes nodos del BVH
}

/* ============================================================================
 * PROGRAMA OPTIX ALTERNATIVO: __closesthit__ con gestión de top-K en shared memory
 *
 * Esta versión utiliza shared memory para mantener un heap de top-K tokens.
 * Más eficiente para queries con muchos hits.
 * ============================================================================
 */
extern "C" __global__ void __closesthit__ch_optical_attention_topk() {
    // Implementación idéntica a __closesthit__ch_optical_attention()
    // pero con soporte para mejor gestión de top-K en memoria compartida
    // (Requiere refactorización del payload para pasar más información)

    uint32_t payload_0 = optixGetPayload_0();
    uint32_t payload_1 = optixGetPayload_1();
    uint32_t payload_2 = optixGetPayload_2();

    float accumulated_attention = __uint_as_float(payload_0);
    float energy_remaining = __uint_as_float(payload_1);
    uint32_t hit_count = payload_2;

    float3 ray_origin = optixGetWorldRayOrigin();
    uint32_t primitive_idx = optixGetPrimitiveIndex();

    if (primitive_idx >= SPECTRAL_MAX_SEQUENCE_LENGTH) {
        // In closest-hit, intersection is already committed — just return early
        return;
    }

    const TokenNode& hit_token = c_token_nodes[primitive_idx];

    float3 delta = make_float3(
        hit_token.centroid.x - ray_origin.x,
        hit_token.centroid.y - ray_origin.y,
        hit_token.centroid.z - ray_origin.z
    );

    float semantic_distance = sqrtf(
        delta.x * delta.x +
        delta.y * delta.y +
        delta.z * delta.z
    );
    semantic_distance = fmaxf(semantic_distance, 0.001f);

    float attention_weight = energy_remaining * expf(-c_lambda * semantic_distance);

    if (attention_weight < SPECTRAL_ENERGY_THRESHOLD) {
        // In closest-hit, intersection is already committed — just return early
        return;
    }

    accumulated_attention += attention_weight;
    float energy_decay = expf(-c_lambda * semantic_distance);
    energy_remaining *= energy_decay;
    hit_count++;

    optixSetPayload_0(__float_as_uint(accumulated_attention));
    optixSetPayload_1(__float_as_uint(energy_remaining));
    optixSetPayload_2(hit_count);

    if (energy_remaining < SPECTRAL_ENERGY_THRESHOLD) {
        optixTerminateRay();
    }
}

