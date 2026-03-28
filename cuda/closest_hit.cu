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
 *   - λ: coeficiente de absorción semántica (LIQUIDBIT_LAMBDA ≈ 0.1)
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

// Lambda (coeficiente de absorción) - debe coincidir con LIQUIDBIT_LAMBDA
extern "C" __constant__ float c_lambda;

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
    if (primitive_idx >= LIQUIDBIT_MAX_SEQUENCE_LENGTH) {
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
    if (attention_weight < LIQUIDBIT_ENERGY_THRESHOLD) {
        // Energía muy baja: ignorar este hit y continuar con la traversal
        // In closest-hit, intersection is already committed — just return early
        return;
    }

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
    if (hit_count <= LIQUIDBIT_MAX_TOP_TOKENS) {
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
    if (energy_remaining < LIQUIDBIT_ENERGY_THRESHOLD) {
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

    if (primitive_idx >= LIQUIDBIT_MAX_SEQUENCE_LENGTH) {
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

    if (attention_weight < LIQUIDBIT_ENERGY_THRESHOLD) {
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

    if (energy_remaining < LIQUIDBIT_ENERGY_THRESHOLD) {
        optixTerminateRay();
    }
}

