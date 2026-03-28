/**
 * ray_generation.cu
 *
 * Programa OptiX RayGen (ray generation shader)
 *
 * Se ejecuta para cada rayo que necesita ser lanzado en la escena.
 * En nuestro caso, genera los rayos que emanan desde un token query
 * en direcciones distribuidas en un hemisferio semántico.
 *
 * Cada rayo representa una "dimensión de pensamiento" o query head,
 * análogo a los multi-head attention heads en Transformers.
 *
 * Responsabilidades:
 * 1. Leer los parámetros del token query
 * 2. Generar direcciones de rayos distribuidas
 * 3. Inicializar el payload del rayo
 * 4. Lanzar optixTrace() para la traversal del BVH
 * 5. Procesar y almacenar los resultados
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
 * CONSTANTES GLOBALES
 * ============================================================================
 */

// Acceleration structure del BVH
extern "C" __constant__ OptixTraversableHandle c_bvh_handle;

// Buffer de TokenNodes
extern "C" __constant__ TokenNode* c_token_nodes;

// Buffer de salida (AttentionResults)
extern "C" __constant__ AttentionResult* c_attention_results;

// Parámetros de configuración
extern "C" __constant__ uint32_t c_num_queries;
extern "C" __constant__ uint32_t c_rays_per_query;
extern "C" __constant__ uint32_t c_num_tokens;

/* ============================================================================
 * FUNCIONES AUXILIARES
 * ============================================================================
 */

/**
 * Genera una dirección normalizada en un hemisferio usando distribución uniforme.
 *
 * Parámetros:
 *   seed: seed pseudoaleatorio (basado en pixel/thread ID)
 *   ray_idx: índice del rayo (para distribución determinística)
 *   num_rays: número total de rayos
 *
 * Retorna: vector normalizado que representa la dirección del rayo
 *
 * Método: Distribución uniforme en hemisferio semántico
 * Se usa una distribución de Fibonacci para mejor cobertura espacial
 * (análogo a la distribución de query heads en atención multi-cabeza)
 */
__device__ float3 generate_hemisphere_direction(
    uint32_t ray_idx,
    uint32_t num_rays
) {
    // Usar distribución de Fibonacci para mejor cobertura
    // que una distribución aleatoria uniforme
    float golden_ratio = 1.618033988749f;
    float inverted_gr = 1.0f / golden_ratio;

    // Ángulo azimutal (alrededor del eje Z)
    float theta = 2.0f * CUDART_PI_F * ray_idx * inverted_gr;

    // Ángulo polar (desde el eje Z hacia el ecuador)
    // Restringido a un hemisferio [0, π/2]
    float phi = acosf(1.0f - (float)ray_idx / (float)num_rays);

    // Convertir a coordenadas cartesianas
    float sin_phi = sinf(phi);
    float cos_phi = cosf(phi);
    float sin_theta = sinf(theta);
    float cos_theta = cosf(theta);

    return make_float3(
        sin_phi * cos_theta,
        sin_phi * sin_theta,
        cos_phi
    );
}

/**
 * Calcula una base ortonormal (u, v, w) para crear rayos en un hemisferio
 * centrado en una dirección arbitraria.
 *
 * Parámetros:
 *   direction: vector normalizado (la dirección "hacia arriba" del hemisferio)
 *   u, v, w: vectores de salida (base ortonormal)
 */
__device__ void create_orthonormal_basis(
    const float3& direction,
    float3& u,
    float3& v,
    float3& w
) {
    w = direction;  // Eje principal

    // Encontrar un vector que NO sea paralelo a w
    float3 orthogonal;
    if (fabsf(w.x) < 0.9f) {
        orthogonal = make_float3(1.0f, 0.0f, 0.0f);
    } else {
        orthogonal = make_float3(0.0f, 1.0f, 0.0f);
    }

    // Primera base usando producto cruz
    u = normalize(cross(orthogonal, w));

    // Segunda base usando producto cruz
    v = cross(w, u);
}

/**
 * Función auxiliar para normalizar un vector 3D
 */
__device__ float3 normalize(const float3& v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len > 1e-6f) {
        return make_float3(v.x / len, v.y / len, v.z / len);
    }
    return v;
}

/**
 * Producto cruz 3D
 */
__device__ float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

/* ============================================================================
 * PROGRAMA OPTIX: __raygen__
 *
 * Se ejecuta una vez por rayo generado. En nuestro caso, genera rayos
 * desde los tokens query.
 *
 * Estructura:
 * 1. Determinar qué token query genera este rayo
 * 2. Generar dirección del rayo (distribución hemisférica)
 * 3. Inicializar payload
 * 4. Lanzar optixTrace()
 * 5. Procesar y almacenar resultados
 * ============================================================================
 */
extern "C" __global__ void __raygen__rg_optical_attention() {
    // ========================================================================
    // ÍNDICES Y CONFIGURACIÓN
    // ========================================================================

    // Calcular índices basados en el trabajo distribuido de OptiX
    // (En OptiX, el trabajo se distribuye en una grilla 1D o 2D)
    uint32_t idx = optixGetLaunchIndex().x;  // Índice del rayo actual
    uint32_t total_rays = optixGetLaunchDimensions().x;  // Total de rayos

    // Determinar a qué token query pertenece este rayo
    uint32_t query_idx = idx / c_rays_per_query;
    uint32_t ray_in_query = idx % c_rays_per_query;

    // Validación de límites
    if (query_idx >= c_num_queries) {
        return;
    }

    // ========================================================================
    // LEER PARÁMETROS DEL TOKEN QUERY
    // ========================================================================

    const TokenNode& query_token = c_token_nodes[query_idx];
    float3 query_position = query_token.centroid;

    // Usar el embedding del token query para determinar dirección base
    // (Esto es opcional, pero proporciona sesgo semántico a los rayos)
    // Para esta versión, usamos distribución uniforme en hemisferio

    // ========================================================================
    // GENERAR DIRECCIÓN DEL RAYO
    //
    // Los rayos se distribuyen uniformemente en un hemisferio semántico.
    // Esto es análogo a los "query heads" en attention multi-cabeza.
    //
    // Interpretación física:
    // - Cada rayo "piensa" en una dirección diferente
    // - Los rayos paralelos forman un "beam" de búsqueda
    // - El hemisferio cubre todas las direcciones posibles de relevancia
    // ========================================================================

    float3 ray_direction = generate_hemisphere_direction(ray_in_query, c_rays_per_query);

    // Normalizar dirección (debería ya estar normalizada)
    ray_direction = normalize(ray_direction);

    // ========================================================================
    // INICIALIZAR PAYLOAD DEL RAYO
    // ========================================================================

    RayPayload ray_payload;
    ray_payload.accumulated_attention = 0.0f;
    ray_payload.energy_remaining = 1.0f;
    ray_payload.hit_count = 0;

    // Almacenar la posición del query en el payload para uso en ClosestHit
    ray_payload.ray_origin_x = __float_as_uint(query_position.x);
    ray_payload.ray_origin_y = __float_as_uint(query_position.y);
    ray_payload.ray_origin_z = __float_as_uint(query_position.z);

    // ========================================================================
    // LANZAR RAYO EN EL BVH
    // ========================================================================

    // Parámetros de la traversal
    float t_min = 0.001f;           // Offset para evitar auto-intersecciones
    float t_max = 1e16f;            // Distancia máxima (infinito efectivo)

    // Convertir payload a 32-bit words para optixTrace
    uint32_t p0 = __float_as_uint(ray_payload.accumulated_attention);
    uint32_t p1 = __float_as_uint(ray_payload.energy_remaining);
    uint32_t p2 = ray_payload.hit_count;

    // Lanzar el rayo
    optixTrace(
        c_bvh_handle,                   // Acceleration structure (BVH)
        query_position,                 // Ray origin (posición del query)
        ray_direction,                  // Ray direction (normalizada)
        t_min,                          // t_min
        t_max,                          // t_max
        0.0f,                           // ray_time (no se usa temporal)
        OptixVisibilityMask(255),       // Visibility mask (todos visibles)
        OPTIX_RAY_FLAG_NONE,            // Ray flags
        0,                              // SBT offset
        0,                              // SBT stride
        0,                              // Miss SBT index
        p0, p1, p2                      // Payloads
    );

    // Después de optixTrace, los payloads (p0, p1, p2) han sido actualizados
    // por ClosestHit o Miss

    // ========================================================================
    // PROCESAR Y ALMACENAR RESULTADOS
    // ========================================================================

    // Reinterpretar los payloads actualizados
    ray_payload.accumulated_attention = __uint_as_float(p0);
    ray_payload.energy_remaining = __uint_as_float(p1);
    ray_payload.hit_count = p2;

    // Almacenar resultado en el buffer de salida
    if (query_idx < c_num_queries) {
        AttentionResult& result = c_attention_results[query_idx];

        // Acumular resultados de este rayo (si es el primer rayo, inicializar)
        if (ray_in_query == 0) {
            result.query_token_id = query_token.token_id;
            result.hit_count = 0;
            result.total_attention = 0.0f;
        }

        // Añadir contribución de este rayo
        result.total_attention += ray_payload.accumulated_attention;
        result.hit_count = max(result.hit_count, ray_payload.hit_count);

        // (Top-K tokens se gestionan en el kernel principal ray_traced_attention_kernel)
    }
}

/* ============================================================================
 * PROGRAMA OPTIX ALTERNATIVO: __raygen__ con distribución gaussiana
 *
 * Versión que distribuye los rayos con una distribución más concentrada
 * (gaussiana en lugar de uniforme en hemisferio).
 *
 * Útil si queremos que el modelo atienda preferentemente a tokens cercanos.
 * ============================================================================
 */
extern "C" __global__ void __raygen__rg_optical_attention_gaussian() {
    // Implementación similar a __raygen__rg_optical_attention(),
    // pero con distribución gaussiana en lugar de uniforme

    uint32_t idx = optixGetLaunchIndex().x;
    uint32_t query_idx = idx / c_rays_per_query;
    uint32_t ray_in_query = idx % c_rays_per_query;

    if (query_idx >= c_num_queries) {
        return;
    }

    const TokenNode& query_token = c_token_nodes[query_idx];
    float3 query_position = query_token.centroid;

    // Distribución gaussiana alrededor de Z (más concentrada que uniforme)
    float golden_ratio = 1.618033988749f;
    float inverted_gr = 1.0f / golden_ratio;

    float theta = 2.0f * CUDART_PI_F * ray_in_query * inverted_gr;
    float phi = acosf(1.0f - powf((float)ray_in_query / (float)c_rays_per_query, 2.0f));

    float sin_phi = sinf(phi);
    float cos_phi = cosf(phi);
    float sin_theta = sinf(theta);
    float cos_theta = cosf(theta);

    float3 ray_direction = normalize(make_float3(
        sin_phi * cos_theta,
        sin_phi * sin_theta,
        cos_phi
    ));

    // Resto del código idéntico a __raygen__rg_optical_attention()
    RayPayload ray_payload;
    ray_payload.accumulated_attention = 0.0f;
    ray_payload.energy_remaining = 1.0f;
    ray_payload.hit_count = 0;
    ray_payload.ray_origin_x = __float_as_uint(query_position.x);
    ray_payload.ray_origin_y = __float_as_uint(query_position.y);
    ray_payload.ray_origin_z = __float_as_uint(query_position.z);

    uint32_t p0 = __float_as_uint(ray_payload.accumulated_attention);
    uint32_t p1 = __float_as_uint(ray_payload.energy_remaining);
    uint32_t p2 = ray_payload.hit_count;

    optixTrace(
        c_bvh_handle,
        query_position,
        ray_direction,
        0.001f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 0, 0,
        p0, p1, p2
    );

    ray_payload.accumulated_attention = __uint_as_float(p0);
    ray_payload.energy_remaining = __uint_as_float(p1);
    ray_payload.hit_count = p2;

    if (query_idx < c_num_queries) {
        AttentionResult& result = c_attention_results[query_idx];
        if (ray_in_query == 0) {
            result.query_token_id = query_token.token_id;
            result.hit_count = 0;
            result.total_attention = 0.0f;
        }
        result.total_attention += ray_payload.accumulated_attention;
        result.hit_count = max(result.hit_count, ray_payload.hit_count);
    }
}

