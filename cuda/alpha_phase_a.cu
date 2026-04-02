/**
 * @file alpha_phase_a.cu
 * @brief FASE A de Alpha BSH: Traversal del árbol BSH mediante ray tracing O(log N)
 *
 * DESCRIPCIÓN GENERAL
 * ====================
 * La FASE A es responsable de encontrar la esfera semántica más relevante a partir
 * de un token query. Utiliza dos estrategias complementarias:
 *
 *   1. KERNEL CUDA FALLBACK (CPU/simple testing):
 *      - Simula el traversal del BSH con un simple algoritmo greedy
 *      - No requiere OptiX, útil para debugging y CPU
 *      - Complejidad: O(log N) amortizado
 *
 *   2. PROGRAMAS OptiX (comentados como pseudocódigo):
 *      - __raygen__: genera UN rayo desde el embedding del query
 *      - __intersection__: test rayo-esfera (fórmula cuadrática)
 *      - __closesthit__: procesa colisión, actualiza payload, continúa en hijos
 *      - __miss__: rayo no encontró nada (mark UINT32_MAX)
 *
 * INNOVACIÓN RESPECTO AL PROYECTO ORIGINAL:
 * ==========================================
 * El proyecto original "SpectralAI Zero-Matrix" emitía MILES de rayos para
 * acumular pesos de atención multi-head. ALPHA BSH solo emite UN rayo, cuyo
 * objetivo es encontrar UNA esfera, no acumular distribuciones.
 *
 * Esto simplifica enormemente la lógica y reduce FLOPs:
 *   - Original: N_rays * (ray traversal + summation) = O(K * N log N), K >> N
 *   - Alpha A:  1 rayo * (traversal) = O(log N)
 *   - Alpha B:  MatMul en esfera encontrada = O(M²)
 *
 * @author SpectralAI Zero-Matrix Team
 * @date 2026
 */

#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

// Para debugging y prototipado, incluimos el header Alpha BSH
#include "../include/alpha_bsh.h"

// ============================================================================
// CONSTANTES LOCALES
// ============================================================================

/// Máximo número de iteraciones en el traversal greedy (seguridad)
#define ALPHA_MAX_TRAVERSAL_ITERATIONS 32

/// Tolerancia numérica para test rayo-esfera
#define ALPHA_RAY_EPSILON 1e-6f

// ============================================================================
// KERNEL CUDA FALLBACK: Traversal Greedy del BSH
// ============================================================================

/**
 * @brief Kernel CUDA que simula la traversal del BSH sin hardware OptiX.
 *
 * Algoritmo:
 * ----------
 * 1. Comienza en la raíz del árbol (sphere_id = 0)
 * 2. Calcula similitud entre el rayo y el centroide actual
 * 3. Si es hoja: devuelve sphere_id (encontrado)
 * 4. Si no es hoja: entre los hijos, elige el más cercano/similar
 * 5. Continúa iterativamente hasta hoja o max_iterations
 *
 * NOTA: Este no es un traversal real de ray-sphere (que requeriría resolver
 * ecuaciones cuadráticas). Es una simplificación greedy que elige el mejor
 * hijo en cada paso. Para un traversal exacto, usar los programas OptiX.
 *
 * @param d_spheres Array GPU de esferas (SemanticSphereAlpha*)
 * @param num_spheres Número total de esferas
 * @param query_point Punto origen del rayo en espacio 3D
 * @param ray_direction Dirección normalizada del rayo
 * @param ray_energy Energía inicial del rayo (1.0)
 * @param lambda_decay Coeficiente de decay exponencial
 * @param d_payload Puntero a AlphaRayPayload donde escribir resultado
 *
 * @complexity O(log N) iteraciones, cada una O(num_children) comparaciones
 *            => O(log N * 8) = O(log N) en práctica (8 = ALPHA_BSH_MAX_CHILDREN)
 */
__global__ void alpha_bsh_traversal_kernel(
    const SemanticSphereAlpha* d_spheres,
    uint32_t num_spheres,
    float3 query_point,
    float3 ray_direction,
    float ray_energy,
    float lambda_decay,
    AlphaRayPayload* d_payload) {

    // Un único thread ejecuta el traversal completo
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Inicializar payload
    d_payload->energy = ray_energy;
    d_payload->hit_sphere_id = UINT32_MAX;  // Initially miss
    d_payload->depth_reached = 0;
    d_payload->best_similarity = 0.0f;

    // Comenzar en raíz (sphere_id = 0)
    uint32_t current_sphere_id = 0;
    float current_energy = ray_energy;

    // Traversal greedy: máximo ALPHA_MAX_TRAVERSAL_ITERATIONS iteraciones
    for (uint32_t iter = 0; iter < ALPHA_MAX_TRAVERSAL_ITERATIONS &&
         current_energy > ALPHA_ENERGY_THRESHOLD; ++iter) {

        if (current_sphere_id >= num_spheres) {
            break;  // Seguridad: sphere_id inválido
        }

        const SemanticSphereAlpha& current_sphere = d_spheres[current_sphere_id];
        d_payload->depth_reached = iter;

        // Calcular similitud entre rayo y centroide actual
        float3 to_center = current_sphere.center - query_point;
        float distance = sqrtf(to_center.x * to_center.x +
                                to_center.y * to_center.y +
                                to_center.z * to_center.z);

        // Normalizar direcciones para product point
        float3 normalized_to_center = {
            to_center.x / (distance + ALPHA_RAY_EPSILON),
            to_center.y / (distance + ALPHA_RAY_EPSILON),
            to_center.z / (distance + ALPHA_RAY_EPSILON)
        };

        float similarity = ray_direction.x * normalized_to_center.x +
                          ray_direction.y * normalized_to_center.y +
                          ray_direction.z * normalized_to_center.z;
        similarity = fmaxf(similarity, 0.0f);  // Clamp [0, 1]

        // Actualizar máxima similitud encontrada
        if (similarity > d_payload->best_similarity) {
            d_payload->best_similarity = similarity;
        }

        // Decay exponencial de energía
        current_energy *= expf(-lambda_decay * distance);

        // Si es hoja: hemos encontrado la esfera!
        if (current_sphere.is_leaf) {
            d_payload->hit_sphere_id = current_sphere_id;
            d_payload->energy = current_energy;
            return;
        }

        // Si no es hoja: encontrar el hijo más similar
        uint32_t best_child_id = UINT32_MAX;
        float best_child_similarity = -1.0f;

        // Bug 2.6 fix: clamp num_children to ALPHA_BSH_MAX_CHILDREN to prevent
        // out-of-bounds access on children_ids[] array
        uint32_t actual_children = min(current_sphere.num_children, (uint32_t)ALPHA_BSH_MAX_CHILDREN);
        for (uint32_t i = 0; i < actual_children; ++i) {
            uint32_t child_id = current_sphere.children_ids[i];
            if (child_id == 0 || child_id >= num_spheres) continue;

            const SemanticSphereAlpha& child = d_spheres[child_id];

            // Calcular similitud con este hijo
            float3 to_child = child.center - query_point;
            float child_dist = sqrtf(to_child.x * to_child.x +
                                      to_child.y * to_child.y +
                                      to_child.z * to_child.z);

            float3 norm_to_child = {
                to_child.x / (child_dist + ALPHA_RAY_EPSILON),
                to_child.y / (child_dist + ALPHA_RAY_EPSILON),
                to_child.z / (child_dist + ALPHA_RAY_EPSILON)
            };

            float child_sim = ray_direction.x * norm_to_child.x +
                              ray_direction.y * norm_to_child.y +
                              ray_direction.z * norm_to_child.z;
            child_sim = fmaxf(child_sim, 0.0f);

            // También considerar el peso semántico del hijo
            child_sim *= child.semantic_weight;

            if (child_sim > best_child_similarity) {
                best_child_similarity = child_sim;
                best_child_id = child_id;
            }
        }

        // Seguir al mejor hijo
        if (best_child_id == UINT32_MAX) {
            // No hay hijos válidos: marcar como miss
            d_payload->hit_sphere_id = UINT32_MAX;
            return;
        }

        current_sphere_id = best_child_id;
    }

    // Si salimos del loop sin encontrar hoja: miss
    if (d_payload->hit_sphere_id == UINT32_MAX) {
        d_payload->energy = 0.0f;
    }
}

// ============================================================================
// FUNCIÓN HOST: Lanzar kernel de traversal
// ============================================================================

/**
 * @brief Host function para lanzar el kernel de traversal del BSH.
 *
 * Esta función:
 *   1. Aloca memoria GPU para AlphaRayPayload si es necesario
 *   2. Lanza el kernel CUDA con un único thread
 *   3. Sincroniza y obtiene el payload resultado
 *   4. Retorna AlphaRayPayload con sphere_id encontrado
 *
 * @param d_spheres Puntero GPU al array de esferas
 * @param num_spheres Número de esferas
 * @param query_point Punto de origen del rayo (query embedding en 3D)
 * @param ray_direction Dirección normalizada del rayo
 * @param ray_energy Energía inicial (típ. 1.0)
 * @param lambda_decay Coeficiente de decay (típ. 0.1)
 *
 * @return AlphaRayPayload con hit_sphere_id (UINT32_MAX si miss)
 */
__host__ AlphaRayPayload launch_alpha_phase_a_kernel(
    const SemanticSphereAlpha* d_spheres,
    uint32_t num_spheres,
    float3 query_point,
    float3 ray_direction,
    float ray_energy,
    float lambda_decay) {

    // Alocar buffer GPU para payload resultado
    // Bug 2.16 fix: check cudaMalloc return value
    AlphaRayPayload* d_payload;
    cudaError_t alloc_err = cudaMalloc(&d_payload, sizeof(AlphaRayPayload));
    if (alloc_err != cudaSuccess) {
        AlphaRayPayload err_payload;
        err_payload.hit_sphere_id = UINT32_MAX;
        err_payload.energy = 0.0f;
        err_payload.depth_reached = 0;
        err_payload.best_similarity = 0.0f;
        return err_payload;
    }

    // Lanzar kernel: 1 block, 1 thread (el traversal es serial)
    alpha_bsh_traversal_kernel<<<1, 1>>>(
        d_spheres,
        num_spheres,
        query_point,
        ray_direction,
        ray_energy,
        lambda_decay,
        d_payload);

    // Sincronizar
    cudaDeviceSynchronize();

    // Copiar payload de volta a host
    AlphaRayPayload h_payload;
    cudaMemcpy(&h_payload, d_payload, sizeof(AlphaRayPayload), cudaMemcpyDeviceToHost);

    // Liberar buffer GPU
    cudaFree(d_payload);

    return h_payload;
}

// ============================================================================
// PSEUDO-CÓDIGO: PROGRAMAS OptiX (comentados, no compilables directamente)
// ============================================================================

/*
 * NOTA: Los siguientes pseudocódigos demuestran la arquitectura de los
 * programas OptiX reales. Para compilarlos, se requiere:
 *   1. NVIDIA OptiX SDK 8.x
 *   2. CUDA 12.x + headers OptiX
 *   3. Pipeline configuration (.ptx, .optix, etc.)
 *   4. CMake with OptiX support
 *
 * Por ahora, los incluimos como pseudocódigo documentado.
 */

/*
// ============================================================================
// OPTIX RAYGEN: Genera UN rayo desde el embedding del query
// ============================================================================

__raygen__alpha_bsh_rg() {
    // Obtener índices del thread
    const uint3 idx = optixGetLaunchIndex();

    // NOTA: En Alpha BSH, solo generamos UN rayo, no miles
    // Por eso idx debe ser (0, 0, 0)

    // Obtener payload (será pasado a intersection/closesthit)
    AlphaRayPayload payload = {};
    payload.energy = 1.0f;
    payload.hit_sphere_id = UINT32_MAX;
    payload.depth_reached = 0;
    payload.best_similarity = 0.0f;

    // Ray origin: query point (asumimos que ya está en 3D)
    // En un sistema real, el query embedding (FP16/FP32) sería proyectado
    // a espacio 3D via PCA (ver token_geometry.h)
    float3 ray_origin = { ... };  // Set by caller

    // Ray direction: calculado desde el embedding del query
    // Estrategia simple: normalizar los primeros 3 componentes del embedding
    // y luego hacer spherical diversification basada en ray_id
    float3 ray_direction = { ... };  // Set by caller

    float ray_tmin = 0.0f;
    float ray_tmax = 1e16f;

    // Lanzar rayo
    // optixTrace( optix_context->traversable,
    //             ray_origin, ray_direction,
    //             ray_tmin, ray_tmax,
    //             ray_time,
    //             OptixVisibilityMask(255),
    //             OPTIX_RAY_FLAG_DISABLE_ANYHIT,  // Solo closest-hit
    //             SBT_RAY_TYPE_RADIANCE,
    //             SBT_RAY_TYPE_COUNT,
    //             SBT_RAY_TYPE_RADIANCE,
    //             (unsigned int)payload );
}

// ============================================================================
// OPTIX INTERSECTION: Test rayo-esfera (fórmula cuadrática exacta)
// ============================================================================

__intersection__alpha_bsh_is() {
    // Obtener ray actual
    float3 ray_origin = optixGetRayOrigin();
    float3 ray_direction = optixGetRayDirection();
    float ray_tmin = optixGetRayTmin();
    float ray_tmax = optixGetRayTmax();

    // Obtener primitive_index (cuál esfera estamos testando)
    int prim_idx = optixGetPrimitiveIndex();

    // Obtener esfera actual (desde SBT, device memory)
    SemanticSphereAlpha sphere = load_sphere_from_sbt(prim_idx);

    // Test rayo-esfera: resolver |o + t*d - c|² = r²
    // Expandiendo: |d|²*t² + 2*<d, o-c>*t + |o-c|²-r² = 0

    float3 oc = ray_origin - sphere.center;
    float a = dot(ray_direction, ray_direction);
    float b = 2.0f * dot(ray_direction, oc);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;

    float discriminant = b*b - 4*a*c;

    if (discriminant >= 0.0f) {
        float t1 = (-b - sqrtf(discriminant)) / (2*a);
        float t2 = (-b + sqrtf(discriminant)) / (2*a);

        // Tomar el hit más cercano dentro del rango [ray_tmin, ray_tmax]
        float t_hit = -1.0f;
        if (t1 >= ray_tmin && t1 <= ray_tmax) t_hit = t1;
        else if (t2 >= ray_tmin && t2 <= ray_tmax) t_hit = t2;

        if (t_hit > 0.0f) {
            // Reportar hit válido
            optixReportIntersection(
                t_hit,             // Hit parameter t
                0,                 // Hit kind (0 = closest hit)
                prim_idx);         // Datos adicionales: cuál esfera
        }
    }
}

// ============================================================================
// OPTIX CLOSESTHIT: Procesa colisión, actualiza payload, continúa en hijos
// ============================================================================

__closesthit__alpha_bsh_ch() {
    // Obtener payload (desde raygen)
    AlphaRayPayload &payload = *(AlphaRayPayload*)optixGetPayload_0();

    // Obtener datos del hit
    int hit_prim_idx = optixGetAttribute_0();
    float3 ray_origin = optixGetRayOrigin();
    float3 ray_direction = optixGetRayDirection();
    float ray_t = optixGetRayTmax();  // distance to hit

    // Obtener la esfera que fue hit
    SemanticSphereAlpha hit_sphere = load_sphere_from_sbt(hit_prim_idx);

    // Calcular punto de impacto
    float3 hit_point = ray_origin + ray_direction * ray_t;
    payload.hit_point = hit_point;

    // Actualizar similitud basada en distancia
    float similarity = distanceToSimilarity(ray_t, ALPHA_LAMBDA_DECAY);
    if (similarity > payload.best_similarity) {
        payload.best_similarity = similarity;
    }

    // Decay de energía
    payload.energy *= expf(-ALPHA_LAMBDA_DECAY * ray_t);

    // Incrementar profundidad alcanzada
    payload.depth_reached++;

    // Si es hoja o energía muy baja: terminar traversal
    if (hit_sphere.is_leaf || payload.energy < ALPHA_ENERGY_THRESHOLD) {
        payload.hit_sphere_id = hit_prim_idx;
        return;  // STOP ray tracing
    }

    // Si no es hoja: continuar traversal hacia los hijos
    // Generar nuevos rayos desde hit_point hacia los hijos
    for (uint32_t i = 0; i < hit_sphere.num_children; ++i) {
        uint32_t child_id = hit_sphere.children_ids[i];
        if (child_id == 0) continue;

        SemanticSphereAlpha child_sphere = load_sphere_from_sbt(child_id);

        // Nueva dirección: hacia el hijo
        float3 new_direction = normalize(child_sphere.center - hit_point);

        // Relanzar rayo desde hit_point
        float ray_tmin_new = 1e-4f;  // Offset pequeño para evitar self-intersection
        float ray_tmax_new = 1e16f;

        // NOTA: En OptiX real, esto sería un optixTrace() recursivo
        // que reutiliza el mismo payload (se actualiza en ClosestHit/Miss)
    }
}

// ============================================================================
// OPTIX MISS: Rayo no encontró nada
// ============================================================================

__miss__alpha_bsh_ms() {
    AlphaRayPayload &payload = *(AlphaRayPayload*)optixGetPayload_0();

    // Marcar como miss: no se encontró esfera
    payload.hit_sphere_id = UINT32_MAX;
    payload.energy = 0.0f;
}

*/

// ============================================================================
// FUNCIÓN AUXILIAR: Proyectar embedding a espacio 3D
// ============================================================================

/**
 * @brief Proyecta un embedding de alta dimensión a espacio 3D.
 *
 * Estrategia simplificada:
 *   - Tomar los primeros 3 componentes del embedding normalizado
 *   - Alternativamente, usar PCA esférica (ver token_geometry.h)
 *
 * En producción, esta proyección debe preservar la métrica coseno del
 * espacio original (ver ARCHITECTURE.md, sección "Proyección Token → Geometría 3D").
 *
 * @param embedding Embedding de entrada (FP32 o FP16)
 * @param embedding_dim Dimensión del embedding
 *
 * @return float3 con la proyección en espacio 3D
 */
__host__ float3 projectEmbeddingTo3D(
    const float* embedding,
    uint32_t embedding_dim) {

    // Normalizar embedding
    float norm = 0.0f;
    for (uint32_t i = 0; i < embedding_dim; ++i) {
        norm += embedding[i] * embedding[i];
    }
    norm = sqrtf(norm + 1e-8f);

    if (norm == 0.0f) {
        // Embedding vacío: devolver punto aleatorio
        return float3{0.5f, 0.5f, 0.5f};
    }

    // Estrategia simple: primeros 3 componentes normalizados
    // (En producción: usar PCA esférica para preservar semántica)
    float3 result;
    result.x = (embedding_dim > 0) ? (embedding[0] / norm) : 0.0f;
    result.y = (embedding_dim > 1) ? (embedding[1] / norm) : 0.0f;
    result.z = (embedding_dim > 2) ? (embedding[2] / norm) : 0.0f;

    return result;
}

/**
 * @brief Normaliza un vector 3D a longitud unitaria.
 *
 * @param v Vector a normalizar
 *
 * @return Vector unitario (||result|| ≈ 1.0)
 */
__host__ __device__ inline float3 normalizeFloat3(float3 v) {
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if (len < 1e-8f) {
        return {0.0f, 0.0f, 0.0f};
    }
    return {v.x / len, v.y / len, v.z / len};
}

// ============================================================================
// FUNCIÓN AUXILIAR GLOBAL: Cálculo de distancia a similitud
// ============================================================================

/**
 * @brief Convierte una distancia 3D en una similitud semántica.
 *
 * Fórmula: similarity = exp(-lambda * distance)
 * Rango: (0.0, 1.0] donde 1.0 es máxima similitud.
 *
 * @param distance Distancia euclídea
 * @param lambda Coeficiente de absorción (típ. 0.1)
 *
 * @return Similitud en [0.0, 1.0]
 */
__host__ __device__ inline float distanceToSimilarity(float distance, float lambda) {
    return expf(-lambda * distance);
}
