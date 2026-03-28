/**
 * @file liquidbit_kernels.cu
 * @brief Programas OptiX del String-Inception Engine
 *
 * ARQUITECTURA:
 * ==============
 * El String-Inception Engine usa 4 niveles de IAS (Instance Acceleration Structures)
 * anidados. La traversal navega automáticamente por los niveles gracias al hardware
 * RT Core — OptiX maneja el stack de traversal de forma nativa.
 *
 * FLUJO DE UN RAYO:
 * =================
 *   1. __raygen__liquidbit: genera el rayo con ω = baseOmega, lanza optixTrace() en el IAS raíz
 *   2. Traversal hardware (RT Cores): el rayo desciende por los niveles IAS automáticamente
 *   3. __closesthit__semantic_portal: en cada colisión:
 *       - Si depth < MAX_DEPTH - 1: aplica AffinePortal, actualiza ω, continúa traversal
 *       - Si depth == MAX_DEPTH - 1 (hoja): calcula resonancia Fourier, acumula resultado
 *   4. __miss__inception: cuando el rayo sale de la escena sin hit → registra energía residual
 *
 * NOTA SOBRE IAS ANIDADOS:
 * =========================
 * OptiX soporta instancias anidadas nativas (IAS dentro de IAS) desde la versión 7.
 * El hardware RT Core maneja el stack de traversal — no necesitamos código explícito
 * para "bajar de nivel". Lo que hacemos en closesthit es:
 *   - Detectar el nivel actual via el campo `depth` de la SemanticSphere
 *   - Aplicar la transformación de contexto (AffinePortal)
 *   - Actualizar el payload ω
 *   - Si es hoja, calcular resonancia
 *
 * El motor de traversal de OptiX ya navega por todos los niveles del IAS automáticamente.
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#include <optix.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include <vector_types.h>
#include <vector_functions.h>

#include "../include/liquidbit_resonance.h"
#include "../include/token_geometry.h"

// ============================================================================
// PARÁMETROS GLOBALES (set por el host antes de optixLaunch)
// ============================================================================

// NOTA: El pipeline host que use estos kernels debe configurar:
//   pipeline_compile_options.pipelineLaunchParamsVariableName = "c_params"
// Este es un pipeline SEPARADO de optix_host.cpp (que usa "params").
extern "C" __constant__ InceptionLaunchParams c_params;

// ============================================================================
// FUNCIONES AUXILIARES DEVICE
// ============================================================================

/**
 * @brief Empaqueta/desempaqueta un InceptionPayload en/desde los registros de OptiX.
 *
 * OptiX usa uint32 para los registros de payload. Convertimos float via __float_as_uint
 * para preservar los bits exactamente.
 */
__device__ __forceinline__
InceptionPayload getPayload() {
    InceptionPayload p;
    p.omega       = __uint_as_float(optixGetPayload_0());
    p.accumulated = __uint_as_float(optixGetPayload_1());
    p.depth       = optixGetPayload_2();
    p.hitCount    = optixGetPayload_3();
    return p;
}

__device__ __forceinline__
void setPayload(const InceptionPayload& p) {
    optixSetPayload_0(__float_as_uint(p.omega));
    optixSetPayload_1(__float_as_uint(p.accumulated));
    optixSetPayload_2(p.depth);
    optixSetPayload_3(p.hitCount);
}

/**
 * @brief Aplica un AffinePortal al vector de frecuencia ω.
 *
 * La transformación es: ω_nuevo = dot(portal.rows[0], make_float4(ω, ω², ω³, 1.0f))
 *
 * Esto es una forma compacta de la transformación afín del contexto.
 * En un sistema completo, el vector espectral sería f ∈ ℝ^64, aquí simplificamos
 * a un escalar ω para eficiencia en el prototipo.
 */
__device__ __forceinline__
float applyPortal(const AffinePortal& portal, float omega) {
    // Expansión polinómica del contexto escalar: [ω, ω², ω³, 1]
    const float4 ctx = make_float4(omega, omega * omega, omega * omega * omega, 1.0f);
    const float4 row = portal.rows[0];
    const float new_omega = row.x * ctx.x + row.y * ctx.y + row.z * ctx.z + row.w * ctx.w;
    // Normalizar a [0, 2π] para mantener la interpretación de frecuencia
    return fmodf(fabsf(new_omega), 2.0f * CUDART_PI_F);
}

// ============================================================================
// PROGRAMA OPTIX: Raygen — String-Inception
// ============================================================================

/**
 * @brief Genera rayos con contexto de frecuencia ω y lanza la traversal del IAS.
 *
 * Un rayo por thread (blockDim.x × gridDim.x threads total = numRays).
 * Cada rayo parte desde la posición semántica del token query y viaja
 * en una dirección determinada por su índice de rayo (similar a query heads).
 *
 * El payload inicial lleva:
 *   - omega = c_params.baseOmega (frecuencia de contexto del prompt actual)
 *   - accumulated = 0.0f
 *   - depth = 0
 *   - hitCount = 0
 */
extern "C" __global__
void __raygen__liquidbit() {
    const uint32_t rayIdx = optixGetLaunchIndex().x;
    if (rayIdx >= c_params.numRays) return;

    // Origen del rayo: posición semántica del token query
    // En el prototipo, los rayos parten del origen; en producción se usará
    // la proyección 3D del token query actual
    const float3 origin = make_float3(0.0f, 0.0f, 0.0f);

    // Dirección del rayo: distribuida en hemisferio según índice
    // Usamos distribución fibonacci en esfera para cobertura uniforme
    const float golden = 2.399963f;  // 2π/φ, ángulo dorado
    const float theta  = acosf(1.0f - 2.0f * (rayIdx + 0.5f) / (float)c_params.numRays);
    const float phi    = golden * (float)rayIdx;

    const float3 direction = make_float3(
        sinf(theta) * cosf(phi),
        sinf(theta) * sinf(phi),
        cosf(theta)
    );

    // Inicializar payload con la frecuencia de contexto base
    InceptionPayload payload;
    payload.omega       = c_params.baseOmega;
    payload.accumulated = 0.0f;
    payload.depth       = 0;
    payload.hitCount    = 0;

    // OptiX 9.1: optixTrace requiere uint32_t& — usar variables temporales
    uint32_t p0 = __float_as_uint(payload.omega);
    uint32_t p1 = __float_as_uint(payload.accumulated);
    uint32_t p2 = payload.depth;
    uint32_t p3 = payload.hitCount;

    // Lanzar rayo en el IAS raíz (nivel 0)
    // OptiX navegará automáticamente por los niveles IAS anidados
    optixTrace(
        c_params.topLevelIAS,
        origin,
        direction,
        0.0f,             // tmin
        1e16f,            // tmax
        0.0f,             // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0u,               // SBT offset
        1u,               // SBT stride
        0u,               // missSBT index
        p0, p1, p2, p3
    );

    // Leer resultado del payload tras la traversal completa.
    // En __raygen__, después de optixTrace() los valores actualizados están
    // en p0..p3 (pasados por referencia). optixGetPayload_*() es ilegal aquí.
    payload.omega       = __uint_as_float(p0);
    payload.accumulated = __uint_as_float(p1);
    payload.depth       = p2;
    payload.hitCount    = p3;

    // Escribir resultado en el buffer de salida
    SpectralAttentionResult& out = c_params.results[rayIdx];
    out.attentionWeight  = payload.accumulated;
    out.finalOmega       = payload.omega;
    out.traversalDepth   = payload.depth;
    out.energyRemaining  = (payload.hitCount > 0) ? (1.0f / (float)payload.hitCount) : 1.0f;
    out.dominantStringId = 0;  // Actualizado en closesthit
    out.exitDirection    = direction;
}

// ============================================================================
// PROGRAMA OPTIX: ClosestHit — Portal Semántico
// ============================================================================

/**
 * @brief Procesa la colisión del rayo con una SemanticSphere.
 *
 * Comportamiento según profundidad:
 *
 *   depth < INCEPTION_MAX_DEPTH - 1 (nodo interno):
 *     → Aplica AffinePortal para transformar ω
 *     → Actualiza el payload y continúa la traversal (el hardware lo hace solo)
 *
 *   depth == INCEPTION_MAX_DEPTH - 1 (nodo hoja):
 *     → Busca la SemanticString asociada al primitivo golpeado
 *     → Calcula resonancia Fourier: W(ω) = outputScale · tanh(Σ a_k·sin(kω) + b_k·cos(kω))
 *     → Acumula el resultado en payload.accumulated
 *
 * La geometría de la esfera se obtiene via optixGetPrimitiveIndex() que devuelve
 * el índice en c_params.spheres[].
 */
extern "C" __global__
void __closesthit__semantic_portal() {
    InceptionPayload payload = getPayload();

    // Obtener la esfera golpeada
    const uint32_t primitiveIdx = optixGetPrimitiveIndex();
    const SemanticSphere& sphere = c_params.spheres[primitiveIdx];

    // Aplicar desplazamiento de frecuencia de esta esfera
    float omega = payload.omega + sphere.frequencyBias;
    // Mantener ω en [0, 2π]
    omega = fmodf(fabsf(omega), 2.0f * CUDART_PI_F);

    payload.hitCount++;

    if (sphere.depth < (uint32_t)(INCEPTION_MAX_DEPTH - 1)) {
        // ---- NODO INTERNO: aplicar portal afín ----
        // Solo aplicamos el portal si existe (childIAS != 0)
        if (sphere.childIAS != 0 && sphere.depth < INCEPTION_MAX_DEPTH) {
            const AffinePortal& portal = c_params.portals[sphere.depth];
            omega = applyPortal(portal, omega);
        }
        payload.omega = omega;
        payload.depth = max(payload.depth, sphere.depth + 1);
    } else {
        // ---- NODO HOJA: calcular resonancia Fourier ----
        // Buscar la SemanticString asociada a este primitivo
        // En el prototipo: relación 1:1 entre primitiveIdx y stringIdx
        if (primitiveIdx < c_params.numStrings) {
            const SemanticString& str = c_params.strings[primitiveIdx];
            const float resonance = semanticStringResonance(str.resonance, omega);

            // Acumulación con decaimiento por profundidad:
            // Los nodos más profundos (más específicos) tienen mayor peso
            const float depthWeight = 1.0f + 0.5f * (float)sphere.depth;
            payload.accumulated += resonance * depthWeight;
            payload.omega = omega;
            payload.depth = max(payload.depth, sphere.depth);

            // Guardar ID de la string más resonante en el payload de salida
            // (reutilizamos hitCount como proxy — en producción usaríamos un payload extra)
        }
    }

    setPayload(payload);
}

// ============================================================================
// PROGRAMA OPTIX: Miss — Energía Residual
// ============================================================================

/**
 * @brief Se ejecuta cuando el rayo sale de la escena sin colisionar.
 *
 * En el String-Inception Engine, un miss significa que el contexto del prompt
 * no tiene resonancia con ninguna esfera semántica en esa dirección.
 * El peso acumulado no se modifica — se devuelve lo que se haya acumulado
 * en hits anteriores (para rayos que golpearon algunos nodos antes de escapar).
 *
 * Si el rayo no golpeó nada en absoluto (hitCount == 0), retorna 0.
 */
extern "C" __global__
void __miss__inception() {
    // El payload se mantiene tal cual — la traversal termina aquí.
    // No modificamos el payload en miss.
    // Los registros de payload ya contienen los acumulados de hits anteriores.
}

// ============================================================================
// PROGRAMA OPTIX: Intersection — Esfera Semántica
// ============================================================================

/**
 * @brief Calcula la intersección rayo-esfera para primitivas custom.
 *
 * OptiX llama a este programa cuando el rayo entra en el AABB de la primitiva.
 * Si el rayo efectivamente intersecta la esfera, llamamos optixReportIntersection(t, 0).
 *
 * La esfera se obtiene de c_params.spheres[optixGetPrimitiveIndex()].
 * El radio y centro están en espacio de objeto (world coords en el prototipo).
 */
extern "C" __global__
void __intersection__sphere() {
    const uint32_t primIdx = optixGetPrimitiveIndex();
    const SemanticSphere& sphere = c_params.spheres[primIdx];

    // FIX CRITICAL: usar world-space porque sphere.center está en coords mundiales.
    // optixGetObjectRayOrigin() daría coordenadas de objeto — incorrecto bajo instancias
    // con transformaciones no-identidad. Para el prototipo usamos world space.
    const float3 origin = optixGetWorldRayOrigin();
    const float3 dir    = optixGetWorldRayDirection();

    // Vector desde el origen del rayo al centro de la esfera
    const float3 oc = make_float3(
        origin.x - sphere.center.x,
        origin.y - sphere.center.y,
        origin.z - sphere.center.z
    );

    // Coeficientes de la ecuación cuadrática: ||oc + t*dir||² = r²
    const float a    = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
    const float b    = 2.0f * (oc.x*dir.x + oc.y*dir.y + oc.z*dir.z);
    const float c    = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z
                       - sphere.radius * sphere.radius;
    const float disc = b*b - 4.0f*a*c;

    if (disc < 0.0f) return;  // No hay intersección real

    const float sqrtDisc = sqrtf(disc);
    const float tmin     = optixGetRayTmin();
    const float tmax     = optixGetRayTmax();

    // Raíz menor (entrada a la esfera)
    float t = (-b - sqrtDisc) / (2.0f * a);
    if (t < tmin) {
        // Raíz mayor (salida — el rayo nació dentro de la esfera)
        t = (-b + sqrtDisc) / (2.0f * a);
    }
    if (t < tmin || t > tmax) return;

    // Reportar intersección — hit_kind = 0 (esfera semántica estándar)
    optixReportIntersection(t, 0u);
}
