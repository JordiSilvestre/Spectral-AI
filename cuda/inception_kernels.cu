/**
 * @file inception_kernels.cu
 * @brief LiquidBit v4.0 — Inception Engine: Salto Dimensional Explícito entre IAS
 *
 * DIFERENCIA CLAVE vs liquidbit_kernels.cu:
 * ==========================================
 * En `liquidbit_kernels.cu`, la traversal IAS anidada es IMPLÍCITA — OptiX maneja
 * el stack automáticamente. Aquí, el salto dimensional es EXPLÍCITO:
 *
 *   En __closesthit__inception_portal (niveles 0-2):
 *     optixTrace(sphere->childIAS, newOrigin, newDirection, ...);
 *
 * Esto permite que cada nivel opere en un SISTEMA DE COORDENADAS INDEPENDIENTE.
 * La transformación AffinePortal convierte el rayo al espacio local del hijo antes
 * de lanzar el optixTrace anidado.
 *
 * FLUJO COMPLETO (4 NIVELES):
 * ============================
 *
 *   Nivel 0 — Dominios (IAS_root):
 *     __raygen__inception → optixTrace(IAS_root)
 *     → closesthit: aplica portal_0, lanza optixTrace(sphere.childIAS = IAS_level1)
 *
 *   Nivel 1 — Subdominios (IAS_level1):
 *     closesthit: aplica portal_1, lanza optixTrace(sphere.childIAS = IAS_level2)
 *
 *   Nivel 2 — Conceptos (IAS_level2):
 *     closesthit: aplica portal_2, lanza optixTrace(sphere.childIAS = IAS_level3)
 *
 *   Nivel 3 — SemanticStrings (GAS_leaf):
 *     closesthit: calcula resonancia Fourier W(ω), acumula en resultado
 *
 *   Miss en cualquier nivel: preserva acumulado, terminación limpia
 *
 * COMPLEJIDAD RESULTANTE:
 * ========================
 *   O(log N) por nivel × 4 niveles = O(4 log N) = O(log N)
 *   Con N=100K → 4 × 17 = 68 pasos de traversal máximo
 *   vs Transformer: O(N²) = 10.000.000.000 operaciones
 *
 * POLISEMIA AUTOMÁTICA:
 * =====================
 *   La frecuencia ω cambia en cada portal (portal.rows[0] transforma el contexto).
 *   La MISMA esfera devuelve distintas resonancias con distintos ω.
 *   Sin duplicar matrices. Sin wormholes extra.
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#include <optix.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <stdio.h>

#include "../include/liquidbit_resonance.h"
#include "../include/inception_engine.h"

// ============================================================================
// PARÁMETROS GLOBALES — Inception Engine
// ============================================================================

// Nombre del variable de launch params para el pipeline inception.
// El host debe configurar:
//   pipeline_compile_options.pipelineLaunchParamsVariableName = "g_inception_params"
extern "C" __constant__ InceptionLaunchParams g_inception_params;

// ============================================================================
// FUNCIONES AUXILIARES DEVICE
// ============================================================================

/**
 * @brief Lee el payload del estado OptiX actual.
 * Solo válido en __closesthit__, __miss__, __anyhit__, __intersection__.
 */
__device__ __forceinline__
InceptionPayload payloadRead() {
    InceptionPayload p;
    p.omega       = __uint_as_float(optixGetPayload_0());
    p.accumulated = __uint_as_float(optixGetPayload_1());
    p.depth       = optixGetPayload_2();
    p.hitCount    = optixGetPayload_3();
    return p;
}

/**
 * @brief Escribe el payload en los registros OptiX.
 * Solo válido en __closesthit__, __miss__, __anyhit__, __intersection__.
 */
__device__ __forceinline__
void payloadWrite(const InceptionPayload& p) {
    optixSetPayload_0(__float_as_uint(p.omega));
    optixSetPayload_1(__float_as_uint(p.accumulated));
    optixSetPayload_2(p.depth);
    optixSetPayload_3(p.hitCount);
}

/**
 * @brief Transforma un rayo usando una AffinePortal.
 *
 * Aplica la transformación al origen y dirección del rayo para pasarlo
 * al sistema de coordenadas local del IAS hijo.
 *
 * La transformación es: [x',y',z'] = rows[0:3] * [x,y,z,1]
 * (filas 0-2 de la matriz 4×4 — la fila 3 es de perspectiva, ignorada)
 *
 * @param portal    Portal afín (transformación 4×4)
 * @param origin    Origen del rayo en el espacio padre
 * @param direction Dirección del rayo en el espacio padre
 * @param newOrigin Origen transformado (espacio hijo) — output
 * @param newDir    Dirección transformada (espacio hijo) — output
 */
__device__ __forceinline__
void applyPortalToRay(
    const AffinePortal& portal,
    float3 origin, float3 direction,
    float3& newOrigin, float3& newDir
) {
    // Transformar origen (punto → incluye traslación de fila 3)
    newOrigin.x = portal.rows[0].x * origin.x + portal.rows[0].y * origin.y
                + portal.rows[0].z * origin.z + portal.rows[0].w;
    newOrigin.y = portal.rows[1].x * origin.x + portal.rows[1].y * origin.y
                + portal.rows[1].z * origin.z + portal.rows[1].w;
    newOrigin.z = portal.rows[2].x * origin.x + portal.rows[2].y * origin.y
                + portal.rows[2].z * origin.z + portal.rows[2].w;

    // Transformar dirección (vector → ignorar traslación)
    newDir.x = portal.rows[0].x * direction.x + portal.rows[0].y * direction.y
             + portal.rows[0].z * direction.z;
    newDir.y = portal.rows[1].x * direction.x + portal.rows[1].y * direction.y
             + portal.rows[1].z * direction.z;
    newDir.z = portal.rows[2].x * direction.x + portal.rows[2].y * direction.y
             + portal.rows[2].z * direction.z;

    // Normalizar dirección (la transformación afín puede cambiar la magnitud)
    const float len = sqrtf(newDir.x*newDir.x + newDir.y*newDir.y + newDir.z*newDir.z);
    if (len > 1e-8f) {
        const float inv = 1.0f / len;
        newDir.x *= inv;
        newDir.y *= inv;
        newDir.z *= inv;
    }
}

/**
 * @brief Transforma la frecuencia ω al espacio local del portal.
 *
 * Usa la fila 3 de la portal matrix como vector de contexto espectral:
 *   ω_nuevo = dot(portal.rows[3], [ω, ω², sin(ω), 1])
 *   → fuerza de la transformación espectral
 *   → resultado normalizado a [0, 2π]
 */
__device__ __forceinline__
float applyPortalToOmega(const AffinePortal& portal, float omega) {
    const float4 ctx = make_float4(omega, omega * omega, sinf(omega), 1.0f);
    const float4 row = portal.rows[3];
    const float new_omega = row.x * ctx.x + row.y * ctx.y
                          + row.z * ctx.z + row.w * ctx.w;
    // Normalizar a [0, 2π]
    return fmodf(fabsf(new_omega), 2.0f * CUDART_PI_F);
}

// ============================================================================
// PROGRAMA OPTIX: Raygen — Inception Engine
// ============================================================================

/**
 * @brief Genera rayos Fibonacci y lanza la traversal del IAS raíz.
 *
 * Cada thread maneja un rayo independiente. Los rayos se distribuyen
 * uniformemente sobre la esfera unidad usando la secuencia de Fibonacci
 * (máxima cobertura con mínimo número de rayos).
 *
 * El payload inicial transporta:
 *   - omega = g_inception_params.baseOmega  (contexto del prompt actual)
 *   - accumulated = 0.0f                    (resonancia acumulada)
 *   - depth = 0                             (nivel inicial: dominio)
 *   - hitCount = 0                          (número de esferas golpeadas)
 *
 * Tras la traversal completa (posiblemente 4 niveles de optixTrace anidados),
 * el resultado se escribe en g_inception_params.results[rayIdx].
 */
extern "C" __global__
void __raygen__inception() {
    const uint32_t rayIdx = optixGetLaunchIndex().x;
    if (rayIdx >= g_inception_params.numRays) return;

    // Distribución Fibonacci en esfera para cobertura uniforme
    // Equivalente a multi-head attention con heads distribuidos en el hemisferio
    const float golden   = 2.399963f;   // 2π / φ (ángulo dorado)
    const float invTotal = 1.0f / (float)g_inception_params.numRays;
    const float theta    = acosf(1.0f - 2.0f * ((float)rayIdx + 0.5f) * invTotal);
    const float phi      = golden * (float)rayIdx;

    const float3 origin = make_float3(0.0f, 0.0f, 0.0f);
    const float3 direction = make_float3(
        sinf(theta) * cosf(phi),
        sinf(theta) * sinf(phi),
        cosf(theta)
    );

    // Payload inicial
    uint32_t p0 = __float_as_uint(g_inception_params.baseOmega);
    uint32_t p1 = __float_as_uint(0.0f);
    uint32_t p2 = 0u;  // depth = 0
    uint32_t p3 = 0u;  // hitCount = 0

    // ── SALTO DIMENSIONAL NIVEL 0: lanzar en IAS_root ──────────────────
    // La traversal desciende automáticamente si closesthit lanza optixTrace
    // en los IAS hijos (inception_closesthit hace esto explícitamente).
    optixTrace(
        g_inception_params.topLevelIAS,
        origin, direction,
        0.0f,             // tmin
        1e16f,            // tmax
        0.0f,             // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0u,               // SBT record offset
        1u,               // SBT record stride
        0u,               // miss SBT index
        p0, p1, p2, p3
    );

    // Recoger resultado del payload (p0..p3 actualizados tras optixTrace)
    SpectralAttentionResult& out = g_inception_params.results[rayIdx];
    out.attentionWeight  = __uint_as_float(p1);  // accumulated
    out.finalOmega       = __uint_as_float(p0);  // omega final
    out.traversalDepth   = p2;                    // max depth alcanzado
    out.energyRemaining  = (p3 > 0) ? 1.0f / (float)p3 : 1.0f;
    out.dominantStringId = 0;
    out.exitDirection    = direction;
}

// ============================================================================
// PROGRAMA OPTIX: ClosestHit — Portal Semántico + Salto Dimensional Explícito
// ============================================================================

/**
 * @brief Procesa una colisión y lanza el salto dimensional explícito al IAS hijo.
 *
 * Este programa implementa el "hack clave" de la arquitectura Inception:
 *
 *   Si nivel 0-2 (nodo interno):
 *     1. Lee la SemanticSphere golpeada (primitiveIndex en c_params.spheres)
 *     2. Aplica AffinePortal: transforma (origin, direction, ω) al espacio hijo
 *     3. Lanza optixTrace(sphere.childIAS, newOrigin, newDir, ...)
 *        → Salto dimensional explícito al espacio de coordenadas del hijo
 *
 *   Si nivel 3 (nodo hoja):
 *     1. Lee la SemanticString asociada
 *     2. Calcula resonancia Fourier: W(ω) = outputScale · tanh(Σ a_k·sin(kω))
 *     3. Acumula en payload.accumulated
 *
 * COMPLEJIDAD:
 *   O(log N) por nivel × 4 niveles = O(log N) total
 *   Los RT Cores hacen el BVH traversal; nosotros solo aplicamos la transformación
 */
extern "C" __global__
void __closesthit__inception_portal() {
    InceptionPayload payload = payloadRead();

    const uint32_t primIdx   = optixGetPrimitiveIndex();
    const SemanticSphere& sphere = g_inception_params.spheres[primIdx];

    // Aplicar sesgo de frecuencia de la esfera
    float omega = payload.omega + sphere.frequencyBias;
    omega = fmodf(fabsf(omega), 2.0f * CUDART_PI_F);
    payload.hitCount++;

    if (sphere.depth < (uint32_t)(INCEPTION_MAX_DEPTH - 1)) {
        // ── NODO INTERNO: Salto Dimensional Explícito ──────────────────
        //
        // El childIAS es el handle del IAS hijo (nivel siguiente).
        // Aplicamos AffinePortal para transformar el rayo al espacio local.
        // Luego lanzamos optixTrace en ese espacio — el "reset de coordenadas".

        if (sphere.childIAS != 0) {
            // Obtener posición y dirección del rayo en espacio actual
            const float3 worldOrigin = optixGetWorldRayOrigin();
            const float3 worldDir    = optixGetWorldRayDirection();

            // Obtener portal de transformación para este nivel
            const AffinePortal& portal = g_inception_params.portals[sphere.depth];

            // Transformar rayo al espacio local del IAS hijo
            float3 childOrigin, childDir;
            applyPortalToRay(portal, worldOrigin, worldDir, childOrigin, childDir);

            // Transformar ω al espacio local
            omega = applyPortalToOmega(portal, omega);
            payload.omega = omega;
            payload.depth = max(payload.depth, sphere.depth + 1);

            // Escribir payload actualizado antes de lanzar el rayo hijo
            // (los registros p0..p3 se pasan por valor al nuevo optixTrace)
            uint32_t p0 = __float_as_uint(payload.omega);
            uint32_t p1 = __float_as_uint(payload.accumulated);
            uint32_t p2 = payload.depth;
            uint32_t p3 = payload.hitCount;

            // ── EL SALTO DIMENSIONAL ────────────────────────────────────
            // Lanzar rayo en el IAS hijo con las coordenadas transformadas.
            // Esto implementa el concepto "esfera dentro de esfera":
            // el rayo ahora opera en el sistema de coordenadas local del dominio hijo.
            //
            // OptiX permite llamar optixTrace desde closesthit (recursión controlada).
            // El hardware RT Core mantiene el stack de traversal.
            optixTrace(
                (OptixTraversableHandle)sphere.childIAS,
                childOrigin, childDir,
                0.0f,
                1e16f,
                0.0f,
                OptixVisibilityMask(255),
                OPTIX_RAY_FLAG_NONE,
                0u, 1u, 0u,
                p0, p1, p2, p3
            );

            // Recoger resultado del rayo hijo
            payload.omega       = __uint_as_float(p0);
            payload.accumulated = __uint_as_float(p1);
            payload.depth       = p2;
            payload.hitCount    = p3;
        } else {
            // childIAS = 0 → nodo interno sin hijos (configuración incompleta)
            // Tratar como hoja: resonancia con las primeras strings
            if (primIdx < g_inception_params.numStrings) {
                const SemanticString& str = g_inception_params.strings[primIdx];
                payload.accumulated += semanticStringResonance(str.resonance, omega);
                payload.omega = omega;
            }
        }
    } else {
        // ── NODO HOJA (nivel 3): Resonancia Fourier ─────────────────────
        //
        // Aquí ya no hay más saltos dimensionales.
        // Calculamos la resonancia Fourier de la SemanticString y acumulamos.
        // W(ω) = outputScale · tanh( Σ_{k=1}^{M} a_k·sin(kω) + b_k·cos(kω) )
        //
        // POLISEMIA RESUELTA:
        // La misma SemanticString con distinto ω devuelve distinta resonancia.
        // No hay duplicación de matrices — solo un escalar ω diferente.

        if (primIdx < g_inception_params.numStrings) {
            const SemanticString& str = g_inception_params.strings[primIdx];

            // Resonancia Fourier on-the-fly (desde registros, sin acceso a HBM)
            const float resonance = semanticStringResonance(str.resonance, omega);

            // Peso de profundidad: nodos hoja = máxima especificidad semántica
            const float depthWeight = 1.0f + 0.3f * (float)sphere.depth;
            payload.accumulated += resonance * depthWeight;
            payload.omega = omega;
            payload.depth = max(payload.depth, (uint32_t)INCEPTION_MAX_DEPTH - 1);
        }
    }

    payloadWrite(payload);
}

// ============================================================================
// PROGRAMA OPTIX: Miss — Terminación Limpia
// ============================================================================

/**
 * @brief Se ejecuta cuando el rayo escapa de la escena (o del IAS hijo) sin colisión.
 *
 * Un miss en el Inception Engine puede ocurrir en cualquier nivel:
 *   - Nivel 0: el rayo no golpeó ningún dominio (concepto ausente del vocabulario)
 *   - Nivel 1-3: el rayo golpeó el dominio pero no encontró subcategoría/concepto
 *
 * En todos los casos, preservamos el acumulado existente y terminamos.
 * El payload vuelve al raygen con los hits que se hayan producido hasta el miss.
 */
extern "C" __global__
void __miss__inception_portal() {
    // No modificar payload — preservar resonancia acumulada en niveles anteriores.
    // La terminación limpia es intencional: el rayo exploró lo que pudo.
}

// ============================================================================
// PROGRAMA OPTIX: Intersection — Esfera Semántica (reutilizado)
// ============================================================================

/**
 * @brief Intersección rayo-esfera para GAS de primitivas custom.
 *
 * Idéntico a __intersection__sphere de liquidbit_kernels.cu.
 * Reutilizado aquí para que inception_kernels.cu sea auto-contenido.
 * El SBT del inception pipeline apunta a este programa.
 */
extern "C" __global__
void __intersection__inception_sphere() {
    const uint32_t primIdx = optixGetPrimitiveIndex();
    const SemanticSphere& sphere = g_inception_params.spheres[primIdx];

    // Usar world-space para coherencia con sphere.center en coords mundiales
    const float3 origin = optixGetWorldRayOrigin();
    const float3 dir    = optixGetWorldRayDirection();

    const float3 oc = make_float3(
        origin.x - sphere.center.x,
        origin.y - sphere.center.y,
        origin.z - sphere.center.z
    );

    const float a    = dir.x*dir.x + dir.y*dir.y + dir.z*dir.z;
    const float b    = 2.0f * (oc.x*dir.x + oc.y*dir.y + oc.z*dir.z);
    const float c    = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z
                       - sphere.radius * sphere.radius;
    const float disc = b*b - 4.0f*a*c;

    if (disc < 0.0f) return;

    const float sqrtDisc = sqrtf(disc);
    const float tmin     = optixGetRayTmin();
    const float tmax     = optixGetRayTmax();

    float t = (-b - sqrtDisc) / (2.0f * a);
    if (t < tmin) {
        t = (-b + sqrtDisc) / (2.0f * a);
    }
    if (t < tmin || t > tmax) return;

    optixReportIntersection(t, 0u);
}
