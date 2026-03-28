/**
 * miss.cu
 *
 * Programa OptiX Miss (miss shader)
 *
 * Se ejecuta cuando un rayo NO intersecta ningún objeto en el BVH.
 * Esto ocurre cuando el rayo se propaga a través de todo el espacio semántico
 * sin golpear ningún token relevante.
 *
 * En el contexto de la atención óptica, un "miss" significa que el rayo
 * no encontró tokens dentro de su "cono de influencia" semántica.
 *
 * Responsabilidades:
 * 1. Marcar el rayo como terminado
 * 2. Establecer el peso de atención final a 0 (el rayo no golpeó nada)
 * 3. Retornar para completar la traversal
 */

#include <optix.h>
#include <cuda_runtime.h>
#include <vector_types.h>

#include "../include/optical_attention.h"

/* ============================================================================
 * PROGRAMA OPTIX: __miss__
 *
 * Se invoca automáticamente cuando optixTrace() no encuentra intersecciones.
 *
 * Parámetros (implícitos):
 *   - optixGetPayload_0/1/2: Datos del rayo
 *   - optixGetWorldRayOrigin(): Punto de lanzamiento del rayo
 *   - optixGetWorldRayDirection(): Dirección del rayo
 *
 * Modifica:
 *   - optixSetPayload_0/1/2: Actualiza los datos del rayo con resultado final
 * ============================================================================
 */
extern "C" __global__ void __miss__ms_optical_attention() {
    // ========================================================================
    // RECUPERAR PAYLOAD DEL RAYO
    // ========================================================================

    uint32_t payload_0 = optixGetPayload_0();
    uint32_t payload_1 = optixGetPayload_1();
    uint32_t payload_2 = optixGetPayload_2();

    float accumulated_attention = __uint_as_float(payload_0);
    float energy_remaining = __uint_as_float(payload_1);
    uint32_t hit_count = payload_2;

    // ========================================================================
    // SEMÁNTICA DE UN "MISS"
    //
    // En el contexto del mecanismo de atención:
    // - El rayo no golpeó ningún token relevante
    // - No hay tokens "cercanos" a la dirección del rayo
    // - No se acumula peso de atención adicional
    //
    // Opciones de manejo:
    // 1. Establecer weight = 0 (el rayo contribuye 0 a la atención)
    // 2. Usar un background weight constante (menos común)
    // 3. Mantener el weight acumulado previo (poco usado)
    //
    // Elegimos opción 1: el miss no contribuye a la atención
    // ========================================================================

    // En un miss, no hay token golpeado, así que no añadimos peso
    // accumulated_attention permanece sin cambios (solo hits cuentan)

    // La energía del rayo tampoco cambia si no hay absorción
    // energy_remaining permanece igual

    // El hit_count no se incrementa (no hubo intersección)
    // hit_count permanece sin cambios

    // ========================================================================
    // FINALIZAR EL RAYO
    // ========================================================================

    // Escribir el payload final (sin cambios respecto a la entrada)
    optixSetPayload_0(__float_as_uint(accumulated_attention));
    optixSetPayload_1(__float_as_uint(energy_remaining));
    optixSetPayload_2(hit_count);

    // El rayo termina automáticamente después del miss
    // No es necesario llamar a optixTerminateRay()
}

/* ============================================================================
 * PROGRAMA OPTIX ALTERNATIVO: __miss__ con background illumination
 *
 * Versión que asigna un pequeño "weight de fondo" para rayos que no golpean nada.
 * Útil si queremos que los rayos que no encuentran tokens tengan cierta contribución.
 *
 * (Probablemente no será necesario en la mayoría de casos, pero aquí disponible)
 * ============================================================================
 */
extern "C" __global__ void __miss__ms_optical_attention_with_background() {
    uint32_t payload_0 = optixGetPayload_0();
    uint32_t payload_1 = optixGetPayload_1();
    uint32_t payload_2 = optixGetPayload_2();

    float accumulated_attention = __uint_as_float(payload_0);
    float energy_remaining = __uint_as_float(payload_1);
    uint32_t hit_count = payload_2;

    // Pequeño weight de fondo para rayos que no encuentran nada
    // Típicamente muy pequeño, ej. 0.001 * energy_remaining
    float background_weight = 0.001f * energy_remaining;

    accumulated_attention += background_weight;

    optixSetPayload_0(__float_as_uint(accumulated_attention));
    optixSetPayload_1(__float_as_uint(energy_remaining));
    optixSetPayload_2(hit_count);
}

