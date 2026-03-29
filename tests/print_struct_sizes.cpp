/**
 * @file tests/print_struct_sizes.cpp
 * @brief Imprime sizeof y offsetof de todos los structs de Inception.
 *
 * Salida JSON para que inference.py la valide automáticamente.
 * Compila con: cmake --build build --target print_struct_sizes
 * Ejecuta:     print_struct_sizes.exe
 */

#include <cstdio>
#include <cstddef>

#include "spectral_resonance.h"
#include "token_geometry.h"

int main() {
    printf("{\n");

    // SemanticSphere
    printf("  \"SemanticSphere\": {\n");
    printf("    \"sizeof\": %zu,\n", sizeof(SemanticSphere));
    printf("    \"alignof\": %zu,\n", alignof(SemanticSphere));
    printf("    \"offsetof_center\": %zu,\n", offsetof(SemanticSphere, center));
    printf("    \"offsetof_radius\": %zu,\n", offsetof(SemanticSphere, radius));
    printf("    \"offsetof_instanceId\": %zu,\n", offsetof(SemanticSphere, instanceId));
    printf("    \"offsetof_childIAS\": %zu,\n", offsetof(SemanticSphere, childIAS));
    printf("    \"offsetof_depth\": %zu,\n", offsetof(SemanticSphere, depth));
    printf("    \"offsetof_frequencyBias\": %zu\n", offsetof(SemanticSphere, frequencyBias));
    printf("  },\n");

    // ResonanceParams
    printf("  \"ResonanceParams\": {\n");
    printf("    \"sizeof\": %zu,\n", sizeof(ResonanceParams));
    printf("    \"alignof\": %zu,\n", alignof(ResonanceParams));
    printf("    \"offsetof_a\": %zu,\n", offsetof(ResonanceParams, a));
    printf("    \"offsetof_b\": %zu,\n", offsetof(ResonanceParams, b));
    printf("    \"offsetof_numModes\": %zu,\n", offsetof(ResonanceParams, numModes));
    printf("    \"offsetof_outputScale\": %zu,\n", offsetof(ResonanceParams, outputScale));
    printf("    \"offsetof_semanticTag\": %zu\n", offsetof(ResonanceParams, semanticTag));
    printf("  },\n");

    // SemanticString
    printf("  \"SemanticString\": {\n");
    printf("    \"sizeof\": %zu,\n", sizeof(SemanticString));
    printf("    \"alignof\": %zu,\n", alignof(SemanticString));
    printf("    \"offsetof_resonance\": %zu,\n", offsetof(SemanticString, resonance));
    printf("    \"offsetof_position\": %zu,\n", offsetof(SemanticString, position));
    printf("    \"offsetof_stringId\": %zu\n", offsetof(SemanticString, stringId));
    printf("  },\n");

    // AffinePortal
    printf("  \"AffinePortal\": {\n");
    printf("    \"sizeof\": %zu,\n", sizeof(AffinePortal));
    printf("    \"alignof\": %zu,\n", alignof(AffinePortal));
    printf("    \"offsetof_rows\": %zu\n", offsetof(AffinePortal, rows));
    printf("  },\n");

    // SpectralAttentionResult
    printf("  \"SpectralAttentionResult\": {\n");
    printf("    \"sizeof\": %zu,\n", sizeof(SpectralAttentionResult));
    printf("    \"alignof\": %zu,\n", alignof(SpectralAttentionResult));
    printf("    \"offsetof_attentionWeight\": %zu,\n", offsetof(SpectralAttentionResult, attentionWeight));
    printf("    \"offsetof_finalOmega\": %zu,\n", offsetof(SpectralAttentionResult, finalOmega));
    printf("    \"offsetof_dominantStringId\": %zu,\n", offsetof(SpectralAttentionResult, dominantStringId));
    printf("    \"offsetof_traversalDepth\": %zu,\n", offsetof(SpectralAttentionResult, traversalDepth));
    printf("    \"offsetof_exitDirection\": %zu,\n", offsetof(SpectralAttentionResult, exitDirection));
    printf("    \"offsetof_energyRemaining\": %zu\n", offsetof(SpectralAttentionResult, energyRemaining));
    printf("  },\n");

    // InceptionLaunchParams
    printf("  \"InceptionLaunchParams\": {\n");
    printf("    \"sizeof\": %zu,\n", sizeof(InceptionLaunchParams));
    printf("    \"alignof\": %zu\n", alignof(InceptionLaunchParams));
    printf("  }\n");

    printf("}\n");
    return 0;
}
