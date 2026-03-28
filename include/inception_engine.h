/**
 * @file inception_engine.h
 * @brief LiquidBit v4.0 "Inception Engine" — Host-side 4-level IAS management
 *
 * ARQUITECTURA (Los 3 Ceros de LiquidBit v4.0):
 * ==============================================
 *   Cero MatMul    — Ray Tracing O(log N) via RT Cores
 *   Cero Memoria   — Aritmética Ternaria {-1, 0, +1} (Fase 2)
 *   Cero Límite    — IAS anidados 4 niveles × 3D = 12 dims efectivas
 *
 * JERARQUÍA DE IAS (INCEPTION FRACTAL):
 * ======================================
 *
 *   IAS_root (nivel 0 — Dominios)
 *   ├── IAS_level1[0] (nivel 1 — Subdominios de "Código")
 *   │   ├── IAS_level2[0] (nivel 2 — Conceptos de "Python")
 *   │   │   └── GAS_leaves (nivel 3 — SemanticStrings concretas)
 *   │   └── IAS_level2[1] (nivel 2 — Conceptos de "Bucles")
 *   └── IAS_level1[1] (nivel 1 — Subdominios de "Música")
 *       └── ...
 *
 * Cada flecha es un OptixInstance con transform[12] — portal de coordenadas.
 * OptiX navega la jerarquía completa SIN código extra del usuario.
 * Complejidad: O(4 × log N) = O(log N).
 *
 * USO TÍPICO:
 * ===========
 * @code
 *   InceptionEngine engine;
 *   engine.init(optixContext, ptxCode);
 *
 *   InceptionScene scene;
 *   scene.addDomain("código",   gloveEmbedding("código"),   0.3f);
 *   scene.addDomain("música",   gloveEmbedding("música"),   0.3f);
 *   scene.addDomain("ciencia",  gloveEmbedding("ciencia"),  0.3f);
 *   // ... añadir subdominios, conceptos y strings
 *
 *   engine.buildScene(scene);
 *   engine.launch(baseOmega, numRays, results);
 *   engine.destroy();
 * @endcode
 *
 * COMPATIBILIDAD:
 * ===============
 *   - OptiX SDK 9.1+ (RTX 5070 Ti / Blackwell sm_120)
 *   - CUDA 12.8+ (compute 8.9 / 12.0)
 *   - C++17
 *
 * @author LiquidBit Zero-Matrix Team
 * @date 2026
 */

#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "liquidbit_resonance.h"

#ifdef __cplusplus
#include <vector>
#include <string>
#include <functional>
#endif

// ============================================================================
// CONSTANTES DE LA ARQUITECTURA INCEPTION
// ============================================================================

/// Máximo de dominios (nivel 0) en una escena
#define INCEPTION_MAX_DOMAINS       64

/// Máximo de subdominios por dominio (nivel 1)
#define INCEPTION_MAX_SUBDOMAINS    64

/// Máximo de conceptos por subdominio (nivel 2)
#define INCEPTION_MAX_CONCEPTS      256

/// Máximo de SemanticStrings por concepto (nivel 3, nodos hoja)
#define INCEPTION_MAX_STRINGS       1024

/// Máximo de rayos por launch
#define INCEPTION_MAX_RAYS          4096

/// Tolerancia de solapamiento para nodos polisémicos (OHBSC)
#define INCEPTION_OVERLAP_ALPHA     0.3f

/// Temperatura inicial para annealing de membresía difusa
#define INCEPTION_FUZZY_TEMP_INIT   1.0f

// ============================================================================
// STRUCTS DE NODOS JERÁRQUICOS (HOST-SIDE)
// ============================================================================

/**
 * @brief Nodo hoja: SemanticString con posición en el árbol.
 *
 * Nivel 3 de la jerarquía. Contiene los coeficientes Fourier aprendidos
 * y la posición 3D en el espacio semántico del concepto padre.
 */
typedef struct {
    SemanticString  string;        ///< Datos Fourier y posición del nodo
    uint32_t        parentId;      ///< ID del concepto padre (nivel 2)
    float           membershipFuzzy; ///< P(string ∈ concepto) — para OHBSC
} InceptionLeaf;

/**
 * @brief Nodo de concepto (nivel 2).
 *
 * Agrupa SemanticStrings que representan facetas del mismo concepto.
 * Un concepto puede ser polisémico (membership > umbral en múltiples subdominios).
 */
typedef struct {
    SemanticSphere  sphere;             ///< Esfera que encierra este concepto en nivel 2
    AffinePortal    portalToLeaves;     ///< Transformación de coordenadas al nivel 3
    uint32_t        leafIds[16];        ///< IDs de SemanticStrings en este concepto
    uint32_t        numLeaves;          ///< Número de leaves activas
    uint32_t        parentSubdomainId;  ///< ID del subdominio padre
    char            label[32];          ///< Etiqueta legible (debug)
    float           polysemyScore;      ///< Entropía de membresía (alto = polisémico)
    uint32_t        wormholeTarget;     ///< ID del concepto gemelo si polisémico (0 = ninguno)
} InceptionConcept;

/**
 * @brief Nodo de subdominio (nivel 1).
 *
 * Agrupa conceptos relacionados dentro de un dominio temático.
 * Ejemplo: "Python" y "Bucles" son subdominios de "Código".
 */
typedef struct {
    SemanticSphere  sphere;              ///< Esfera de nivel 1 en el espacio del dominio
    AffinePortal    portalToConcepts;    ///< Transformación de coordenadas al nivel 2
    uint32_t        conceptIds[32];      ///< IDs de conceptos en este subdominio
    uint32_t        numConcepts;         ///< Número de conceptos activos
    uint32_t        parentDomainId;      ///< ID del dominio padre
    char            label[32];           ///< Etiqueta legible (debug)
} InceptionSubdomain;

/**
 * @brief Nodo de dominio (nivel 0, raíz).
 *
 * Los dominios son las categorías semánticas de más alto nivel.
 * Ejemplo: "Código", "Música", "Ciencia", "Finanzas", etc.
 *
 * La `portalToSubdomains` transforma el vector de contexto global f
 * en el espacio semántico local del dominio. Esto es el "salto dimensional":
 * el rayo cambia de coordenadas al entrar al dominio.
 */
typedef struct {
    SemanticSphere  sphere;                  ///< Esfera de nivel 0 en el espacio raíz
    AffinePortal    portalToSubdomains;      ///< Transformación de coordenadas al nivel 1
    uint32_t        subdomainIds[16];        ///< IDs de subdominios en este dominio
    uint32_t        numSubdomains;           ///< Número de subdominios activos
    char            label[32];              ///< Etiqueta legible (debug)
    float3          embedding3d;            ///< Posición 3D derivada de GloVe/embedding
} InceptionDomain;

// ============================================================================
// SCENA INCEPTION (COLECCIÓN COMPLETA)
// ============================================================================

/**
 * @brief Escena completa del Inception Engine.
 *
 * Contiene todos los nodos en los 4 niveles de la jerarquía.
 * Se pasa a InceptionEngine::buildScene() para construir los IAS en GPU.
 */
typedef struct {
    /* Nivel 0 — Dominios */
    InceptionDomain    domains[INCEPTION_MAX_DOMAINS];
    uint32_t           numDomains;

    /* Nivel 1 — Subdominios */
    InceptionSubdomain subdomains[INCEPTION_MAX_SUBDOMAINS];
    uint32_t           numSubdomains;

    /* Nivel 2 — Conceptos */
    InceptionConcept   concepts[INCEPTION_MAX_CONCEPTS];
    uint32_t           numConcepts;

    /* Nivel 3 — SemanticStrings (hojas) */
    InceptionLeaf      leaves[INCEPTION_MAX_STRINGS];
    uint32_t           numLeaves;

    /* Portales globales (uno por nivel de transición) */
    AffinePortal       globalPortals[INCEPTION_MAX_DEPTH];

    /* Frecuencia base del contexto */
    float              baseOmega;
} InceptionScene;

// ============================================================================
// INCEPTION ENGINE (HOST-SIDE — C++ only)
// ============================================================================

#ifdef __cplusplus

/**
 * @brief Motor principal del Inception Engine.
 *
 * Gestiona:
 *   1. Inicialización del pipeline OptiX (una sola vez)
 *   2. Construcción de IAS multinivel desde una InceptionScene
 *   3. Lanzamiento de rayos y recolección de resultados
 *   4. Liberación de recursos GPU
 *
 * Thread safety: NO thread-safe. Un motor por contexto.
 */
class InceptionEngine {
public:
    InceptionEngine() = default;
    ~InceptionEngine() { if (m_initialized) destroy(); }

    /* No copiable */
    InceptionEngine(const InceptionEngine&) = delete;
    InceptionEngine& operator=(const InceptionEngine&) = delete;

    /* Movible */
    InceptionEngine(InceptionEngine&&) = default;
    InceptionEngine& operator=(InceptionEngine&&) = default;

    // ─────────────────────────────────────────────────────────────────
    // Ciclo de vida
    // ─────────────────────────────────────────────────────────────────

    /**
     * @brief Inicializa el pipeline OptiX con los entry points del PTX.
     *
     * @param context    Contexto OptiX creado por el caller
     * @param ptxData    Contenido del archivo liquidbit_kernels.ptx
     * @param ptxSize    Tamaño del PTX en bytes
     * @return           0 en éxito, código de error OptiX en fallo
     */
    int init(OptixDeviceContext context, const char* ptxData, size_t ptxSize);

    /**
     * @brief Libera todos los recursos GPU y OptiX.
     */
    void destroy();

    // ─────────────────────────────────────────────────────────────────
    // Construcción de escena
    // ─────────────────────────────────────────────────────────────────

    /**
     * @brief Construye los IAS de los 4 niveles en GPU desde una InceptionScene.
     *
     * Secuencia de construcción:
     *   1. GAS (nivel 3): AABB por cada InceptionLeaf::string.position
     *   2. IAS_level3 (nivel 2→3): instancias de GAS por concepto
     *   3. IAS_level2 (nivel 1→2): instancias de IAS_level3 por subdominio
     *   4. IAS_level1 (nivel 0→1): instancias de IAS_level2 por dominio
     *   5. IAS_root (raíz): instancias de IAS_level1
     *
     * @param scene  Escena con nodos en los 4 niveles
     * @return       0 en éxito
     */
    int buildScene(const InceptionScene& scene);

    /**
     * @brief Actualiza las posiciones de las hojas sin reconstruir la jerarquía.
     *
     * Útil durante training cuando los embeddings cambian pero la estructura
     * del árbol permanece estable.
     *
     * @param leaves      Nuevas posiciones de las SemanticStrings
     * @param numLeaves   Número de leaves a actualizar
     * @return            0 en éxito
     */
    int updateLeafPositions(const InceptionLeaf* leaves, uint32_t numLeaves);

    // ─────────────────────────────────────────────────────────────────
    // Ejecución
    // ─────────────────────────────────────────────────────────────────

    /**
     * @brief Lanza el pipeline de ray tracing Inception.
     *
     * Genera `numRays` rayos Fibonacci desde el origen con frecuencia `baseOmega`.
     * Cada rayo navega los 4 niveles de IAS y calcula resonancia Fourier en las hojas.
     *
     * @param baseOmega   Frecuencia de contexto base [0, 2π]
     * @param numRays     Número de rayos a lanzar
     * @param stream      CUDA stream (0 = default)
     * @return            0 en éxito
     */
    int launch(float baseOmega, uint32_t numRays, cudaStream_t stream = 0);

    /**
     * @brief Copia los resultados del último launch desde GPU a CPU.
     *
     * @param outResults  Buffer de salida (tamaño >= numRays)
     * @param numRays     Número de resultados a copiar
     * @return            0 en éxito
     */
    int copyResults(SpectralAttentionResult* outResults, uint32_t numRays);

    // ─────────────────────────────────────────────────────────────────
    // Utilidades de construcción de escena
    // ─────────────────────────────────────────────────────────────────

    /**
     * @brief Construye una InceptionScene desde embeddings GloVe-300d.
     *
     * Usa OHBSC (Overlapping Hierarchical Bounding Sphere Clustering) con
     * membresía difusa para asignar tokens a nodos.
     *
     * @param embeddings   Array de embeddings [numTokens × dim]
     * @param numTokens    Número de tokens
     * @param dim          Dimensión del embedding (300 para GloVe-300d)
     * @param outScene     Escena construida
     * @return             0 en éxito
     */
    static int buildSceneFromEmbeddings(
        const float* embeddings, uint32_t numTokens, uint32_t dim,
        InceptionScene* outScene
    );

    /**
     * @brief Genera coeficientes Fourier iniciales desde un embedding.
     *
     * Proyecta el embedding al espacio de coeficientes (a[], b[]) usando
     * las primeras 2*M componentes PCA del embedding.
     *
     * @param embedding    Embedding de dimensión D
     * @param dim          Dimensión D del embedding
     * @param outParams    Parámetros de resonancia calculados
     */
    static void embeddingToFourier(
        const float* embedding, uint32_t dim,
        ResonanceParams* outParams
    );

    /**
     * @brief Crea un AffinePortal identidad (sin transformación).
     */
    static AffinePortal identityPortal();

    /**
     * @brief Crea un AffinePortal desde rotación + traslación.
     *
     * @param rotation     Matriz 3×3 de rotación (row-major)
     * @param translation  Vector de traslación 3D
     */
    static AffinePortal makePortal(const float rotation[9], float3 translation);

    // ─────────────────────────────────────────────────────────────────
    // Estado
    // ─────────────────────────────────────────────────────────────────

    bool isInitialized()  const { return m_initialized; }
    bool hasScene()       const { return m_sceneBuilt; }
    uint32_t numDomains() const { return m_numDomains; }

private:
    // ── OptiX handles ─────────────────────────────────────────────────
    OptixDeviceContext      m_context         = nullptr;
    OptixModule             m_module          = nullptr;
    OptixProgramGroup       m_raygenPG        = nullptr;
    OptixProgramGroup       m_closesthitPG    = nullptr;
    OptixProgramGroup       m_missPG          = nullptr;
    OptixPipeline           m_pipeline        = nullptr;
    OptixShaderBindingTable m_sbt             = {};

    // ── IAS handles (4 niveles) ───────────────────────────────────────
    OptixTraversableHandle  m_rootIAS         = 0;  ///< Nivel 0 (raíz)
    OptixTraversableHandle  m_domainIAS       = 0;  ///< Nivel 1
    OptixTraversableHandle  m_subdomainIAS    = 0;  ///< Nivel 2
    OptixTraversableHandle  m_conceptIAS      = 0;  ///< Nivel 3
    OptixTraversableHandle  m_leafGAS         = 0;  ///< GAS de hojas

    // ── Buffers GPU ───────────────────────────────────────────────────
    CUdeviceptr  m_d_rootIASBuf          = 0;
    CUdeviceptr  m_d_domainIASBuf        = 0;
    CUdeviceptr  m_d_subdomainIASBuf     = 0;
    CUdeviceptr  m_d_conceptIASBuf       = 0;
    CUdeviceptr  m_d_leafGASBuf          = 0;
    CUdeviceptr  m_d_spheresBuf          = 0;
    CUdeviceptr  m_d_portalsBuf          = 0;
    CUdeviceptr  m_d_stringsBuf          = 0;
    CUdeviceptr  m_d_resultsBuf          = 0;
    CUdeviceptr  m_d_launchParamsBuf     = 0;
    CUdeviceptr  m_d_sbtRaygenBuf        = 0;
    CUdeviceptr  m_d_sbtMissBuf          = 0;
    CUdeviceptr  m_d_sbtHitBuf           = 0;

    // ── Estado ────────────────────────────────────────────────────────
    bool     m_initialized  = false;
    bool     m_sceneBuilt   = false;
    uint32_t m_numDomains   = 0;
    uint32_t m_numLeaves    = 0;
    uint32_t m_maxRays      = INCEPTION_MAX_RAYS;

    // ── Helpers privados ──────────────────────────────────────────────
    int  _buildLeafGAS(const InceptionLeaf* leaves, uint32_t numLeaves);
    int  _buildConceptIAS(const InceptionConcept* concepts, uint32_t n);
    int  _buildSubdomainIAS(const InceptionSubdomain* subdomains, uint32_t n);
    int  _buildDomainIAS(const InceptionDomain* domains, uint32_t n);
    int  _buildRootIAS(const InceptionDomain* domains, uint32_t n);
    int  _uploadSceneData(const InceptionScene& scene);
    int  _setupSBT();
    void _freeBuf(CUdeviceptr& buf);
};

// ============================================================================
// UTILIDADES DE CONSTRUCCIÓN (inline helpers)
// ============================================================================

/**
 * @brief Crea una InceptionScene vacía con portales identidad.
 */
inline InceptionScene makeEmptyScene(float baseOmega = 0.785398f) {
    InceptionScene s;
    memset(&s, 0, sizeof(s));
    s.baseOmega = baseOmega;
    for (int i = 0; i < INCEPTION_MAX_DEPTH; ++i) {
        AffinePortal& p = s.globalPortals[i];
        memset(&p, 0, sizeof(p));
        p.rows[0].x = 1.0f;
        p.rows[1].y = 1.0f;
        p.rows[2].z = 1.0f;
        p.rows[3].w = 1.0f;
    }
    return s;
}

/**
 * @brief Crea una SemanticSphere en la posición dada.
 *
 * @param cx,cy,cz  Centro de la esfera
 * @param radius    Radio semántico
 * @param id        ID único dentro del IAS padre
 * @param depth     Nivel en la jerarquía (0=raíz, 3=hoja)
 * @param freqBias  Sesgo de frecuencia Δω
 */
inline SemanticSphere makeSphere(
    float cx, float cy, float cz, float radius,
    uint32_t id, uint32_t depth, float freqBias = 0.0f
) {
    SemanticSphere s;
    s.center       = make_float3(cx, cy, cz);
    s.radius       = radius;
    s.instanceId   = id;
    s.childIAS     = 0;
    s.depth        = depth;
    s.frequencyBias = freqBias;
    return s;
}

/**
 * @brief Crea unos ResonanceParams con coeficientes uniformes pequeños.
 *
 * Sirve como inicialización antes del training.
 */
inline ResonanceParams makeDefaultResonanceParams(
    uint32_t numModes = RESONANCE_NUM_MODES,
    float scale = 1.0f,
    uint32_t tag = 0
) {
    ResonanceParams p;
    for (uint32_t k = 0; k < RESONANCE_NUM_MODES; ++k) {
        p.a[k] = (k < numModes) ? 0.01f : 0.0f;
        p.b[k] = (k < numModes) ? 0.01f : 0.0f;
    }
    p.numModes    = numModes;
    p.outputScale = scale;
    p.semanticTag = tag;
    p._pad        = 0;
    return p;
}

/**
 * @brief Crea una SemanticString en la posición dada con coeficientes por defecto.
 */
inline SemanticString makeSemanticString(
    float px, float py, float pz,
    uint32_t stringId,
    uint32_t numModes = RESONANCE_NUM_MODES
) {
    SemanticString ss;
    ss.resonance = makeDefaultResonanceParams(numModes, 1.0f, stringId);
    ss.position  = make_float3(px, py, pz);
    ss.stringId  = stringId;
    return ss;
}

#endif // __cplusplus
