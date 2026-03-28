#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════╗
║  LiquidBit Zero-Matrix — Embeddings Downloader v2.0                    ║
║  NUEVA IDEA (doc 1.pdf/2.pdf): gensim downloader en 1 línea            ║
║  ─────────────────────────────────────────────────────────────────────  ║
║  Métodos de carga disponibles:                                         ║
║                                                                         ║
║  1. gensim (recomendado, pip install gensim):                          ║
║       api.load("glove-wiki-gigaword-300")   ← automático               ║
║       api.load("word2vec-google-news-300")  ← alternativa              ║
║                                                                         ║
║  2. GloVe manual (si tienes el .txt descargado):                       ║
║       python3 download_embeddings_v2.py --source glove-file            ║
║       --path ~/Downloads/glove.6B.50d.txt                              ║
║                                                                         ║
║  3. Sintético (fallback sin internet):                                 ║
║       python3 download_embeddings_v2.py --source synthetic             ║
║                                                                         ║
║  Salida:                                                                ║
║    embeddings_3d.npy  — [N, 3] posiciones para el BVH                 ║
║    embeddings_full.npy — [N, D] embeddings originales (para MatMul)   ║
║    vocab.txt          — vocabulario                                    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import argparse
import numpy as np
import time

# ══════════════════════════════════════════════════════════════════════════
# DESCARGA GENSIM (idea nueva de los documentos)
# ══════════════════════════════════════════════════════════════════════════

def load_with_gensim(model_name: str = "glove-wiki-gigaword-50"):
    """
    Descarga embeddings usando gensim.downloader.

    Modelos disponibles (ordenados por tamaño):
      • glove-wiki-gigaword-50    →  50d, ~66MB  ← ideal para prototipos
      • glove-wiki-gigaword-100   → 100d, ~128MB
      • glove-wiki-gigaword-300   → 300d, ~376MB
      • word2vec-google-news-300  → 300d, ~1.6GB ← máxima calidad
      • fasttext-wiki-news-subwords-300 → 300d

    Uso:
        import gensim.downloader as api
        model = api.load("glove-wiki-gigaword-50")
        vec = model["python"]   # [50,]
    """
    try:
        import gensim.downloader as api
    except ImportError:
        print("  gensim no encontrado. Instalar con:")
        print("  pip install gensim")
        return None, None

    print(f"  Descargando {model_name} via gensim...")
    print(f"  (Se cachea en ~/.local/share/gensim-data, solo se descarga 1 vez)")
    t0 = time.perf_counter()

    try:
        model = api.load(model_name)
        elapsed = time.perf_counter() - t0
        print(f"  ✅ Cargado en {elapsed:.1f}s — {len(model)} palabras, dim={model.vector_size}")
        return model, model.vector_size
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None, None


def gensim_to_arrays(model, max_words: int = 50000):
    """
    Convierte gensim KeyedVectors a arrays numpy.
    Preserva las max_words palabras más frecuentes (primer índice = más frecuente).

    Returns:
        vocab      : list[str]          — palabras
        embeddings : np.ndarray [N, D]  — vectores FP32
    """
    vocab = []
    vecs  = []

    for i, word in enumerate(model.index_to_key):
        if i >= max_words:
            break
        vocab.append(word)
        vecs.append(model[word])

    embeddings = np.array(vecs, dtype=np.float32)
    print(f"  Exportados: {len(vocab)} palabras × {embeddings.shape[1]}d")
    return vocab, embeddings


# ══════════════════════════════════════════════════════════════════════════
# CARGA ARCHIVO GLOVE MANUAL
# ══════════════════════════════════════════════════════════════════════════

def load_glove_file(path: str, max_words: int = 50000):
    """
    Carga un archivo GloVe en formato texto (Stanford NLP).
    Descarga manual: https://nlp.stanford.edu/data/glove.6B.zip
    """
    if not os.path.exists(path):
        print(f"  ❌ Archivo no encontrado: {path}")
        print(f"  Descarga desde: https://nlp.stanford.edu/data/glove.6B.zip")
        return None, None

    print(f"  Cargando GloVe desde {path}...")
    vocab = []
    vecs  = []
    t0    = time.perf_counter()

    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_words:
                break
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            vocab.append(parts[0])
            vecs.append(np.array(parts[1:], dtype=np.float32))

    elapsed = time.perf_counter() - t0
    embeddings = np.array(vecs, dtype=np.float32)
    print(f"  ✅ Cargado en {elapsed:.1f}s — {len(vocab)} palabras × {embeddings.shape[1]}d")
    return vocab, embeddings


# ══════════════════════════════════════════════════════════════════════════
# EMBEDDINGS SINTÉTICOS (fallback)
# ══════════════════════════════════════════════════════════════════════════

def generate_synthetic(dim: int = 50, n_words: int = 500):
    """
    Genera embeddings sintéticos con estructura semántica correcta.
    Garantiza clusters bien separados para probar el BVH.
    """
    np.random.seed(2026)

    clusters = {
        "programacion": {
            "words": ["python", "code", "loop", "for", "while", "function", "class",
                      "variable", "array", "algorithm", "debug", "compiler", "syntax",
                      "module", "import", "return", "if", "else", "def", "int"],
            "center": np.array([1.0, 0.0, 0.0, 0.0, 0.0] + [0.0]*(dim-5), dtype=np.float32),
        },
        "musica": {
            "words": ["music", "rhythm", "beat", "tempo", "melody", "chord", "note",
                      "song", "guitar", "piano", "bass", "drum", "harmony", "scale",
                      "octave", "frequency", "sound", "audio", "pitch", "tone"],
            "center": np.array([0.0, 1.0, 0.0, 0.0, 0.0] + [0.0]*(dim-5), dtype=np.float32),
        },
        "fisica": {
            "words": ["physics", "force", "mass", "energy", "orbit", "wave", "field",
                      "quantum", "gravity", "velocity", "momentum", "charge", "particle",
                      "light", "photon", "electron", "proton", "atom", "nuclear", "plasma"],
            "center": np.array([0.0, 0.0, 1.0, 0.0, 0.0] + [0.0]*(dim-5), dtype=np.float32),
        },
        "lenguaje": {
            "words": ["word", "sentence", "grammar", "language", "text", "paragraph",
                      "meaning", "semantic", "syntax_lang", "token", "vocabulary", "corpus",
                      "context", "embedding", "vector", "attention", "transformer",
                      "translation", "document", "phrase"],
            "center": np.array([0.0, 0.0, 0.0, 1.0, 0.0] + [0.0]*(dim-5), dtype=np.float32),
        },
        "matematicas": {
            "words": ["matrix", "vector_m", "derivative", "integral", "equation",
                      "theorem", "proof", "algebra", "calculus", "geometry", "topology",
                      "graph", "set", "function_m", "limit", "series", "probability",
                      "statistics", "linear", "nonlinear"],
            "center": np.array([0.0, 0.0, 0.0, 0.0, 1.0] + [0.0]*(dim-5), dtype=np.float32),
        },
        "general": {  # palabras comunes (relleno)
            "words": [f"word_{i}" for i in range(n_words - 100)],
            "center": np.zeros(dim, dtype=np.float32),
        },
    }

    vocab      = []
    embeddings = []

    for cluster_name, cluster_data in clusters.items():
        center = cluster_data["center"]
        for w in cluster_data["words"]:
            noise = np.random.randn(dim).astype(np.float32) * 0.3
            vec   = center + noise
            # Normalizar a norma ≈ 1
            vec  /= (np.linalg.norm(vec) + 1e-8)
            vocab.append(w)
            embeddings.append(vec)

    # Normalizar todos
    embeddings = np.array(embeddings, dtype=np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings /= (norms + 1e-8)

    print(f"  ✅ Sintético: {len(vocab)} palabras × {dim}d (5 clusters semánticos)")
    return vocab, embeddings


# ══════════════════════════════════════════════════════════════════════════
# PCA MANUAL D → 3D
# ══════════════════════════════════════════════════════════════════════════

def pca_to_3d(embeddings: np.ndarray):
    """
    Proyección PCA D → 3D preservando máxima varianza.
    Sin sklearn — solo numpy (compatible con nuestro stack C++/CUDA).
    """
    # Centrar
    mean  = embeddings.mean(axis=0)
    X     = embeddings - mean

    # Covarianza
    cov   = (X.T @ X) / len(X)

    # SVD (más estable que eig para matrices grandes)
    U, S, Vt = np.linalg.svd(cov, full_matrices=False)

    # Proyectar a 3D con los 3 ejes de mayor varianza
    components = Vt[:3, :]  # [3, D]
    proj_3d    = X @ components.T  # [N, 3]

    # Normalizar a cubo [-5, 5]
    scale  = np.abs(proj_3d).max()
    proj_3d = proj_3d / (scale + 1e-8) * 5.0

    variance_explained = S[:3].sum() / S.sum() * 100
    print(f"  PCA: {embeddings.shape[1]}D → 3D | varianza explicada: {variance_explained:.1f}%")
    return proj_3d.astype(np.float32), components


# ══════════════════════════════════════════════════════════════════════════
# VALIDACIÓN DE CLUSTERS
# ══════════════════════════════════════════════════════════════════════════

def validate_clusters(vocab, embeddings_3d):
    """Verifica que palabras similares queden cerca en 3D."""
    test_groups = [
        ["python", "code", "loop", "function"],
        ["music", "rhythm", "beat", "melody"],
        ["physics", "force", "energy", "orbit"],
    ]

    vocab_set = set(vocab)
    print("  Validación de clusters semánticos en 3D:")

    all_ok = True
    for group in test_groups:
        available = [w for w in group if w in vocab_set]
        if len(available) < 2:
            continue

        indices = [vocab.index(w) for w in available]
        positions = embeddings_3d[indices]

        # Distancia promedio intra-cluster
        dists = []
        for i in range(len(positions)):
            for j in range(i+1, len(positions)):
                d = np.linalg.norm(positions[i] - positions[j])
                dists.append(d)

        mean_dist = np.mean(dists) if dists else 0
        status    = "✅" if mean_dist < 3.0 else "⚠️"
        if mean_dist >= 3.0:
            all_ok = False
        print(f"  {status} {group[0][:10]:<10} grupo — dist_media={mean_dist:.2f}")

    return all_ok


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LiquidBit Embeddings Downloader v2.0")
    parser.add_argument("--source", choices=["gensim", "glove-file", "synthetic"],
                        default="gensim",
                        help="Fuente de embeddings (default: gensim)")
    parser.add_argument("--model", default="glove-wiki-gigaword-50",
                        help="Modelo gensim (solo con --source gensim)")
    parser.add_argument("--path", default="",
                        help="Ruta al archivo GloVe .txt (solo con --source glove-file)")
    parser.add_argument("--max-words", type=int, default=5000,
                        help="Máx. palabras a cargar (default: 5000)")
    parser.add_argument("--output-dir", default=".",
                        help="Directorio de salida")
    args = parser.parse_args()

    print()
    print("╔══════════════════════════════════════════════════════════════════════════╗")
    print("║    LiquidBit Zero-Matrix — Embeddings Downloader v2.0                  ║")
    print(f"║    Fuente: {args.source:<60}║")
    print("╚══════════════════════════════════════════════════════════════════════════╝")
    print()

    vocab, embeddings = None, None

    # ── Seleccionar fuente ─────────────────────────────────────────────────
    if args.source == "gensim":
        model, dim = load_with_gensim(args.model)
        if model is not None:
            vocab, embeddings = gensim_to_arrays(model, max_words=args.max_words)
        else:
            print("  → Fallback a embeddings sintéticos")
            args.source = "synthetic"

    if args.source == "glove-file":
        vocab, embeddings = load_glove_file(args.path, max_words=args.max_words)
        if vocab is None:
            print("  → Fallback a embeddings sintéticos")
            args.source = "synthetic"

    if args.source == "synthetic" or vocab is None:
        vocab, embeddings = generate_synthetic(dim=50, n_words=args.max_words)

    print()
    print(f"  Vocabulario: {len(vocab)} palabras | Dimensión: {embeddings.shape[1]}d")

    # ── PCA D → 3D ────────────────────────────────────────────────────────
    print()
    print("  Proyección PCA D → 3D...")
    embeddings_3d, pca_components = pca_to_3d(embeddings)

    # ── Validación ────────────────────────────────────────────────────────
    print()
    validate_clusters(vocab, embeddings_3d)

    # ── Guardar ───────────────────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    path_3d    = os.path.join(args.output_dir, "embeddings_3d.npy")
    path_full  = os.path.join(args.output_dir, "embeddings_full.npy")
    path_vocab = os.path.join(args.output_dir, "vocab.txt")
    path_pca   = os.path.join(args.output_dir, "pca_components.npy")

    np.save(path_3d,   embeddings_3d)
    np.save(path_full, embeddings)
    np.save(path_pca,  pca_components)

    with open(path_vocab, 'w', encoding='utf-8') as f:
        f.write('\n'.join(vocab))

    print()
    print("  Archivos guardados:")
    print(f"  ✅ {path_3d}     — {embeddings_3d.shape}  FP32 (input para BVH)")
    print(f"  ✅ {path_full}   — {embeddings.shape} FP32 (input para MatMul)")
    print(f"  ✅ {path_vocab}")
    print(f"  ✅ {path_pca}    — {pca_components.shape} (para proyectar nuevos tokens)")

    print(f"""
  INTEGRACIÓN CON LiquidBit:
  ─────────────────────────
  # Cargar en Python:
  import numpy as np
  emb_3d   = np.load("embeddings_3d.npy")   # posiciones para el BVH
  emb_full = np.load("embeddings_full.npy") # embeddings para el MatMul
  vocab    = open("vocab.txt").read().split("\\n")

  # Proyectar nuevo token (en inferencia):
  pca = np.load("pca_components.npy")  # [3, D]
  new_vec_3d = (new_embedding - mean) @ pca.T  # → [3,]

  # Cargar en C++ (embedding_bridge.py → binario):
  # liquidbit::BVHBuilder::loadEmbeddings("embeddings_3d.npy", vocab_size)
    """)

    # ── Instrucciones gensim ───────────────────────────────────────────────
    print("  INSTALAR GENSIM (para embeddings reales):")
    print("  pip install gensim")
    print()
    print("  Modelos recomendados:")
    print("  • glove-wiki-gigaword-50   →  50d, rápido, ideal para BVH")
    print("  • glove-wiki-gigaword-300  → 300d, mejor calidad semántica")
    print("  • word2vec-google-news-300 → 300d, mejor para código/técnico")
    print()
    print("  Ejecutar:")
    print("  python3 download_embeddings_v2.py --source gensim --model glove-wiki-gigaword-50")
    print()


if __name__ == "__main__":
    main()
