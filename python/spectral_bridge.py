#!/usr/bin/env python3
"""
spectral_bridge.py — Python ↔ C++ bridge para SpectralAI Zero-Matrix
======================================================================

Conecta los embeddings pre-entrenados (GloVe-50d, ya descargados) con el
pipeline C++/CUDA. Serializa secuencias de tokens al formato binario exacto
que espera el código C++ (struct TokenNode en token_geometry.h).

Funcionalidades:
  1. Cargar embeddings_full.npy + pca_components.npy + vocab.txt
  2. Tokenizar una frase → lista de token IDs
  3. Proyectar embeddings a posiciones 3D (PCA ya entrenada)
  4. Serializar TokenNodes en binario FP16-compatible que lee C++
  5. Validar la topología: verificar que tokens semánticamente cercanos
     tienen posiciones 3D cercanas (sanity check)
  6. CLI para generar archivos .bin listos para el benchmark C++

Formato binario (compatible con token_geometry.h — struct TokenNode):
  Header:  uint32 num_tokens  (4 bytes)
  Tokens:  por cada token —
    uint32  token_id          (4)
    uint32  position_in_seq   (4)
    float3  centroid          (12)
    float3  aabb_min          (12)
    float3  aabb_max          (12)
    float   semantic_radius   (4)
    half[256] embedding       (512)   ← FP16 real, no float32
    float   attention_weight  (4)
    float   energy_remaining  (4)
    Total por token: 568 bytes

Uso:
    python spectral_bridge.py "the algorithm loops over the array"
    python spectral_bridge.py --sentence "..." --output seq.bin --validate
    python spectral_bridge.py --benchmark-seqs --sizes 100 1000 10000
"""

import struct
import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Rutas por defecto (relativas al directorio del script)
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR    = Path(__file__).parent
VOCAB_PATH    = SCRIPT_DIR / "vocab.txt"
EMB_FULL_PATH = SCRIPT_DIR / "embeddings_full.npy"
EMB_3D_PATH   = SCRIPT_DIR / "embeddings_3d.npy"
PCA_PATH      = SCRIPT_DIR / "pca_components.npy"

# Alineación del struct C++ (múltiplo de 4 bytes garantizado)
TOKEN_NODE_BINARY_SIZE = 568  # bytes por TokenNode

# ═══════════════════════════════════════════════════════════════════════════
# Carga de recursos
# ═══════════════════════════════════════════════════════════════════════════

class EmbeddingDB:
    """
    Carga y sirve embeddings GloVe-50d pre-entrenados.

    Atributos:
        vocab:      List[str]              — palabras en orden (índice = token_id)
        word2id:    Dict[str, int]         — lookup inverso
        emb_full:   np.ndarray [N, D]      — embeddings originales float32
        emb_3d:     np.ndarray [N, 3]      — proyecciones PCA 3D float32
        pca:        np.ndarray [3, D]      — componentes PCA para proyectar nuevas palabras
        embed_dim:  int                    — dimensión original (50)
    """

    def __init__(
        self,
        vocab_path: Path = VOCAB_PATH,
        emb_full_path: Path = EMB_FULL_PATH,
        emb_3d_path: Path = EMB_3D_PATH,
        pca_path: Path = PCA_PATH,
    ):
        if not vocab_path.exists():
            raise FileNotFoundError(f"vocab.txt no encontrado: {vocab_path}")
        if not emb_full_path.exists():
            raise FileNotFoundError(f"embeddings_full.npy no encontrado: {emb_full_path}")

        self.vocab: List[str] = vocab_path.read_text(encoding="utf-8").splitlines()
        self.word2id: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}

        self.emb_full: np.ndarray = np.load(str(emb_full_path))   # [N, D]
        self.embed_dim: int = self.emb_full.shape[1]

        # Proyecciones 3D — cargadas o calculadas al vuelo
        if emb_3d_path.exists():
            self.emb_3d: np.ndarray = np.load(str(emb_3d_path))  # [N, 3]
        else:
            self.emb_3d = None

        # Componentes PCA para proyectar tokens fuera del vocabulario
        if pca_path.exists():
            self.pca: np.ndarray = np.load(str(pca_path))  # [3, D]
        else:
            self.pca = None

        print(f"[bridge] Vocabulario: {len(self.vocab)} tokens, "
              f"dim={self.embed_dim}, 3D={'listo' if self.emb_3d is not None else 'no'}")

    def get_embedding(self, token_id: int) -> np.ndarray:
        """Devuelve embedding float32 [D] para el token_id."""
        return self.emb_full[token_id]

    def get_position_3d(self, token_id: int) -> np.ndarray:
        """Devuelve posición 3D float32 [3] para el token_id."""
        if self.emb_3d is not None:
            return self.emb_3d[token_id]
        # Fallback: proyectar con PCA
        if self.pca is not None:
            mean = self.emb_full.mean(axis=0)
            return (self.emb_full[token_id] - mean) @ self.pca.T
        # Último fallback: primeras 3 dimensiones normalizadas
        v = self.emb_full[token_id, :3].copy()
        norm = np.linalg.norm(v) + 1e-6
        return v / norm

    def tokenize(self, sentence: str) -> List[Tuple[int, str]]:
        """
        Tokeniza una frase en (token_id, palabra).

        Palabras fuera del vocabulario → token_id del token '<UNK>' si existe,
        o se omiten con un aviso.

        Returns:
            List[(token_id, word)]
        """
        unk_id = self.word2id.get("<UNK>", self.word2id.get("unk", None))
        result = []
        for raw_word in sentence.lower().split():
            word = raw_word.strip(".,!?;:\"'()[]{}")
            if not word:
                continue
            if word in self.word2id:
                result.append((self.word2id[word], word))
            elif unk_id is not None:
                result.append((unk_id, word))
            else:
                print(f"[bridge] OOV ignorado: '{word}'", file=sys.stderr)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Serialización binaria
# ═══════════════════════════════════════════════════════════════════════════

def _embedding_to_fp16_bytes(embedding: np.ndarray, target_dim: int = 256) -> bytes:
    """
    Comprime un embedding float32 [D] a half-float [256].

    - Si D < 256: pad con ceros
    - Si D > 256: truncar (los primeros 256 componentes son los más importantes)
    - Convierte a numpy float16 → 2 bytes por valor, little-endian
    """
    buf = np.zeros(target_dim, dtype=np.float32)
    copy_len = min(len(embedding), target_dim)
    buf[:copy_len] = embedding[:copy_len]

    # Normalizar para maximizar precisión en FP16 (evitar overflow/underflow)
    max_val = np.abs(buf).max()
    if max_val > 0:
        buf = buf / max_val  # Normalizar a [-1, 1]

    return buf.astype(np.float16).tobytes()


def _compute_semantic_radius(embedding: np.ndarray) -> float:
    """
    Radio semántico = f(varianza local del embedding).

    Tokens polisémicos (ej. 'bank') → varianza alta → radio mayor.
    Rango: [0.02, 0.25]
    """
    norm = float(np.linalg.norm(embedding))
    std  = float(np.std(embedding))
    raw  = 0.02 + 0.23 * np.tanh(std * 3.0)
    return float(np.clip(raw, 0.02, 0.25))


def token_node_to_bytes(
    token_id: int,
    position_in_seq: int,
    embedding: np.ndarray,
    position_3d: np.ndarray,
) -> bytes:
    """
    Serializa un TokenNode al formato binario exacto de token_geometry.h.

    Formato (568 bytes):
        uint32  token_id          4
        uint32  position_in_seq   4
        float3  centroid          12
        float3  aabb_min          12
        float3  aabb_max          12
        float   semantic_radius   4
        half[256] embedding       512
        float   attention_weight  4
        float   energy_remaining  4
    """
    r = _compute_semantic_radius(embedding)
    cx, cy, cz = float(position_3d[0]), float(position_3d[1]), float(position_3d[2])

    buf = struct.pack(
        "<II",             # token_id, position_in_seq
        token_id, position_in_seq
    )
    buf += struct.pack("<fff", cx, cy, cz)               # centroid
    buf += struct.pack("<fff", cx - r, cy - r, cz - r)  # aabb_min
    buf += struct.pack("<fff", cx + r, cy + r, cz + r)  # aabb_max
    buf += struct.pack("<f", r)                          # semantic_radius
    buf += _embedding_to_fp16_bytes(embedding, 256)      # half[256] = 512 bytes
    buf += struct.pack("<ff", 0.0, 1.0)                  # attention_weight, energy_remaining

    assert len(buf) == TOKEN_NODE_BINARY_SIZE, f"Size mismatch: {len(buf)} vs {TOKEN_NODE_BINARY_SIZE}"
    return buf


def serialize_sequence(
    db: EmbeddingDB,
    tokens: List[Tuple[int, str]],
    output_path: Path,
) -> int:
    """
    Serializa una secuencia de tokens al archivo binario.

    Formato:
        uint32  num_tokens         4 bytes
        TokenNode[num_tokens]      568 bytes cada uno

    Returns:
        Número de tokens escritos
    """
    with open(str(output_path), "wb") as f:
        f.write(struct.pack("<I", len(tokens)))
        for pos, (token_id, word) in enumerate(tokens):
            embedding   = db.get_embedding(token_id)
            position_3d = db.get_position_3d(token_id)
            node_bytes  = token_node_to_bytes(token_id, pos, embedding, position_3d)
            f.write(node_bytes)

    size_kb = output_path.stat().st_size / 1024
    print(f"[bridge] Escrito: {output_path.name} — {len(tokens)} tokens, {size_kb:.1f} KB")
    return len(tokens)


# ═══════════════════════════════════════════════════════════════════════════
# Validación semántica (sanity check)
# ═══════════════════════════════════════════════════════════════════════════

def validate_topology(db: EmbeddingDB, n_pairs: int = 20) -> bool:
    """
    Verifica que la topología 3D preserva la similitud coseno original.

    Para N pares aleatorios de tokens:
      1. Calcula similitud coseno en espacio original (D-dim)
      2. Calcula distancia euclídea en espacio 3D
    Debería haber correlación negativa: similitud alta ↔ distancia pequeña.

    Returns:
        True si la correlación de Pearson es > 0.5 (buena preservación)
    """
    N = len(db.vocab)
    np.random.seed(42)
    idx_a = np.random.randint(0, N, n_pairs)
    idx_b = np.random.randint(0, N, n_pairs)

    cosine_sims = []
    distances_3d = []

    for a, b in zip(idx_a, idx_b):
        ea = db.emb_full[a]
        eb = db.emb_full[b]

        # Similitud coseno
        na, nb = np.linalg.norm(ea), np.linalg.norm(eb)
        cos_sim = float(np.dot(ea, eb) / (na * nb + 1e-8))

        # Distancia 3D
        pa, pb = db.get_position_3d(a), db.get_position_3d(b)
        dist3d = float(np.linalg.norm(pa - pb))

        cosine_sims.append(cos_sim)
        distances_3d.append(dist3d)

    cosine_sims = np.array(cosine_sims)
    distances_3d = np.array(distances_3d)

    # Correlación de Pearson entre similitud coseno y -distancia 3D
    correlation = float(np.corrcoef(cosine_sims, -distances_3d)[0, 1])

    print(f"[bridge] Validacion topologia: correlacion coseno<->(-dist3D) = {correlation:.3f}")
    if correlation > 0.5:
        print("[bridge] ✓ Topología preservada correctamente (>0.5)")
        return True
    else:
        print("[bridge] ⚠ Correlación baja — revisar PCA o escala de embeddings", file=sys.stderr)
        return False


def print_nearest_neighbors(db: EmbeddingDB, query_word: str, k: int = 5) -> None:
    """
    Imprime los k vecinos más cercanos en el espacio 3D para una palabra query.
    Útil para verificar que la proyección tiene sentido semántico.
    """
    if query_word not in db.word2id:
        print(f"[bridge] '{query_word}' no está en el vocabulario", file=sys.stderr)
        return

    qid = db.word2id[query_word]
    q3d = db.get_position_3d(qid)

    # Distancias euclídeas a todos los tokens
    dists = np.linalg.norm(db.emb_3d - q3d, axis=1)
    nearest = np.argsort(dists)[:k + 1]  # +1 porque el primero es el mismo

    print(f"\n[bridge] Vecinos 3D de '{query_word}':")
    for rank, nid in enumerate(nearest):
        if nid == qid:
            continue
        word = db.vocab[nid]
        dist = dists[nid]
        # Calcular similitud coseno en espacio original
        eq = db.emb_full[qid]
        en = db.emb_full[nid]
        cos = float(np.dot(eq, en) / (np.linalg.norm(eq) * np.linalg.norm(en) + 1e-8))
        print(f"  {rank}. '{word:15s}' dist3D={dist:.4f}  cosine={cos:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Generador de secuencias para benchmarks
# ═══════════════════════════════════════════════════════════════════════════

def generate_benchmark_sequence(db: EmbeddingDB, n_tokens: int, output_path: Path) -> float:
    """
    Genera una secuencia sintética de N tokens cíclicos (de los 500 disponibles).

    El ciclo asegura variedad semántica real (no tokens repetidos en exceso).
    Útil para medir rendimiento del BVH C++ con distintos tamaños de secuencia.

    Returns:
        Tiempo de serialización en segundos
    """
    N_vocab = len(db.vocab)
    token_ids = [(i % N_vocab, db.vocab[i % N_vocab]) for i in range(n_tokens)]

    t0 = time.perf_counter()
    written = serialize_sequence(db, token_ids, output_path)
    elapsed = time.perf_counter() - t0

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[bridge] Benchmark seq N={n_tokens}: {size_mb:.2f} MB, serializado en {elapsed*1000:.1f} ms")
    return elapsed


# ═══════════════════════════════════════════════════════════════════════════
# Exportar métricas de la topología (para LEARNINGS.md)
# ═══════════════════════════════════════════════════════════════════════════

def export_topology_stats(db: EmbeddingDB, output_path: Path) -> dict:
    """
    Calcula y guarda estadísticas de la proyección 3D.

    Guarda JSON con:
      - correlación coseno↔dist3D (métrica de preservación)
      - distribución de radios semánticos
      - bounds del espacio 3D
      - ejemplo de vecinos para palabras clave
    """
    N = len(db.vocab)

    # Calcular radios semánticos para todos los tokens
    radii = np.array([_compute_semantic_radius(db.emb_full[i]) for i in range(N)])

    # Bounds del espacio 3D
    pos3d = db.emb_3d
    bounds = {
        "x": [float(pos3d[:, 0].min()), float(pos3d[:, 0].max())],
        "y": [float(pos3d[:, 1].min()), float(pos3d[:, 1].max())],
        "z": [float(pos3d[:, 2].min()), float(pos3d[:, 2].max())],
    }

    # Correlación coseno↔dist3D (muestra de 200 pares)
    np.random.seed(0)
    idx_a = np.random.randint(0, N, 200)
    idx_b = np.random.randint(0, N, 200)
    cos_sims, dists = [], []
    for a, b in zip(idx_a, idx_b):
        ea, eb = db.emb_full[a], db.emb_full[b]
        cos_sims.append(float(np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-8)))
        dists.append(float(np.linalg.norm(pos3d[a] - pos3d[b])))
    correlation = float(np.corrcoef(cos_sims, [-d for d in dists])[0, 1])

    stats = {
        "vocab_size":   N,
        "embed_dim":    int(db.embed_dim),
        "correlation_cosine_vs_neg_dist3d": correlation,
        "radius_min":   float(radii.min()),
        "radius_max":   float(radii.max()),
        "radius_mean":  float(radii.mean()),
        "space_bounds": bounds,
        "token_node_bytes": TOKEN_NODE_BINARY_SIZE,
        "memory_1k_tokens_kb":  round(1000 * TOKEN_NODE_BINARY_SIZE / 1024, 1),
        "memory_100k_tokens_mb": round(100000 * TOKEN_NODE_BINARY_SIZE / (1024**2), 1),
    }

    with open(str(output_path), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"[bridge] Estadísticas exportadas: {output_path}")
    return stats


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SpectralAI Bridge: Python → C++ token serializer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Serializar una frase
  python spectral_bridge.py "the algorithm loops over the array"

  # Frase custom a archivo específico con validación
  python spectral_bridge.py --sentence "for loop array function" --output seq.bin --validate

  # Generar secuencias de benchmark para C++
  python spectral_bridge.py --benchmark-seqs --sizes 100 1000 10000 100000

  # Ver vecinos semánticos de una palabra
  python spectral_bridge.py --neighbors algorithm --k 8

  # Exportar estadísticas de topología
  python spectral_bridge.py --stats
        """
    )

    parser.add_argument(
        "sentence", nargs="?",
        help="Frase a tokenizar y serializar (argumento posicional)"
    )
    parser.add_argument("--sentence", dest="sentence_flag", type=str,
                        help="Frase a tokenizar (alternativa explícita)")
    parser.add_argument("--output", "-o", type=str, default="sequence.bin",
                        help="Archivo de salida (default: sequence.bin)")
    parser.add_argument("--validate", action="store_true",
                        help="Validar que la topología 3D preserva similitud coseno")
    parser.add_argument("--neighbors", type=str, metavar="WORD",
                        help="Mostrar k vecinos 3D de WORD")
    parser.add_argument("--k", type=int, default=5,
                        help="Número de vecinos a mostrar (default: 5)")
    parser.add_argument("--benchmark-seqs", action="store_true",
                        help="Generar secuencias de benchmark de varios tamaños")
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[100, 1000, 10000],
                        help="Tamaños de secuencia para --benchmark-seqs")
    parser.add_argument("--stats", action="store_true",
                        help="Exportar estadísticas de topología a topology_stats.json")
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Directorio de datos (default: directorio del script)"
    )

    args = parser.parse_args()

    # Resolver directorio de datos
    data_dir = Path(args.data_dir) if args.data_dir else SCRIPT_DIR

    # Cargar DB
    try:
        db = EmbeddingDB(
            vocab_path    = data_dir / "vocab.txt",
            emb_full_path = data_dir / "embeddings_full.npy",
            emb_3d_path   = data_dir / "embeddings_3d.npy",
            pca_path      = data_dir / "pca_components.npy",
        )
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        print("[ERROR] Ejecuta download_embeddings_v2.py primero.", file=sys.stderr)
        sys.exit(1)

    # ── Validación de topología ──────────────────────────────────────────
    if args.validate:
        validate_topology(db)

    # ── Vecinos ─────────────────────────────────────────────────────────
    if args.neighbors:
        print_nearest_neighbors(db, args.neighbors, k=args.k)

    # ── Estadísticas ────────────────────────────────────────────────────
    if args.stats:
        stats = export_topology_stats(db, SCRIPT_DIR / "topology_stats.json")
        print(f"\n[bridge] Resumen topología:")
        print(f"  Correlación coseno↔(-dist3D) : {stats['correlation_cosine_vs_neg_dist3d']:.3f}")
        print(f"  Radio semántico [min,max]     : [{stats['radius_min']:.3f}, {stats['radius_max']:.3f}]")
        print(f"  Bounds 3D X                   : {stats['space_bounds']['x']}")
        print(f"  Bounds 3D Y                   : {stats['space_bounds']['y']}")
        print(f"  Memoria por token             : {stats['token_node_bytes']} bytes")
        print(f"  Memoria 100K tokens           : {stats['memory_100k_tokens_mb']} MB")

    # ── Benchmark sequences ─────────────────────────────────────────────
    if args.benchmark_seqs:
        out_dir = SCRIPT_DIR / "benchmark_data"
        out_dir.mkdir(exist_ok=True)
        print(f"\n[bridge] Generando secuencias de benchmark en {out_dir}/...")
        for n in args.sizes:
            out_path = out_dir / f"seq_{n:07d}.bin"
            generate_benchmark_sequence(db, n, out_path)
        print("[bridge] ¡Secuencias listas para el benchmark C++!")
        return

    # ── Serializar frase ────────────────────────────────────────────────
    sentence = args.sentence or args.sentence_flag
    if sentence:
        tokens = db.tokenize(sentence)
        if not tokens:
            print("[ERROR] Ningún token del vocabulario encontrado en la frase.", file=sys.stderr)
            sys.exit(1)

        print(f"\n[bridge] Frase: \"{sentence}\"")
        print(f"[bridge] Tokens ({len(tokens)}): {[(w, tid) for tid, w in tokens]}")

        output_path = SCRIPT_DIR / args.output
        serialize_sequence(db, tokens, output_path)

        # Mostrar vecinos de la primera palabra como sanity check
        if tokens:
            print_nearest_neighbors(db, tokens[0][1], k=3)
    elif not (args.validate or args.neighbors or args.stats or args.benchmark_seqs):
        parser.print_help()


if __name__ == "__main__":
    main()
