#!/usr/bin/env python3
"""
inference.py — Pipeline de inferencia SpectralAI String-Inception con embeddings reales
========================================================================================

Conecta los embeddings GloVe-50d (ya descargados) con el pipeline OptiX real.

FLUJO:
  frase → tokenizar → posición 3D (PCA) → SemanticSphere + SemanticString
       → scene.bin → inception_runner.exe → results.bin → pesos de atención

USO:
    python inference.py "the algorithm loops over the array"
    python inference.py "jazz music improvisation" --num-rays 32
    python inference.py --demo  # corre 3 frases de ejemplo y compara

SALIDA:
    Tabla con los tokens, su posición 3D, y el attentionWeight calculado por los RT Cores.

REQUISITOS:
    pip install numpy scikit-learn
    spectral_kernels.ptx  (compilado por CMake con SPECTRAL_BUILD_INCEPTION=ON)
    inception_runner.exe   (compilado con CMake target inception_runner)
"""

import sys
import os
import struct
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Windows: forzar UTF-8 para stdout/stderr (evita UnicodeEncodeError en cp1252)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ─────────────────────────────────────────────────────────────────
# Rutas
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR   = Path(__file__).parent
PROJECT_DIR  = SCRIPT_DIR.parent
BUILD_DIR    = PROJECT_DIR / "build" / "Release"
BUILD_ROOT   = PROJECT_DIR / "build"
PTX_PATH     = BUILD_ROOT / "spectral_kernels.ptx"   # generado por custom_command (no va a Release/)
RUNNER_EXE   = BUILD_DIR / "inception_runner.exe"
SCENE_FILE   = SCRIPT_DIR / "scene.bin"
RESULTS_FILE = SCRIPT_DIR / "results.bin"

# ─────────────────────────────────────────────────────────────────
# Structs C++ (binary layout EXACTO verificado con print_struct_sizes.exe)
# ─────────────────────────────────────────────────────────────────
# SemanticSphere (alignas(16), sizeof=32)
#   center[0..11]  radius[12..15]
#   instanceId[16..19]  childIAS[20..23]  depth[24..27]  frequencyBias[28..31]
SPHERE_FMT  = "<3f f I I I f"   # 32 bytes
SPHERE_SIZE = struct.calcsize(SPHERE_FMT)
assert SPHERE_SIZE == 32, f"SemanticSphere size mismatch: {SPHERE_SIZE}"

# ResonanceParams (alignas(32), sizeof=96)
#   a[0..31]  b[32..63]  numModes[64..67]  outputScale[68..71]
#   semanticTag[72..75]  _pad[76..79]  + 16 bytes C++ tail padding → 96 total
RESONANCE_FMT     = "<8f 8f I f I I"  # 80 bytes de datos
RESONANCE_PAD     = 16               # 16 bytes de padding a múltiplo de 32
RESONANCE_SIZE    = struct.calcsize(RESONANCE_FMT) + RESONANCE_PAD  # = 96
assert RESONANCE_SIZE == 96, f"ResonanceParams size mismatch: {RESONANCE_SIZE}"

# SemanticString (alignas(32), sizeof=128)
#   resonance[0..95]  position[96..107]  stringId[108..111]
#   + 16 bytes C++ tail padding → 128 total
STRING_BODY_FMT  = "<3f I"    # position (12) + stringId (4) = 16 bytes
STRING_BODY_SIZE = struct.calcsize(STRING_BODY_FMT)
STRING_SIZE      = RESONANCE_SIZE + STRING_BODY_SIZE + 16  # 96 + 16 + 16 = 128
assert STRING_SIZE == 128, f"SemanticString size mismatch: {STRING_SIZE}"

# AffinePortal (alignas(64), sizeof=64)
#   float4 rows[4] = 16 floats
PORTAL_FMT  = "<16f"
PORTAL_SIZE = struct.calcsize(PORTAL_FMT)
assert PORTAL_SIZE == 64, f"AffinePortal size mismatch: {PORTAL_SIZE}"

# SpectralAttentionResult (alignas(4), sizeof=32)
#   attentionWeight[0]  finalOmega[4]
#   dominantStringId[8]  traversalDepth[12]
#   exitDirection[16..27]  energyRemaining[28..31]
RESULT_FMT  = "<f f I I 3f f"  # 32 bytes
RESULT_SIZE = struct.calcsize(RESULT_FMT)
assert RESULT_SIZE == 32, f"SpectralAttentionResult size mismatch: {RESULT_SIZE}"

# Scene file magic header
SCENE_MAGIC   = 0x4C425354   # 'LBST'
SCENE_VERSION = 1

# ─────────────────────────────────────────────────────────────────
# Carga de embeddings
# ─────────────────────────────────────────────────────────────────

class EmbeddingDB:
    """Carga vocab + embeddings GloVe + proyecciones PCA 3D."""

    def __init__(self):
        vocab_path  = SCRIPT_DIR / "vocab.txt"
        emb_path    = SCRIPT_DIR / "embeddings_full.npy"
        emb3d_path  = SCRIPT_DIR / "embeddings_3d.npy"
        pca_path    = SCRIPT_DIR / "pca_components.npy"

        if not vocab_path.exists() or not emb_path.exists():
            raise FileNotFoundError(
                "Faltan vocab.txt o embeddings_full.npy. "
                "Ejecuta: python download_embeddings.py"
            )

        self.vocab   = vocab_path.read_text(encoding="utf-8").splitlines()
        self.word2id = {w: i for i, w in enumerate(self.vocab)}
        self.emb     = np.load(str(emb_path))        # [N, D]
        self.dim     = self.emb.shape[1]

        # Proyecciones 3D ya calculadas
        self.emb3d  = np.load(str(emb3d_path)) if emb3d_path.exists() else None
        self.pca    = np.load(str(pca_path))   if pca_path.exists()   else None

        print(f"[embed] {len(self.vocab)} tokens, dim={self.dim}, "
              f"3D={'ok' if self.emb3d is not None else 'calculando'}")

    def get_3d(self, token_id: int) -> np.ndarray:
        """Posición 3D float32 [3] para el token."""
        if self.emb3d is not None:
            return self.emb3d[token_id].astype(np.float32)
        # Fallback: PCA on-the-fly
        if self.pca is not None:
            mean = self.emb.mean(axis=0)
            return ((self.emb[token_id] - mean) @ self.pca.T).astype(np.float32)
        v = self.emb[token_id, :3].astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-6)

    def tokenize(self, sentence: str) -> List[Tuple[int, str]]:
        """Frase → [(token_id, word), ...]"""
        unk = self.word2id.get("<UNK>", self.word2id.get("unk"))
        result = []
        for raw in sentence.lower().split():
            word = raw.strip(".,!?;:\"'()[]{}")
            if not word:
                continue
            tid = self.word2id.get(word, unk)
            if tid is not None:
                result.append((tid, word))
            else:
                print(f"[embed] OOV ignorado: '{word}'", file=sys.stderr)
        return result

    def cosine_sim(self, id_a: int, id_b: int) -> float:
        a, b = self.emb[id_a], self.emb[id_b]
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ─────────────────────────────────────────────────────────────────
# Fourier coefficients desde embeddings
# ─────────────────────────────────────────────────────────────────

def embedding_to_fourier(emb_vec: np.ndarray, num_modes: int = 8) -> Tuple[List[float], List[float]]:
    """
    Convierte un embedding D-dimensional a coeficientes Fourier (a[], b[]).

    Estrategia: tomar las primeras 2·M dimensiones del embedding normalizado
    y asignarlas alternadamente a a[k] y b[k].

    Esto no es entrenado — son coeficientes inicializados "semánticamente".
    El training futuro los refinará. Por ahora sirven para validar el pipeline.
    """
    v = emb_vec / (np.linalg.norm(emb_vec) + 1e-8)
    # Tomar 2*M valores del embedding (las dims más informativas por PCA)
    needed = 2 * num_modes
    if len(v) >= needed:
        vals = v[:needed].tolist()
    else:
        vals = (v.tolist() + [0.0] * needed)[:needed]

    a = [vals[2*k]     for k in range(num_modes)]
    b = [vals[2*k + 1] for k in range(num_modes)]
    return a, b


# ─────────────────────────────────────────────────────────────────
# Serialización de escena
# ─────────────────────────────────────────────────────────────────

def pack_sphere(cx: float, cy: float, cz: float,
                radius: float, instance_id: int,
                depth: int, freq_bias: float) -> bytes:
    return struct.pack(SPHERE_FMT,
        cx, cy, cz, radius,
        instance_id,
        0,          # childIAS = 0 (hojas en este prototipo)
        depth,
        freq_bias)

def pack_resonance(a: List[float], b: List[float],
                   num_modes: int, scale: float, tag: int) -> bytes:
    # 80 bytes de datos + 16 bytes de tail padding = 96 bytes (alignas(32))
    data = struct.pack(RESONANCE_FMT, *a, *b, num_modes, scale, tag, 0)
    return data + bytes(RESONANCE_PAD)

def pack_string(a: List[float], b: List[float], num_modes: int,
                scale: float, tag: int,
                px: float, py: float, pz: float,
                string_id: int) -> bytes:
    res  = pack_resonance(a, b, num_modes, scale, tag)          # 96 bytes
    body = struct.pack(STRING_BODY_FMT, px, py, pz, string_id)  # 16 bytes
    # 96 + 16 = 112 bytes + 16 bytes tail padding = 128 bytes (alignas(32))
    return res + body + bytes(16)

def pack_portal_identity() -> bytes:
    """Portal identidad 4×4 (no transforma ω)."""
    return struct.pack(PORTAL_FMT,
        1,0,0,0,  0,1,0,0,  0,0,1,0,  0,0,0,1)

def write_scene(path: Path, spheres_data: List[bytes],
                strings_data: List[bytes],
                portals_data: List[bytes],
                base_omega: float, num_rays: int):
    with open(path, "wb") as f:
        # Header
        f.write(struct.pack("<III", SCENE_MAGIC, SCENE_VERSION,
                            len(spheres_data)))
        f.write(struct.pack("<III", len(strings_data),
                            len(portals_data), num_rays))
        f.write(struct.pack("<f", base_omega))
        # Data
        for s in spheres_data:
            f.write(s)
        for s in strings_data:
            f.write(s)
        for p in portals_data:
            f.write(p)

RESULTS_MAGIC = 0x4C425253  # 'LBRS'

def read_results(path: Path, num_rays: int) -> List[dict]:
    results = []
    with open(path, "rb") as f:
        # Header: magic(4) + numResults(4) = 8 bytes
        hdr = f.read(8)
        if len(hdr) < 8:
            print(f"[results] Archivo de resultados truncado o vacío: {path}", file=sys.stderr)
            return []
        magic, count = struct.unpack("<II", hdr)
        if magic != RESULTS_MAGIC:
            print(f"[results] Magic incorrecto: 0x{magic:08X} (esperado 0x{RESULTS_MAGIC:08X})",
                  file=sys.stderr)
            return []
        num_to_read = min(count, num_rays)
        for i in range(num_to_read):
            data = f.read(RESULT_SIZE)
            if len(data) < RESULT_SIZE:
                break
            aw, fw, dom, depth, ex, ey, ez, energy = struct.unpack(RESULT_FMT, data)
            results.append({
                "ray": i,
                "attentionWeight": aw,
                "finalOmega": fw,
                "dominantStringId": dom,
                "traversalDepth": depth,
                "exitDir": (ex, ey, ez),
                "energyRemaining": energy,
            })
    return results


# ─────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────

def run_inference(sentence: str,
                  num_rays: int = 16,
                  base_omega: float = None,
                  db: EmbeddingDB = None) -> Optional[List[dict]]:
    """
    Ejecuta el pipeline completo para una frase.

    1. Tokeniza la frase
    2. Proyecta tokens a posiciones 3D
    3. Crea SemanticSpheres + SemanticStrings
    4. Escribe scene.bin
    5. Llama inception_runner.exe
    6. Lee y devuelve results.bin
    """
    if db is None:
        db = EmbeddingDB()

    tokens = db.tokenize(sentence)
    if not tokens:
        print(f"[infer] No se encontraron tokens para: '{sentence}'")
        return None

    print(f"\n[infer] Frase: '{sentence}'")
    print(f"[infer] Tokens ({len(tokens)}): {[w for _, w in tokens]}")

    # Usar omega basada en la dimensión semántica media de la frase
    if base_omega is None:
        emb_mean = np.mean([db.emb[tid] for tid, _ in tokens], axis=0)
        base_omega = float(np.pi / 4 + 0.1 * np.linalg.norm(emb_mean) / db.dim)

    # Construir esferas y strings
    spheres_data, strings_data = [], []
    token_3d_positions = []

    # Calcular posiciones 3D crudas
    raw_positions = [db.get_3d(tid) for tid, _ in tokens]

    # Normalizar la escena: escalar a distancia media = 2.5 desde el origen
    # Esto garantiza que los rayos desde (0,0,0) puedan impactar las esferas
    # con un radio razonable de 1.2 unidades.
    dists = [np.linalg.norm(p) + 1e-6 for p in raw_positions]
    mean_dist = np.mean(dists)
    scale = 2.5 / max(mean_dist, 0.1)

    for i, (tid, word) in enumerate(tokens):
        pos3d = raw_positions[i] * scale
        token_3d_positions.append((pos3d, word, tid))

        # Radio grande para garantizar cobertura con 16 rayos en esfera fibonacci
        # r=1.2 a distancia=2.5 → ángulo sólido ~0.72 sr → ~16 * 0.72/4π ≈ 0.9 hits/esfera
        radius = 1.2
        freq_bias = float(i * 0.15 % (2 * np.pi))

        spheres_data.append(pack_sphere(
            float(pos3d[0]), float(pos3d[1]), float(pos3d[2]),
            radius, i, 3, freq_bias   # depth=3 = nodo hoja
        ))

        # Coeficientes Fourier desde el embedding del token
        a, b = embedding_to_fourier(db.emb[tid])
        strings_data.append(pack_string(
            a, b, 8, 1.0, tid,
            float(pos3d[0]), float(pos3d[1]), float(pos3d[2]),
            i
        ))
        print(f"  [{i}] {word:12s}: pos=({pos3d[0]:.2f},{pos3d[1]:.2f},{pos3d[2]:.2f}) r=1.2")

    # 4 portales identidad
    portals_data = [pack_portal_identity() for _ in range(4)]

    # Escribir escena
    write_scene(SCENE_FILE, spheres_data, strings_data, portals_data,
                base_omega, num_rays)
    print(f"[infer] Escena escrita: {len(tokens)} esferas, omega={base_omega:.3f}")

    # Verificar ejecutable
    if not RUNNER_EXE.exists():
        print(f"[infer] inception_runner.exe no encontrado en {RUNNER_EXE}")
        print(f"        Compile con: cmake --build build --target inception_runner")
        return None
    if not PTX_PATH.exists():
        print(f"[infer] spectral_kernels.ptx no encontrado en {PTX_PATH}")
        return None

    # Llamar C++
    t0 = time.perf_counter()
    result = subprocess.run(
        [str(RUNNER_EXE), str(PTX_PATH), str(SCENE_FILE), str(RESULTS_FILE)],
        capture_output=True, text=True
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000

    if result.returncode != 0:
        print(f"[infer] ERROR en inception_runner: {result.stderr}")
        return None

    # Leer resultados
    results = read_results(RESULTS_FILE, num_rays)

    # Mostrar tabla
    print(f"\n  {'Ray':>3} | {'Token':>12} | {'Attention':>10} | {'Omega':>7} | {'Depth':>5}")
    print(f"  {'---':>3}-|-{'---':>12}-|-{'---':>10}-|-{'---':>7}-|-{'---':>5}")

    # Asignar tokens a rayos (los primeros num_tokens rayos apuntan a tokens)
    hits = [r for r in results if r["traversalDepth"] > 0]
    for r in results:
        tok_label = tokens[min(r["ray"], len(tokens)-1)][1] if r["ray"] < len(tokens) else "-"
        hit_str = "HIT" if r["traversalDepth"] > 0 else "miss"
        print(f"  {r['ray']:>3} | {tok_label:>12} | {r['attentionWeight']:>10.4f} | "
              f"{r['finalOmega']:>7.4f} | {hit_str:>5}")

    print(f"\n  Hits: {len(hits)}/{num_rays}  |  Tiempo C++: {elapsed_ms:.1f} ms")
    return results


def run_demo(db: EmbeddingDB):
    """Corre 3 frases de ejemplo para comparar los pesos de atención."""
    sentences = [
        "the algorithm loops over the array",
        "jazz music improvisation harmony",
        "quantum gravity space time curvature",
    ]
    print("\n" + "="*60)
    print("DEMO: 3 frases con contextos semánticos distintos")
    print("="*60)

    all_results = {}
    for s in sentences:
        res = run_inference(s, num_rays=8, db=db)
        all_results[s] = res

    # Comparar pesos medios de atención entre frases
    print("\n" + "="*60)
    print("RESUMEN — Attention weight medio por frase:")
    for s, res in all_results.items():
        if res:
            hits = [r["attentionWeight"] for r in res if r["traversalDepth"] > 0]
            mean_w = np.mean(hits) if hits else 0.0
            print(f"  {mean_w:.4f}  ←  '{s}'")


# ─────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="SpectralAI String-Inception — inferencia con embeddings reales")
    parser.add_argument("sentence", nargs="?", default=None,
                        help="Frase a procesar")
    parser.add_argument("--num-rays", type=int, default=16,
                        help="Número de rayos OptiX (default: 16)")
    parser.add_argument("--omega", type=float, default=None,
                        help="Frecuencia de contexto base ω₀ (default: auto)")
    parser.add_argument("--demo", action="store_true",
                        help="Corre 3 frases de ejemplo")
    args = parser.parse_args()

    db = EmbeddingDB()

    if args.demo:
        run_demo(db)
    elif args.sentence:
        run_inference(args.sentence, num_rays=args.num_rays,
                      base_omega=args.omega, db=db)
    else:
        parser.print_help()
        print("\nEjemplo: python inference.py 'the cat sat on the mat'")


if __name__ == "__main__":
    main()
