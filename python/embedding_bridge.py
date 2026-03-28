#!/usr/bin/env python3
"""
embedding_bridge.py — Bridge Python para cargar embeddings y crear espacio semántico 3D

Este script proporciona funciones para:
  1. Cargar embeddings pre-entrenados (GloVe, Word2Vec, etc.)
  2. Aplicar PCA para proyección a 3D
  3. Generar TokenNodes que se pueden serializar al código C++
  4. Visualizar la distribución semántica en 3D

Uso:
    python3 embedding_bridge.py --load-glove embeddings.txt --output tokens.bin
    python3 embedding_bridge.py --sample-vocab 100 --visualize
"""

import sys
import struct
import argparse
import numpy as np
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass

# ============================================================================
# Importar bibliotecas opcionales
# ============================================================================

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARNING] sklearn not available. PCA will be implemented manually.", file=sys.stderr)

try:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[INFO] matplotlib not available. 3D visualization disabled.", file=sys.stderr)

# ============================================================================
# Estructura TokenNode (mirror de C++)
# ============================================================================

@dataclass
class TokenNode:
    """
    Espejo de la estructura TokenNode del código C++.

    Atributos:
        token_id: ID único del token en el vocabulario
        position_in_seq: Posición en la secuencia (inicialmente 0)
        centroid: Posición 3D (tuple de 3 floats)
        aabb_min: Esquina mínima del AABB
        aabb_max: Esquina máxima del AABB
        semantic_radius: Radio semántico
        embedding_fp16: Embedding comprimido a FP16 (array de 256 half-floats)
        attention_weight: Inicialmente 0.0
        energy_remaining: Inicialmente 1.0
    """
    token_id: int
    position_in_seq: int
    centroid: Tuple[float, float, float]
    aabb_min: Tuple[float, float, float]
    aabb_max: Tuple[float, float, float]
    semantic_radius: float
    embedding_fp16: np.ndarray  # Array de 256 float32 (simulando half-float)
    attention_weight: float = 0.0
    energy_remaining: float = 1.0

    def to_binary(self) -> bytes:
        """
        Serializa el TokenNode a formato binario compatible con C++.

        Formato:
            - uint32_t token_id (4 bytes)
            - uint32_t position_in_seq (4 bytes)
            - float3 centroid (12 bytes: 3 * 4)
            - float3 aabb_min (12 bytes)
            - float3 aabb_max (12 bytes)
            - float semantic_radius (4 bytes)
            - half[256] embedding (512 bytes: 256 * 2, simulado como float32)
            - float attention_weight (4 bytes)
            - float energy_remaining (4 bytes)
            Total: 580 bytes por token
        """
        buf = bytearray()

        # uint32_t token_id
        buf.extend(struct.pack('<I', self.token_id))

        # uint32_t position_in_seq
        buf.extend(struct.pack('<I', self.position_in_seq))

        # float3 centroid
        buf.extend(struct.pack('<fff', *self.centroid))

        # float3 aabb_min
        buf.extend(struct.pack('<fff', *self.aabb_min))

        # float3 aabb_max
        buf.extend(struct.pack('<fff', *self.aabb_max))

        # float semantic_radius
        buf.extend(struct.pack('<f', self.semantic_radius))

        # half[256] embedding (simulado como 256 float32 en lugar de float16)
        for val in self.embedding_fp16:
            buf.extend(struct.pack('<f', float(val)))

        # float attention_weight
        buf.extend(struct.pack('<f', self.attention_weight))

        # float energy_remaining
        buf.extend(struct.pack('<f', self.energy_remaining))

        return bytes(buf)

    @staticmethod
    def from_binary(data: bytes, offset: int = 0) -> Tuple['TokenNode', int]:
        """
        Deserializa un TokenNode desde datos binarios.

        Args:
            data: Buffer binario
            offset: Offset de inicio en el buffer

        Returns:
            (TokenNode, nuevo_offset)
        """
        def unpack(fmt, length):
            nonlocal offset
            result = struct.unpack_from(fmt, data, offset)
            offset += length
            return result

        token_id = unpack('<I', 4)[0]
        position_in_seq = unpack('<I', 4)[0]

        centroid = unpack('<fff', 12)

        aabb_min = unpack('<fff', 12)
        aabb_max = unpack('<fff', 12)

        semantic_radius = unpack('<f', 4)[0]

        embedding_fp16 = []
        for _ in range(256):
            val = unpack('<f', 4)[0]
            embedding_fp16.append(val)
        embedding_fp16 = np.array(embedding_fp16, dtype=np.float32)

        attention_weight = unpack('<f', 4)[0]
        energy_remaining = unpack('<f', 4)[0]

        node = TokenNode(
            token_id=token_id,
            position_in_seq=position_in_seq,
            centroid=centroid,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            semantic_radius=semantic_radius,
            embedding_fp16=embedding_fp16,
            attention_weight=attention_weight,
            energy_remaining=energy_remaining,
        )

        return node, offset


# ============================================================================
# Funciones de carga de embeddings
# ============================================================================

def load_glove_embeddings(filepath: str, max_vocab: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Carga embeddings en formato GloVe (formato de texto).

    Formato GloVe:
        word1 0.1234 0.5678 ... (embedding_dim values)
        word2 0.9876 0.5432 ...
        ...

    Args:
        filepath: Ruta al archivo GloVe
        max_vocab: Número máximo de palabras a cargar (None = todas)

    Returns:
        Dict[palabra] → np.ndarray (embedding)
    """
    embeddings = {}
    embed_dim = None

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_vocab and i >= max_vocab:
                    break

                parts = line.rstrip().split()
                if len(parts) < 2:
                    continue

                word = parts[0]
                try:
                    values = np.array([float(x) for x in parts[1:]], dtype=np.float32)
                except ValueError:
                    continue

                if embed_dim is None:
                    embed_dim = len(values)
                elif len(values) != embed_dim:
                    print(f"[WARNING] Line {i}: dimension mismatch, skipping word '{word}'", file=sys.stderr)
                    continue

                embeddings[word] = values

        print(f"[INFO] Loaded {len(embeddings)} embeddings from {filepath}")
        print(f"[INFO] Embedding dimension: {embed_dim}")

    except FileNotFoundError:
        print(f"[ERROR] File not found: {filepath}", file=sys.stderr)
        return {}

    return embeddings


def load_word2vec_embeddings(filepath: str, binary: bool = True, max_vocab: Optional[int] = None) -> Dict[str, np.ndarray]:
    """
    Carga embeddings en formato Word2Vec (binario).

    Nota: Este es un formato simplificado. Para producción, usar gensim.

    Args:
        filepath: Ruta al archivo Word2Vec
        binary: Si es formato binario (True) o texto (False)
        max_vocab: Número máximo de palabras a cargar

    Returns:
        Dict[palabra] → np.ndarray (embedding)
    """
    if binary:
        print("[INFO] Binary Word2Vec loading not fully implemented.", file=sys.stderr)
        print("[INFO] Falling back to text format assumption.", file=sys.stderr)

    # Asumir formato de texto para el prototipo
    return load_glove_embeddings(filepath, max_vocab)


# ============================================================================
# PCA manual (fallback si sklearn no disponible)
# ============================================================================

def pca_manual(data: np.ndarray, n_components: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementa PCA simple manualmente usando eigendecomposición.

    Args:
        data: Array de datos (n_samples, n_features)
        n_components: Número de componentes principales

    Returns:
        (transformed_data, components)
        - transformed_data: Array (n_samples, n_components)
        - components: Array (n_components, n_features) de vectores propios
    """
    print(f"[INFO] Running manual PCA with {n_components} components...", file=sys.stderr)

    # Centrar datos
    mean = np.mean(data, axis=0)
    data_centered = data - mean

    # Calcular matriz de covarianza
    cov_matrix = np.cov(data_centered.T)

    # Calcular eigenvalores y eigenvectores
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Ordenar en orden decreciente de eigenvalores
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Seleccionar n_components
    components = eigenvectors[:, :n_components].T

    # Proyectar datos
    transformed = data_centered @ components.T

    print(f"[INFO] PCA explained variance ratio: {eigenvalues[:n_components] / np.sum(eigenvalues)}", file=sys.stderr)

    return transformed, components


def apply_pca(embeddings_dict: Dict[str, np.ndarray], n_components: int = 3) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Aplica PCA a un diccionario de embeddings.

    Args:
        embeddings_dict: Dict[palabra] → np.ndarray
        n_components: Número de componentes (típicamente 3 para 3D)

    Returns:
        (Dict[palabra] → proyección 3D, componentes)
    """
    if not embeddings_dict:
        print("[ERROR] No embeddings provided.", file=sys.stderr)
        return {}, np.array([])

    # Stack de embeddings
    words = list(embeddings_dict.keys())
    embedding_matrix = np.array([embeddings_dict[w] for w in words])

    print(f"[INFO] PCA input: {embedding_matrix.shape}")

    # Aplicar PCA
    if HAS_SKLEARN:
        print("[INFO] Using sklearn PCA...")
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(embedding_matrix)
        components = pca.components_
    else:
        transformed, components = pca_manual(embedding_matrix, n_components)

    # Reconstruir diccionario
    result = {}
    for word, proj in zip(words, transformed):
        result[word] = proj

    return result, components


# ============================================================================
# Proyección de embedding a 3D y creación de TokenNode
# ============================================================================

def project_embedding_to_3d(embedding: np.ndarray, components: np.ndarray) -> np.ndarray:
    """
    Proyecta un embedding a 3D usando PCA componentes.

    Args:
        embedding: Embedding de alta dimensión (D,)
        components: PCA components (3, D)

    Returns:
        Proyección 3D (3,)
    """
    return embedding @ components.T


def create_token_node(
    word: str,
    token_id: int,
    embedding: np.ndarray,
    projection_3d: np.ndarray,
    position_in_seq: int = 0
) -> TokenNode:
    """
    Crea un TokenNode completo.

    Args:
        word: Palabra (solo para logging)
        token_id: ID del token
        embedding: Embedding original (para almacenamiento)
        projection_3d: Proyección 3D (centroide)
        position_in_seq: Posición en secuencia

    Returns:
        TokenNode inicializado
    """
    centroid = tuple(projection_3d.astype(np.float32))

    # Calcular radio semántico basado en magnitud del embedding
    embed_norm = np.linalg.norm(embedding[:64])  # Usar primeras 64 dimensiones
    semantic_radius = max(0.01, min(0.2, 0.01 + 0.19 * np.tanh(embed_norm)))

    # Crear AABB
    r = semantic_radius
    aabb_min = (centroid[0] - r, centroid[1] - r, centroid[2] - r)
    aabb_max = (centroid[0] + r, centroid[1] + r, centroid[2] + r)

    # Comprimir embedding a 256 dimensiones (simulado en FP16)
    # En código real, esto sería conversión a float16. Aquí usamos float32.
    embedding_compressed = np.zeros(256, dtype=np.float32)
    copy_dim = min(len(embedding), 256)
    embedding_compressed[:copy_dim] = embedding[:copy_dim]

    node = TokenNode(
        token_id=token_id,
        position_in_seq=position_in_seq,
        centroid=centroid,
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        semantic_radius=semantic_radius,
        embedding_fp16=embedding_compressed,
    )

    return node


# ============================================================================
# Visualización 3D
# ============================================================================

def visualize_semantic_space(
    nodes: List[TokenNode],
    words: List[str],
    title: str = "Semantic Space (3D Projection)"
) -> None:
    """
    Visualiza el espacio semántico 3D usando matplotlib.

    Args:
        nodes: Lista de TokenNodes
        words: Lista de palabras correspondientes
        title: Título del gráfico
    """
    if not HAS_MATPLOTLIB:
        print("[WARNING] matplotlib not available. Skipping visualization.", file=sys.stderr)
        return

    # Extraer coordenadas
    centroids = np.array([node.centroid for node in nodes])
    radii = np.array([node.semantic_radius for node in nodes])

    # Crear figura 3D
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot con colores por radio semántico
    scatter = ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c=radii,
        s=100,
        cmap='viridis',
        alpha=0.6
    )

    # Labels para puntos seleccionados (cada 5 puntos para evitar clutter)
    for i in range(0, len(words), max(1, len(words) // 20)):
        ax.text(
            centroids[i, 0],
            centroids[i, 1],
            centroids[i, 2],
            words[i],
            fontsize=8
        )

    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title(title)

    plt.colorbar(scatter, ax=ax, label='Semantic Radius')
    plt.tight_layout()
    plt.savefig('semantic_space_3d.png', dpi=150)
    print("[INFO] Saved visualization to semantic_space_3d.png", file=sys.stderr)
    plt.show()


# ============================================================================
# Funciones de I/O
# ============================================================================

def save_token_nodes_binary(nodes: List[TokenNode], filepath: str) -> bool:
    """
    Guarda TokenNodes a archivo binario.

    Formato:
        - uint32_t num_nodes (4 bytes)
        - TokenNode[num_nodes] (580 bytes cada uno)
    """
    try:
        with open(filepath, 'wb') as f:
            # Escribir número de nodos
            f.write(struct.pack('<I', len(nodes)))

            # Escribir cada nodo
            for node in nodes:
                f.write(node.to_binary())

        print(f"[INFO] Saved {len(nodes)} nodes to {filepath}")
        return True

    except IOError as e:
        print(f"[ERROR] Failed to save tokens: {e}", file=sys.stderr)
        return False


def load_token_nodes_binary(filepath: str) -> List[TokenNode]:
    """
    Carga TokenNodes desde archivo binario.
    """
    nodes = []

    try:
        with open(filepath, 'rb') as f:
            # Leer número de nodos
            num_nodes_data = f.read(4)
            if len(num_nodes_data) < 4:
                print("[ERROR] File too short", file=sys.stderr)
                return nodes

            num_nodes = struct.unpack('<I', num_nodes_data)[0]
            print(f"[INFO] Loading {num_nodes} nodes...")

            # Leer todos los nodos
            data = f.read()
            offset = 0

            for _ in range(num_nodes):
                node, offset = TokenNode.from_binary(data, offset)
                nodes.append(node)

        print(f"[INFO] Loaded {len(nodes)} nodes from {filepath}")
        return nodes

    except IOError as e:
        print(f"[ERROR] Failed to load tokens: {e}", file=sys.stderr)
        return nodes


# ============================================================================
# Ejemplo: Vocabulario de código
# ============================================================================

def create_sample_code_vocab() -> Dict[str, np.ndarray]:
    """
    Crea un vocabulario de muestra con palabras clave de código.

    Cada palabra tiene un embedding sintético (100-dimensional)
    basado en su "complejidad semántica".
    """
    keywords = {
        # Control flow
        'if': 1.0,
        'else': 0.95,
        'for': 1.1,
        'while': 1.05,
        'do': 0.8,

        # Functions and classes
        'function': 2.0,
        'class': 1.9,
        'def': 1.8,
        'return': 1.5,
        'yield': 1.4,

        # Data structures
        'array': 1.3,
        'list': 1.2,
        'dict': 1.25,
        'set': 1.15,
        'tuple': 1.1,

        # Operations
        'add': 0.5,
        'subtract': 0.5,
        'multiply': 0.6,
        'divide': 0.6,

        # Logic
        'and': 0.7,
        'or': 0.7,
        'not': 0.6,

        # Other keywords
        'import': 1.7,
        'from': 1.6,
        'as': 0.4,
        'try': 1.3,
        'except': 1.2,
        'finally': 1.1,
        'pass': 0.3,
        'break': 0.8,
        'continue': 0.8,
    }

    # Crear embeddings sintéticos
    embeddings = {}
    for word, complexity in keywords.items():
        # Embedding pseudo-aleatorio basado en palabra
        seed = sum(ord(c) for c in word)
        np.random.seed(seed % (2**31))

        # Crear embedding con componente dominante = complejidad
        embedding = np.random.randn(100).astype(np.float32) * 0.1
        embedding[0] = complexity  # Componente dominante
        embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

        embeddings[word] = embedding

    return embeddings


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="EmbeddingBridge: Proyecta embeddings al espacio 3D para LiquidBit Zero-Matrix"
    )

    parser.add_argument('--load-glove', type=str, help='Ruta al archivo GloVe')
    parser.add_argument('--load-word2vec', type=str, help='Ruta al archivo Word2Vec')
    parser.add_argument('--sample-vocab', type=int, default=None,
                        help='Usar vocabulario de muestra de N palabras de código')
    parser.add_argument('--output', type=str, default='token_nodes.bin',
                        help='Archivo de salida (binario)')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualizar espacio semántico 3D')
    parser.add_argument('--max-vocab', type=int, default=None,
                        help='Limitar vocabulario a N palabras')

    args = parser.parse_args()

    # ==================================================================================
    # PASO 1: Cargar embeddings
    # ==================================================================================
    embeddings_dict = {}

    if args.sample_vocab:
        print(f"[INFO] Creating sample code vocabulary with {args.sample_vocab} words...")
        embeddings_dict = create_sample_code_vocab()

        # Limitar a muestra si se requiere
        if args.sample_vocab and len(embeddings_dict) > args.sample_vocab:
            keys = list(embeddings_dict.keys())[:args.sample_vocab]
            embeddings_dict = {k: embeddings_dict[k] for k in keys}

    elif args.load_glove:
        print(f"[INFO] Loading GloVe embeddings from {args.load_glove}...")
        embeddings_dict = load_glove_embeddings(args.load_glove, args.max_vocab)

    elif args.load_word2vec:
        print(f"[INFO] Loading Word2Vec embeddings from {args.load_word2vec}...")
        embeddings_dict = load_word2vec_embeddings(args.load_word2vec, max_vocab=args.max_vocab)

    if not embeddings_dict:
        print("[ERROR] No embeddings loaded. Use --sample-vocab, --load-glove, or --load-word2vec", file=sys.stderr)
        sys.exit(1)

    # ==================================================================================
    # PASO 2: Aplicar PCA a 3D
    # ==================================================================================
    print(f"[INFO] Applying PCA to {len(embeddings_dict)} embeddings...")
    embeddings_3d, components = apply_pca(embeddings_dict, n_components=3)

    # ==================================================================================
    # PASO 3: Crear TokenNodes
    # ==================================================================================
    print(f"[INFO] Creating TokenNodes...")
    nodes = []
    words = list(embeddings_dict.keys())

    for token_id, word in enumerate(words):
        embedding = embeddings_dict[word]
        projection_3d = embeddings_3d[word]

        node = create_token_node(
            word=word,
            token_id=token_id,
            embedding=embedding,
            projection_3d=projection_3d,
            position_in_seq=token_id,
        )

        nodes.append(node)

    # ==================================================================================
    # PASO 4: Guardar a archivo binario
    # ==================================================================================
    if not save_token_nodes_binary(nodes, args.output):
        sys.exit(1)

    # ==================================================================================
    # PASO 5: Visualizar (opcional)
    # ==================================================================================
    if args.visualize:
        print("[INFO] Visualizing semantic space...")
        visualize_semantic_space(nodes, words, title=f"Semantic Space ({len(words)} tokens)")

    print("[INFO] Done!")


if __name__ == '__main__':
    main()
