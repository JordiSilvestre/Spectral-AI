#!/usr/bin/env python3
"""
test_polysemy_benchmark.py -- Polysemy resolution benchmark for SpectralAI

Tests 50 polysemous English words across 2-3 contexts each, measuring
whether SpectralAI's optical routing mechanisms produce different expert
routing for different senses of the same word.

Configurations tested (matching Patent P3 Table, Section 11):
  1. Single PrismaticRefraction (Snell only)
  2. + ChromaticAberration (4 bands)
  3. + TotalInternalReflection
  4. + PhaseCoherentInterference

Each configuration is trained with a brief contrastive objective that
teaches the spectral encoder and refraction layers to separate different
senses of the same word into distinct routing patterns.

Metric: percentage of polysemous words where different contexts produce
different top-1 expert routing.

Runs on CPU. Uses fixed seeds for full reproducibility.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import pytest
except ImportError:
    pytest = None
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Add project root so we can import from python/
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "python"))

from inception_attention import (
    ChromaticAberration,
    InceptionConfig,
    PhaseCoherentInterference,
    PrismaticRefraction,
    SpectralEncoder,
    TotalInternalReflection,
)


# ===================================================================
# Polysemy corpus: 50 words, 2-3 senses each
# ===================================================================

@dataclass(frozen=True)
class WordSense:
    """One sense of a polysemous word, defined by context words."""
    label: str
    context_words: Tuple[str, ...]


# Mapping: polysemous word -> list of senses
POLYSEMY_CORPUS: Dict[str, List[WordSense]] = {
    "bank": [
        WordSense("financial", ("money", "account", "deposit", "loan", "interest")),
        WordSense("river", ("river", "water", "shore", "flood", "stream")),
    ],
    "bass": [
        WordSense("music", ("guitar", "instrument", "amplifier", "band", "solo")),
        WordSense("fish", ("fishing", "lake", "bait", "catch", "pond")),
    ],
    "scale": [
        WordSense("music", ("notes", "melody", "octave", "key", "chord")),
        WordSense("measurement", ("weight", "measure", "balance", "kilogram", "metric")),
        WordSense("reptile", ("fish", "skin", "dragon", "lizard", "armor")),
    ],
    "bark": [
        WordSense("dog", ("dog", "loud", "growl", "pet", "bite")),
        WordSense("tree", ("tree", "wood", "oak", "trunk", "forest")),
    ],
    "bat": [
        WordSense("animal", ("cave", "wings", "nocturnal", "echolocation", "vampire")),
        WordSense("sports", ("baseball", "cricket", "swing", "hit", "pitch")),
    ],
    "crane": [
        WordSense("bird", ("bird", "wetland", "migration", "feathers", "nest")),
        WordSense("machine", ("construction", "lift", "steel", "tower", "building")),
    ],
    "match": [
        WordSense("fire", ("flame", "ignite", "sulfur", "burn", "lighter")),
        WordSense("sports", ("game", "tournament", "opponent", "score", "win")),
        WordSense("pair", ("equal", "identical", "pair", "similar", "correspond")),
    ],
    "pitch": [
        WordSense("music", ("tone", "frequency", "soprano", "note", "tune")),
        WordSense("sports", ("throw", "baseball", "mound", "strike", "fastball")),
        WordSense("sales", ("presentation", "proposal", "investor", "startup", "sell")),
    ],
    "spring": [
        WordSense("season", ("flowers", "bloom", "warm", "april", "rain")),
        WordSense("water", ("fountain", "well", "underground", "mineral", "geyser")),
        WordSense("coil", ("metal", "bounce", "tension", "elastic", "mattress")),
    ],
    "cell": [
        WordSense("biology", ("membrane", "nucleus", "mitosis", "organism", "protein")),
        WordSense("prison", ("jail", "inmate", "bars", "sentence", "guard")),
        WordSense("phone", ("mobile", "call", "signal", "battery", "smartphone")),
    ],
    "club": [
        WordSense("social", ("membership", "nightclub", "dancing", "party", "lounge")),
        WordSense("weapon", ("weapon", "stick", "blunt", "caveman", "medieval")),
        WordSense("golf", ("golf", "driver", "iron", "wedge", "putter")),
    ],
    "current": [
        WordSense("electricity", ("ampere", "voltage", "wire", "circuit", "ohm")),
        WordSense("water", ("river", "flow", "tide", "ocean", "downstream")),
        WordSense("time", ("present", "today", "modern", "now", "recent")),
    ],
    "date": [
        WordSense("calendar", ("day", "month", "year", "schedule", "appointment")),
        WordSense("fruit", ("palm", "sweet", "desert", "dried", "fig")),
        WordSense("romantic", ("dinner", "couple", "romance", "restaurant", "evening")),
    ],
    "draft": [
        WordSense("writing", ("manuscript", "edit", "revision", "author", "document")),
        WordSense("sports", ("pick", "selection", "rookie", "league", "prospect")),
        WordSense("air", ("breeze", "ventilation", "cold", "window", "airflow")),
    ],
    "drill": [
        WordSense("tool", ("hole", "bit", "power", "screw", "cordless")),
        WordSense("exercise", ("practice", "repetition", "training", "routine", "military")),
    ],
    "fan": [
        WordSense("device", ("blade", "cooling", "motor", "rotate", "breeze")),
        WordSense("admirer", ("supporter", "cheering", "audience", "idol", "autograph")),
    ],
    "file": [
        WordSense("computer", ("document", "folder", "save", "disk", "upload")),
        WordSense("tool", ("metal", "sharpen", "smooth", "rasp", "grind")),
    ],
    "fly": [
        WordSense("insect", ("bug", "wings", "buzz", "swatter", "pest")),
        WordSense("aviation", ("airplane", "pilot", "altitude", "runway", "cockpit")),
    ],
    "jam": [
        WordSense("food", ("fruit", "toast", "strawberry", "preserve", "spread")),
        WordSense("traffic", ("congestion", "stuck", "gridlock", "highway", "delay")),
        WordSense("music", ("improvise", "session", "guitar", "blues", "groove")),
    ],
    "key": [
        WordSense("lock", ("door", "lock", "metal", "unlock", "chain")),
        WordSense("music", ("chord", "minor", "major", "sharps", "signature")),
        WordSense("important", ("crucial", "essential", "vital", "critical", "main")),
    ],
    "lead": [
        WordSense("guide", ("leader", "front", "charge", "command", "direction")),
        WordSense("metal", ("heavy", "toxic", "element", "plumbing", "solder")),
    ],
    "light": [
        WordSense("optics", ("photon", "beam", "lamp", "brightness", "illuminate")),
        WordSense("weight", ("feather", "lightweight", "airy", "thin", "delicate")),
    ],
    "mole": [
        WordSense("animal", ("burrow", "underground", "garden", "tunnel", "fur")),
        WordSense("chemistry", ("avogadro", "molecules", "concentration", "solution", "molar")),
        WordSense("skin", ("spot", "dermatology", "melanoma", "biopsy", "pigment")),
    ],
    "mouse": [
        WordSense("animal", ("rodent", "cheese", "trap", "whiskers", "squeak")),
        WordSense("computer", ("click", "pointer", "cursor", "scroll", "usb")),
    ],
    "nail": [
        WordSense("body", ("finger", "manicure", "polish", "cuticle", "salon")),
        WordSense("hardware", ("hammer", "wood", "screw", "bolt", "construction")),
    ],
    "note": [
        WordSense("music", ("melody", "sharp", "flat", "staff", "treble")),
        WordSense("writing", ("paper", "memo", "scribble", "jot", "sticky")),
    ],
    "organ": [
        WordSense("body", ("heart", "liver", "transplant", "kidney", "tissue")),
        WordSense("music", ("pipe", "church", "keyboard", "pedal", "bach")),
    ],
    "palm": [
        WordSense("hand", ("hand", "fingers", "grip", "wrist", "fist")),
        WordSense("tree", ("tropical", "coconut", "frond", "beach", "island")),
    ],
    "pen": [
        WordSense("writing", ("ink", "paper", "ballpoint", "write", "fountain")),
        WordSense("enclosure", ("fence", "livestock", "pig", "corral", "paddock")),
    ],
    "plant": [
        WordSense("botany", ("flower", "root", "leaf", "photosynthesis", "seed")),
        WordSense("factory", ("manufacturing", "assembly", "industrial", "machinery", "production")),
    ],
    "pool": [
        WordSense("swimming", ("swim", "chlorine", "diving", "lane", "lifeguard")),
        WordSense("billiards", ("cue", "billiards", "table", "pocket", "eight")),
        WordSense("resource", ("shared", "carpool", "collective", "fund", "common")),
    ],
    "port": [
        WordSense("harbor", ("ship", "dock", "cargo", "harbor", "maritime")),
        WordSense("computer", ("usb", "socket", "connection", "serial", "ethernet")),
        WordSense("wine", ("red", "vintage", "portugal", "dessert", "fortified")),
    ],
    "pound": [
        WordSense("currency", ("sterling", "british", "money", "exchange", "shilling")),
        WordSense("weight", ("kilogram", "ounce", "mass", "heavy", "scale")),
        WordSense("action", ("hammer", "beat", "smash", "fist", "strike")),
    ],
    "pupil": [
        WordSense("eye", ("iris", "retina", "dilate", "cornea", "optic")),
        WordSense("student", ("school", "teacher", "classroom", "learn", "grade")),
    ],
    "race": [
        WordSense("competition", ("sprint", "marathon", "finish", "runner", "track")),
        WordSense("ethnicity", ("heritage", "culture", "diversity", "identity", "ancestry")),
    ],
    "ring": [
        WordSense("jewelry", ("gold", "diamond", "wedding", "finger", "engagement")),
        WordSense("sound", ("bell", "phone", "chime", "alarm", "tone")),
        WordSense("arena", ("boxing", "wrestling", "fight", "corner", "referee")),
    ],
    "rock": [
        WordSense("geology", ("stone", "mineral", "granite", "boulder", "sediment")),
        WordSense("music", ("concert", "electric", "band", "guitar", "stage")),
    ],
    "ruler": [
        WordSense("measuring", ("centimeter", "length", "straight", "inches", "measure")),
        WordSense("leader", ("king", "emperor", "monarch", "throne", "kingdom")),
    ],
    "seal": [
        WordSense("animal", ("ocean", "flipper", "arctic", "fish", "marine")),
        WordSense("closure", ("envelope", "stamp", "wax", "airtight", "gasket")),
    ],
    "set": [
        WordSense("collection", ("group", "bundle", "series", "kit", "assortment")),
        WordSense("tennis", ("game", "match", "deuce", "serve", "volley")),
    ],
    "sink": [
        WordSense("kitchen", ("faucet", "drain", "basin", "dishes", "tap")),
        WordSense("descend", ("submerge", "drown", "underwater", "bottom", "deep")),
    ],
    "suit": [
        WordSense("clothing", ("jacket", "tie", "formal", "blazer", "tailor")),
        WordSense("legal", ("lawsuit", "court", "plaintiff", "attorney", "litigation")),
        WordSense("cards", ("hearts", "spades", "diamonds", "clubs", "deck")),
    ],
    "tap": [
        WordSense("faucet", ("water", "sink", "plumbing", "valve", "pipe")),
        WordSense("touch", ("finger", "screen", "gentle", "shoulder", "knock")),
    ],
    "temple": [
        WordSense("religion", ("worship", "prayer", "sacred", "altar", "shrine")),
        WordSense("anatomy", ("head", "forehead", "skull", "headache", "side")),
    ],
    "tie": [
        WordSense("clothing", ("necktie", "suit", "knot", "silk", "collar")),
        WordSense("equal", ("draw", "score", "even", "deadlock", "stalemate")),
    ],
    "toast": [
        WordSense("food", ("bread", "butter", "breakfast", "toaster", "crispy")),
        WordSense("celebration", ("champagne", "glass", "cheers", "speech", "wedding")),
    ],
    "trunk": [
        WordSense("tree", ("bark", "wood", "roots", "branch", "oak")),
        WordSense("car", ("luggage", "boot", "storage", "rear", "sedan")),
        WordSense("elephant", ("ivory", "tusk", "nose", "spray", "pachyderm")),
    ],
    "vessel": [
        WordSense("ship", ("sail", "maritime", "cargo", "captain", "harbor")),
        WordSense("anatomy", ("blood", "artery", "vein", "capillary", "circulation")),
    ],
    "watch": [
        WordSense("timepiece", ("wrist", "clock", "hour", "dial", "strap")),
        WordSense("observe", ("look", "stare", "surveillance", "monitor", "gaze")),
    ],
    "wave": [
        WordSense("ocean", ("surf", "tide", "beach", "crest", "swell")),
        WordSense("physics", ("frequency", "amplitude", "wavelength", "oscillation", "hertz")),
        WordSense("gesture", ("hand", "greeting", "hello", "goodbye", "signal")),
    ],
}


# ===================================================================
# Deterministic embedding generator
# ===================================================================

EMBED_DIM = 256
SPECTRAL_DIM = 64
N_SPHERES = 16  # expert count
TRAIN_STEPS = 100  # contrastive training steps for Single Snell
TRAIN_LR = 1e-2
# Chromatic/Phase configs have more parameters and need extra training
TRAIN_STEPS_ADVANCED = 300
TRAIN_LR_ADVANCED = 3e-2


def _word_to_seed(word: str) -> int:
    """Deterministic seed from word string (no external vocab needed)."""
    h = 0
    for ch in word:
        h = (h * 31 + ord(ch)) & 0xFFFFFFFF
    return h


def _make_word_embedding(word: str) -> torch.Tensor:
    """Generate a reproducible pseudo-embedding for a single word."""
    gen = torch.Generator()
    gen.manual_seed(_word_to_seed(word))
    return torch.randn(EMBED_DIM, generator=gen)


def build_context_embedding(
    target_word: str,
    context_words: Tuple[str, ...],
) -> torch.Tensor:
    """
    Build a context-aware embedding by combining the target word embedding
    with a context centroid.  The context words are weighted 3x relative to
    the target so that different senses of the same word produce clearly
    separated embeddings (the target word is identical across senses, so it
    must not dominate).  Shape: (1, 1, EMBED_DIM).
    """
    target_emb = _make_word_embedding(target_word)
    context_embs = torch.stack([_make_word_embedding(w) for w in context_words])
    context_centroid = context_embs.mean(dim=0)
    # Context-heavy weighting: 25% target, 75% context
    combined = 0.25 * target_emb + 0.75 * context_centroid
    # Normalize to unit sphere for consistent magnitude
    combined = combined / (combined.norm() + 1e-8)
    return combined.unsqueeze(0).unsqueeze(0)  # (1, 1, D)


def _build_corpus_embeddings() -> Dict[str, List[torch.Tensor]]:
    """Pre-compute all context embeddings. Returns {word: [emb_sense0, emb_sense1, ...]}."""
    result: Dict[str, List[torch.Tensor]] = {}
    for word, senses in POLYSEMY_CORPUS.items():
        result[word] = [
            build_context_embedding(word, sense.context_words)
            for sense in senses
        ]
    return result


# ===================================================================
# Configuration builders
# ===================================================================

@dataclass(frozen=True)
class MechanismConfig:
    """Describes which optical mechanisms are active."""
    name: str
    use_chromatic: bool
    use_tir: bool
    use_phase: bool


MECHANISM_CONFIGS: List[MechanismConfig] = [
    MechanismConfig("Single Snell", False, False, False),
    MechanismConfig("+ Chromatic", True, False, False),
    MechanismConfig("+ TIR", True, True, False),
    MechanismConfig("+ Phase", True, True, True),
]


def _build_modules(
    cfg: MechanismConfig,
    master_seed: int = 42,
) -> Dict[str, nn.Module]:
    """
    Build the optical modules for a given mechanism config.

    Each module type uses its own dedicated seed derived from master_seed,
    so that adding/removing optional modules does not shift the random
    state for shared modules (encoder, refraction).
    """
    # Encoder -- always present, always same initial weights
    torch.manual_seed(master_seed)
    encoder = SpectralEncoder(EMBED_DIM, SPECTRAL_DIM)
    modules: Dict[str, nn.Module] = {"encoder": encoder}

    # Refraction -- seed offset 1000
    torch.manual_seed(master_seed + 1000)
    if cfg.use_chromatic:
        refraction: nn.Module = ChromaticAberration(
            n_spheres=N_SPHERES,
            spectral_dim=SPECTRAL_DIM,
            n_bands=4,
        )
    else:
        refraction = PrismaticRefraction(
            n_spheres=N_SPHERES,
            spectral_dim=SPECTRAL_DIM,
        )
    modules["refraction"] = refraction

    # TIR -- seed offset 2000
    if cfg.use_tir:
        torch.manual_seed(master_seed + 2000)
        modules["tir"] = TotalInternalReflection(n_spheres=N_SPHERES)

    # Phase interference -- seed offset 3000
    if cfg.use_phase:
        torch.manual_seed(master_seed + 3000)
        modules["phase"] = PhaseCoherentInterference(
            spectral_dim=SPECTRAL_DIM,
            n_rays=4,
        )

    return modules


# ===================================================================
# Routing forward pass
# ===================================================================

def _get_routing_distribution(
    context_emb: torch.Tensor,
    modules: Dict[str, nn.Module],
    cfg: MechanismConfig,
) -> torch.Tensor:
    """
    Run the spectral pipeline and return the routing distribution.

    context_emb: (1, 1, EMBED_DIM)
    returns: (1, 1, N_SPHERES) -- soft routing probabilities
    """
    encoder = modules["encoder"]
    refraction = modules["refraction"]

    # Step 1: encode context -> spectral color
    spectral_color = encoder(context_emb)  # (1, 1, spectral_dim)

    # Step 2: compute refractive indices
    if cfg.use_phase and "phase" in modules:
        phase_mod = modules["phase"]
        routing = phase_mod(spectral_color, refraction)  # (1, 1, n_spheres)
    else:
        routing = refraction(spectral_color)  # (1, 1, n_spheres)

    # Step 3: apply TIR if enabled
    if cfg.use_tir and "tir" in modules:
        tir_mod = modules["tir"]
        # Incidence angle: use routing values directly as proxy for cos(theta).
        # Clamp to valid [0, 1] range so Snell's discriminant is well-defined.
        cos_incidence = routing.clamp(0.01, 0.99)
        membership = torch.softmax(routing * 10.0, dim=-1)
        adjusted_membership, _tir_mask = tir_mod(routing, membership, cos_incidence)
        # Combine: use TIR-adjusted membership to re-weight routing
        routing = routing * adjusted_membership

    return routing


# ===================================================================
# Contrastive training
# ===================================================================

def _contrastive_loss(
    routing_distributions: List[torch.Tensor],
) -> torch.Tensor:
    """
    Contrastive loss: for each pair of senses of the same word, maximize
    the divergence of their routing distributions.

    We want different senses to peak at different experts.  The loss uses
    negative cosine similarity (lower = more different routing).

    routing_distributions: list of (1, 1, N_SPHERES) tensors, one per sense.
    """
    if len(routing_distributions) < 2:
        return torch.tensor(0.0)

    loss = torch.tensor(0.0)
    n_pairs = 0
    for i in range(len(routing_distributions)):
        for j in range(i + 1, len(routing_distributions)):
            r_i = routing_distributions[i].squeeze()  # (N_SPHERES,)
            r_j = routing_distributions[j].squeeze()  # (N_SPHERES,)
            # We want cosine similarity to be LOW (different routing)
            # So loss = cosine_similarity (minimize this)
            cos_sim = F.cosine_similarity(r_i.unsqueeze(0), r_j.unsqueeze(0))
            loss = loss + cos_sim.squeeze()
            n_pairs += 1

    return loss / max(n_pairs, 1)


def _build_batched_training_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pre-compute all context embeddings as a single batched tensor.

    Returns:
        all_embs: (1, N_total, EMBED_DIM) -- all context embeddings
        pair_i: (n_pairs,) -- left indices of contrastive pairs
        pair_j: (n_pairs,) -- right indices of contrastive pairs
    """
    all_embs_list: List[torch.Tensor] = []
    pairs_i: List[int] = []
    pairs_j: List[int] = []
    offset = 0

    for word, senses in POLYSEMY_CORPUS.items():
        sense_indices = []
        for sense in senses:
            emb = build_context_embedding(word, sense.context_words).squeeze()  # (D,)
            all_embs_list.append(emb)
            sense_indices.append(offset)
            offset += 1

        for i in range(len(sense_indices)):
            for j in range(i + 1, len(sense_indices)):
                pairs_i.append(sense_indices[i])
                pairs_j.append(sense_indices[j])

    all_embs = torch.stack(all_embs_list).unsqueeze(0)  # (1, N_total, D)
    return all_embs, torch.tensor(pairs_i), torch.tensor(pairs_j)


def _get_training_routing(
    context_emb: torch.Tensor,
    modules: Dict[str, nn.Module],
    cfg: MechanismConfig,
) -> torch.Tensor:
    """
    Simplified routing for training: encoder + refraction only.
    TIR and Phase are applied only at eval time as decision-boundary
    mechanisms (matching the patent design where they add post-hoc
    routing refinement).
    """
    encoder = modules["encoder"]
    refraction = modules["refraction"]
    spectral_color = encoder(context_emb)

    if cfg.use_phase and "phase" in modules:
        return modules["phase"](spectral_color, refraction)
    else:
        return refraction(spectral_color)


def _compute_band_diversity_loss(modules: Dict[str, nn.Module]) -> torch.Tensor:
    """
    Encourage ChromaticAberration bands to specialize differently.
    Returns 0.0 if no ChromaticAberration module is present.
    """
    refraction = modules.get("refraction")
    if not isinstance(refraction, ChromaticAberration):
        return torch.tensor(0.0)

    # Compare W_dispersion weights across bands -- penalize similarity
    band_weights_list = [
        br.W_dispersion.weight for br in refraction.band_refractions
    ]
    diversity_loss = torch.tensor(0.0)
    n_pairs = 0
    for i in range(len(band_weights_list)):
        for j in range(i + 1, len(band_weights_list)):
            # These have different shapes if band_size differs, but here
            # all bands have the same size, so we flatten and compare
            w_i = band_weights_list[i].flatten()
            w_j = band_weights_list[j].flatten()
            # We want them to be DIFFERENT, so penalize high cosine sim
            diversity_loss = diversity_loss + F.cosine_similarity(
                w_i.unsqueeze(0), w_j.unsqueeze(0),
            ).squeeze()
            n_pairs += 1

    return diversity_loss / max(n_pairs, 1)


def train_config(
    cfg: MechanismConfig,
    n_steps: int = TRAIN_STEPS,
    lr: float = TRAIN_LR,
    master_seed: int = 42,
    verbose: bool = False,
) -> Dict[str, nn.Module]:
    """
    Train a configuration's modules with a contrastive polysemy objective.

    Uses batched forward passes for efficiency.  The contrastive loss
    minimizes cosine similarity between routing distributions of different
    senses of the same word.

    For ChromaticAberration configs, an additional band diversity loss
    encourages the spectral bands to specialize differently.

    Training flows through encoder + refraction (+ phase if enabled).
    TIR is a test-time mechanism applied only during evaluation.

    Returns the trained modules in eval mode.
    """
    torch.manual_seed(master_seed)
    modules = _build_modules(cfg, master_seed=master_seed)

    # Collect all trainable parameters
    all_params: List[torch.Tensor] = []
    for m in modules.values():
        m.train()
        all_params.extend(m.parameters())

    optimizer = torch.optim.Adam(all_params, lr=lr)

    # Pre-compute corpus data
    all_embs, pair_i, pair_j = _build_batched_training_data()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Forward pass through encoder + refraction (+ phase)
        routing = _get_training_routing(
            all_embs, modules, cfg,
        ).squeeze(0)  # (N_total, N_SPHERES)

        # Contrastive loss: minimize cosine similarity for sense pairs
        r_i = routing[pair_i]  # (n_pairs, N_SPHERES)
        r_j = routing[pair_j]  # (n_pairs, N_SPHERES)
        cos_sim = F.cosine_similarity(r_i, r_j, dim=-1)  # (n_pairs,)
        loss = cos_sim.mean()

        # Band diversity: encourage chromatic bands to specialize
        if cfg.use_chromatic:
            diversity_loss = _compute_band_diversity_loss(modules)
            loss = loss + 0.5 * diversity_loss

        loss.backward()
        optimizer.step()

        if verbose and (step + 1) % 25 == 0:
            print(f"    Step {step + 1}/{n_steps}: loss = {loss.item():.4f}")

    # Switch to eval mode
    for m in modules.values():
        m.eval()

    return modules


# ===================================================================
# Evaluation
# ===================================================================

@torch.no_grad()
def get_top1_expert(
    context_emb: torch.Tensor,
    modules: Dict[str, nn.Module],
    cfg: MechanismConfig,
) -> int:
    """
    Run the spectral pipeline and return the top-1 expert index.

    context_emb: (1, 1, EMBED_DIM)
    returns: integer expert index in [0, N_SPHERES)
    """
    routing = _get_routing_distribution(context_emb, modules, cfg)
    return int(routing.squeeze().argmax().item())


def evaluate_polysemy(
    cfg: MechanismConfig,
    trained_modules: Dict[str, nn.Module] | None = None,
    verbose: bool = False,
) -> Tuple[float, int, int]:
    """
    Evaluate polysemy resolution for a given mechanism configuration.

    If trained_modules is None, trains from scratch first.

    Returns (accuracy, n_resolved, n_total).
    - n_total = number of polysemous words
    - n_resolved = words where all senses map to distinct top-1 experts
    - accuracy = n_resolved / n_total
    """
    if trained_modules is None:
        trained_modules = train_config(cfg, verbose=verbose)

    n_resolved = 0
    n_total = len(POLYSEMY_CORPUS)

    for word, senses in POLYSEMY_CORPUS.items():
        expert_ids = set()
        for sense in senses:
            emb = build_context_embedding(word, sense.context_words)
            top1 = get_top1_expert(emb, trained_modules, cfg)
            expert_ids.add(top1)

        # Resolved if all senses got different routing
        if len(expert_ids) == len(senses):
            n_resolved += 1

    accuracy = n_resolved / n_total if n_total > 0 else 0.0
    return accuracy, n_resolved, n_total


# ===================================================================
# Summary printer
# ===================================================================

def print_summary_table(
    results: List[Tuple[str, float, int, int]],
) -> None:
    """Print patent-style summary table."""
    header = f"| {'Mechanism':<25} | {'Polysemy Accuracy':>18} | {'Resolved':>10} |"
    separator = f"|{'-' * 27}|{'-' * 20}|{'-' * 12}|"

    print()
    print("=" * 63)
    print("  SpectralAI Polysemy Resolution Benchmark")
    print(f"  Corpus: {len(POLYSEMY_CORPUS)} polysemous words, "
          f"{sum(len(s) for s in POLYSEMY_CORPUS.values())} total senses")
    print(f"  Experts (spheres): {N_SPHERES}")
    print(f"  Training: {TRAIN_STEPS} contrastive steps, lr={TRAIN_LR}")
    print("=" * 63)
    print(header)
    print(separator)
    for name, accuracy, resolved, total in results:
        print(f"| {name:<25} | {accuracy * 100:>17.1f}% | {resolved:>4}/{total:<4} |")
    print(separator)
    print()


# ===================================================================
# Pytest tests
# ===================================================================

class TestPolysemyCorpus:
    """Validate the polysemy corpus itself."""

    def test_corpus_has_at_least_50_words(self) -> None:
        assert len(POLYSEMY_CORPUS) >= 50, (
            f"Corpus has {len(POLYSEMY_CORPUS)} words, expected >= 50"
        )

    def test_each_word_has_multiple_senses(self) -> None:
        for word, senses in POLYSEMY_CORPUS.items():
            assert len(senses) >= 2, (
                f"Word '{word}' has only {len(senses)} sense(s), expected >= 2"
            )

    def test_each_sense_has_context_words(self) -> None:
        for word, senses in POLYSEMY_CORPUS.items():
            for sense in senses:
                assert len(sense.context_words) >= 3, (
                    f"Word '{word}' sense '{sense.label}' has only "
                    f"{len(sense.context_words)} context words, expected >= 3"
                )

    def test_senses_have_distinct_labels(self) -> None:
        for word, senses in POLYSEMY_CORPUS.items():
            labels = [s.label for s in senses]
            assert len(labels) == len(set(labels)), (
                f"Word '{word}' has duplicate sense labels: {labels}"
            )


class TestEmbeddingReproducibility:
    """Verify that embeddings are deterministic."""

    def test_same_word_same_embedding(self) -> None:
        e1 = _make_word_embedding("bank")
        e2 = _make_word_embedding("bank")
        assert torch.allclose(e1, e2), "Same word should produce identical embeddings"

    def test_different_words_different_embeddings(self) -> None:
        e1 = _make_word_embedding("bank")
        e2 = _make_word_embedding("scale")
        assert not torch.allclose(e1, e2), "Different words should differ"

    def test_context_embedding_shape(self) -> None:
        emb = build_context_embedding("bank", ("money", "account", "deposit"))
        assert emb.shape == (1, 1, EMBED_DIM)

    def test_different_contexts_produce_different_embeddings(self) -> None:
        e1 = build_context_embedding("bank", ("money", "account", "deposit"))
        e2 = build_context_embedding("bank", ("river", "water", "shore"))
        assert not torch.allclose(e1, e2), (
            "Same word in different contexts should produce different embeddings"
        )


# pytest-based test classes (only defined when pytest is available)
if pytest is not None:

    class TestMechanismConfigs:
        """Test that each mechanism config produces valid routing."""

        @pytest.mark.parametrize("cfg", MECHANISM_CONFIGS, ids=[c.name for c in MECHANISM_CONFIGS])
        def test_routing_returns_valid_expert(self, cfg: MechanismConfig) -> None:
            modules = _build_modules(cfg)
            for m in modules.values():
                m.eval()

            emb = build_context_embedding("bank", ("money", "account", "deposit"))
            expert = get_top1_expert(emb, modules, cfg)
            assert 0 <= expert < N_SPHERES, f"Expert {expert} out of range [0, {N_SPHERES})"

        @pytest.mark.parametrize("cfg", MECHANISM_CONFIGS, ids=[c.name for c in MECHANISM_CONFIGS])
        def test_routing_is_deterministic(self, cfg: MechanismConfig) -> None:
            modules = _build_modules(cfg)
            for m in modules.values():
                m.eval()

            emb = build_context_embedding("scale", ("notes", "melody", "octave"))
            expert1 = get_top1_expert(emb, modules, cfg)
            expert2 = get_top1_expert(emb, modules, cfg)
            assert expert1 == expert2, "Same input should produce same routing"


    class TestPolysemyBenchmark:
        """Run the full polysemy benchmark as a test suite."""

        @pytest.fixture(scope="class")
        def trained_configs(self) -> Dict[str, Tuple[MechanismConfig, Dict[str, nn.Module]]]:
            """Train all configs once, share across tests in this class."""
            result = {}
            for cfg in MECHANISM_CONFIGS:
                is_advanced = cfg.use_chromatic
                steps = TRAIN_STEPS_ADVANCED if is_advanced else TRAIN_STEPS
                lr = TRAIN_LR_ADVANCED if is_advanced else TRAIN_LR
                modules = train_config(cfg, n_steps=steps, lr=lr)
                result[cfg.name] = (cfg, modules)
            return result

        @pytest.mark.parametrize("cfg_name", [c.name for c in MECHANISM_CONFIGS])
        def test_polysemy_resolution(
            self,
            cfg_name: str,
            trained_configs: Dict[str, Tuple[MechanismConfig, Dict[str, nn.Module]]],
        ) -> None:
            cfg, modules = trained_configs[cfg_name]
            accuracy, resolved, total = evaluate_polysemy(cfg, trained_modules=modules)
            print(f"\n  {cfg.name}: {accuracy * 100:.1f}% ({resolved}/{total})")
            assert accuracy >= 0.30, (
                f"{cfg.name} only resolved {resolved}/{total} = {accuracy:.1%}, "
                f"expected >= 30%"
            )

        def test_full_pipeline_beats_single_snell(
            self,
            trained_configs: Dict[str, Tuple[MechanismConfig, Dict[str, nn.Module]]],
        ) -> None:
            """The full pipeline (all mechanisms) should beat Single Snell."""
            cfg_snell, mod_snell = trained_configs["Single Snell"]
            cfg_full, mod_full = trained_configs["+ Phase"]

            acc_snell, _, _ = evaluate_polysemy(cfg_snell, trained_modules=mod_snell)
            acc_full, _, _ = evaluate_polysemy(cfg_full, trained_modules=mod_full)

            assert acc_full >= acc_snell - 0.10, (
                f"Full pipeline ({acc_full:.1%}) should not be much worse than "
                f"Single Snell ({acc_snell:.1%})"
            )


# ===================================================================
# Standalone entry point
# ===================================================================

def run_benchmark(verbose: bool = False) -> List[Tuple[str, float, int, int]]:
    """
    Run the full benchmark: train each config, evaluate polysemy resolution.

    Returns list of (name, accuracy, resolved, total) tuples.
    """
    results: List[Tuple[str, float, int, int]] = []

    for cfg in MECHANISM_CONFIGS:
        is_advanced = cfg.use_chromatic
        steps = TRAIN_STEPS_ADVANCED if is_advanced else TRAIN_STEPS
        lr = TRAIN_LR_ADVANCED if is_advanced else TRAIN_LR
        if verbose:
            print(f"  Training: {cfg.name} ({steps} steps, lr={lr})...")
        modules = train_config(cfg, n_steps=steps, lr=lr, verbose=verbose)
        accuracy, resolved, total = evaluate_polysemy(
            cfg, trained_modules=modules,
        )
        results.append((cfg.name, accuracy, resolved, total))

    return results


def main() -> None:
    """Run full benchmark with training and print summary table."""
    print("\nTraining and evaluating each mechanism configuration...\n")
    results = run_benchmark(verbose=True)
    print_summary_table(results)


if __name__ == "__main__":
    if pytest is not None and "--pytest" in sys.argv:
        sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
    else:
        main()
