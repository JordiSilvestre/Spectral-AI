#!/usr/bin/env python3
"""
multi_domain_dataset.py — Dataset Multi-Dominio para LiquidBit v5.0 FASE 4
4 dominios con etiquetas para supervisar el routing.

Dominios:
  0 = General (WikiText-2)
  1 = Codigo Python (codeparrot subset)
  2 = Ciencia (abstracts de papers)
  3 = Legal (contratos / leyes)

Cada muestra devuelve (tokens, domain_id).
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from typing import Tuple

CACHE_DIR = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────
# Tokenizer compartido (GPT-2 BPE via tiktoken)
# ─────────────────────────────────────────────────────────────────

def get_encoder():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


# ─────────────────────────────────────────────────────────────────
# Dataset base por dominio
# ─────────────────────────────────────────────────────────────────

class DomainDataset(Dataset):
    """Dataset de un solo dominio con etiqueta."""

    def __init__(
        self,
        domain_id: int,
        domain_name: str,
        tokens: np.ndarray,
        seq_len: int = 256,
    ):
        self.domain_id = domain_id
        self.domain_name = domain_name
        self.seq_len = seq_len
        self.tokens = tokens
        stride = seq_len // 2
        self.stride = stride
        self.n_samples = max(1, (len(tokens) - seq_len - 1) // stride)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, i) -> Tuple[torch.Tensor, int]:
        start = i * self.stride
        end = start + self.seq_len + 1
        chunk = self.tokens[start:end].astype(np.int64)
        return torch.from_numpy(chunk), self.domain_id


# ─────────────────────────────────────────────────────────────────
# Loaders por dominio
# ─────────────────────────────────────────────────────────────────

def load_wikitext(split: str = "train") -> np.ndarray:
    """WikiText-2 — dominio general."""
    cache = CACHE_DIR / f"wikitext2_{split}_tokens.npy"
    if cache.exists():
        return np.load(str(cache))

    print(f"[Wiki] Tokenizando WikiText-2 {split}...")
    from datasets import load_dataset
    enc = get_encoder()
    # WikiText-2 uses "validation" not "val"
    hf_split = "validation" if split == "val" else split
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=hf_split)
    all_tokens = []
    for row in ds:
        text = row["text"].strip()
        if text:
            all_tokens.extend(enc.encode(text))
    tokens = np.array(all_tokens, dtype=np.int32)
    np.save(str(cache), tokens)
    return tokens


def load_code(split: str = "train", max_tokens: int = 2_500_000) -> np.ndarray:
    """Codigo Python — subset de codeparrot o generado sinteticamente."""
    cache = CACHE_DIR / f"code_{split}_tokens.npy"
    if cache.exists():
        return np.load(str(cache))

    print(f"[Code] Generando dataset de codigo Python {split}...")
    enc = get_encoder()

    # Generar codigo sintetico variado (no depende de HuggingFace)
    # Esto es suficiente para demostrar que el router discrimina
    code_templates = [
        # Funciones basicas
        "def {fn}({args}):\n    {body}\n    return {ret}\n",
        # Clases
        "class {cls}:\n    def __init__(self, {args}):\n        {init}\n\n    def {fn}(self):\n        {body}\n",
        # Imports + uso
        "import {mod}\n\n{var} = {mod}.{fn}({args})\nprint({var})\n",
        # List comprehension
        "{var} = [{expr} for {it} in range({n})]\n",
        # Try/except
        "try:\n    {body}\nexcept {exc} as e:\n    print(f'Error: {{e}}')\n",
        # Async
        "async def {fn}({args}):\n    result = await {call}\n    return result\n",
        # Decorators
        "@{dec}\ndef {fn}({args}):\n    {body}\n",
        # Dict/JSON
        "{var} = {{\n    '{k1}': {v1},\n    '{k2}': {v2},\n}}\n",
    ]

    fns = ["calculate", "process", "transform", "validate", "parse", "encode",
           "decode", "serialize", "compress", "optimize", "fetch", "query",
           "filter_data", "sort_items", "merge_lists", "build_tree",
           "train_model", "predict", "evaluate", "load_config"]
    classes = ["DataProcessor", "ModelTrainer", "APIClient", "FileHandler",
               "CacheManager", "TokenParser", "GraphBuilder", "EventLoop"]
    mods = ["numpy", "torch", "json", "os", "sys", "math", "re", "pathlib",
            "collections", "itertools", "functools", "typing", "dataclasses"]
    vars_ = ["result", "data", "output", "items", "values", "config", "state"]
    args = ["x", "data", "config", "n=10", "verbose=False", "**kwargs",
            "self, x, y", "arr, target", "model, input_data"]
    bodies = ["x = x + 1", "data.append(x)", "result = sum(items)",
              "output = [x**2 for x in data]", "config.update(kwargs)",
              "return sorted(items, key=lambda x: x.value)"]
    excs = ["ValueError", "KeyError", "TypeError", "IOError", "RuntimeError"]

    rng = np.random.RandomState(42 if split == "train" else 123)
    all_tokens = []

    while len(all_tokens) < max_tokens:
        tmpl = rng.choice(code_templates)
        code = tmpl.format(
            fn=rng.choice(fns), cls=rng.choice(classes), mod=rng.choice(mods),
            var=rng.choice(vars_), args=rng.choice(args), body=rng.choice(bodies),
            ret=rng.choice(vars_), init="self.data = []", call="fetch(url)",
            dec="staticmethod", k1="name", k2="value", v1="42", v2="'hello'",
            exc=rng.choice(excs), expr="x**2", it="x", n=str(rng.randint(5, 100)),
        )
        all_tokens.extend(enc.encode(code))

    tokens = np.array(all_tokens[:max_tokens], dtype=np.int32)
    np.save(str(cache), tokens)
    print(f"[Code] {len(tokens):,} tokens generados")
    return tokens


def load_science(split: str = "train", max_tokens: int = 2_500_000) -> np.ndarray:
    """Texto cientifico — abstracts sinteticos de papers."""
    cache = CACHE_DIR / f"science_{split}_tokens.npy"
    if cache.exists():
        return np.load(str(cache))

    print(f"[Science] Generando dataset cientifico {split}...")
    enc = get_encoder()

    topics = [
        ("quantum computing", "qubit", "entanglement", "decoherence", "gate fidelity"),
        ("machine learning", "gradient descent", "backpropagation", "loss function", "overfitting"),
        ("neuroscience", "neural pathway", "synaptic plasticity", "cortical region", "dopamine"),
        ("astrophysics", "black hole", "neutron star", "gravitational wave", "dark matter"),
        ("molecular biology", "protein folding", "gene expression", "CRISPR", "RNA sequencing"),
        ("materials science", "crystal structure", "thermal conductivity", "tensile strength", "alloy"),
        ("climate science", "carbon dioxide", "temperature anomaly", "ice core", "radiative forcing"),
        ("particle physics", "Higgs boson", "muon", "cross-section", "luminosity"),
    ]

    templates = [
        "We present a novel approach to {topic} that achieves state-of-the-art results. "
        "Our method leverages {t1} and {t2} to improve {t3} by {pct}%. "
        "Experiments on {n} datasets demonstrate the effectiveness of our approach. "
        "The key insight is that {t4} can be combined with {t1} to reduce {t2} overhead.\n\n",

        "In this paper, we investigate the relationship between {t1} and {t2} in the context of {topic}. "
        "Our theoretical analysis shows that {t3} scales as O(log n) with respect to {t4}. "
        "We validate our findings through extensive experiments on synthetic and real-world data. "
        "Results indicate a {pct}% improvement over existing baselines.\n\n",

        "Abstract: {topic} remains a fundamental challenge in modern science. "
        "We propose a framework based on {t1} that addresses the limitations of {t2}. "
        "The framework incorporates {t3} and {t4} to achieve robust performance. "
        "Our experiments show {pct}% accuracy on standard benchmarks with {n}x speedup.\n\n",

        "Recent advances in {topic} have enabled new applications in {t1} and {t2}. "
        "However, existing methods suffer from {t3} degradation at scale. "
        "We introduce a novel {t4} mechanism that maintains performance "
        "while reducing computational cost by {pct}%. "
        "Evaluation on {n} benchmarks confirms the superiority of our method.\n\n",
    ]

    rng = np.random.RandomState(42 if split == "train" else 123)
    all_tokens = []

    while len(all_tokens) < max_tokens:
        topic_group = topics[rng.randint(len(topics))]
        tmpl = rng.choice(templates)
        text = tmpl.format(
            topic=topic_group[0],
            t1=topic_group[rng.randint(1, len(topic_group))],
            t2=topic_group[rng.randint(1, len(topic_group))],
            t3=topic_group[rng.randint(1, len(topic_group))],
            t4=topic_group[rng.randint(1, len(topic_group))],
            pct=rng.randint(5, 95),
            n=rng.randint(3, 20),
        )
        all_tokens.extend(enc.encode(text))

    tokens = np.array(all_tokens[:max_tokens], dtype=np.int32)
    np.save(str(cache), tokens)
    print(f"[Science] {len(tokens):,} tokens generados")
    return tokens


def load_legal(split: str = "train", max_tokens: int = 2_500_000) -> np.ndarray:
    """Texto legal — contratos y leyes sinteticos."""
    cache = CACHE_DIR / f"legal_{split}_tokens.npy"
    if cache.exists():
        return np.load(str(cache))

    print(f"[Legal] Generando dataset legal {split}...")
    enc = get_encoder()

    templates = [
        "SECTION {sec}. {title}.\n"
        "(a) The {party} shall {obligation} in accordance with the provisions "
        "set forth in Article {art} of this Agreement. "
        "Any {violation} of this section shall constitute a material breach.\n"
        "(b) Notwithstanding the foregoing, the {party} may {exception} "
        "provided that written notice is given {days} days in advance.\n\n",

        "ARTICLE {art}. REPRESENTATIONS AND WARRANTIES.\n"
        "The {party} hereby represents and warrants that:\n"
        "(i) it has full power and authority to execute this {doc};\n"
        "(ii) the execution and delivery of this {doc} does not violate "
        "any {law} applicable to the {party};\n"
        "(iii) all {disclosure} provided to the {counterparty} is true, "
        "accurate, and complete in all material respects.\n\n",

        "WHEREAS, the {party} desires to {purpose}; and\n"
        "WHEREAS, the {counterparty} has agreed to {counter_purpose} "
        "subject to the terms and conditions herein;\n"
        "NOW, THEREFORE, in consideration of the mutual covenants "
        "contained herein, the parties agree as follows:\n\n",

        "LIMITATION OF LIABILITY. IN NO EVENT SHALL THE {party} "
        "BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR "
        "CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION "
        "WITH THIS {doc}, WHETHER BASED ON WARRANTY, CONTRACT, "
        "TORT, OR ANY OTHER LEGAL THEORY. THE AGGREGATE LIABILITY "
        "SHALL NOT EXCEED ${amount}.\n\n",
    ]

    parties = ["Licensor", "Licensee", "Company", "Contractor", "Client",
               "Employer", "Employee", "Seller", "Buyer", "Tenant"]
    obligations = ["deliver the goods", "provide the services",
                   "maintain confidentiality", "indemnify the other party",
                   "comply with applicable laws", "pay the agreed compensation"]
    docs = ["Agreement", "Contract", "License", "Memorandum", "Deed"]
    laws = ["federal regulation", "state statute", "municipal ordinance",
            "EU directive", "international treaty"]
    purposes = ["license the intellectual property", "acquire the assets",
                "establish a joint venture", "retain professional services"]

    rng = np.random.RandomState(42 if split == "train" else 123)
    all_tokens = []

    while len(all_tokens) < max_tokens:
        tmpl = rng.choice(templates)
        text = tmpl.format(
            sec=rng.randint(1, 30),
            title=rng.choice(["DEFINITIONS", "SCOPE", "TERM", "PAYMENT",
                              "TERMINATION", "CONFIDENTIALITY", "INDEMNIFICATION"]),
            party=rng.choice(parties),
            counterparty=rng.choice(parties),
            obligation=rng.choice(obligations),
            art=rng.randint(1, 15),
            violation="breach",
            exception="terminate this agreement",
            days=rng.choice([30, 60, 90]),
            doc=rng.choice(docs),
            law=rng.choice(laws),
            disclosure="information",
            purpose=rng.choice(purposes),
            counter_purpose=rng.choice(purposes),
            amount=f"{rng.randint(1, 100) * 10_000:,}",
        )
        all_tokens.extend(enc.encode(text))

    tokens = np.array(all_tokens[:max_tokens], dtype=np.int32)
    np.save(str(cache), tokens)
    print(f"[Legal] {len(tokens):,} tokens generados")
    return tokens


# ─────────────────────────────────────────────────────────────────
# Dataset Multi-Dominio combinado
# ─────────────────────────────────────────────────────────────────

DOMAIN_NAMES = ["general", "code", "science", "legal"]
N_DOMAINS = len(DOMAIN_NAMES)


def create_multi_domain_dataset(
    split: str = "train",
    seq_len: int = 256,
    max_tokens_per_domain: int = 2_500_000,
) -> ConcatDataset:
    """Crear dataset combinado de 4 dominios."""

    loaders = [
        (0, "general", load_wikitext),
        (1, "code",    load_code),
        (2, "science", load_science),
        (3, "legal",   load_legal),
    ]

    datasets = []
    for domain_id, name, loader_fn in loaders:
        if name == "general":
            tokens = loader_fn(split)
        else:
            tokens = loader_fn(split, max_tokens=max_tokens_per_domain)
        ds = DomainDataset(domain_id, name, tokens, seq_len)
        datasets.append(ds)
        print(f"  [{name}] {len(tokens):,} tokens, {len(ds)} muestras")

    combined = ConcatDataset(datasets)
    print(f"  [TOTAL] {len(combined)} muestras")
    return combined


def collate_with_domain(batch):
    """Collate que separa tokens y domain_ids."""
    tokens = torch.stack([b[0] for b in batch])
    domains = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return tokens, domains


# ─────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Creando Multi-Domain Dataset ===\n")
    ds = create_multi_domain_dataset("train", seq_len=256)

    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_with_domain)

    batch_tokens, batch_domains = next(iter(dl))
    print(f"\nBatch tokens: {batch_tokens.shape}")
    print(f"Batch domains: {batch_domains.tolist()}")

    # Verificar distribucion
    from collections import Counter
    domain_counts = Counter()
    for i in range(len(ds)):
        _, d = ds[i]
        domain_counts[d] += 1
    print(f"\nDistribucion por dominio:")
    for d in range(N_DOMAINS):
        print(f"  {DOMAIN_NAMES[d]}: {domain_counts[d]:,} muestras")
