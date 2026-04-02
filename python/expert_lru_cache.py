#!/usr/bin/env python3
"""
expert_lru_cache.py — LRU Cache for Ternary Expert Modules

Manages GPU memory by keeping only the top-k most recently used experts
on the GPU. Cache misses trigger CPU→GPU transfer with LRU eviction.

Like a GPU texture cache: only the active "textures" (experts) live in
VRAM. The rest stay in system RAM, ready to be loaded on demand.

Usage:
    cache = ExpertLRUCache(expert_modules, max_gpu_slots=13, device=device)
    expert = cache.get(expert_id)            # single expert
    experts = cache.get_multi([58, 57, 40])  # batch for top-k

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

from collections import OrderedDict
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class ExpertLRUCache:
    """
    LRU cache for expert modules on GPU.

    Keeps at most `max_gpu_slots` experts on the target device (GPU).
    All other experts remain on CPU. On cache miss, the requested expert
    is moved to GPU and the least-recently-used expert is evicted to CPU.

    Memory budget: max_gpu_slots × expert_size_bytes
    - For int8 experts (~6 MB each): 13 slots ≈ 80 MB
    - For CUDA packed (~0.77 MB each): 64 slots = all fit

    The cache does NOT own the modules — they remain in the caller's
    nn.ModuleDict. The cache only manages device placement.
    """

    def __init__(
        self,
        expert_modules: nn.ModuleDict,
        max_gpu_slots: int = 13,
        device: torch.device = torch.device("cuda"),
    ):
        self._modules = expert_modules
        self._max_slots = max_gpu_slots
        self._device = device

        # OrderedDict preserves insertion order — last = most recent
        # Key: expert_id (str), Value: True (presence marker)
        self._gpu_resident: OrderedDict = OrderedDict()

        # Stats for monitoring
        self._hits = 0
        self._misses = 0
        self._evictions = 0

        # Move ALL experts to CPU initially
        for key, module in self._modules.items():
            module.cpu()

    def get(self, expert_id: int) -> nn.Module:
        """
        Get an expert module on the target GPU device.
        Loads from CPU on cache miss, evicts LRU if cache is full.
        """
        key = str(expert_id)
        if key not in self._modules:
            raise KeyError(f"Expert {expert_id} not found in module registry")

        if key in self._gpu_resident:
            # Cache hit — move to end (most recently used)
            self._gpu_resident.move_to_end(key)
            self._hits += 1
        else:
            # Cache miss — load to GPU
            self._misses += 1
            self._ensure_slot()
            self._modules[key].to(self._device)
            self._gpu_resident[key] = True

        return self._modules[key]

    def get_multi(self, expert_ids: List[int]) -> List[nn.Module]:
        """
        Get multiple experts on GPU (for top-k routing).
        Returns list of modules in the same order as expert_ids.
        """
        return [self.get(eid) for eid in expert_ids]

    def _ensure_slot(self):
        """Evict LRU expert to CPU if cache is full."""
        while len(self._gpu_resident) >= self._max_slots:
            # Pop oldest (least recently used)
            evict_key, _ = self._gpu_resident.popitem(last=False)
            self._modules[evict_key].cpu()
            self._evictions += 1

    def preload(self, expert_ids: List[int]):
        """
        Pre-load a set of experts onto GPU (e.g., at pipeline init).
        Useful for warming the cache before generation starts.
        """
        for eid in expert_ids:
            self.get(eid)

    def stats(self) -> Dict:
        """Cache performance statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": f"{hit_rate:.1%}",
            "gpu_resident": len(self._gpu_resident),
            "max_slots": self._max_slots,
            "gpu_resident_ids": list(self._gpu_resident.keys()),
        }

    def reset_stats(self):
        """Reset hit/miss counters."""
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def gpu_memory_mb(self) -> float:
        """Estimated GPU memory used by cached experts."""
        total_bytes = 0
        for key in self._gpu_resident:
            module = self._modules[key]
            if hasattr(module, 'memory_bytes'):
                total_bytes += module.memory_bytes()
            else:
                # Estimate from buffers
                for buf in module.buffers():
                    total_bytes += buf.numel() * buf.element_size()
        return total_bytes / (1024 * 1024)
