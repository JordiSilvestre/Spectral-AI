#!/usr/bin/env python3
"""
test_checkpoint_paths.py -- Verify checkpoint path resolution logic.

Ensures that per-layer checkpoints (olmoe_distill_layer8/) are found
before the global fallback (olmoe_distill/bvh_router_best.pt).

This test exists because a critical bug (2026-04-02) caused ALL layers
to use the global checkpoint instead of per-layer ones, destroying
model quality (HellaSwag dropped to random 25.8%).

Root cause: scripts searched olmoe_distill/layer8/ (wrong) instead of
olmoe_distill_layer8/ (correct sibling directory naming).
"""

import os
import tempfile

import pytest


def resolve_checkpoint_path(
    router_dir: str, layer_idx: int
) -> str:
    """Resolve checkpoint path using the same logic as eval scripts.

    This is the canonical path resolution extracted from eval_hellaswag.py
    and sweep_prefilter.py. Any changes to those scripts MUST be reflected here.
    """
    base_dir = os.path.dirname(router_dir)
    base_name = os.path.basename(router_dir)

    # 1. Per-layer sibling directory (CORRECT primary path)
    ckpt_path = os.path.join(
        base_dir, f"{base_name}_layer{layer_idx}", "bvh_router_best.pt"
    )
    if os.path.exists(ckpt_path):
        return ckpt_path

    # 2. Subdirectory format (legacy fallback)
    ckpt_path = os.path.join(
        router_dir, f"layer{layer_idx}", "bvh_router_best.pt"
    )
    if os.path.exists(ckpt_path):
        return ckpt_path

    # 3. Global checkpoint (last resort)
    ckpt_path = os.path.join(router_dir, "bvh_router_best.pt")
    if os.path.exists(ckpt_path):
        return ckpt_path

    return None


@pytest.fixture
def checkpoint_tree(tmp_path):
    """Create a realistic checkpoint directory structure."""
    # Per-layer checkpoints (correct naming)
    for layer in range(16):
        layer_dir = tmp_path / f"olmoe_distill_layer{layer}"
        layer_dir.mkdir()
        (layer_dir / "bvh_router_best.pt").write_text(f"layer_{layer}")

    # Global checkpoint (fallback)
    global_dir = tmp_path / "olmoe_distill"
    global_dir.mkdir()
    (global_dir / "bvh_router_best.pt").write_text("global")

    return tmp_path


class TestCheckpointPathResolution:
    """Verify per-layer checkpoints are found before global fallback."""

    def test_per_layer_found_first(self, checkpoint_tree):
        """Per-layer checkpoint MUST be selected over global."""
        router_dir = str(checkpoint_tree / "olmoe_distill")
        for layer_idx in range(16):
            path = resolve_checkpoint_path(router_dir, layer_idx)
            assert path is not None, f"No checkpoint found for layer {layer_idx}"
            assert f"_layer{layer_idx}" in path, (
                f"Layer {layer_idx}: expected per-layer path, got global: {path}"
            )

    def test_per_layer_content_is_correct(self, checkpoint_tree):
        """Each layer gets its own checkpoint, not the global one."""
        router_dir = str(checkpoint_tree / "olmoe_distill")
        for layer_idx in range(16):
            path = resolve_checkpoint_path(router_dir, layer_idx)
            content = open(path).read()
            assert content == f"layer_{layer_idx}", (
                f"Layer {layer_idx}: got content '{content}', "
                f"expected 'layer_{layer_idx}' — wrong checkpoint loaded!"
            )

    def test_global_fallback_when_no_per_layer(self, tmp_path):
        """Falls back to global when per-layer doesn't exist."""
        global_dir = tmp_path / "olmoe_distill"
        global_dir.mkdir()
        (global_dir / "bvh_router_best.pt").write_text("global")

        router_dir = str(global_dir)
        path = resolve_checkpoint_path(router_dir, 8)
        assert path is not None
        content = open(path).read()
        assert content == "global"

    def test_none_when_nothing_exists(self, tmp_path):
        """Returns None when no checkpoint exists at all."""
        router_dir = str(tmp_path / "nonexistent")
        os.makedirs(router_dir, exist_ok=True)
        path = resolve_checkpoint_path(router_dir, 8)
        assert path is None

    def test_all_16_layers_are_distinct(self, checkpoint_tree):
        """No two layers should resolve to the same checkpoint file."""
        router_dir = str(checkpoint_tree / "olmoe_distill")
        paths = set()
        for layer_idx in range(16):
            path = resolve_checkpoint_path(router_dir, layer_idx)
            assert path not in paths, (
                f"Layer {layer_idx} resolved to same path as another layer: {path}"
            )
            paths.add(path)
        assert len(paths) == 16


class TestRealCheckpoints:
    """Verify actual checkpoint structure on disk (skip if not available)."""

    @pytest.mark.skipif(
        not os.path.isdir("checkpoints/olmoe_distill"),
        reason="Checkpoints not available in this environment",
    )
    def test_real_per_layer_checkpoints_exist(self):
        """All 16 per-layer checkpoints should exist."""
        missing = []
        for layer_idx in range(16):
            path = resolve_checkpoint_path("checkpoints/olmoe_distill", layer_idx)
            if path is None:
                missing.append(layer_idx)
            elif "global" in open(path).read() if os.path.getsize(path) < 100 else False:
                missing.append(layer_idx)
        assert not missing, f"Missing per-layer checkpoints for layers: {missing}"

    @pytest.mark.skipif(
        not os.path.isdir("checkpoints/olmoe_distill"),
        reason="Checkpoints not available in this environment",
    )
    def test_real_paths_are_per_layer_not_global(self):
        """Real checkpoint resolution must pick per-layer, not global."""
        router_dir = "checkpoints/olmoe_distill"
        for layer_idx in range(16):
            path = resolve_checkpoint_path(router_dir, layer_idx)
            if path is not None:
                assert f"_layer{layer_idx}" in path or f"layer{layer_idx}" in os.path.dirname(path), (
                    f"Layer {layer_idx} resolved to global checkpoint: {path}"
                )
