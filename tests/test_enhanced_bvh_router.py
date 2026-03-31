"""
test_enhanced_bvh_router.py -- Comprehensive tests for EnhancedBVHRouter

Tests the 3-level hierarchical BVH router (4x4x4 = 64 experts) from
python/olmoe_bvh_distill.py. All tests are CPU-only, no GPU required.

Copyright (c) 2026 SpectralAI Studio -- Apache 2.0
"""

import sys
from pathlib import Path
from typing import Tuple

import pytest
import torch
import torch.nn as nn

# Add project python/ to path so we can import the router
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from olmoe_bvh_distill import EnhancedBVHRouter, HierarchicalLevel


# ─────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────

@pytest.fixture
def seed() -> int:
    """Fixed seed for reproducibility."""
    torch.manual_seed(42)
    return 42


@pytest.fixture
def default_router(seed: int) -> EnhancedBVHRouter:
    """Router with default params, spectral_mode=False."""
    return EnhancedBVHRouter(
        input_dim=2048,
        n_level1=4,
        n_level2=4,
        n_level3=4,
        feature_dim=256,
        spectral_mode=False,
    )


@pytest.fixture
def spectral_router(seed: int) -> EnhancedBVHRouter:
    """Router with spectral_mode=True."""
    return EnhancedBVHRouter(
        input_dim=2048,
        n_level1=4,
        n_level2=4,
        n_level3=4,
        feature_dim=256,
        spectral_mode=True,
        spectral_dim=64,
    )


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Batch of 8 samples with input_dim=2048."""
    torch.manual_seed(99)
    return torch.randn(8, 2048)


# ─────────────────────────────────────────────────────────────────
# 1. Initialization tests
# ─────────────────────────────────────────────────────────────────

class TestInitialization:
    """Verify that routers are created with correct structure."""

    def test_default_params_create_valid_router(self, default_router: EnhancedBVHRouter) -> None:
        assert default_router.n_level1 == 4
        assert default_router.n_level2 == 4
        assert default_router.n_level3 == 4
        assert default_router.n_experts == 64
        assert default_router.feature_dim == 256
        assert default_router.spectral_mode is False

    def test_hierarchy_4x4x4_gives_64_experts(self) -> None:
        torch.manual_seed(42)
        router = EnhancedBVHRouter(
            input_dim=2048, n_level1=4, n_level2=4, n_level3=4,
        )
        assert router.n_experts == 4 * 4 * 4
        assert router.n_experts == 64

    def test_custom_hierarchy(self) -> None:
        torch.manual_seed(42)
        router = EnhancedBVHRouter(
            input_dim=2048, n_level1=2, n_level2=8, n_level3=4,
        )
        assert router.n_experts == 2 * 8 * 4
        assert router.n_experts == 64

    def test_spectral_mode_creates_spectral_encoder(self, spectral_router: EnhancedBVHRouter) -> None:
        assert spectral_router.spectral_enabled is True
        assert hasattr(spectral_router, "spectral_encoder")
        assert hasattr(spectral_router, "prismatic_refraction")
        assert hasattr(spectral_router, "spectral_gate")
        assert hasattr(spectral_router, "post_routing_norm")

    def test_non_spectral_has_no_spectral_encoder(self, default_router: EnhancedBVHRouter) -> None:
        assert default_router.spectral_mode is False
        assert not hasattr(default_router, "spectral_encoder")
        assert not hasattr(default_router, "prismatic_refraction")

    def test_encoder_hidden_parameter_respected(self) -> None:
        torch.manual_seed(42)
        router = EnhancedBVHRouter(
            input_dim=2048, spectral_mode=True,
            spectral_dim=64, encoder_hidden=512,
        )
        # spectral_encoder is Sequential: Linear(256, 512), GELU, Linear(512, 64), Tanh
        first_layer = router.spectral_encoder[0]
        assert isinstance(first_layer, nn.Linear)
        assert first_layer.out_features == 512

    def test_encoder_hidden_default_uses_max_of_128_and_spectral_dim(self) -> None:
        torch.manual_seed(42)
        # spectral_dim=64 -> encoder_hidden = max(128, 64) = 128
        router_small = EnhancedBVHRouter(
            input_dim=2048, spectral_mode=True, spectral_dim=64,
        )
        assert router_small.spectral_encoder[0].out_features == 128

        torch.manual_seed(42)
        # spectral_dim=256 -> encoder_hidden = max(128, 256) = 256
        router_large = EnhancedBVHRouter(
            input_dim=2048, spectral_mode=True, spectral_dim=256,
        )
        assert router_large.spectral_encoder[0].out_features == 256

    def test_three_hierarchical_levels_exist(self, default_router: EnhancedBVHRouter) -> None:
        assert isinstance(default_router.level1, HierarchicalLevel)
        assert isinstance(default_router.level2, HierarchicalLevel)
        assert isinstance(default_router.level3, HierarchicalLevel)

    def test_temperature_buffer_initialized(self, default_router: EnhancedBVHRouter) -> None:
        assert hasattr(default_router, "temperature")
        assert default_router.temperature.item() == pytest.approx(1.0)

    def test_expert_counts_buffer_initialized_zero(self, default_router: EnhancedBVHRouter) -> None:
        assert hasattr(default_router, "expert_counts")
        assert default_router.expert_counts.shape == (64,)
        assert default_router.expert_counts.sum().item() == 0.0

    def test_input_proj_structure(self, default_router: EnhancedBVHRouter) -> None:
        proj = default_router.input_proj
        # Linear(2048,512) -> GELU -> Linear(512,256) -> LayerNorm(256)
        assert isinstance(proj[0], nn.Linear)
        assert proj[0].in_features == 2048
        assert proj[0].out_features == 512
        assert isinstance(proj[2], nn.Linear)
        assert proj[2].in_features == 512
        assert proj[2].out_features == 256


# ─────────────────────────────────────────────────────────────────
# 2. Forward pass shape tests
# ─────────────────────────────────────────────────────────────────

class TestForwardPassShapes:
    """Verify output tensor shapes for various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 256])
    def test_output_shapes(self, default_router: EnhancedBVHRouter, batch_size: int) -> None:
        torch.manual_seed(42)
        x = torch.randn(batch_size, 2048)
        default_router.eval()
        probs, expert_ids = default_router(x)

        assert probs.shape == (batch_size, 64)
        assert expert_ids.shape == (batch_size,)

    def test_probs_sum_to_one_eval(self, default_router: EnhancedBVHRouter, sample_input: torch.Tensor) -> None:
        default_router.eval()
        probs, _ = default_router(sample_input)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_expert_ids_in_valid_range(self, default_router: EnhancedBVHRouter, sample_input: torch.Tensor) -> None:
        default_router.eval()
        _, expert_ids = default_router(sample_input)
        assert (expert_ids >= 0).all()
        assert (expert_ids < 64).all()

    def test_last_logits_set_after_forward(
        self, default_router: EnhancedBVHRouter, sample_input: torch.Tensor,
    ) -> None:
        default_router.eval()
        default_router(sample_input)
        assert hasattr(default_router, "_last_logits")
        assert default_router._last_logits.shape == (sample_input.shape[0], 64)

    @pytest.mark.parametrize("batch_size", [1, 32, 256])
    def test_spectral_output_shapes(self, spectral_router: EnhancedBVHRouter, batch_size: int) -> None:
        torch.manual_seed(42)
        x = torch.randn(batch_size, 2048)
        spectral_router.eval()
        probs, expert_ids = spectral_router(x)

        assert probs.shape == (batch_size, 64)
        assert expert_ids.shape == (batch_size,)


# ─────────────────────────────────────────────────────────────────
# 3. Spectral mode tests
# ─────────────────────────────────────────────────────────────────

class TestSpectralMode:
    """Verify spectral encoding affects routing."""

    def test_non_spectral_uses_raw_input(self, default_router: EnhancedBVHRouter) -> None:
        """Non-spectral router should NOT have spectral_encoder."""
        assert not hasattr(default_router, "spectral_encoder")

    def test_spectral_projects_through_encoder(self, spectral_router: EnhancedBVHRouter) -> None:
        """Spectral router should use spectral_encoder during forward."""
        torch.manual_seed(42)
        x = torch.randn(4, 2048)
        spectral_router.eval()

        # Manually check encoder produces output
        h = spectral_router.input_proj(x)
        color = spectral_router.spectral_encoder(h)
        assert color.shape == (4, 64)
        # Tanh output should be in [-1, 1]
        assert color.min() >= -1.0
        assert color.max() <= 1.0

    @pytest.mark.parametrize("spectral_dim", [64, 128, 256])
    def test_different_spectral_dims(self, spectral_dim: int) -> None:
        torch.manual_seed(42)
        router = EnhancedBVHRouter(
            input_dim=2048, spectral_mode=True, spectral_dim=spectral_dim,
        )
        assert router.spectral_dim == spectral_dim
        # Last linear in spectral_encoder outputs spectral_dim
        last_linear = router.spectral_encoder[2]
        assert isinstance(last_linear, nn.Linear)
        assert last_linear.out_features == spectral_dim

    def test_spectral_vs_non_spectral_differ(self) -> None:
        """Same input should produce different outputs with spectral on vs off."""
        torch.manual_seed(42)
        x = torch.randn(4, 2048)

        torch.manual_seed(42)
        router_off = EnhancedBVHRouter(input_dim=2048, spectral_mode=False)
        router_off.eval()

        torch.manual_seed(42)
        router_on = EnhancedBVHRouter(input_dim=2048, spectral_mode=True)
        router_on.eval()

        probs_off, _ = router_off(x)
        probs_on, _ = router_on(x)

        # They should differ because spectral adds extra bias terms
        assert not torch.allclose(probs_off, probs_on, atol=1e-3)


# ─────────────────────────────────────────────────────────────────
# 4. Train vs eval behavior
# ─────────────────────────────────────────────────────────────────

class TestTrainEvalBehavior:
    """Verify training uses Gumbel noise and eval is deterministic."""

    def test_eval_is_deterministic(self, default_router: EnhancedBVHRouter) -> None:
        default_router.eval()
        x = torch.randn(4, 2048)

        probs1, ids1 = default_router(x)
        probs2, ids2 = default_router(x)

        assert torch.allclose(probs1, probs2)
        assert torch.equal(ids1, ids2)

    def test_train_has_stochastic_gumbel(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        x = torch.randn(4, 2048)

        # Run twice -- Gumbel noise should make outputs differ
        probs1, _ = default_router(x)
        probs2, _ = default_router(x)

        # With overwhelming probability, Gumbel noise produces different probs
        assert not torch.allclose(probs1, probs2, atol=1e-6)

    def test_training_mode_updates_expert_counts(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        default_router.reset_expert_counts()
        x = torch.randn(16, 2048)
        default_router(x)
        assert default_router.expert_counts.sum().item() == 16.0

    def test_eval_mode_does_not_update_expert_counts(self, default_router: EnhancedBVHRouter) -> None:
        default_router.eval()
        default_router.reset_expert_counts()
        x = torch.randn(16, 2048)
        default_router(x)
        assert default_router.expert_counts.sum().item() == 0.0


# ─────────────────────────────────────────────────────────────────
# 5. Gradient flow tests
# ─────────────────────────────────────────────────────────────────

class TestGradientFlow:
    """Verify gradients flow through the full hierarchy."""

    def test_gradients_flow_to_all_named_parameters(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        torch.manual_seed(42)
        x = torch.randn(8, 2048)

        probs, _ = default_router(x)
        loss = probs.sum()
        loss.backward()

        params_without_grad = []
        for name, param in default_router.named_parameters():
            if param.requires_grad and param.grad is None:
                params_without_grad.append(name)

        # log_radii don't participate in forward pass (geometric decoration)
        expected_no_grad = {"level1.log_radii", "level2.log_radii", "level3.log_radii"}
        unexpected_no_grad = [n for n in params_without_grad if n not in expected_no_grad]
        assert len(unexpected_no_grad) == 0, (
            f"Unexpected parameters without gradients: {unexpected_no_grad}"
        )

    def test_gradients_flow_to_input_proj(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        x = torch.randn(4, 2048)
        probs, _ = default_router(x)
        probs.sum().backward()

        assert default_router.input_proj[0].weight.grad is not None
        assert default_router.input_proj[0].weight.grad.abs().sum() > 0

    def test_gradients_flow_to_each_level(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        x = torch.randn(4, 2048)
        probs, _ = default_router(x)
        probs.sum().backward()

        for level_name in ["level1", "level2", "level3"]:
            level = getattr(default_router, level_name)
            assert level.centers.grad is not None, f"{level_name}.centers has no grad"
            assert level.to_3d.weight.grad is not None, f"{level_name}.to_3d has no grad"

    def test_spectral_encoder_receives_gradients(self, spectral_router: EnhancedBVHRouter) -> None:
        spectral_router.train()
        torch.manual_seed(42)
        x = torch.randn(8, 2048)
        probs, _ = spectral_router(x)
        probs.sum().backward()

        # Check spectral_encoder layers have gradients
        for i, layer in enumerate(spectral_router.spectral_encoder):
            if hasattr(layer, "weight"):
                assert layer.weight.grad is not None, (
                    f"spectral_encoder[{i}].weight has no grad"
                )
                assert layer.weight.grad.abs().sum() > 0

    def test_spectral_gate_receives_gradients(self, spectral_router: EnhancedBVHRouter) -> None:
        spectral_router.train()
        x = torch.randn(4, 2048)
        probs, _ = spectral_router(x)
        probs.sum().backward()

        assert spectral_router.spectral_gate.weight.grad is not None
        assert spectral_router.spectral_gate.weight.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────
# 6. Load balancing tests
# ─────────────────────────────────────────────────────────────────

class TestLoadBalancing:
    """Verify load balancing loss and expert count tracking."""

    def test_load_balancing_loss_returns_scalar(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        x = torch.randn(64, 2048)
        default_router(x)
        loss = default_router.load_balancing_loss()

        assert loss.dim() == 0  # scalar
        assert loss.dtype == torch.float32

    def test_load_balancing_loss_positive_after_forward(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        default_router.reset_expert_counts()
        x = torch.randn(64, 2048)
        default_router(x)
        loss = default_router.load_balancing_loss()

        # Unless perfectly balanced (astronomically unlikely), loss > 0
        assert loss.item() > 0.0

    def test_load_balancing_loss_zero_when_no_counts(self, default_router: EnhancedBVHRouter) -> None:
        default_router.reset_expert_counts()
        loss = default_router.load_balancing_loss()
        assert loss.item() == 0.0

    def test_reset_expert_counts_zeros_all(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        x = torch.randn(32, 2048)
        default_router(x)
        assert default_router.expert_counts.sum().item() > 0

        default_router.reset_expert_counts()
        assert default_router.expert_counts.sum().item() == 0.0

    def test_expert_counts_accumulate_across_batches(self, default_router: EnhancedBVHRouter) -> None:
        default_router.train()
        default_router.reset_expert_counts()

        x1 = torch.randn(10, 2048)
        x2 = torch.randn(20, 2048)
        default_router(x1)
        default_router(x2)

        assert default_router.expert_counts.sum().item() == 30.0


# ─────────────────────────────────────────────────────────────────
# 7. Temperature annealing tests
# ─────────────────────────────────────────────────────────────────

class TestTemperatureAnnealing:
    """Verify temperature annealing and its effect on routing."""

    def test_anneal_temperature_decreases(self, default_router: EnhancedBVHRouter) -> None:
        initial = default_router.temperature.item()
        default_router.anneal_temperature(decay=0.95)
        after = default_router.temperature.item()

        assert after < initial
        assert after == pytest.approx(initial * 0.95)

    def test_anneal_temperature_clamps_at_minimum(self, default_router: EnhancedBVHRouter) -> None:
        # Anneal many times -- should not go below 0.1
        for _ in range(200):
            default_router.anneal_temperature(decay=0.9)
        assert default_router.temperature.item() >= 0.1

    def test_anneal_temperature_custom_decay(self, default_router: EnhancedBVHRouter) -> None:
        default_router.anneal_temperature(decay=0.5)
        assert default_router.temperature.item() == pytest.approx(0.5)

    def test_lower_temperature_produces_sharper_routing(self, default_router: EnhancedBVHRouter) -> None:
        default_router.eval()
        torch.manual_seed(42)
        x = torch.randn(16, 2048)

        # High temperature
        default_router.temperature.fill_(2.0)
        probs_warm, _ = default_router(x)
        entropy_warm = -(probs_warm * (probs_warm + 1e-10).log()).sum(dim=-1).mean()

        # Low temperature
        default_router.temperature.fill_(0.1)
        probs_cold, _ = default_router(x)
        entropy_cold = -(probs_cold * (probs_cold + 1e-10).log()).sum(dim=-1).mean()

        # Lower temperature = lower entropy (sharper distribution)
        assert entropy_cold < entropy_warm


# ─────────────────────────────────────────────────────────────────
# 8. State dict save/load tests
# ─────────────────────────────────────────────────────────────────

class TestStateDictSaveLoad:
    """Verify model can be serialized and deserialized."""

    def test_save_and_reload_state_dict(self, default_router: EnhancedBVHRouter) -> None:
        state = default_router.state_dict()

        torch.manual_seed(99)
        new_router = EnhancedBVHRouter(
            input_dim=2048, n_level1=4, n_level2=4, n_level3=4, feature_dim=256,
        )
        new_router.load_state_dict(state)

        # Same weights → same output
        new_router.eval()
        default_router.eval()
        x = torch.randn(4, 2048)
        probs_orig, ids_orig = default_router(x)
        probs_loaded, ids_loaded = new_router(x)

        assert torch.allclose(probs_orig, probs_loaded, atol=1e-6)
        assert torch.equal(ids_orig, ids_loaded)

    def test_spectral_save_and_reload(self, spectral_router: EnhancedBVHRouter) -> None:
        state = spectral_router.state_dict()

        torch.manual_seed(99)
        new_router = EnhancedBVHRouter(
            input_dim=2048, n_level1=4, n_level2=4, n_level3=4,
            feature_dim=256, spectral_mode=True, spectral_dim=64,
        )
        new_router.load_state_dict(state)

        new_router.eval()
        spectral_router.eval()
        torch.manual_seed(42)
        x = torch.randn(4, 2048)
        probs_orig, _ = spectral_router(x)
        probs_loaded, _ = new_router(x)

        assert torch.allclose(probs_orig, probs_loaded, atol=1e-6)

    def test_state_dict_contains_all_keys(self, default_router: EnhancedBVHRouter) -> None:
        state = default_router.state_dict()

        # Check critical keys exist
        expected_prefixes = [
            "input_proj.",
            "level1.",
            "level2.",
            "level3.",
            "expert_head.",
        ]
        for prefix in expected_prefixes:
            matching = [k for k in state if k.startswith(prefix)]
            assert len(matching) > 0, f"No keys with prefix '{prefix}' in state_dict"

        # Buffers
        assert "temperature" in state
        assert "expert_counts" in state

    def test_spectral_state_dict_extra_keys(self, spectral_router: EnhancedBVHRouter) -> None:
        state = spectral_router.state_dict()

        spectral_prefixes = [
            "spectral_encoder.",
            "prismatic_refraction.",
            "spectral_gate.",
            "post_routing_norm.",
        ]
        for prefix in spectral_prefixes:
            matching = [k for k in state if k.startswith(prefix)]
            assert len(matching) > 0, f"No keys with prefix '{prefix}' in state_dict"

    def test_temperature_persists_in_state_dict(self, default_router: EnhancedBVHRouter) -> None:
        default_router.anneal_temperature(decay=0.5)  # temp = 0.5
        state = default_router.state_dict()
        assert "temperature" in state
        assert state["temperature"].item() == pytest.approx(0.5)

        # Reload into identical architecture (must match fixture params)
        new_router = EnhancedBVHRouter(
            input_dim=2048, n_level1=4, n_level2=4, n_level3=4,
            feature_dim=256, spectral_mode=False,
        )
        new_router.load_state_dict(state)
        assert new_router.temperature.item() == pytest.approx(0.5)


# ─────────────────────────────────────────────────────────────────
# 9. HierarchicalLevel unit tests
# ─────────────────────────────────────────────────────────────────

class TestHierarchicalLevel:
    """Unit tests for the individual HierarchicalLevel module."""

    def test_output_shapes(self) -> None:
        torch.manual_seed(42)
        level = HierarchicalLevel(input_dim=256, n_children=4, feature_dim=128)
        x = torch.randn(8, 256)
        probs, features, pos_3d = level(x)

        assert probs.shape == (8, 4)
        assert features.shape == (8, 128)
        assert pos_3d.shape == (8, 3)

    def test_spectral_mode_creates_smooth_hit(self) -> None:
        torch.manual_seed(42)
        level = HierarchicalLevel(
            input_dim=256, n_children=4, feature_dim=128, spectral_mode=True,
        )
        assert hasattr(level, "smooth_hit")

    def test_non_spectral_has_no_smooth_hit(self) -> None:
        torch.manual_seed(42)
        level = HierarchicalLevel(
            input_dim=256, n_children=4, feature_dim=128, spectral_mode=False,
        )
        assert not hasattr(level, "smooth_hit")

    def test_eval_probs_sum_to_one(self) -> None:
        torch.manual_seed(42)
        level = HierarchicalLevel(input_dim=256, n_children=4, feature_dim=128)
        level.eval()
        x = torch.randn(8, 256)
        probs, _, _ = level(x)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_train_uses_gumbel(self) -> None:
        torch.manual_seed(42)
        level = HierarchicalLevel(input_dim=256, n_children=4, feature_dim=128)
        level.train()
        x = torch.randn(4, 256)
        p1, _, _ = level(x)
        p2, _, _ = level(x)
        # Gumbel noise should produce different results
        assert not torch.allclose(p1, p2, atol=1e-6)


# ─────────────────────────────────────────────────────────────────
# 10. Edge cases
# ─────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Edge cases and robustness checks."""

    def test_single_sample_batch(self, default_router: EnhancedBVHRouter) -> None:
        default_router.eval()
        x = torch.randn(1, 2048)
        probs, expert_ids = default_router(x)
        assert probs.shape == (1, 64)
        assert expert_ids.shape == (1,)

    def test_zero_input(self, default_router: EnhancedBVHRouter) -> None:
        default_router.eval()
        x = torch.zeros(4, 2048)
        probs, expert_ids = default_router(x)
        # Should not produce NaN or Inf
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()

    def test_large_input_values(self, default_router: EnhancedBVHRouter) -> None:
        default_router.eval()
        x = torch.randn(4, 2048) * 100.0
        probs, expert_ids = default_router(x)
        assert not torch.isnan(probs).any()
        assert not torch.isinf(probs).any()

    def test_probs_non_negative(self, default_router: EnhancedBVHRouter) -> None:
        default_router.eval()
        x = torch.randn(16, 2048)
        probs, _ = default_router(x)
        assert (probs >= 0).all()

    def test_custom_temperature_init(self) -> None:
        torch.manual_seed(42)
        router = EnhancedBVHRouter(input_dim=2048, temperature_init=0.5)
        assert router.temperature.item() == pytest.approx(0.5)
