"""Tests for BVHRouter (bvh_router.py) — core hierarchical routing.

All tests run on CPU — no GPU required.
Run: python -m pytest tests/test_bvh_router.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pytest
import torch
import torch.nn.functional as F

from bvh_router import BVHRouter, RouterConfig, RoutingResult


# ===== Fixtures =====

@pytest.fixture
def cfg():
    return RouterConfig(embed_dim=64, spectral_dim=16,
                        n_level1=4, n_level2=4, n_level3=4)

@pytest.fixture
def router(cfg):
    return BVHRouter(cfg)

@pytest.fixture
def batch():
    torch.manual_seed(42)
    return torch.randn(8, 64)


# ===== RouterConfig =====

class TestRouterConfig:
    def test_expert_count(self):
        cfg = RouterConfig(n_level1=4, n_level2=4, n_level3=4)
        assert cfg.n_experts == 64

    def test_hierarchy_totals(self):
        cfg = RouterConfig(n_level1=3, n_level2=5, n_level3=2)
        assert cfg.total_l1 == 3
        assert cfg.total_l2 == 15
        assert cfg.total_l3 == 30
        assert cfg.n_experts == 30


# ===== Forward Pass Shapes =====

class TestForwardShapes:
    def test_returns_routing_result(self, router, batch):
        router.train()
        result = router(batch)
        assert isinstance(result, RoutingResult)

    def test_expert_id_shape(self, router, batch):
        result = router(batch)
        assert result.expert_id.shape == (8,)

    def test_expert_probs_shape(self, router, batch, cfg):
        result = router(batch)
        assert result.expert_probs.shape == (8, cfg.n_experts)

    def test_route_path_shape(self, router, batch):
        result = router(batch)
        assert result.route_path.shape == (8, 3)

    def test_confidence_shape(self, router, batch):
        result = router(batch)
        assert result.confidence.shape == (8,)

    def test_probs_are_normalized(self, router, batch):
        router.eval()
        result = router(batch, hard=True)
        # One-hot in eval mode → sums to 1
        sums = result.expert_probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(8), atol=1e-5)

    def test_expert_id_in_valid_range(self, router, batch, cfg):
        result = router(batch)
        assert (result.expert_id >= 0).all()
        assert (result.expert_id < cfg.n_experts).all()

    def test_single_sample(self, router, cfg):
        x = torch.randn(1, cfg.embed_dim)
        result = router(x)
        assert result.expert_id.shape == (1,)


# ===== Training vs Inference =====

class TestTrainEval:
    def test_eval_is_deterministic(self, router, batch):
        router.eval()
        r1 = router(batch, hard=True)
        r2 = router(batch, hard=True)
        assert torch.equal(r1.expert_id, r2.expert_id)
        assert torch.equal(r1.expert_probs, r2.expert_probs)

    def test_train_uses_gumbel(self, router, batch):
        """Train mode uses Gumbel-Softmax — probs should NOT be one-hot."""
        router.train()
        result = router(batch)
        # At least some probs should be non-zero for multiple experts
        nonzero_per_sample = (result.expert_probs > 0.01).sum(dim=-1)
        # With Gumbel-Softmax soft, most samples have >1 expert with prob > 0.01
        assert nonzero_per_sample.float().mean() > 1.0

    def test_eval_uses_argmax(self, router, batch):
        """Eval mode should produce one-hot probs (hard routing)."""
        router.eval()
        result = router(batch, hard=True)
        # One-hot: exactly one 1.0 per row
        maxvals = result.expert_probs.max(dim=-1).values
        assert torch.allclose(maxvals, torch.ones(8))


# ===== Gradient Flow =====

class TestGradients:
    def test_gradients_flow_to_input(self, router):
        router.train()
        x = torch.randn(4, 64, requires_grad=True)
        result = router(x)
        loss = result.expert_probs.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_gradients_flow_to_params(self, router, batch):
        router.train()
        result = router(batch)
        loss = result.expert_probs.sum()
        loss.backward()
        # Check at least one parameter has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in router.parameters())
        assert has_grad


# ===== Temperature Annealing =====

class TestTemperature:
    def test_anneal_reduces_temperature(self, router):
        t_before = router.temperature.item()
        router.anneal_temperature()
        t_after = router.temperature.item()
        assert t_after < t_before

    def test_anneal_respects_minimum(self, router, cfg):
        for _ in range(200):
            router.anneal_temperature()
        assert router.temperature.item() >= cfg.temperature_min

    def test_anneal_follows_decay(self, router, cfg):
        t_before = router.temperature.item()
        router.anneal_temperature()
        expected = t_before * cfg.temperature_decay
        assert abs(router.temperature.item() - expected) < 1e-6


# ===== Load Balancing =====

class TestLoadBalancing:
    def test_initial_balance_loss_is_zero(self, router):
        loss = router.load_balancing_loss()
        assert loss.item() == 0.0

    def test_balance_loss_after_forward(self, router, batch):
        router.train()
        router(batch)
        loss = router.load_balancing_loss()
        assert loss.item() >= 0.0

    def test_reset_expert_counts(self, router, batch):
        router.train()
        router(batch)
        assert router.expert_counts.sum() > 0
        router.reset_expert_counts()
        assert router.expert_counts.sum() == 0


# ===== Routing Diversity =====

class TestDiversity:
    def test_different_inputs_different_routes(self, router, cfg):
        """Sufficiently different inputs should route to different experts."""
        router.eval()
        torch.manual_seed(123)
        # Generate widely separated inputs
        x = torch.randn(32, cfg.embed_dim) * 5.0
        result = router(x, hard=True)
        unique_experts = result.expert_id.unique().numel()
        # With 32 diverse inputs and 64 experts, should see at least 2 different routes
        assert unique_experts >= 2, f"Only {unique_experts} unique expert(s) — routing collapsed"
