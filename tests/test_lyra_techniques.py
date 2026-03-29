"""Tests for Lyra-AGI techniques adapted for LiquidBit.

All tests run on CPU — no GPU required.
Run: python -m pytest tests/test_lyra_techniques.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from lyra_techniques import (
    SmoothTernarySTE, ternary_ste,
    get_ste_beta, set_ste_beta, BetaScheduler,
    SmoothBVHHit,
    RMSNorm,
    LiquidTimeGate,
    get_dual_lr_param_groups,
    MetabolicBVH,
)


# ===== SmoothTernarySTE Tests =====

class TestSmoothSTE:
    def setup_method(self):
        self._orig_beta = get_ste_beta()

    def teardown_method(self):
        set_ste_beta(self._orig_beta)

    def test_hard_ternary_at_high_beta(self):
        """High beta → hard ternary {-1, 0, +1}."""
        set_ste_beta(20.0)
        x = torch.tensor([-0.8, -0.3, 0.0, 0.3, 0.8])
        t = ternary_ste(x)
        assert t.tolist() == [-1.0, 0.0, 0.0, 0.0, 1.0]

    def test_soft_at_low_beta(self):
        """Low beta → smooth values between 0 and 1."""
        set_ste_beta(1.0)
        x = torch.tensor([0.8], requires_grad=True)
        t = ternary_ste(x)
        assert 0.0 < t.item() < 1.0, f"Expected soft value, got {t.item()}"

    def test_gradient_flows(self):
        """STE must pass gradients through."""
        set_ste_beta(2.0)
        x = torch.tensor([0.7, -0.7, 0.1], requires_grad=True)
        t = ternary_ste(x)
        loss = t.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0, "Gradients should be non-zero"

    def test_gradient_suppression_for_saturated(self):
        """Highly saturated values should get smaller gradients."""
        set_ste_beta(2.0)
        x_sat = torch.tensor([5.0], requires_grad=True)
        x_mid = torch.tensor([0.5], requires_grad=True)

        t_sat = ternary_ste(x_sat)
        t_sat.backward()

        t_mid = ternary_ste(x_mid)
        t_mid.backward()

        assert abs(x_sat.grad.item()) < abs(x_mid.grad.item()), \
            "Saturated values should have smaller gradients"

    def test_zero_input(self):
        """Zero input should map to zero output."""
        set_ste_beta(5.0)
        x = torch.tensor([0.0])
        t = ternary_ste(x)
        assert abs(t.item()) < 0.01

    def test_symmetry(self):
        """Positive and negative should be symmetric."""
        set_ste_beta(3.0)
        x = torch.tensor([0.7, -0.7])
        t = ternary_ste(x)
        assert abs(t[0].item() + t[1].item()) < 1e-6

    def test_batch_works(self):
        """Should handle batched inputs."""
        set_ste_beta(2.0)
        x = torch.randn(32, 64, requires_grad=True)
        t = ternary_ste(x)
        assert t.shape == x.shape
        t.sum().backward()
        assert x.grad.shape == x.shape

    def test_bf16_no_nan(self):
        """BF16 should not produce NaN (FP16 would)."""
        set_ste_beta(5.0)
        x = torch.randn(128, 128, dtype=torch.bfloat16, requires_grad=True)
        t = ternary_ste(x)
        loss = t.sum()
        loss.backward()
        assert not torch.isnan(loss), "BF16 should not produce NaN"
        assert not torch.isnan(x.grad).any(), "BF16 gradients should not be NaN"


# ===== SmoothBVHHit Tests =====

class TestSmoothBVHHit:
    def test_close_gets_high_attention(self):
        """Nodes closer than radius should get high attention."""
        hit_fn = SmoothBVHHit(lambda_decay=0.1)
        set_ste_beta(5.0)
        distances = torch.tensor([[0.1, 0.5, 2.0]])  # close, medium, far
        radii = torch.tensor([1.0, 1.0, 1.0])
        energy = torch.tensor([1.0])
        weights = hit_fn(distances, radii, energy)
        assert weights[0, 0] > weights[0, 1] > weights[0, 2]

    def test_differentiable(self):
        """Gradients should flow through soft hit."""
        hit_fn = SmoothBVHHit()
        set_ste_beta(2.0)
        distances = torch.tensor([[0.5, 1.5]], requires_grad=True)
        radii = torch.tensor([1.0, 1.0])
        energy = torch.tensor([1.0])
        weights = hit_fn(distances, radii, energy)
        weights.sum().backward()
        assert distances.grad is not None
        assert distances.grad.abs().sum() > 0

    def test_high_beta_approaches_hard(self):
        """At high beta, soft hit should approximate hard hit/miss."""
        hit_fn = SmoothBVHHit(lambda_decay=0.1)
        set_ste_beta(50.0)
        distances = torch.tensor([[0.5, 1.5]])  # inside radius, outside radius
        radii = torch.tensor([1.0, 1.0])
        energy = torch.tensor([1.0])
        weights = hit_fn(distances, radii, energy)
        assert weights[0, 0] > 0.8, "Inside radius should be ~1"
        assert weights[0, 1] < 0.1, "Outside radius should be ~0"

    def test_zero_energy_zero_attention(self):
        """Zero energy ray should produce zero attention."""
        hit_fn = SmoothBVHHit()
        set_ste_beta(5.0)
        distances = torch.tensor([[0.5]])
        radii = torch.tensor([1.0])
        energy = torch.tensor([0.0])
        weights = hit_fn(distances, radii, energy)
        assert abs(weights.item()) < 1e-6


# ===== RMSNorm (SubLN) Tests =====

class TestRMSNorm:
    def test_output_scale(self):
        """Output should have approximately unit RMS."""
        norm = RMSNorm(64)
        x = torch.randn(4, 16, 64) * 10  # large scale input
        out = norm(x)
        rms = out.pow(2).mean(-1).sqrt()
        assert rms.mean().item() < 3.0, "RMS should be normalized"

    def test_gradient_flows(self):
        """Gradients should flow through normalization."""
        norm = RMSNorm(32)
        x = torch.randn(2, 8, 32, requires_grad=True)
        out = norm(x)
        out.sum().backward()
        assert x.grad is not None

    def test_learnable_weight(self):
        """Weight parameter should be trainable."""
        norm = RMSNorm(16)
        assert norm.weight.requires_grad
        assert norm.weight.shape == (16,)


# ===== LiquidTimeGate Tests =====

class TestLiquidTimeGate:
    def test_output_shape(self):
        """Output shape should match input."""
        gate = LiquidTimeGate(64)
        x = torch.randn(2, 16, 64)
        out = gate(x)
        assert out.shape == x.shape

    def test_2d_passthrough(self):
        """2D input (no temporal dim) should pass through unchanged."""
        gate = LiquidTimeGate(32)
        x = torch.randn(4, 32)
        out = gate(x)
        assert torch.equal(x, out)

    def test_gradient_flows(self):
        """time_a and time_b should receive gradients."""
        gate = LiquidTimeGate(16)
        x = torch.randn(2, 8, 16)
        out = gate(x)
        out.sum().backward()
        assert gate.time_a.grad is not None
        assert gate.time_b.grad is not None

    def test_local_gate_attenuates_distant(self):
        """Negative time_a should attenuate early positions."""
        gate = LiquidTimeGate(1)
        gate.time_a.data.fill_(-1.0)
        gate.time_b.data.fill_(0.0)
        x = torch.ones(1, 10, 1)
        out = gate(x)
        # First position (most distant) should be attenuated
        assert out[0, 0, 0] < out[0, 9, 0], \
            "LOCAL gate should attenuate distant positions"

    def test_global_gate_preserves_distant(self):
        """Positive time_a should preserve early positions."""
        gate = LiquidTimeGate(1)
        gate.time_a.data.fill_(1.0)
        gate.time_b.data.fill_(0.0)
        x = torch.ones(1, 10, 1)
        out = gate(x)
        # First position should have higher weight than last
        assert out[0, 0, 0] > out[0, 9, 0], \
            "GLOBAL gate should favor distant positions"

    def test_stats_counts(self):
        """gate_stats should count LOCAL/GLOBAL/UNIFORM."""
        gate = LiquidTimeGate(10)
        gate.time_a.data[:3] = -0.5   # 3 LOCAL
        gate.time_a.data[3:7] = 0.5   # 4 GLOBAL
        gate.time_a.data[7:] = 0.0    # 3 UNIFORM
        stats = gate.gate_stats()
        assert stats["n_local"] == 3
        assert stats["n_global"] == 4
        assert stats["n_uniform"] == 3


# ===== DualLR Tests =====

class TestDualLR:
    def test_separates_bvh_and_float(self):
        """Should create separate groups for BVH and float params."""
        model = nn.Module()
        model.D_cont = nn.Parameter(torch.randn(8, 8))
        model.linear = nn.Linear(8, 8)
        model.norm_weight = nn.Parameter(torch.ones(8))

        groups = get_dual_lr_param_groups(model, lr=3e-4, bvh_lr_mult=0.1)

        names = [g["name"] for g in groups]
        assert "bvh_discrete" in names
        assert "float_decay" in names

        bvh_group = next(g for g in groups if g["name"] == "bvh_discrete")
        assert bvh_group["lr"] == 3e-4 * 0.1
        assert bvh_group["weight_decay"] == 0.0

    def test_empty_bvh_group(self):
        """Should work when no BVH params exist."""
        model = nn.Linear(8, 8)
        groups = get_dual_lr_param_groups(model, lr=3e-4)
        names = [g["name"] for g in groups]
        assert "bvh_discrete" not in names

    def test_optimizer_creates_successfully(self):
        """AdamW should accept the param groups."""
        model = nn.Module()
        model.D_cont = nn.Parameter(torch.randn(4, 4))
        model.fc = nn.Linear(4, 4)
        groups = get_dual_lr_param_groups(model, lr=3e-4)
        opt = torch.optim.AdamW(groups, betas=(0.9, 0.95))
        assert opt is not None


# ===== MetabolicBVH Tests =====

class TestMetabolicBVH:
    def test_initial_state(self):
        """All nodes should start active with full reserves."""
        mbvh = MetabolicBVH(64)
        stats = mbvh.stats()
        assert stats["n_active"] == 64
        assert stats["n_pruned"] == 0
        assert stats["sparsity"] == 0.0

    def test_age_increases(self):
        """Age should increase for nodes not hit."""
        mbvh = MetabolicBVH(4, max_age=10)
        mbvh.step()
        mbvh.step()
        assert mbvh.age[0] == 2
        assert mbvh.age[1] == 2

    def test_hit_resets_age(self):
        """Hit nodes should have age reset to 0."""
        mbvh = MetabolicBVH(4, max_age=10)
        mbvh.step()
        mbvh.step()
        mbvh.record_hits(np.array([0, 2]))
        mbvh.step()
        assert mbvh.age[0] == 0  # was hit
        assert mbvh.age[1] == 3  # was not hit
        assert mbvh.age[2] == 0  # was hit

    def test_old_nodes_pruned(self):
        """Nodes exceeding max_age should be pruned."""
        mbvh = MetabolicBVH(4, max_age=3)
        for _ in range(4):
            mbvh.step()
        stats = mbvh.stats()
        assert stats["n_active"] == 0, "All nodes should be pruned after max_age"

    def test_hit_prevents_pruning(self):
        """Hit nodes should survive past max_age."""
        mbvh = MetabolicBVH(4, max_age=3)
        for _ in range(4):
            mbvh.record_hits(np.array([0]))  # keep node 0 alive
            mbvh.step()
        assert mbvh.active[0] == True, "Hit node should survive"
        assert mbvh.active[1] == False, "Unhit node should be pruned"

    def test_energy_depletes_with_children(self):
        """Nodes with many children should lose energy faster."""
        mbvh = MetabolicBVH(4, max_age=10000, energy_cost=0.1, energy_regen=0.05)
        children = np.array([10.0, 1.0, 1.0, 1.0])  # node 0 has 10 children
        for _ in range(3):
            mbvh.step(children_counts=children)
        assert mbvh.reserves[0] < mbvh.reserves[1], \
            "Node with more children should have less reserves"

    def test_revive(self):
        """Pruned nodes should be revivable."""
        mbvh = MetabolicBVH(4, max_age=2)
        for _ in range(3):
            mbvh.step()
        assert mbvh.active[0] == False
        mbvh.revive(np.array([0]))
        assert mbvh.active[0] == True
        assert mbvh.reserves[0] == 1.0


# ===== BetaScheduler Tests =====

class TestBetaScheduler:
    def setup_method(self):
        self._orig_beta = get_ste_beta()

    def teardown_method(self):
        set_ste_beta(self._orig_beta)

    def test_warmup_stays_at_1(self):
        """During warmup, beta should stay at 1.0."""
        sched = BetaScheduler(max_beta=10.0, warmup_steps=100, total_steps=1000)
        for step in range(100):
            beta = sched.step(step)
            assert beta == 1.0

    def test_anneals_to_max(self):
        """After warmup, beta should reach max_beta."""
        sched = BetaScheduler(max_beta=10.0, warmup_steps=100, total_steps=1000)
        beta = sched.step(1000)
        assert abs(beta - 10.0) < 0.1

    def test_linear_progression(self):
        """Beta should increase linearly after warmup."""
        sched = BetaScheduler(max_beta=10.0, warmup_steps=0, total_steps=100)
        beta_25 = sched.step(25)
        beta_50 = sched.step(50)
        beta_75 = sched.step(75)
        assert beta_25 < beta_50 < beta_75

    def test_cap_at_15(self):
        """Max beta should be capped at 15 for safety."""
        sched = BetaScheduler(max_beta=100.0)
        beta = sched.step(sched.total_steps)
        assert beta <= 15.0


# ===== Integration Test =====

class TestIntegration:
    def test_full_forward_pass(self):
        """SmoothSTE + SubLN + LiquidTimeGate in a mini pipeline."""
        set_ste_beta(2.0)
        B, T, D = 2, 8, 32

        # Simulate BVH routing
        D_cont = torch.randn(D, D, requires_grad=True)
        x = torch.randn(B, T, D)

        # 1. Ternary routing
        D_ternary = ternary_ste(D_cont)
        h = x @ D_ternary.T

        # 2. SubLN (mandatory)
        sub_ln = RMSNorm(D)
        h = sub_ln(h)

        # 3. LiquidTimeGate
        gate = LiquidTimeGate(D)
        h = gate(h)

        # 4. Residual
        out = x + h

        # Should be differentiable end-to-end
        loss = out.sum()
        loss.backward()

        assert D_cont.grad is not None, "D_cont should receive gradients"
        assert not torch.isnan(D_cont.grad).any(), "No NaN gradients"
        assert D_cont.grad.abs().sum() > 0, "Non-zero gradients"

    def test_training_loop_stability(self):
        """Mini training loop should not produce NaN."""
        set_ste_beta(1.0)
        D = 16
        D_cont = nn.Parameter(torch.randn(D, D) * 0.5)
        sub_ln = RMSNorm(D)
        gate = LiquidTimeGate(D)
        target = torch.randn(2, 4, D)

        params = list(sub_ln.parameters()) + list(gate.parameters()) + [D_cont]
        opt = torch.optim.AdamW(params, lr=1e-3)
        sched = BetaScheduler(max_beta=10.0, warmup_steps=5, total_steps=20)

        for step in range(20):
            sched.step(step)
            opt.zero_grad()
            D_ternary = ternary_ste(D_cont)
            x = torch.randn(2, 4, D)
            h = x @ D_ternary.T
            h = sub_ln(h)
            h = gate(h)
            loss = F.mse_loss(h, target)
            loss.backward()
            opt.step()
            D_cont.data.clamp_(-2, 2)  # mandatory clamp

            assert not torch.isnan(loss), f"NaN at step {step}"

        assert loss.item() < 100, "Loss should be bounded"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
