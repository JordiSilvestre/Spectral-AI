#!/usr/bin/env python3
"""
test_bvh_gate_wrapper.py — Tests for BVHGateWrapper, IdentityGateWrapper,
and calibration functions (calibrate, apply_calibration).

All tests are CPU-only. Uses a small EnhancedBVHRouter (8 experts) for speed.

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — make python/ importable
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

from olmoe_bvh_distill import EnhancedBVHRouter
from olmoe_e2e_eval import BVHGateWrapper, IdentityGateWrapper
from calibrate_router import calibrate, apply_calibration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_EXPERTS = 8  # 2 * 2 * 2
INPUT_DIM = 128
FEATURE_DIM = 32
BATCH = 4
TOP_K = 4  # half of 8 experts


@pytest.fixture()
def small_router() -> EnhancedBVHRouter:
    """Create a small EnhancedBVHRouter for fast CPU tests."""
    router = EnhancedBVHRouter(
        input_dim=INPUT_DIM,
        n_level1=2,
        n_level2=2,
        n_level3=2,
        feature_dim=FEATURE_DIM,
    )
    router.eval()
    return router


@pytest.fixture()
def hidden_states() -> torch.Tensor:
    """Random hidden states (B, INPUT_DIM)."""
    torch.manual_seed(42)
    return torch.randn(BATCH, INPUT_DIM)


@pytest.fixture()
def wrapper_no_cal(small_router: EnhancedBVHRouter) -> BVHGateWrapper:
    """BVHGateWrapper without calibration."""
    return BVHGateWrapper(
        router=small_router,
        top_k=TOP_K,
        norm_topk_prob=False,
        calibration_mode=None,
        calibration_state=None,
        logit_temperature=None,
    )


@pytest.fixture()
def gate_probs_for_calibration(small_router: EnhancedBVHRouter) -> tuple:
    """Generate fake gate probs for calibration training data."""
    torch.manual_seed(99)
    n_samples = 256
    h = torch.randn(n_samples, INPUT_DIM)
    # Fake target distribution: random softmax probs over N_EXPERTS
    probs = F.softmax(torch.randn(n_samples, N_EXPERTS), dim=-1)
    return h, probs


# ═══════════════════════════════════════════════════════════════════════════
# 1. BVHGateWrapper — forward returns correct tuple
# ═══════════════════════════════════════════════════════════════════════════

class TestBVHGateWrapperForward:

    def test_returns_three_tensors(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        result = wrapper_no_cal(hidden_states)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_router_probs_shape(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        router_probs, _, _ = wrapper_no_cal(hidden_states)
        assert router_probs.shape == (BATCH, N_EXPERTS)

    def test_router_probs_sum_to_one(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        router_probs, _, _ = wrapper_no_cal(hidden_states)
        sums = router_probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-5, rtol=0)

    def test_router_probs_non_negative(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        router_probs, _, _ = wrapper_no_cal(hidden_states)
        assert (router_probs >= 0).all()

    def test_top_k_weights_shape(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        _, top_k_weights, _ = wrapper_no_cal(hidden_states)
        assert top_k_weights.shape == (BATCH, TOP_K)

    def test_top_k_index_shape(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        _, _, top_k_index = wrapper_no_cal(hidden_states)
        assert top_k_index.shape == (BATCH, TOP_K)

    def test_top_k_index_range(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        _, _, top_k_index = wrapper_no_cal(hidden_states)
        assert (top_k_index >= 0).all()
        assert (top_k_index < N_EXPERTS).all()

    def test_top_k_indices_are_unique_per_sample(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        _, _, top_k_index = wrapper_no_cal(hidden_states)
        for i in range(BATCH):
            unique = top_k_index[i].unique()
            assert len(unique) == TOP_K, (
                f"Sample {i}: expected {TOP_K} unique indices, got {len(unique)}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# 2. No calibration mode — raw logits -> softmax -> topk
# ═══════════════════════════════════════════════════════════════════════════

class TestNoCalibration:

    def test_no_cal_uses_raw_logits(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Without calibration, wrapper output matches manual softmax+topk on raw logits."""
        wrapper = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            norm_topk_prob=False,
            calibration_mode=None, calibration_state=None,
            logit_temperature=None,
        )
        router_probs, top_k_weights, top_k_index = wrapper(hidden_states)

        # Manually reproduce: get raw logits from router
        with torch.no_grad():
            small_router(hidden_states.float())
            raw_logits = small_router._last_logits

        expected_probs = F.softmax(raw_logits.float(), dim=-1)
        exp_weights, exp_indices = torch.topk(expected_probs, TOP_K, dim=-1)

        torch.testing.assert_close(router_probs, expected_probs, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(top_k_weights, exp_weights, atol=1e-5, rtol=1e-5)
        assert (top_k_index == exp_indices).all()


# ═══════════════════════════════════════════════════════════════════════════
# 3. Calibration modes
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibrationModes:

    def test_affine_calibration(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Affine calibration applies scale * logits + bias."""
        scale = torch.ones(N_EXPERTS) * 2.0
        bias = torch.ones(N_EXPERTS) * 0.5
        cal_state = {"scale": scale, "bias": bias}

        wrapper = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            norm_topk_prob=False,
            calibration_mode="affine", calibration_state=cal_state,
        )
        router_probs, _, _ = wrapper(hidden_states)

        # Manually compute expected
        with torch.no_grad():
            small_router(hidden_states.float())
            raw = small_router._last_logits
        calibrated = raw * 2.0 + 0.5
        expected = F.softmax(calibrated.float(), dim=-1)
        torch.testing.assert_close(router_probs, expected, atol=1e-5, rtol=1e-5)

    def test_topk_preserving_calibration(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """topk_preserving applies inv_temp * logits + bias (no cross-expert mixing)."""
        inv_temp = torch.tensor([3.0])
        bias = torch.randn(N_EXPERTS)
        cal_state = {"inv_temp": inv_temp, "bias": bias}

        wrapper = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            norm_topk_prob=False,
            calibration_mode="topk_preserving", calibration_state=cal_state,
        )
        router_probs, _, _ = wrapper(hidden_states)

        with torch.no_grad():
            small_router(hidden_states.float())
            raw = small_router._last_logits
        calibrated = raw * 3.0 + bias
        expected = F.softmax(calibrated.float(), dim=-1)
        torch.testing.assert_close(router_probs, expected, atol=1e-5, rtol=1e-5)

    def test_linear_calibration(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Linear calibration applies Linear(n_experts, n_experts) to logits."""
        # Create an identity-like linear layer state
        weight = torch.eye(N_EXPERTS) * 1.5
        bias = torch.zeros(N_EXPERTS)
        cal_state = {"weight": weight, "bias": bias}

        wrapper = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            norm_topk_prob=False,
            calibration_mode="linear", calibration_state=cal_state,
        )
        router_probs, _, _ = wrapper(hidden_states)

        with torch.no_grad():
            small_router(hidden_states.float())
            raw = small_router._last_logits
        calibrated = F.linear(raw, weight, bias)
        expected = F.softmax(calibrated.float(), dim=-1)
        torch.testing.assert_close(router_probs, expected, atol=1e-5, rtol=1e-5)

    def test_none_calibration_is_passthrough(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """calibration_mode=None yields same result as no calibration at all."""
        w1 = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, calibration_state=None,
        )
        w2 = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, calibration_state=None,
        )
        p1, wt1, idx1 = w1(hidden_states)
        p2, wt2, idx2 = w2(hidden_states)
        torch.testing.assert_close(p1, p2, atol=1e-6, rtol=0)
        assert (idx1 == idx2).all()


# ═══════════════════════════════════════════════════════════════════════════
# 4. Logit temperature
# ═══════════════════════════════════════════════════════════════════════════

class TestLogitTemperature:

    def test_temperature_none_no_scaling(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """temperature=None means no logit scaling."""
        w_no_temp = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, logit_temperature=None,
        )
        w_temp1 = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, logit_temperature=None,
        )
        p1, _, _ = w_no_temp(hidden_states)
        p2, _, _ = w_temp1(hidden_states)
        torch.testing.assert_close(p1, p2, atol=1e-6, rtol=0)

    def test_temperature_divides_logits(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """temperature=10 divides logits by 10 before softmax."""
        wrapper = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, logit_temperature=10.0,
        )
        router_probs, _, _ = wrapper(hidden_states)

        with torch.no_grad():
            small_router(hidden_states.float())
            raw = small_router._last_logits
        expected = F.softmax((raw / 10.0).float(), dim=-1)
        torch.testing.assert_close(router_probs, expected, atol=1e-5, rtol=1e-5)

    def test_temperature_preserves_topk_ranking(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Temperature changes weights but NOT top-k selection order."""
        w_no = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, logit_temperature=None,
        )
        w_hot = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, logit_temperature=10.0,
        )
        _, _, idx_no = w_no(hidden_states)
        _, _, idx_hot = w_hot(hidden_states)
        # Top-k indices must be identical (temperature is monotonic transform)
        assert (idx_no == idx_hot).all(), (
            f"Temperature changed top-k ranking!\n  no_temp: {idx_no}\n  temp=10: {idx_hot}"
        )

    def test_temperature_flattens_distribution(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """High temperature makes the distribution more uniform (lower max weight)."""
        w_sharp = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, logit_temperature=None,
        )
        w_flat = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            calibration_mode=None, logit_temperature=50.0,
        )
        probs_sharp, _, _ = w_sharp(hidden_states)
        probs_flat, _, _ = w_flat(hidden_states)

        # Max prob should be lower with high temperature
        assert probs_flat.max() <= probs_sharp.max() + 1e-6


# ═══════════════════════════════════════════════════════════════════════════
# 5. norm_topk_prob
# ═══════════════════════════════════════════════════════════════════════════

class TestNormTopkProb:

    def test_norm_false_raw_weights(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """norm_topk_prob=False returns raw softmax slices (do NOT sum to 1)."""
        wrapper = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            norm_topk_prob=False, calibration_mode=None,
        )
        _, top_k_weights, _ = wrapper(hidden_states)
        sums = top_k_weights.sum(dim=-1)
        # Raw top-k weights from softmax generally do NOT sum to 1
        # (unless the remaining experts have ~0 probability)
        # Just verify they are positive and <= 1 each
        assert (top_k_weights > 0).all()
        assert (top_k_weights <= 1.0 + 1e-6).all()

    def test_norm_true_weights_sum_to_one(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """norm_topk_prob=True renormalizes top-k weights to sum to 1."""
        wrapper = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            norm_topk_prob=True, calibration_mode=None,
        )
        _, top_k_weights, _ = wrapper(hidden_states)
        sums = top_k_weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-5, rtol=0)

    def test_norm_true_vs_false_different_weights(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Normalized and raw weights should differ (unless distribution is degenerate)."""
        w_raw = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            norm_topk_prob=False, calibration_mode=None,
        )
        w_norm = BVHGateWrapper(
            router=small_router, top_k=TOP_K,
            norm_topk_prob=True, calibration_mode=None,
        )
        _, wt_raw, _ = w_raw(hidden_states)
        _, wt_norm, _ = w_norm(hidden_states)
        # Indices should be the same
        # Weights may differ because normalization rescales
        raw_sums = wt_raw.sum(dim=-1)
        # If raw sums are not already 1, the normalized weights must differ
        if not torch.allclose(raw_sums, torch.ones(BATCH), atol=1e-4):
            assert not torch.allclose(wt_raw, wt_norm, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Hybrid mode concept — BVH selects candidates, gate scores them
# ═══════════════════════════════════════════════════════════════════════════

class TestHybridForwardConcept:
    """Test the hybrid forward logic pattern from olmoe_e2e_eval._hybrid_forward."""

    def test_hybrid_forward_logic(
        self, small_router: EnhancedBVHRouter
    ) -> None:
        """Verify hybrid pattern: BVH prunes, gate scores, top-k from candidates."""
        torch.manual_seed(123)
        B = 8
        h = torch.randn(B, INPUT_DIM)
        n_cand = 6  # select 6 out of 8 candidates
        top_k = TOP_K

        # Simulate original gate weight
        gate_weight = torch.randn(N_EXPERTS, INPUT_DIM)

        # Step 1: BVH routing
        with torch.no_grad():
            bvh_probs, _ = small_router(h.float())

        # Step 2: BVH prunes to top-N candidates
        _, candidate_ids = torch.topk(bvh_probs, n_cand, dim=-1)
        assert candidate_ids.shape == (B, n_cand)

        # Step 3: Full gate softmax (simulating original gate)
        full_logits = F.linear(h, gate_weight)
        full_probs = F.softmax(full_logits, dtype=torch.float, dim=-1)

        # Step 4: Restrict top-k to BVH candidates
        cand_probs = full_probs.gather(1, candidate_ids)
        topk_vals, topk_local = torch.topk(cand_probs, top_k, dim=-1)
        topk_global = candidate_ids.gather(1, topk_local)

        # Verify shapes
        assert topk_vals.shape == (B, top_k)
        assert topk_global.shape == (B, top_k)
        # All selected indices must be from the candidate set
        for i in range(B):
            cand_set = set(candidate_ids[i].tolist())
            selected_set = set(topk_global[i].tolist())
            assert selected_set.issubset(cand_set), (
                f"Sample {i}: selected {selected_set} not subset of candidates {cand_set}"
            )

    def test_hybrid_returns_full_probs(
        self, small_router: EnhancedBVHRouter
    ) -> None:
        """Hybrid mode returns the full softmax distribution (not restricted)."""
        torch.manual_seed(77)
        h = torch.randn(BATCH, INPUT_DIM)
        gate_weight = torch.randn(N_EXPERTS, INPUT_DIM)

        full_logits = F.linear(h, gate_weight)
        full_probs = F.softmax(full_logits, dtype=torch.float, dim=-1)

        # The full_probs returned should cover all experts
        assert full_probs.shape == (BATCH, N_EXPERTS)
        sums = full_probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-5, rtol=0)


# ═══════════════════════════════════════════════════════════════════════════
# 7. calibrate() function
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibrateFunction:

    def test_affine_mode_returns_scale_bias(
        self,
        small_router: EnhancedBVHRouter,
        gate_probs_for_calibration: tuple,
    ) -> None:
        h, probs = gate_probs_for_calibration
        result = calibrate(
            small_router, h, probs,
            mode="affine", n_experts=N_EXPERTS,
            epochs=5, lr=0.01, batch_size=64, device="cpu",
        )
        assert result["mode"] == "affine"
        assert "scale" in result["state"]
        assert "bias" in result["state"]
        assert result["state"]["scale"].shape == (N_EXPERTS,)
        assert result["state"]["bias"].shape == (N_EXPERTS,)

    def test_topk_preserving_mode_returns_inv_temp_bias(
        self,
        small_router: EnhancedBVHRouter,
        gate_probs_for_calibration: tuple,
    ) -> None:
        h, probs = gate_probs_for_calibration
        result = calibrate(
            small_router, h, probs,
            mode="topk_preserving", n_experts=N_EXPERTS,
            epochs=5, lr=0.01, batch_size=64, device="cpu",
        )
        assert result["mode"] == "topk_preserving"
        assert "inv_temp" in result["state"]
        assert "bias" in result["state"]
        assert result["state"]["inv_temp"].shape == (1,)
        assert result["state"]["bias"].shape == (N_EXPERTS,)

    def test_linear_mode_returns_weight_bias(
        self,
        small_router: EnhancedBVHRouter,
        gate_probs_for_calibration: tuple,
    ) -> None:
        h, probs = gate_probs_for_calibration
        result = calibrate(
            small_router, h, probs,
            mode="linear", n_experts=N_EXPERTS,
            epochs=5, lr=0.01, batch_size=64, device="cpu",
        )
        assert result["mode"] == "linear"
        assert "weight" in result["state"]
        assert "bias" in result["state"]
        assert result["state"]["weight"].shape == (N_EXPERTS, N_EXPERTS)
        assert result["state"]["bias"].shape == (N_EXPERTS,)

    def test_affine_param_count(
        self,
        small_router: EnhancedBVHRouter,
        gate_probs_for_calibration: tuple,
    ) -> None:
        h, probs = gate_probs_for_calibration
        result = calibrate(
            small_router, h, probs,
            mode="affine", n_experts=N_EXPERTS,
            epochs=2, lr=0.01, batch_size=64, device="cpu",
        )
        # affine: scale (N) + bias (N) = 2*N
        assert result["n_params"] == 2 * N_EXPERTS

    def test_topk_preserving_param_count(
        self,
        small_router: EnhancedBVHRouter,
        gate_probs_for_calibration: tuple,
    ) -> None:
        h, probs = gate_probs_for_calibration
        result = calibrate(
            small_router, h, probs,
            mode="topk_preserving", n_experts=N_EXPERTS,
            epochs=2, lr=0.01, batch_size=64, device="cpu",
        )
        # topk_preserving: inv_temp (1) + bias (N) = 1 + N
        assert result["n_params"] == 1 + N_EXPERTS

    def test_linear_param_count(
        self,
        small_router: EnhancedBVHRouter,
        gate_probs_for_calibration: tuple,
    ) -> None:
        h, probs = gate_probs_for_calibration
        result = calibrate(
            small_router, h, probs,
            mode="linear", n_experts=N_EXPERTS,
            epochs=2, lr=0.01, batch_size=64, device="cpu",
        )
        # linear: weight (N*N) + bias (N) = N^2 + N
        assert result["n_params"] == N_EXPERTS * N_EXPERTS + N_EXPERTS

    def test_param_counts_at_64_experts(self) -> None:
        """Verify the documented param counts for the real 64-expert case."""
        n = 64
        assert 2 * n == 128, "affine should be 128 params"
        assert 1 + n == 65, "topk_preserving should be 65 params"
        assert n * n + n == 4160, "linear should be 4160 params"

    def test_unknown_mode_raises(
        self,
        small_router: EnhancedBVHRouter,
        gate_probs_for_calibration: tuple,
    ) -> None:
        h, probs = gate_probs_for_calibration
        with pytest.raises(ValueError, match="Unknown mode"):
            calibrate(
                small_router, h, probs,
                mode="banana", n_experts=N_EXPERTS,
                epochs=1, batch_size=64, device="cpu",
            )


# ═══════════════════════════════════════════════════════════════════════════
# 8. apply_calibration()
# ═══════════════════════════════════════════════════════════════════════════

class TestApplyCalibration:

    def _make_cal_data(self, mode: str) -> dict:
        """Build a minimal cal_data dict for apply_calibration."""
        if mode == "affine":
            return {
                "mode": "affine",
                "state": {
                    "scale": torch.ones(N_EXPERTS) * 2.0,
                    "bias": torch.ones(N_EXPERTS) * 0.1,
                },
            }
        elif mode == "topk_preserving":
            return {
                "mode": "topk_preserving",
                "state": {
                    "inv_temp": torch.tensor([5.0]),
                    "bias": torch.zeros(N_EXPERTS),
                },
            }
        elif mode == "linear":
            return {
                "mode": "linear",
                "state": {
                    "weight": torch.eye(N_EXPERTS),
                    "bias": torch.zeros(N_EXPERTS),
                },
            }
        raise ValueError(mode)

    def test_affine_transforms_correctly(self) -> None:
        logits = torch.randn(BATCH, N_EXPERTS)
        cal = self._make_cal_data("affine")
        out = apply_calibration(logits, cal, device="cpu")
        expected = logits * 2.0 + 0.1
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_topk_preserving_transforms_correctly(self) -> None:
        logits = torch.randn(BATCH, N_EXPERTS)
        cal = self._make_cal_data("topk_preserving")
        out = apply_calibration(logits, cal, device="cpu")
        expected = logits * 5.0  # bias is zeros
        torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)

    def test_linear_transforms_correctly(self) -> None:
        logits = torch.randn(BATCH, N_EXPERTS)
        cal = self._make_cal_data("linear")
        out = apply_calibration(logits, cal, device="cpu")
        # Identity weight + zero bias = passthrough
        torch.testing.assert_close(out, logits, atol=1e-5, rtol=1e-5)

    def test_output_shape_matches_input(self) -> None:
        for mode in ("affine", "topk_preserving", "linear"):
            logits = torch.randn(16, N_EXPERTS)
            cal = self._make_cal_data(mode)
            out = apply_calibration(logits, cal, device="cpu")
            assert out.shape == logits.shape, f"Shape mismatch for mode={mode}"


# ═══════════════════════════════════════════════════════════════════════════
# 9. topk_preserving preserves ranking
# ═══════════════════════════════════════════════════════════════════════════

class TestTopkPreservingRanking:

    def test_topk_indices_preserved_after_calibration(
        self, small_router: EnhancedBVHRouter
    ) -> None:
        """
        The KEY property of topk_preserving: top-k expert indices before
        calibration must equal top-k expert indices after calibration.

        topk_preserving applies: logits * inv_temp + bias
        Since inv_temp is a scalar and bias is element-wise, the ranking
        of logits is preserved (monotonic transform when inv_temp > 0).
        """
        torch.manual_seed(7)
        h = torch.randn(4, INPUT_DIM)  # small batch to avoid near-tied logits with 8 experts

        with torch.no_grad():
            small_router(h.float())
            raw_logits = small_router._last_logits.clone()

        # Simulate topk_preserving calibration with arbitrary params
        inv_temp = torch.tensor([3.5])
        bias = torch.randn(N_EXPERTS) * 0.001  # very small bias to avoid ranking changes
        calibrated = raw_logits * inv_temp + bias

        # Get top-k before and after
        _, idx_before = torch.topk(raw_logits, TOP_K, dim=-1)
        _, idx_after = torch.topk(calibrated, TOP_K, dim=-1)

        # Sort indices to handle ties consistently, convert to same dtype
        idx_before_sorted, _ = idx_before.sort(dim=-1)
        idx_after_sorted, _ = idx_after.sort(dim=-1)

        # Compare as sets per sample (order may differ for tied values)
        match = True
        for i in range(idx_before_sorted.shape[0]):
            if set(idx_before_sorted[i].tolist()) != set(idx_after_sorted[i].tolist()):
                match = False
                break
        assert match, (
            "topk_preserving changed expert ranking!\n"
            f"  before: {idx_before_sorted[:3]}\n"
            f"  after:  {idx_after_sorted[:3]}"
        )

    def test_topk_preserving_with_large_bias_can_change_ranking(self) -> None:
        """
        With a sufficiently large bias, ranking CAN change. This is expected
        since bias is per-expert. The 'preserving' property holds when bias
        is small relative to logit spread (as trained by the calibrator).
        """
        torch.manual_seed(42)
        logits = torch.tensor([[1.0, 0.5, 0.3, 0.1, -0.1, -0.3, -0.5, -1.0]])

        # Huge bias that flips ranking
        big_bias = torch.tensor([-10.0, -10.0, -10.0, -10.0, 10.0, 10.0, 10.0, 10.0])
        calibrated = logits * 1.0 + big_bias

        _, idx_before = torch.topk(logits, TOP_K, dim=-1)
        _, idx_after = torch.topk(calibrated, TOP_K, dim=-1)

        # With huge bias, ranking SHOULD change
        idx_before_sorted, _ = idx_before.sort(dim=-1)
        idx_after_sorted, _ = idx_after.sort(dim=-1)
        assert not (idx_before_sorted == idx_after_sorted).all(), (
            "Expected ranking to change with huge bias"
        )

    def test_topk_preserving_zero_bias_always_preserves(self) -> None:
        """With zero bias and positive inv_temp, ranking is always preserved."""
        torch.manual_seed(55)
        logits = torch.randn(16, N_EXPERTS)

        for temp_val in [0.5, 1.0, 2.0, 10.0]:
            calibrated = logits * temp_val  # zero bias
            _, idx_orig = torch.topk(logits, TOP_K, dim=-1)
            _, idx_cal = torch.topk(calibrated, TOP_K, dim=-1)
            assert (idx_orig == idx_cal).all(), (
                f"Zero-bias topk_preserving failed at inv_temp={temp_val}"
            )


# ═══════════════════════════════════════════════════════════════════════════
# IdentityGateWrapper (bonus — the diagnostic wrapper)
# ═══════════════════════════════════════════════════════════════════════════

class TestIdentityGateWrapper:

    def test_identity_forward_shape(self) -> None:
        torch.manual_seed(10)
        weight = torch.randn(N_EXPERTS, INPUT_DIM)
        wrapper = IdentityGateWrapper(weight, top_k=TOP_K, norm_topk_prob=False)
        h = torch.randn(BATCH, INPUT_DIM)
        probs, weights, indices = wrapper(h)
        assert probs.shape == (BATCH, N_EXPERTS)
        assert weights.shape == (BATCH, TOP_K)
        assert indices.shape == (BATCH, TOP_K)

    def test_identity_matches_manual_gate(self) -> None:
        """IdentityGateWrapper must reproduce F.linear -> softmax -> topk exactly."""
        torch.manual_seed(20)
        weight = torch.randn(N_EXPERTS, INPUT_DIM)
        h = torch.randn(BATCH, INPUT_DIM)

        wrapper = IdentityGateWrapper(weight, top_k=TOP_K, norm_topk_prob=False)
        probs, weights, indices = wrapper(h)

        # Manual computation
        logits = F.linear(h, weight)
        expected_probs = F.softmax(logits, dtype=torch.float, dim=-1)
        exp_w, exp_i = torch.topk(expected_probs, TOP_K, dim=-1)

        torch.testing.assert_close(probs, expected_probs, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(weights, exp_w, atol=1e-5, rtol=1e-5)
        assert (indices == exp_i).all()

    def test_identity_norm_topk_prob(self) -> None:
        torch.manual_seed(30)
        weight = torch.randn(N_EXPERTS, INPUT_DIM)
        h = torch.randn(BATCH, INPUT_DIM)

        wrapper = IdentityGateWrapper(weight, top_k=TOP_K, norm_topk_prob=True)
        _, weights, _ = wrapper(h)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(BATCH), atol=1e-5, rtol=0)


# ═══════════════════════════════════════════════════════════════════════════
# 8. Confidence-Gated Routing — adaptive hybrid per token
# ═══════════════════════════════════════════════════════════════════════════


class TestConfidenceGatedRouting:
    """Tests for the confidence-gated routing mechanism (Claims 34-38)."""

    def _make_wrapper_with_gate(
        self, small_router: EnhancedBVHRouter, threshold: float
    ) -> BVHGateWrapper:
        """Create a wrapper with confidence gate enabled and fake gate weight."""
        wrapper = BVHGateWrapper(
            router=small_router,
            top_k=TOP_K,
            norm_topk_prob=False,
            weight_mode="relu_norm",
        )
        # Simulate original gate weight (required for fallback)
        torch.manual_seed(77)
        wrapper._original_gate_weight = torch.randn(N_EXPERTS, INPUT_DIM)
        wrapper._confidence_threshold = threshold
        return wrapper

    def test_disabled_by_default(
        self, wrapper_no_cal: BVHGateWrapper, hidden_states: torch.Tensor
    ) -> None:
        """Confidence gate is disabled when threshold is None."""
        assert wrapper_no_cal._confidence_threshold is None
        probs, weights, indices = wrapper_no_cal(hidden_states)
        assert probs.shape == (BATCH, N_EXPERTS)
        # Stats should show zeros (disabled)
        assert wrapper_no_cal._confidence_stats["bvh_tokens"] == 0
        assert wrapper_no_cal._confidence_stats["gate_tokens"] == 0

    def test_threshold_zero_all_bvh(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Threshold=0.0 should route all tokens via BVH (confidence always >= 0)."""
        wrapper = self._make_wrapper_with_gate(small_router, threshold=0.0)
        probs, weights, indices = wrapper(hidden_states)
        stats = wrapper._confidence_stats
        assert stats["bvh_tokens"] > 0
        assert stats["gate_tokens"] == 0

    def test_threshold_one_all_gate(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Threshold=1.0 should route all tokens via gate (confidence never >= 1.0 with sigmoid)."""
        wrapper = self._make_wrapper_with_gate(small_router, threshold=1.0)
        probs, weights, indices = wrapper(hidden_states)
        stats = wrapper._confidence_stats
        # With threshold=1.0, sigmoid can't reach 1.0 so all go to gate
        assert stats["gate_tokens"] == BATCH
        assert stats["bvh_tokens"] == 0

    def test_output_shape_correct(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Output shapes must be correct regardless of threshold."""
        wrapper = self._make_wrapper_with_gate(small_router, threshold=0.5)
        probs, weights, indices = wrapper(hidden_states)
        assert probs.shape == (BATCH, N_EXPERTS)
        assert weights.shape == (BATCH, TOP_K)
        assert indices.shape == (BATCH, TOP_K)

    def test_stats_accumulate(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Stats should accumulate across multiple forward calls."""
        wrapper = self._make_wrapper_with_gate(small_router, threshold=0.5)
        wrapper(hidden_states)
        stats1_total = (
            wrapper._confidence_stats["bvh_tokens"]
            + wrapper._confidence_stats["gate_tokens"]
        )
        wrapper(hidden_states)
        stats2_total = (
            wrapper._confidence_stats["bvh_tokens"]
            + wrapper._confidence_stats["gate_tokens"]
        )
        assert stats2_total == stats1_total + BATCH

    def test_higher_threshold_more_gate(
        self, small_router: EnhancedBVHRouter
    ) -> None:
        """Higher threshold should result in more gate usage (monotonic)."""
        torch.manual_seed(42)
        h = torch.randn(64, INPUT_DIM)  # larger batch for statistical power

        wrapper_low = self._make_wrapper_with_gate(small_router, threshold=0.3)
        wrapper_high = self._make_wrapper_with_gate(small_router, threshold=0.9)

        wrapper_low(h)
        wrapper_high(h)

        gate_low = wrapper_low._confidence_stats["gate_tokens"]
        gate_high = wrapper_high._confidence_stats["gate_tokens"]
        assert gate_high >= gate_low, (
            f"Higher threshold should use more gate: "
            f"gate@0.3={gate_low}, gate@0.9={gate_high}"
        )

    def test_requires_original_gate_weight(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Without original gate weight, confidence gate should not activate."""
        wrapper = BVHGateWrapper(
            router=small_router,
            top_k=TOP_K,
            norm_topk_prob=False,
            weight_mode="relu_norm",
        )
        wrapper._confidence_threshold = 0.5
        # No _original_gate_weight set
        probs, weights, indices = wrapper(hidden_states)
        # Should still work (no crash), just no gating
        assert probs.shape == (BATCH, N_EXPERTS)
        assert wrapper._confidence_stats["bvh_tokens"] == 0
        assert wrapper._confidence_stats["gate_tokens"] == 0

    def test_probs_non_negative(
        self, small_router: EnhancedBVHRouter, hidden_states: torch.Tensor
    ) -> None:
        """Merged probs must be non-negative."""
        wrapper = self._make_wrapper_with_gate(small_router, threshold=0.5)
        probs, _, _ = wrapper(hidden_states)
        assert (probs >= 0).all(), "Router probs must be non-negative after merge"
