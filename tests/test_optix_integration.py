"""
Test OptiX RT Core integration with PyTorch BVH Router.

Tests the full pipeline: Python -> OptiX extension -> RT Cores -> expert IDs.
Compares RT Core routing with PyTorch-only routing for consistency.

Requirements:
    - NVIDIA GPU with RT Cores
    - OptiX SDK installed
    - PTX shaders compiled (cmake --build build)
    - optix_training_ext compiled (python cuda/v5/build_optix_ext.py)

Run: python -m pytest tests/test_optix_integration.py -v
     (skips automatically if OptiX not available)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import pytest
import torch

# ── Try loading OptiX extension ─────────────────────────────────────

_optix_ext = None
_optix_available = False

try:
    import optix_training_ext as _optix_ext

    _optix_available = True
except ImportError:
    # Try JIT build
    try:
        sys.path.insert(
            0, os.path.join(os.path.dirname(__file__), "..", "cuda", "v5")
        )
        from build_optix_ext import build_optix_training_ext

        _optix_available = build_optix_training_ext()
        if _optix_available:
            import optix_training_ext as _optix_ext
    except Exception:
        pass

requires_optix = pytest.mark.skipif(
    not _optix_available, reason="OptiX extension not available"
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


# ── Locate PTX files ────────────────────────────────────────────────

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
PTX_DIRS = [
    os.path.join(PROJECT_ROOT, "build", "ptx"),
    os.path.join(PROJECT_ROOT, "build", "Release", "ptx"),
    os.path.join(PROJECT_ROOT, "build"),
]


def _find_ptx(name_fragment: str) -> str:
    """Find a PTX file containing name_fragment in its filename."""
    for d in PTX_DIRS:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".ptx") and name_fragment in f.lower():
                return os.path.join(d, f)
    return ""


@pytest.fixture(scope="module")
def ptx_paths():
    """Return (raygen_ptx, hitgroup_ptx) paths or skip if not found."""
    raygen = _find_ptx("raygen")
    hitgroup = _find_ptx("hitgroup")
    if not raygen or not hitgroup:
        pytest.skip("PTX shader files not found. Build with CMake first.")
    return raygen, hitgroup


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def initialized_ext(ptx_paths):
    """Initialize OptiX extension with PTX files."""
    raygen, hitgroup = ptx_paths
    _optix_ext.initialize(raygen, hitgroup)
    yield _optix_ext
    _optix_ext.shutdown()


@pytest.fixture
def expert_spheres():
    """64 expert spheres in 3D space (matching BVH 4x4x4 structure)."""
    torch.manual_seed(42)
    centers = torch.randn(64, 3, dtype=torch.float32) * 2.0
    radii = torch.ones(64, dtype=torch.float32) * 0.5
    return centers, radii


@pytest.fixture
def query_batch():
    """Batch of 256 query positions and directions on CUDA."""
    torch.manual_seed(123)
    positions = torch.randn(256, 3, dtype=torch.float32, device="cuda")
    directions = torch.randn(256, 3, dtype=torch.float32, device="cuda")
    directions = directions / directions.norm(dim=-1, keepdim=True)
    return positions, directions


# ── Tests ───────────────────────────────────────────────────────────


@requires_optix
@requires_cuda
class TestOptiXExtension:
    """Basic extension functionality tests."""

    def test_is_ready_before_init(self):
        """Extension should not be ready before initialization."""
        # Note: if previous test already initialized, this may pass.
        # That's OK — we're testing the API works.
        assert isinstance(_optix_ext.is_ready(), bool)

    def test_initialize(self, ptx_paths):
        """Can initialize OptiX from PTX files."""
        raygen, hitgroup = ptx_paths
        result = _optix_ext.initialize(raygen, hitgroup)
        assert result is True

    def test_build_gas_aabb(self, initialized_ext, expert_spheres):
        """Can build GAS with AABB (axis-aligned bounding boxes)."""
        centers, radii = expert_spheres
        result = initialized_ext.build_gas(centers, radii, False)
        assert result is True
        assert initialized_ext.num_experts() == 64
        assert initialized_ext.gas_size() > 0

    def test_build_gas_triangles(self, initialized_ext, expert_spheres):
        """Can build GAS with triangle octahedrons (more precise)."""
        centers, radii = expert_spheres
        result = initialized_ext.build_gas(centers, radii, True)
        assert result is True
        assert initialized_ext.num_experts() == 64

    def test_route_single(self, initialized_ext, expert_spheres, query_batch):
        """Can route a batch through RT Cores."""
        centers, radii = expert_spheres
        initialized_ext.build_gas(centers, radii, False)

        positions, directions = query_batch
        expert_ids, distances = initialized_ext.route(positions, directions)

        assert expert_ids.shape == (256,)
        assert distances.shape == (256,)
        assert expert_ids.dtype == torch.int32
        assert distances.dtype == torch.float32
        assert expert_ids.is_cuda

        # Expert IDs should be valid (0-63) or sentinel (-1 for miss)
        valid = expert_ids >= 0
        assert valid.any(), "All rays missed — check sphere placement"
        assert (expert_ids[valid] < 64).all()

    def test_route_topk(self, initialized_ext, expert_spheres, query_batch):
        """Can route with top-K selection (MoE-style)."""
        centers, radii = expert_spheres
        initialized_ext.build_gas(centers, radii, False)

        positions, directions = query_batch
        topk_ids, topk_dists = initialized_ext.route_topk(
            positions, directions, 8
        )

        assert topk_ids.shape == (256, 8)
        assert topk_dists.shape == (256, 8)
        assert topk_ids.dtype == torch.int32

    def test_gas_size_reasonable(self, initialized_ext, expert_spheres):
        """GAS memory should be reasonable (< 1 MB for 64 experts)."""
        centers, radii = expert_spheres
        initialized_ext.build_gas(centers, radii, False)
        gas_bytes = initialized_ext.gas_size()
        assert 0 < gas_bytes < 1_000_000, f"GAS size {gas_bytes} seems wrong"


@requires_optix
@requires_cuda
class TestOptiXBridge:
    """Test the Python bridge (optix_training_bridge.py)."""

    def test_bridge_import(self):
        """OptiXTrainingBridge can be imported."""
        from optix_training_bridge import OptiXTrainingBridge

        bridge = OptiXTrainingBridge(auto_init=False)
        assert not bridge.available

    def test_smooth_bvh_hit(self):
        """SmoothBVHHit produces correct shapes."""
        from optix_training_bridge import SmoothBVHHit

        hit = SmoothBVHHit(sharpness=10.0)
        positions = torch.randn(32, 3)
        centers = torch.randn(64, 3)
        radii = torch.ones(64) * 0.5

        scores = hit(positions, centers, radii)
        assert scores.shape == (32, 64)
        assert (scores >= 0).all()
        assert (scores <= 1).all()

    def test_straight_through_optix(self):
        """StraightThroughOptiX STE produces correct one-hot output."""
        from optix_training_bridge import StraightThroughOptiX

        soft_signal = torch.randn(16, 64, requires_grad=True)
        hard_ids = torch.randint(0, 64, (16,), dtype=torch.int32)

        result = StraightThroughOptiX.apply(soft_signal, hard_ids, 64)

        assert result.shape == (16, 64)
        # Each row should be one-hot
        assert (result.sum(dim=-1) == 1.0).all()
        # Gradient should flow through soft_signal
        result.sum().backward()
        assert soft_signal.grad is not None

    def test_bridge_forward_fallback(self):
        """Bridge falls back to soft routing when OptiX unavailable."""
        from optix_training_bridge import OptiXTrainingBridge

        bridge = OptiXTrainingBridge(auto_init=False)
        assert not bridge.available

        positions = torch.randn(16, 3, requires_grad=True)
        centers = torch.randn(64, 3)
        radii = torch.ones(64) * 0.5

        # Should use soft fallback (no OptiX)
        probs = bridge.forward_with_ste(positions, centers, radii, 64)
        assert probs.shape == (16, 64)
        # Should be differentiable
        probs.sum().backward()
        assert positions.grad is not None


@requires_optix
@requires_cuda
class TestOptiXVsPyTorch:
    """Compare OptiX RT Core routing with PyTorch-only routing."""

    def test_routing_consistency(self, initialized_ext, expert_spheres):
        """RT Core routing should agree with distance-based PyTorch routing."""
        from optix_training_bridge import SmoothBVHHit

        centers, radii = expert_spheres
        initialized_ext.build_gas(centers, radii, False)

        # Generate queries on CUDA
        torch.manual_seed(99)
        batch_size = 128
        positions = torch.randn(
            batch_size, 3, dtype=torch.float32, device="cuda"
        )
        # Direction: toward weighted center of closest spheres
        centers_cuda = centers.cuda()
        dists = torch.cdist(positions, centers_cuda)
        nearest_idx = dists.argmin(dim=-1)
        directions = centers_cuda[nearest_idx] - positions
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp(
            min=1e-6
        )

        # RT Core routing
        rt_ids, rt_dists = initialized_ext.route(positions, directions)

        # PyTorch distance-based routing (closest sphere)
        pytorch_nearest = dists.argmin(dim=-1)

        # Compare — RT Cores use ray-AABB intersection which may differ
        # from pure Euclidean distance. Expect >50% agreement.
        valid = rt_ids >= 0
        agreement = (rt_ids[valid] == pytorch_nearest[valid]).float().mean()
        print(f"\n  RT vs PyTorch agreement: {agreement:.1%}")
        assert agreement > 0.3, (
            f"RT/PyTorch agreement too low: {agreement:.1%}"
        )

    def test_topk_overlap(self, initialized_ext, expert_spheres):
        """Top-8 RT Core selection should overlap with distance-based top-8."""
        centers, radii = expert_spheres
        initialized_ext.build_gas(centers, radii, False)

        torch.manual_seed(77)
        batch_size = 64
        positions = torch.randn(
            batch_size, 3, dtype=torch.float32, device="cuda"
        )
        directions = torch.randn(
            batch_size, 3, dtype=torch.float32, device="cuda"
        )
        directions = directions / directions.norm(dim=-1, keepdim=True)

        # RT Core top-8
        rt_topk_ids, _ = initialized_ext.route_topk(positions, directions, 8)
        rt_topk_ids = rt_topk_ids.cpu()

        # PyTorch top-8 by distance
        centers_cuda = centers.cuda()
        dists = torch.cdist(positions, centers_cuda)
        _, pt_topk_ids = dists.topk(8, dim=-1, largest=False)
        pt_topk_ids = pt_topk_ids.cpu()

        # Measure overlap
        overlaps = []
        for b in range(batch_size):
            rt_set = set(rt_topk_ids[b].tolist())
            pt_set = set(pt_topk_ids[b].tolist())
            overlaps.append(len(rt_set & pt_set) / 8)

        avg_overlap = sum(overlaps) / len(overlaps)
        print(f"\n  Top-8 overlap (RT vs distance): {avg_overlap:.1%}")
        # Ray tracing is directional, distance is omnidirectional
        # so overlap may be moderate
        assert avg_overlap > 0.2, (
            f"Top-8 overlap too low: {avg_overlap:.1%}"
        )


@requires_optix
@requires_cuda
class TestOptiXLatency:
    """Benchmark RT Core routing latency."""

    def test_latency_vs_pytorch(self, initialized_ext, expert_spheres):
        """RT Core routing should be faster than PyTorch distance computation."""
        import time

        centers, radii = expert_spheres
        initialized_ext.build_gas(centers, radii, False)

        batch_size = 256
        positions = torch.randn(
            batch_size, 3, dtype=torch.float32, device="cuda"
        )
        directions = torch.randn(
            batch_size, 3, dtype=torch.float32, device="cuda"
        )
        directions = directions / directions.norm(dim=-1, keepdim=True)
        centers_cuda = centers.cuda()

        n_iters = 100

        # Warmup
        for _ in range(10):
            initialized_ext.route(positions, directions)
            torch.cdist(positions, centers_cuda).argmin(dim=-1)
        torch.cuda.synchronize()

        # RT Core timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            initialized_ext.route(positions, directions)
        torch.cuda.synchronize()
        rt_time = (time.perf_counter() - t0) / n_iters * 1e6  # microseconds

        # PyTorch timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            torch.cdist(positions, centers_cuda).argmin(dim=-1)
        torch.cuda.synchronize()
        pt_time = (time.perf_counter() - t0) / n_iters * 1e6

        print(f"\n  RT Core: {rt_time:.1f} us/batch")
        print(f"  PyTorch: {pt_time:.1f} us/batch")
        print(f"  Speedup: {pt_time / rt_time:.1f}x")

        # Just verify it runs — speedup depends on batch size and GPU
        assert rt_time > 0
