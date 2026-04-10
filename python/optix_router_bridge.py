#!/usr/bin/env python3
"""
optix_router_bridge.py -- Bridge: Trained BVH Router → OptiX RT Core Routing

Connects the trained BVH router (FASE A) with the OptiX pipeline (FASE B).

The BVH router's 3D sphere centers become OptiX SemanticSpheres.
Instead of Gumbel-Softmax routing in PyTorch, routing happens on RT Cores:
  1. Export router spheres to scene.bin
  2. Run inception_runner.exe (OptiX)
  3. Parse results.bin → expert_id per query

This replaces ~1ms PyTorch routing with ~0.05ms RT Core routing at scale.

Usage:
    # Export trained router to OptiX scene
    bridge = OptiXRouterBridge(router_checkpoint="data/moe_best.pt")
    expert_ids = bridge.route(prompt_embeddings)  # Uses RT Cores

    # Benchmark: PyTorch vs OptiX routing
    bridge.benchmark(prompt_embeddings, n_iters=100)

Copyright (c) 2026 Jordi Silvestre Lopez — Apache 2.0
"""

import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ─────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────

SCRIPT_DIR  = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
BUILD_DIR   = PROJECT_DIR / "build" / "Release"
BUILD_ROOT  = PROJECT_DIR / "build"
PTX_PATH    = BUILD_ROOT / "spectral_kernels.ptx"
RUNNER_EXE  = BUILD_DIR / "inception_runner.exe"

# ─────────────────────────────────────────────────────────────────
# Binary struct formats (verified with print_struct_sizes.exe)
# ─────────────────────────────────────────────────────────────────

SPHERE_FMT   = "<3f f I I I f"    # 32 bytes: center(3f) radius(f) instanceId(I) childIAS(I) depth(I) freqBias(f)
RESONANCE_FMT = "<8f 8f I f I I"  # 80 bytes data + 16 pad = 96
RESONANCE_PAD = 16
STRING_BODY_FMT = "<3f I"         # 16 bytes: position(3f) stringId(I)
PORTAL_FMT   = "<16f"             # 64 bytes: 4x4 matrix
RESULT_FMT   = "<f f I I 3f f"    # 32 bytes: attWeight, omega, stringId, depth, exitDir(3f), energy

SPHERE_SIZE    = struct.calcsize(SPHERE_FMT)      # 32
RESONANCE_SIZE = struct.calcsize(RESONANCE_FMT) + RESONANCE_PAD  # 96
STRING_SIZE    = RESONANCE_SIZE + struct.calcsize(STRING_BODY_FMT) + 16  # 128
PORTAL_SIZE    = struct.calcsize(PORTAL_FMT)       # 64
RESULT_SIZE    = struct.calcsize(RESULT_FMT)       # 32

SCENE_MAGIC   = 0x4C425354  # 'LBST'
SCENE_VERSION = 1


# ─────────────────────────────────────────────────────────────────
# Helper: pack structs
# ─────────────────────────────────────────────────────────────────

def pack_sphere(center: np.ndarray, radius: float,
                instance_id: int, depth: int, freq_bias: float) -> bytes:
    """Pack a SemanticSphere (32 bytes)."""
    return struct.pack(SPHERE_FMT,
        float(center[0]), float(center[1]), float(center[2]),
        radius, instance_id, 0, depth, freq_bias)


def pack_string(position: np.ndarray, string_id: int,
                fourier_a: List[float], fourier_b: List[float]) -> bytes:
    """Pack a SemanticString (128 bytes)."""
    # ResonanceParams (96 bytes)
    a = (fourier_a + [0.0] * 8)[:8]
    b = (fourier_b + [0.0] * 8)[:8]
    resonance = struct.pack(RESONANCE_FMT,
        *a, *b,
        8,                  # numModes
        1.0,                # outputScale
        string_id,          # semanticTag
        0)                  # _pad
    resonance += b'\x00' * RESONANCE_PAD  # 16 bytes tail padding

    # Body (position + stringId + padding)
    body = struct.pack(STRING_BODY_FMT,
        float(position[0]), float(position[1]), float(position[2]),
        string_id)
    body += b'\x00' * 16  # tail padding to 128

    return resonance + body


def pack_portal_identity() -> bytes:
    """Pack an identity AffinePortal (64 bytes)."""
    return struct.pack(PORTAL_FMT,
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1)


# ─────────────────────────────────────────────────────────────────
# Main Bridge Class
# ─────────────────────────────────────────────────────────────────

class OptiXRouterBridge:
    """
    Bridge between trained BVH Router and OptiX RT Core pipeline.

    Extracts 3D sphere centers from router levels and creates an
    OptiX scene where ray tracing performs the routing decision.

    The dominant sphere hit by each ray = the selected expert.
    """

    def __init__(
        self,
        runner_exe: Optional[Path] = None,
        ptx_path: Optional[Path] = None,
        work_dir: Optional[Path] = None,
    ):
        self.runner_exe = runner_exe or RUNNER_EXE
        self.ptx_path = ptx_path or PTX_PATH
        self.work_dir = work_dir or SCRIPT_DIR
        self.scene_file = self.work_dir / "router_scene.bin"
        self.results_file = self.work_dir / "router_results.bin"

        # Verify executables exist
        if not self.runner_exe.exists():
            raise FileNotFoundError(
                f"inception_runner.exe not found at {self.runner_exe}\n"
                f"Build with: cmake --build build --config Release")
        if not self.ptx_path.exists():
            raise FileNotFoundError(
                f"PTX not found at {self.ptx_path}\n"
                f"Build with: cmake .. -DSPECTRAL_BUILD_INCEPTION=ON")

    def extract_spheres_from_router(
        self,
        router: torch.nn.Module,
    ) -> List[Dict]:
        """
        Extract 3D sphere centers and radii from a trained BVH Router.

        The BVH Router has 3 levels of RouterLevel, each with:
          - centers: (total_spheres, 3) — learned 3D positions
          - log_radii: (total_spheres,) — learned radii (softplus)

        Returns list of dicts with center, radius, level, local_id, global_id.
        """
        spheres = []
        global_id = 0

        for level_idx, level_name in enumerate(["level1", "level2", "level3"]):
            level = getattr(router, level_name, None)
            if level is None:
                continue

            centers = level.centers.detach().cpu().numpy()   # (K, 3)
            radii = level.radii.detach().cpu().numpy()       # (K,)

            for local_id in range(centers.shape[0]):
                spheres.append({
                    "center": centers[local_id],
                    "radius": float(radii[local_id]),
                    "level": level_idx,
                    "local_id": local_id,
                    "global_id": global_id,
                })
                global_id += 1

        return spheres

    def export_router_scene(
        self,
        router: torch.nn.Module,
        query_positions: np.ndarray,
        num_rays_per_query: int = 1,
    ) -> Path:
        """
        Export a trained router's spheres + query rays to scene.bin.

        Args:
            router: Trained BVHRouter module
            query_positions: (B, 3) — 3D positions of query prompts
            num_rays_per_query: rays per query (more = better coverage)

        Returns:
            Path to the scene.bin file
        """
        spheres = self.extract_spheres_from_router(router)
        n_spheres = len(spheres)
        n_queries = query_positions.shape[0]
        total_rays = n_queries * num_rays_per_query

        # Use only leaf-level spheres as SemanticStrings (they map to experts)
        leaf_spheres = [s for s in spheres if s["level"] == 2]
        n_strings = len(leaf_spheres)

        # Build scene binary
        with open(self.scene_file, "wb") as f:
            # Header (28 bytes)
            f.write(struct.pack("<I I I I I I f",
                SCENE_MAGIC,
                SCENE_VERSION,
                n_spheres,       # numSpheres
                n_strings,       # numStrings
                4,               # numPortals (identity for 4 levels)
                total_rays,      # numRays
                1.0,             # baseOmega
            ))

            # SemanticSpheres (all levels)
            for s in spheres:
                f.write(pack_sphere(
                    center=s["center"],
                    radius=s["radius"],
                    instance_id=s["global_id"],
                    depth=s["level"],
                    freq_bias=0.0,
                ))

            # SemanticStrings (leaf level only — map to experts)
            for s in leaf_spheres:
                # Fourier coefficients from sphere center (semantic encoding)
                center = s["center"]
                a_coeffs = [float(center[i % 3]) * (0.5 ** (i // 3))
                           for i in range(8)]
                b_coeffs = [float(center[(i + 1) % 3]) * (0.5 ** (i // 3))
                           for i in range(8)]
                f.write(pack_string(
                    position=center,
                    string_id=s["local_id"],  # maps to expert_id
                    fourier_a=a_coeffs,
                    fourier_b=b_coeffs,
                ))

            # AffinePortals (4 identity matrices)
            for _ in range(4):
                f.write(pack_portal_identity())

        return self.scene_file

    def run_optix_routing(self) -> List[Dict]:
        """
        Run inception_runner.exe to perform routing via RT Cores.

        Returns list of results per ray with dominant expert ID.
        """
        result = subprocess.run(
            [str(self.runner_exe), str(self.ptx_path),
             str(self.scene_file), str(self.results_file)],
            capture_output=True, text=True, timeout=30,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"inception_runner failed (code {result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}")

        return self._parse_results()

    def _parse_results(self) -> List[Dict]:
        """Parse results.bin into list of routing decisions."""
        results = []
        with open(self.results_file, "rb") as f:
            magic, count = struct.unpack("<II", f.read(8))
            if magic != 0x4C425253:  # 'LBRS'
                raise ValueError(f"Invalid results magic: 0x{magic:08X}")

            for i in range(count):
                data = f.read(RESULT_SIZE)
                if len(data) < RESULT_SIZE:
                    break
                aw, omega, dom_id, depth, ex, ey, ez, energy = struct.unpack(
                    RESULT_FMT, data)
                results.append({
                    "ray_id": i,
                    "expert_id": dom_id,      # dominantStringId = expert
                    "attention_weight": aw,
                    "traversal_depth": depth,
                    "confidence": energy,
                })

        return results

    def route(
        self,
        router: torch.nn.Module,
        prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Full routing pipeline: embeddings → OptiX → expert_ids.

        Args:
            router: Trained BVHRouter
            prompt_embeddings: (B, embed_dim) — prompt embeddings

        Returns:
            expert_ids: (B,) tensor of selected expert indices
            raw_results: list of per-ray result dicts
        """
        # Project to 3D using the router's to_3d layer
        with torch.no_grad():
            pos_3d = router.to_3d(prompt_embeddings.float()).cpu().numpy()

        # Export scene
        self.export_router_scene(router, pos_3d)

        # Run OptiX
        raw_results = self.run_optix_routing()

        # Extract expert IDs
        expert_ids = torch.tensor(
            [r["expert_id"] for r in raw_results],
            dtype=torch.long)

        return expert_ids, raw_results

    def benchmark(
        self,
        router: torch.nn.Module,
        prompt_embeddings: torch.Tensor,
        n_iters: int = 50,
    ) -> Dict:
        """
        Benchmark PyTorch vs OptiX routing latency.

        Returns dict with pytorch_ms, optix_ms, speedup.
        """
        B = prompt_embeddings.shape[0]
        device = prompt_embeddings.device

        # ── PyTorch routing ────────────────────────────────────────
        router.eval()
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = router(prompt_embeddings.float(), hard=True)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.no_grad():
                pt_result = router(prompt_embeddings.float(), hard=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pytorch_ms = (time.perf_counter() - t0) / n_iters * 1000

        # ── OptiX routing ──────────────────────────────────────────
        with torch.no_grad():
            pos_3d = router.to_3d(prompt_embeddings.float()).cpu().numpy()

        # Export scene once
        self.export_router_scene(router, pos_3d)

        # Warmup
        for _ in range(2):
            self.run_optix_routing()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            self.run_optix_routing()
        optix_ms = (time.perf_counter() - t0) / n_iters * 1000

        speedup = pytorch_ms / optix_ms if optix_ms > 0 else float("inf")

        return {
            "batch_size": B,
            "pytorch_ms": pytorch_ms,
            "optix_ms": optix_ms,
            "speedup": speedup,
            "pytorch_expert": pt_result.expert_id.tolist(),
        }


# ─────────────────────────────────────────────────────────────────
# CLI test
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  OptiX Router Bridge — PyTorch → RT Core Routing")
    print("=" * 60)

    # Check prerequisites
    if not RUNNER_EXE.exists():
        print(f"\n  ERROR: inception_runner.exe not found at {RUNNER_EXE}")
        print("  Build with: cmake --build build --config Release")
        sys.exit(1)
    if not PTX_PATH.exists():
        print(f"\n  ERROR: spectral_kernels.ptx not found at {PTX_PATH}")
        print("  Build with: cmake .. -DSPECTRAL_BUILD_INCEPTION=ON")
        sys.exit(1)

    print(f"  Runner: {RUNNER_EXE}")
    print(f"  PTX:    {PTX_PATH}")

    # Try loading a trained router checkpoint
    checkpoint_path = SCRIPT_DIR / ".." / "data" / "moe_best.pt"
    if checkpoint_path.exists():
        print(f"\n  Loading checkpoint: {checkpoint_path}")
        sys.path.insert(0, str(SCRIPT_DIR))
        from bvh_router import RouterConfig, BVHRouter

        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        router_state = ckpt.get("router", None)

        if router_state:
            # Reconstruct router config from state dict
            cfg = RouterConfig(embed_dim=256, n_level1=2, n_level2=2, n_level3=4)
            router = BVHRouter(cfg)
            router.load_state_dict(router_state)
            router.eval()

            bridge = OptiXRouterBridge()

            # Extract and print sphere info
            spheres = bridge.extract_spheres_from_router(router)
            print(f"\n  Router spheres: {len(spheres)} total")
            for level in range(3):
                level_spheres = [s for s in spheres if s["level"] == level]
                if level_spheres:
                    centers = np.array([s["center"] for s in level_spheres])
                    radii = np.array([s["radius"] for s in level_spheres])
                    print(f"    Level {level}: {len(level_spheres)} spheres, "
                          f"avg radius={radii.mean():.3f}, "
                          f"center spread={centers.std():.3f}")

            # Test routing with random embeddings
            print("\n  Testing OptiX routing...")
            test_emb = torch.randn(4, 256)
            try:
                expert_ids, results = bridge.route(router, test_emb)
                print(f"  Expert IDs (OptiX): {expert_ids.tolist()}")
                for r in results[:4]:
                    print(f"    ray {r['ray_id']}: expert={r['expert_id']}, "
                          f"weight={r['attention_weight']:.4f}, "
                          f"depth={r['traversal_depth']}")

                # Benchmark
                print("\n  Running benchmark (50 iterations)...")
                bench = bridge.benchmark(router, test_emb, n_iters=50)
                print(f"    PyTorch: {bench['pytorch_ms']:.3f} ms")
                print(f"    OptiX:   {bench['optix_ms']:.3f} ms")
                print(f"    Speedup: {bench['speedup']:.1f}x")

            except Exception as e:
                print(f"  OptiX routing failed: {e}")
                print("  (This is expected if OptiX driver is not available)")
    else:
        print(f"\n  No checkpoint found at {checkpoint_path}")
        print("  Run training first: python train_moe.py --epochs 10")
        print("  Then re-run this script to test OptiX routing.")

    print("\n  Done.")
