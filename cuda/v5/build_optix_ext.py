"""
build_optix_ext.py -- Compile the OptiX RT Core Training Extension for PyTorch
SpectralAI v5.0

This extension bridges OptiX RT Core hardware to PyTorch's training loop,
enabling hardware-accelerated BVH traversal for expert routing.

Prerequisites:
    - NVIDIA GPU with RT Cores (RTX 20xx+)
    - OptiX SDK 8.x or 9.x installed
    - CUDA Toolkit 12.x
    - PyTorch with CUDA support
    - PTX shaders compiled (cmake --build build)

Usage:
    cd /path/to/spectral-ai
    python cuda/v5/build_optix_ext.py

    # Or specify OptiX path explicitly:
    OPTIX_DIR="C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0" \
        python cuda/v5/build_optix_ext.py

Copyright (c) 2026 SpectralAI Studio -- Apache 2.0
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Resolve project root (spectral-ai/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)

import torch
import torch.utils.cpp_extension as cpp_ext


# ── Environment Detection ──────────────────────────────────────────────

def _detect_optix_include() -> Optional[str]:
    """Find OptiX include directory from environment or standard paths."""
    # 1. Explicit environment variable
    env_dir = os.environ.get("OPTIX_DIR") or os.environ.get("OptiX_INSTALL_DIR")
    if env_dir:
        inc = Path(env_dir) / "include"
        if (inc / "optix.h").exists():
            return str(inc)

    # 2. Standard installation paths
    candidates = [
        Path("C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include"),
        Path("C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0/include"),
        Path("/opt/nvidia/optix/include"),
        Path("/usr/local/optix/include"),
        Path.home() / "optix" / "include",
    ]

    for c in candidates:
        if (c / "optix.h").exists():
            return str(c)

    return None


def _detect_ptx_paths() -> Tuple[Optional[str], Optional[str]]:
    """Find compiled PTX shader files from cmake build."""
    search_dirs = [
        PROJECT_ROOT / "build" / "ptx",
        PROJECT_ROOT / "build" / "Release" / "ptx",
        PROJECT_ROOT / "build" / "Debug" / "ptx",
        PROJECT_ROOT / "build",
    ]

    raygen_path = None
    hitgroup_path = None

    for d in search_dirs:
        if not d.exists():
            continue
        for f in d.glob("*.ptx"):
            name = f.stem.lower()
            if "raygen" in name and "router" in name:
                raygen_path = str(f)
            elif "hitgroup" in name and "router" in name:
                hitgroup_path = str(f)

    return raygen_path, hitgroup_path


def _get_cuda_arch_flags() -> List[str]:
    """Detect GPU compute capability and return nvcc arch flags."""
    if not torch.cuda.is_available():
        return []
    cap = torch.cuda.get_device_capability(0)
    arch = f"{cap[0]}{cap[1]}"
    return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]


# ── Build the Extension ────────────────────────────────────────────────

def build_optix_training_ext() -> bool:
    """Compile and load the optix_training_ext PyTorch extension."""

    print("=" * 70)
    print("SpectralAI OptiX RT Core Training Extension — Build")
    print("=" * 70)
    print(f"PyTorch:        {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU:            {torch.cuda.get_device_name(0)}")
        cap = torch.cuda.get_device_capability(0)
        print(f"Compute cap:    {cap[0]}.{cap[1]}")

    # Check OptiX
    optix_inc = _detect_optix_include()
    if optix_inc is None:
        print("\n[ERROR] OptiX SDK not found.")
        print("Set OPTIX_DIR environment variable to your OptiX installation.")
        print("Example: set OPTIX_DIR=C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 9.1.0")
        return False
    print(f"OptiX include:  {optix_inc}")

    # ── Spaces-in-path workaround for nvcc ────────────────────────────────
    # nvcc fatal: "A single input file is required" — triggered when ANY -I flag
    # path contains spaces.  Ninja passes flags unquoted; nvcc splits on the space.
    # Fix: create no-spaces symlinks in /tmp for every affected path.

    def _safe_path(original: str, link_name: str) -> str:
        """If *original* has spaces, create /tmp/<link_name> -> original."""
        if " " not in original:
            return original
        safe = f"/tmp/{link_name}"
        try:
            if os.path.islink(safe):
                os.unlink(safe)
            os.symlink(original, safe)
            print(f"  Symlinked: {safe} -> {original}")
            return safe
        except OSError as exc:
            print(f"  [WARNING] Cannot symlink {original}: {exc}")
            return original

    # Actual symlink calls happen later, after include_dir is defined (line ~183).
    # For now, just apply to optix_inc which is already available:
    optix_inc = _safe_path(optix_inc, "optix_sdk_inc")

    # Check PTX files
    raygen_ptx, hitgroup_ptx = _detect_ptx_paths()
    if raygen_ptx:
        print(f"PTX raygen:     {raygen_ptx}")
    if hitgroup_ptx:
        print(f"PTX hitgroup:   {hitgroup_ptx}")
    if not raygen_ptx or not hitgroup_ptx:
        print("\n[WARNING] PTX shaders not found. Build with CMake first:")
        print("  cd build && cmake .. && cmake --build . --config Release")
        print("The extension will compile, but you'll need PTX at runtime.")

    # Source files
    ext_cu = PROJECT_ROOT / "cuda" / "v5" / "optix_training_ext.cu"
    router_host = PROJECT_ROOT / "cuda" / "optix_router_host.cpp"

    if not ext_cu.exists():
        print(f"\n[ERROR] Source not found: {ext_cu}")
        return False
    if not router_host.exists():
        print(f"\n[ERROR] Source not found: {router_host}")
        return False

    # Build directory (no spaces in path)
    build_dir = os.path.expanduser("~/.cache/torch_extensions/optix_training_ext")
    os.makedirs(build_dir, exist_ok=True)

    # On Windows, the .cu file contains NO device code (no __global__ /
    # __device__).  Compiling as .cpp avoids nvcc entirely, which avoids
    # the CCCL / MSVC header conflicts (CUDA 13 CCCL vs PyTorch cu128).
    _is_win = sys.platform == "win32"
    if _is_win:
        safe_ext = Path(build_dir) / "optix_training_ext.cpp"
    else:
        safe_ext = Path(build_dir) / "optix_training_ext.cu"
    shutil.copy2(ext_cu, safe_ext)

    # The .cu file does `#include "../optix_router_host.cpp"` (relative path).
    # Since safe_ext is in build_dir, we need the host file one level up.
    parent_of_build = Path(build_dir).parent
    safe_host = parent_of_build / "optix_router_host.cpp"
    shutil.copy2(router_host, safe_host)
    print(f"Host source:    {safe_host}")

    # Also copy any required headers — apply space-safe symlinks
    include_dir = _safe_path(str(PROJECT_ROOT / "include"), "spectral_include")
    cuda_dir_str = _safe_path(str(PROJECT_ROOT / "cuda"), "spectral_cuda")

    # ── Workaround for spaces in torch package path ─────────────────────
    # nvcc/ninja/ld all break when -L or -I paths contain spaces.
    # PyTorch's cpp_extension derives paths from torch.__file__ in MULTIPLE
    # places (library_paths(), _prepare_ldflags, ninja file generation).
    # Patching library_paths alone is insufficient — we must redirect the
    # ROOT reference: torch.__file__ and torch.__path__[0].
    # Strategy: symlink the entire torch package dir to a safe path, then
    # temporarily point torch.__file__ / torch.__path__ through it.
    torch_pkg_dir = str(Path(torch.__file__).parent)  # .../site-packages/torch
    torch_lib_dir = str(Path(torch_pkg_dir) / "lib")
    needs_patch = " " in torch_pkg_dir
    safe_torch_lib = torch_lib_dir

    _original_torch_file = torch.__file__
    _original_torch_path = list(torch.__path__) if torch.__path__ else []

    if needs_patch:
        safe_torch_pkg = "/tmp/_torch_pkg_optix"
        if sys.platform == "win32":
            safe_torch_pkg = os.path.join(
                os.environ.get("TEMP", "C:\\Temp"), "_torch_pkg_optix"
            )
        if os.path.islink(safe_torch_pkg):
            os.unlink(safe_torch_pkg)
        try:
            os.symlink(torch_pkg_dir, safe_torch_pkg)
            print(f"  Torch pkg symlinked: {safe_torch_pkg} -> {torch_pkg_dir}")

            # Redirect ALL internal torch path references through the symlink
            torch.__file__ = torch.__file__.replace(torch_pkg_dir, safe_torch_pkg)
            if torch.__path__:
                torch.__path__[0] = safe_torch_pkg

            safe_torch_lib = str(Path(safe_torch_pkg) / "lib")
            print(f"  Torch lib now:       {safe_torch_lib}")

            # Patch module-level path variables computed at import time.
            # These are the REAL source of the second -L flag with spaces.
            if hasattr(cpp_ext, '_TORCH_PATH'):
                cpp_ext._TORCH_PATH = safe_torch_pkg
                print(f"  _TORCH_PATH patched: {safe_torch_pkg}")
            if hasattr(cpp_ext, 'TORCH_LIB_PATH'):
                cpp_ext.TORCH_LIB_PATH = safe_torch_lib
                print(f"  TORCH_LIB_PATH patched: {safe_torch_lib}")

            # Also patch library_paths for extra safety
            _orig_lib_paths = cpp_ext.library_paths
            def _patched_lib_paths(cuda: bool = False) -> List[str]:
                paths = _orig_lib_paths(cuda)
                return [
                    p.replace(torch_pkg_dir, safe_torch_pkg)
                    if torch_pkg_dir in p else p
                    for p in paths
                ]
            cpp_ext.library_paths = _patched_lib_paths
        except OSError as exc:
            print(f"[WARNING] Cannot create symlink for torch package path: {exc}")

    # ── Platform-aware compile flags ─────────────────────────────────────
    is_windows = sys.platform == "win32"
    cuda_arch_flags = _get_cuda_arch_flags()

    if is_windows:
        # On Windows we compile as .cpp (pure host code, no device kernels)
        # to bypass nvcc entirely and avoid CCCL / MSVC header conflicts.
        # We only need extra_cflags and extra_ldflags.
        cuda_home_win = os.environ.get(
            "CUDA_HOME",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        )
        cuda_inc_win = str(Path(cuda_home_win) / "include")

        extra_cuda_cflags = []  # Not used (no .cu file on Windows)
        extra_cflags = [
            f"/I{optix_inc}",
            f"/I{include_dir}",
            f"/I{cuda_dir_str}",
            f"/I{cuda_inc_win}",
            "/DOPTIX_TRAINING_STANDALONE",
            "/O2",
            "/EHsc",
        ]
    else:
        extra_cuda_cflags = [
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            f"-I{optix_inc}",
            f"-I{include_dir}",
            f"-I{cuda_dir_str}",
            "-DOPTIX_TRAINING_STANDALONE",
        ] + cuda_arch_flags

        extra_cflags = [
            f"-I{optix_inc}",
            f"-I{include_dir}",
            f"-I{cuda_dir_str}",
            "-DOPTIX_TRAINING_STANDALONE",
        ]

    # ── CUDA Driver API library ───────────────────────────────────────────
    # OptiX uses cuDeviceGet, cuCtxGetCurrent, etc. from the Driver API.
    # On Linux this is libcuda.so; on Windows it's nvcuda.lib / nvcuda.dll.
    extra_ldflags = []

    if is_windows:
        # On Windows, nvcuda.dll is in System32 and nvcuda.lib is in the
        # CUDA toolkit.  The linker finds it automatically via CUDA_HOME.
        cuda_home_win = os.environ.get(
            "CUDA_HOME",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2",
        )
        nvcuda_lib_dir = Path(cuda_home_win) / "lib" / "x64"
        if (nvcuda_lib_dir / "cuda.lib").exists():
            extra_ldflags.append(f"/LIBPATH:{nvcuda_lib_dir}")
            extra_ldflags.append("cuda.lib")
            extra_ldflags.append("cudart.lib")   # CUDA Runtime API
            extra_ldflags.append("advapi32.lib")  # Windows Registry (OptiX DLL loading)
            print(f"  CUDA Driver lib:     {nvcuda_lib_dir / 'cuda.lib'}")
            print(f"  CUDA Runtime lib:    {nvcuda_lib_dir / 'cudart.lib'}")
        else:
            print(f"  [WARNING] cuda.lib not found at {nvcuda_lib_dir}")
            print("  OptiX init may fail at runtime.")
    else:
        # Linux / WSL2
        cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
        cuda_driver_lib_candidates = [
            Path(cuda_home) / "lib64" / "stubs",
            Path(cuda_home) / "lib" / "stubs",
            Path(cuda_home) / "targets" / "x86_64-linux" / "lib" / "stubs",
            Path(cuda_home) / "compat",
            Path("/usr/lib/wsl/lib"),
            Path("/usr/lib/x86_64-linux-gnu"),
            Path(cuda_home) / "lib64",
            Path(cuda_home) / "lib",
        ]

        cuda_driver_found = False
        for cand in cuda_driver_lib_candidates:
            if cand.exists() and any(cand.glob("libcuda.so*")):
                safe_cand = _safe_path(str(cand), "cuda_driver_lib")
                extra_ldflags.append(f"-L{safe_cand}")
                extra_ldflags.append("-lcuda")
                print(f"  CUDA Driver lib:     {cand} (via {safe_cand})")
                cuda_driver_found = True
                break

        if not cuda_driver_found:
            print("  [WARNING] libcuda.so not found — OptiX init may fail at runtime")
            print("  Set CUDA_HOME or ensure libcuda.so is in your library path")
            extra_ldflags.append("-lcuda")

    if needs_patch:
        extra_ldflags.append(f"-L{safe_torch_lib}")

    print(f"\n{'='*70}")
    print("Compiling optix_training_ext ...")
    print(f"{'='*70}")

    try:
        ext = cpp_ext.load(
            name="optix_training_ext",
            sources=[str(safe_ext)],
            extra_cuda_cflags=extra_cuda_cflags,
            extra_cflags=extra_cflags,
            extra_ldflags=extra_ldflags,
            build_directory=build_dir,
            verbose=True,
        )
    except Exception as exc:
        print(f"\n[ERROR] Compilation failed: {exc}")
        print("\nTroubleshooting:")
        print("  1. Ensure CUDA Toolkit 12.x is installed")
        print("  2. Ensure OptiX SDK 8.x/9.x is installed")
        print("  3. Check OPTIX_DIR points to the correct path")
        print("  4. Try: pip install ninja  (for faster builds)")
        return False
    finally:
        # Restore torch paths so other imports aren't affected
        if needs_patch:
            torch.__file__ = _original_torch_file
            if _original_torch_path:
                torch.__path__[:] = _original_torch_path

    print(f"\n{'='*70}")
    print("Extension compiled successfully!")
    print(f"Module: {ext}")
    print(f"Functions: initialize, build_gas, route, route_topk, is_ready, shutdown")
    print(f"{'='*70}")

    # Register in sys.modules so `import optix_training_ext` works
    sys.modules["optix_training_ext"] = ext

    return True


# ── Quick Verification Test ────────────────────────────────────────────

def run_quick_test() -> bool:
    """Verify the extension works with a basic test."""
    try:
        import optix_training_ext as ext
    except ImportError:
        print("[ERROR] optix_training_ext not importable after build")
        return False

    print("\n=== Quick Verification Test ===")

    # Check initial state
    ready = ext.is_ready()
    print(f"is_ready() before init: {ready}")

    # Try to initialize (will fail if PTX not found, which is OK)
    raygen_ptx, hitgroup_ptx = _detect_ptx_paths()
    if raygen_ptx and hitgroup_ptx:
        print(f"Initializing with PTX files...")
        try:
            ext.initialize(raygen_ptx, hitgroup_ptx)
            print("initialize(): OK")

            # Build a small test GAS
            import torch
            num_experts = 8
            centers = torch.randn(num_experts, 3, dtype=torch.float32)
            radii = torch.ones(num_experts, dtype=torch.float32) * 0.5

            ext.build_gas(centers, radii, False)
            print(f"build_gas(): OK ({num_experts} experts)")
            print(f"  GAS size: {ext.gas_size() / 1024:.1f} KB")
            print(f"  num_experts: {ext.num_experts()}")

            # Route test batch
            batch_size = 16
            positions = torch.randn(batch_size, 3, dtype=torch.float32, device="cuda")
            directions = torch.randn(batch_size, 3, dtype=torch.float32, device="cuda")
            directions = directions / directions.norm(dim=-1, keepdim=True)

            expert_ids, distances = ext.route(positions, directions)
            print(f"route(): OK")
            print(f"  expert_ids: {expert_ids[:8].tolist()}")
            print(f"  distances:  {[f'{d:.3f}' for d in distances[:8].tolist()]}")

            # Top-K test
            topk_ids, topk_dists = ext.route_topk(positions, directions, 4)
            print(f"route_topk(k=4): OK, shape={topk_ids.shape}")

            # Cleanup
            ext.shutdown()
            print("shutdown(): OK")

            print("\n=== ALL TESTS PASSED ===")
            return True

        except RuntimeError as exc:
            print(f"[WARNING] Runtime test failed: {exc}")
            print("This may be normal if OptiX drivers are not available.")
            return False
    else:
        print("[INFO] PTX files not found. Skipping runtime test.")
        print("Build PTX with: cd build && cmake --build . --config Release")
        print("Extension compiled OK — will work once PTX is available.")
        return True


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    success = build_optix_training_ext()
    if success:
        run_quick_test()
    else:
        sys.exit(1)
