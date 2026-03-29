#!/usr/bin/env python3
"""
async_pipeline_bridge.py — Python bridge for the tri-core async pipeline.

Simulates the triple-buffered pipeline in Python for correctness testing
before the CUDA implementation is fully compiled. Also provides the Python
API that wraps the compiled C library.

Pipeline stages:
  Stage 0: RT Core routing  (or CUDA kernel fallback)
  Stage 1: Scatter + calibration (CUDA cores)
  Stage 2: Expert forward pass (Tensor cores via cuBLAS)

Triple buffering: stages N, N-1, N-2 run concurrently on different streams.

Copyright (c) 2026 LiquidBit Studio — Apache 2.0
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the async pipeline."""
    num_experts: int = 64
    top_k: int = 8
    hidden_dim: int = 2048
    inter_dim: int = 1024
    max_batch: int = 512
    num_buffers: int = 3  # Triple buffering
    use_cuda_streams: bool = True


@dataclass
class PipelineSlot:
    """One slot in the triple buffer."""
    hidden: Optional[torch.Tensor] = None
    expert_ids: Optional[torch.Tensor] = None
    expert_weights: Optional[torch.Tensor] = None
    dispatch_indices: Optional[dict] = None  # expert_id -> list of token indices
    output: Optional[torch.Tensor] = None
    batch_size: int = 0


class AsyncPipelineSimulator:
    """
    Pure-Python simulation of the tri-core async pipeline.

    Used for correctness verification before the CUDA implementation.
    Also useful for profiling the pipeline stages independently.
    """

    def __init__(
        self,
        config: PipelineConfig,
        router: torch.nn.Module,
        experts: torch.nn.ModuleList,
        calibration_state: Optional[dict] = None,
        device: str = "cuda",
    ):
        self.config = config
        self.router = router
        self.experts = experts
        self.device = device

        # Calibration layer (64->64 linear)
        self.cal_layer: Optional[torch.nn.Linear] = None
        if calibration_state is not None:
            self.cal_layer = torch.nn.Linear(
                config.num_experts, config.num_experts
            )
            self.cal_layer.load_state_dict(calibration_state)
            self.cal_layer = self.cal_layer.to(device).eval()

        # Triple buffer
        self.slots = [PipelineSlot() for _ in range(config.num_buffers)]
        self.step = 0

        # CUDA streams for actual overlap (if available)
        self.streams: Optional[list] = None
        if config.use_cuda_streams and torch.cuda.is_available():
            self.streams = [torch.cuda.Stream() for _ in range(3)]

    def _stage_route(self, slot: PipelineSlot, hidden: torch.Tensor) -> None:
        """Stage 0: Route tokens to experts via BVH router."""
        slot.hidden = hidden
        slot.batch_size = hidden.shape[0]

        with torch.no_grad():
            probs, _ = self.router(hidden)

            # Apply calibration if available
            if self.cal_layer is not None:
                raw_logits = self.router._last_logits
                cal_logits = self.cal_layer(raw_logits)
                probs = F.softmax(cal_logits, dim=-1)

            # Top-K selection
            weights, ids = torch.topk(probs, self.config.top_k, dim=-1)
            slot.expert_ids = ids
            slot.expert_weights = weights

    def _stage_scatter(self, slot: PipelineSlot) -> None:
        """Stage 1: Scatter tokens by expert assignment."""
        dispatch: dict[int, list[int]] = {}
        ids = slot.expert_ids  # [batch, top_k]

        for token_idx in range(slot.batch_size):
            for k in range(self.config.top_k):
                expert_id = ids[token_idx, k].item()
                if expert_id not in dispatch:
                    dispatch[expert_id] = []
                dispatch[expert_id].append(token_idx)

        slot.dispatch_indices = dispatch

    def _stage_expert(self, slot: PipelineSlot) -> None:
        """Stage 2: Run expert forward passes and combine outputs."""
        hidden_dim = slot.hidden.shape[1]
        output = torch.zeros(
            slot.batch_size, hidden_dim,
            device=self.device, dtype=slot.hidden.dtype
        )

        with torch.no_grad():
            for expert_id, token_indices in slot.dispatch_indices.items():
                if expert_id >= len(self.experts):
                    continue

                indices_t = torch.tensor(token_indices, device=self.device)
                expert_input = slot.hidden[indices_t]

                # Forward through expert
                expert_output = self.experts[expert_id](expert_input)

                # Weighted accumulate (find which k-slot each token used this expert)
                for local_idx, global_idx in enumerate(token_indices):
                    k_mask = (slot.expert_ids[global_idx] == expert_id)
                    weight = slot.expert_weights[global_idx][k_mask].sum()
                    output[global_idx] += weight * expert_output[local_idx].detach()

        slot.output = output

    def forward_sequential(self, hidden: torch.Tensor) -> torch.Tensor:
        """Sequential (non-pipelined) forward for correctness baseline."""
        slot = PipelineSlot()
        self._stage_route(slot, hidden)
        self._stage_scatter(slot)
        self._stage_expert(slot)
        return slot.output

    def forward_pipelined(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Pipelined forward using CUDA streams.

        In steady state, all 3 stages overlap on different buffer slots.
        """
        if self.streams is None:
            return self.forward_sequential(hidden)

        slot_idx = self.step % self.config.num_buffers
        slot = self.slots[slot_idx]

        # Stage 0: Route (high priority stream)
        with torch.cuda.stream(self.streams[0]):
            self._stage_route(slot, hidden)

        # Stage 1: Scatter on previous slot (medium priority stream)
        if self.step >= 1:
            prev_slot = self.slots[(self.step - 1) % self.config.num_buffers]
            self.streams[1].wait_stream(self.streams[0])
            with torch.cuda.stream(self.streams[1]):
                self._stage_scatter(prev_slot)

        # Stage 2: Expert on slot two steps back (low priority stream)
        if self.step >= 2:
            pprev_slot = self.slots[(self.step - 2) % self.config.num_buffers]
            self.streams[2].wait_stream(self.streams[1])
            with torch.cuda.stream(self.streams[2]):
                self._stage_expert(pprev_slot)

        self.step += 1

        # Return output from completed slot (2 steps behind)
        if self.step >= 3:
            done_slot = self.slots[(self.step - 3) % self.config.num_buffers]
            torch.cuda.current_stream().wait_stream(self.streams[2])
            return done_slot.output

        return None  # Pipeline warming up

    def flush(self) -> list[torch.Tensor]:
        """Flush remaining pipeline stages after last input."""
        outputs = []
        for _ in range(2):  # Drain 2 remaining stages
            self.step += 1
            if self.step >= 3:
                slot = self.slots[(self.step - 3) % self.config.num_buffers]
                if slot.dispatch_indices is None:
                    self._stage_scatter(slot)
                if slot.output is None:
                    self._stage_expert(slot)
                outputs.append(slot.output)
        return outputs

    def benchmark(
        self,
        batch_size: int = 64,
        num_steps: int = 100,
        warmup: int = 10,
    ) -> dict:
        """Benchmark pipeline stages independently and combined."""
        import time

        hidden = torch.randn(
            batch_size, self.config.hidden_dim,
            device=self.device
        )

        # Warmup
        for _ in range(warmup):
            _ = self.forward_sequential(hidden)
        torch.cuda.synchronize()

        # Benchmark sequential
        t0 = time.perf_counter()
        for _ in range(num_steps):
            _ = self.forward_sequential(hidden)
        torch.cuda.synchronize()
        seq_time = (time.perf_counter() - t0) / num_steps

        # Benchmark pipelined
        self.step = 0
        for _ in range(warmup):
            _ = self.forward_pipelined(hidden)
        torch.cuda.synchronize()

        self.step = 0
        t0 = time.perf_counter()
        for _ in range(num_steps + 2):  # +2 for warmup
            _ = self.forward_pipelined(hidden)
        torch.cuda.synchronize()
        pipe_time = (time.perf_counter() - t0) / num_steps

        return {
            "sequential_ms": seq_time * 1000,
            "pipelined_ms": pipe_time * 1000,
            "speedup": seq_time / pipe_time if pipe_time > 0 else 0,
            "batch_size": batch_size,
            "num_steps": num_steps,
        }
