"""Deterministic per-stage random streams.

All random draws are made on the CPU so a given seed gives identical inputs on
any device; arrays are moved to the compute device afterwards.
"""

import numpy as np
import torch

STAGES = ("envelope", "turing", "atoms", "reference", "noise_half1", "noise_half2")


def stage_generator(seed: int, stage: str, attempt: int = 0) -> torch.Generator:
    ss = np.random.SeedSequence([int(seed), STAGES.index(stage), int(attempt)])
    g = torch.Generator(device="cpu")
    g.manual_seed(int(ss.generate_state(1, np.uint64)[0] >> np.uint64(1)))
    return g


def randn(shape, gen: torch.Generator, device) -> torch.Tensor:
    return torch.randn(shape, generator=gen, dtype=torch.float32).to(device)


def rand(shape, gen: torch.Generator, device) -> torch.Tensor:
    return torch.rand(shape, generator=gen, dtype=torch.float32).to(device)
