"""Verifier-only deterministic hidden fault generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fault:
    zero_offset: tuple[float, ...]
    gain: tuple[float, ...]
    friction: tuple[float, ...]
    delay_steps: int
    payload_scale: float
    weak_joint: int
    derate_factor: float
    crosstalk: tuple[tuple[float, ...], ...]
    impulse_joint: int
    impulse_torque: float


def sample_fault(seed: int) -> Fault:
    rng = np.random.default_rng(seed)
    gain = rng.uniform(0.72, 1.22, 7)
    weak_joint = int(rng.integers(0, 7))
    gain[weak_joint] *= rng.uniform(0.62, 0.78)
    coupling = np.eye(7)
    for joint in range(6):
        coupling[joint, joint + 1] = rng.uniform(-0.11, 0.11)
        coupling[joint + 1, joint] = rng.uniform(-0.07, 0.07)
    return Fault(
        tuple(rng.uniform(-0.14, 0.14, 7)),
        tuple(gain),
        tuple(rng.uniform(0.0, 3.0, 7)),
        int(rng.integers(0, 4)),
        float(rng.uniform(1.0, 2.6)),
        weak_joint,
        float(rng.uniform(0.62, 0.78)),
        tuple(tuple(float(value) for value in row) for row in coupling),
        int(rng.integers(0, 4)),
        float(rng.choice((-1.0, 1.0)) * rng.uniform(22.0, 38.0)),
    )
