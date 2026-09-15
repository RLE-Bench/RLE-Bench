"""Pure verifier metrics with saturating rewards."""

from __future__ import annotations

import numpy as np

from . import config


def descending(value: float, excellent: float, fail: float) -> float:
    return float(np.clip((fail - value) / (fail - excellent), 0.0, 1.0))


def tracking_score(position_rmse: float) -> float:
    return descending(position_rmse, config.TRACK_EXCELLENT_M, config.MAX_POSITION_RMSE_M)


def contact_score(position_rmse: float, force_rmse: float, peak_force: float) -> float:
    position = descending(position_rmse, 0.022, 0.14)
    force = descending(force_rmse, config.CONTACT_FORCE_EXCELLENT_N, config.CONTACT_FORCE_FAIL_N)
    peak = descending(peak_force, 18.0, config.MAX_CONTACT_FORCE_N)
    return 0.45 * position + 0.40 * force + 0.15 * peak


def recovery_score(position_rmse: float, recovery_seconds: float) -> float:
    return 0.55 * tracking_score(position_rmse) + 0.45 * descending(recovery_seconds, config.RECOVERY_EXCELLENT_S, config.RECOVERY_FAIL_S)
