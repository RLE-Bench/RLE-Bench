"""Analytic optima and counters for the task03 tabletop scenes (pure math)."""

from __future__ import annotations
import os
import pytest
if os.environ.get("RLEBENCH_TEST_TASK") != "task03":
    pytest.skip("family-specific metrics", allow_module_level=True)


import itertools

import pytest

from harness.tabletop import analytic as A


# -- tower ---------------------------------------------------------------------

def test_tower_optimum_sums_stackable_heights_only():
    pieces = [
        A.Piece("box", 0.064), A.Piece("plate", 0.012), A.Piece("cylinder", 0.082),
        A.Piece("rod", 0.120), A.Piece("sphere", 0.060), A.Piece("capsule", 0.096),
        A.Piece("wedge", 0.050),
    ]
    assert A.tower_optimum(pieces) == pytest.approx(0.064 + 0.012 + 0.082 + 0.120)


def test_tower_optimum_matches_brute_force_over_subsets():
    """The optimum equals the best over every subset/order under the physical
    rule 'only flat-stackable pieces add height': brute force on a small set."""
    pieces = [A.Piece("box", 0.05), A.Piece("sphere", 0.08),
              A.Piece("plate", 0.01), A.Piece("wedge", 0.09)]
    stackable = {"box", "plate", "cylinder", "rod"}
    best = 0.0
    for r in range(1, len(pieces) + 1):
        for combo in itertools.combinations(pieces, r):
            if all(p.kind in stackable for p in combo):
                best = max(best, sum(p.height for p in combo))
    assert A.tower_optimum(pieces) == pytest.approx(best)


def test_tower_optimum_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        A.tower_optimum([A.Piece("pyramid", 0.1)])


# -- cantilever ------------------------------------------------------------------

def test_harmonic_overhang_closed_form():
    L = 0.12
    assert A.harmonic_overhang(1, L) == pytest.approx(L / 2)
    assert A.harmonic_overhang(4, L) == pytest.approx((L / 2) * (25 / 12))


def test_harmonic_overhang_needs_a_block():
    with pytest.raises(ValueError):
        A.harmonic_overhang(0, 0.12)


# -- balance ---------------------------------------------------------------------

def test_weighing_optimum_is_ternary():
    assert A.weighing_optimum(3) == 1
    assert A.weighing_optimum(9) == 2
    assert A.weighing_optimum(12) == 3
    assert A.weighing_optimum(27) == 3
    assert A.weighing_optimum(28) == 4


def test_weighing_efficiency_regret():
    assert A.weighing_efficiency(3, 3) == 1.0
    assert A.weighing_efficiency(1, 3) == 1.0
    assert A.weighing_efficiency(6, 3) == pytest.approx(0.5)


def _flags(*runs):
    """[(value, count), ...] -> flat flag list."""
    out = []
    for value, count in runs:
        out += [value] * count
    return out


def test_weighing_counter_counts_debounced_intervals():
    flags = _flags((False, 10), (True, 30), (False, 30), (True, 30), (False, 10))
    assert A.count_weighings(flags, min_on=25, min_off=25) == 2


def test_weighing_counter_ignores_short_blips():
    # a 5-step contact flicker neither starts a weighing...
    assert A.count_weighings(_flags((True, 5), (False, 50)), min_on=25) == 0
    # ...nor splits one in two
    flags = _flags((True, 30), (False, 5), (True, 30), (False, 30))
    assert A.count_weighings(flags, min_on=25, min_off=25) == 1


def test_weighing_counter_counts_balanced_weighings():
    # a level beam with both pans loaded is still one weighing (its outcome is
    # "equal", which is information); the counter reads load, not tilt
    assert A.count_weighings(_flags((True, 100)), min_on=25) == 1


def test_weighing_counter_distinct_configs_count_separately():
    """Rearranging coins while both pans stay loaded is still a new weighing --
    the exploit that interval counting would miss."""
    c = A.WeighingCounter(min_on=5, min_off=5)
    for _ in range(10):
        c.update(True, config=("ab", "cd"))
    for _ in range(10):
        c.update(True, config=("a", "c"))       # reconfigured, never unloaded
    assert c.count == 2
    # repeating the LAST comparison adds nothing
    for _ in range(10):
        c.update(False)
    for _ in range(10):
        c.update(True, config=("a", "c"))
    assert c.count == 2
    # a config flicker neither counts nor forgets the standing comparison
    c.update(True, config=("x", "y"))
    for _ in range(10):
        c.update(True, config=("a", "c"))
    assert c.count == 2


def test_weighing_counter_incremental_matches_batch():
    flags = _flags((True, 26), (False, 26), (True, 24), (False, 2), (True, 26),
                   (False, 26), (True, 26))
    counter = A.WeighingCounter(min_on=25, min_off=25)
    for f in flags:
        counter.update(f)
    assert counter.count == A.count_weighings(flags, min_on=25, min_off=25)


@pytest.mark.parametrize("quality", [0, .2, .6, 1])
def test_quality_reward(quality):
    from tasks.task03.tabletop.reward import reward
    assert reward(quality) == quality


@pytest.mark.parametrize("quality", [-.1, 1.1, float("nan"), float("inf")])
def test_invalid_quality(quality):
    from tasks.task03.tabletop.reward import apply, reward
    with pytest.raises(ValueError, match="quality"):
        reward(quality)
    with pytest.raises(ValueError, match="quality"):
        apply({}, quality)


