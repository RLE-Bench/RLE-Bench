import os
import pytest
if os.environ.get("RLEBENCH_TEST_TASK") != "task03":
    pytest.skip("family-specific checks", allow_module_level=True)

import numpy as np
from scipy.spatial.transform import Rotation
from harness.tabletop.pocket import metrics as M
SOLVED = np.tile(np.eye(3), (8, 1, 1))

def test_all_faces_turns_inverses_and_four_turns():
    for axis in range(3):
        for sign in (-1, 1):
            for turns in (-1, 1, 2):
                move = (axis, sign, turns)
                state = M.apply_moves(SOLVED, [move])
                assert M.inspect_cube(state)["legal"]
                assert not M.inspect_cube(state)["solved"]
                np.testing.assert_array_equal(M.apply_moves(state, [(axis, sign, -turns)]), SOLVED)
            np.testing.assert_array_equal(M.apply_moves(SOLVED, [(axis, sign, 1)]*4), SOLVED)


def test_solved_is_global_rotation_invariant_and_odd_permutations_are_legal():
    for rotation in M.ROTATIONS:
        assert M.inspect_cube(rotation @ SOLVED)["solved"]
    state = M.apply_moves(SOLVED, [(0, 1, 1)])
    positions = np.einsum("bij,bj->bi", state, M.POSITIONS)
    perm = [next(i for i, p in enumerate(M.POSITIONS) if np.array_equal(p, q)) for q in positions]
    assert sum(a > b for i, a in enumerate(perm) for b in perm[i+1:]) % 2 == 1
    assert M.inspect_cube(state)["legal"]


def test_scramble_inverse_and_independent_rotation_reference():
    for seed in range(30):
        moves = M.scramble(seed, 20)
        state = M.apply_moves(SOLVED, moves)
        reference = SOLVED.copy()
        for axis, sign, turns in moves:
            turn = Rotation.from_rotvec(np.eye(3)[axis] * sign * turns * np.pi / 2).as_matrix()
            for i, position in enumerate(M.POSITIONS):
                if (reference[i] @ position)[axis] * sign > .5:
                    reference[i] = turn @ reference[i]
        np.testing.assert_allclose(state, reference, atol=1e-12)
        assert M.inspect_cube(state)["legal"]
        inverse = [(a, s, -t) for a, s, t in reversed(moves)]
        assert M.inspect_cube(M.apply_moves(state, inverse))["solved"]


def test_impossible_twist_overlap_mirror_and_misalignment_rejected():
    state = SOLVED.copy()
    state[0] = Rotation.from_rotvec(M.POSITIONS[0]/np.sqrt(3) * 2*np.pi/3).as_matrix()
    assert not M.inspect_cube(state)["legal"]
    state = SOLVED.copy()
    state[0] = next(r for r in M.ROTATIONS if np.array_equal(r @ M.POSITIONS[0], M.POSITIONS[1]))
    assert not M.inspect_cube(state)["legal"]
    state = SOLVED.copy()
    state[0, 0] *= -1
    assert not M.inspect_cube(state)["legal"]
    state = SOLVED.copy()
    state[0] = Rotation.from_rotvec([.1, 0, 0]).as_matrix()
    assert not M.inspect_cube(state)["legal"]
    assert not M.inspect_cube(np.full((8, 3, 3), np.nan))["legal"]
