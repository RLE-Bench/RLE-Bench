"""Drop validation is read-only; a valid rescue preserves the private scramble."""
import os

import numpy as np
import pytest

if os.environ.get("RLEBENCH_TEST_TASK") != "task03":
    pytest.skip("family-specific physics", allow_module_level=True)
pytest.importorskip("robosuite")

from harness.adapter import Adapter
from harness.tabletop.pocket.recovery import canonical


def test_recovery_preflight_and_metered_action():
    adapter = Adapter(task="RubikCube", seed=17, mode="pocket")
    try:
        adapter.reset()
        env = adapter.env
        model, data = env.sim.model._model, env.sim.data._data
        before = data.qpos.copy()
        stamp = data.time
        assert not adapter.recovery_available()
        np.testing.assert_array_equal(data.qpos, before)
        assert data.time == stamp
        expected = env._checkpoint.copy()
        address = model.joint("cube_free").qposadr[0]
        data.qpos[address+2] = .79
        env.sim.forward()
        assert adapter.recovery_available()
        result = adapter.act(action=[0.]*6+[-1.]+[0.]*6+[-1.], recover=True)
        assert result["counters"]["recoveries"] == 1
        assert result["counters"]["actual_qtm"] == 0
        np.testing.assert_array_equal(canonical(data.xmat[env._cubie_ids].reshape(8, 3, 3)), expected)
        assert data.time-stamp == pytest.approx(1/env.control_freq)
        assert not adapter.recovery_available()
    finally:
        adapter.close()
