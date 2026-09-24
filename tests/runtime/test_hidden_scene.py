"""Independent inertia and hidden-state rendering regressions."""
import os
import pytest

if os.environ.get("RLEBENCH_TEST_TASK") != "task03":
    pytest.skip("family-specific physics", allow_module_level=True)
os.environ.setdefault("MUJOCO_GL", "egl")

from harness.tabletop.hidden_com.config import CASES


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("quadrant", list("ABCD"))
def test_mass_properties_against_independent_compound_body(quadrant, case):
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    import numpy as np
    from scipy.spatial.transform import Rotation
    from harness.tabletop.hidden_com.scene import mass_properties, SIGNS, HANDLE_PARTS
    x, y = np.array(SIGNS[quadrant]) * case.offset_xy
    handle_xml = "".join(f'<geom type="box" size="{" ".join(map(str, half))}" mass="{mass}" pos="{" ".join(map(str, pos))}"/>'
                         for _, mass, half, pos in HANDLE_PARTS)
    model = mujoco.MjModel.from_xml_string(f'''<mujoco><worldbody><body>
      <freejoint/><geom type="box" size=".075 .075 .025" mass="{case.shell_mass}"/>
      <geom type="box" size="{' '.join(map(str, case.ballast_half))}" mass="{case.ballast_mass}" pos="{x} {y} 0"/>
    {handle_xml}</body></worldbody></mujoco>''')
    mass, center, inertia = mass_properties(quadrant, case)
    axes = Rotation.from_quat(model.body_iquat[1], scalar_first=True).as_matrix()
    np.testing.assert_allclose(mass, model.body_mass[1], atol=1e-12)
    np.testing.assert_allclose(center, model.body_ipos[1], atol=1e-12)
    np.testing.assert_allclose(inertia, axes @ np.diag(model.body_inertia[1]) @ axes.T, atol=2e-10)
    assert np.linalg.eigvalsh(inertia).min() > 0


def test_box_markings_preserve_inertia_and_cannot_collide():
    mujoco = pytest.importorskip("mujoco")
    pytest.importorskip("robosuite")
    import numpy as np
    import xml.etree.ElementTree as ET
    from harness.tabletop.hidden_com.scene import add_labels

    root = ET.fromstring('<mujoco><worldbody><body><freejoint/>'
                         '<geom type="box" size=".075 .075 .025" mass=".5"/>'
                         '</body></worldbody></mujoco>')
    original = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    add_labels(root.find("worldbody/body"))
    marked = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    for field in ("body_mass", "body_ipos"):
        np.testing.assert_array_equal(getattr(marked, field), getattr(original, field))
    from scipy.spatial.transform import Rotation
    def inertia(model):
        axes = Rotation.from_quat(model.body_iquat[1], scalar_first=True).as_matrix()
        return axes @ np.diag(model.body_inertia[1]) @ axes.T
    np.testing.assert_allclose(inertia(marked), inertia(original), atol=1e-15)
    assert not marked.geom_contype[1:].any()
    assert not marked.geom_conaffinity[1:].any()


@pytest.mark.parametrize("case", CASES)
def test_initial_rgb_does_not_reveal_quadrant(case):
    pytest.importorskip("robosuite")
    import numpy as np
    from harness.tabletop.hidden_com.scene import HiddenCOM
    from harness.tabletop.hidden_com.render import render
    reference = None
    for quadrant in "ABCD":
        env = HiddenCOM(quadrant, case=case)
        try:
            env.reset()
            pixels = np.asarray(render(env, "top"))
            if reference is None:
                reference = pixels.copy()
            else:
                np.testing.assert_array_equal(pixels, reference)
        finally:
            env.close()

