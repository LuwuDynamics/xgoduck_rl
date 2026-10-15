"""Walk and full-contact variants share the transferred CAD inertials."""

import mujoco
import numpy as np
from mjlab_microduck.robot.xgoduck_constants import (
    XGODUCK_ALLCOLLISIONS_XML,
    XGODUCK_WALK_XML,
)


def test_xgoduck_allcollisions_matches_walk_inertials():
    walk = mujoco.MjModel.from_xml_path(str(XGODUCK_WALK_XML))
    allc = mujoco.MjModel.from_xml_path(str(XGODUCK_ALLCOLLISIONS_XML))
    assert walk.nbody == allc.nbody
    for i in range(walk.nbody):
        name = walk.body(i).name
        j = mujoco.mj_name2id(allc, mujoco.mjtObj.mjOBJ_BODY, name)
        assert j >= 0, name
        np.testing.assert_allclose(
            allc.body_mass[j], walk.body_mass[i], atol=1e-9, err_msg=f"{name} mass"
        )
        np.testing.assert_allclose(
            allc.body_ipos[j], walk.body_ipos[i], atol=1e-9, err_msg=f"{name} com"
        )
        np.testing.assert_allclose(
            allc.body_inertia[j], walk.body_inertia[i], atol=1e-12, err_msg=f"{name} I"
        )
