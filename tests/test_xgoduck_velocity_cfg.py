"""XgoDuck walk-task DR / pose-weight overrides; MicroDuck stays on the old recipe."""

from mjlab_microduck.tasks import _xgoduck_velocity_cfg
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
)


def test_xgoduck_velocity_friction_push_noise_and_pose_weight():
    cfg = _xgoduck_velocity_cfg()
    actor = cfg.observations["actor"].terms

    assert cfg.events["foot_friction"].params["ranges"] == (0.4, 1.5)

    vr = cfg.events["push_robot"].params["velocity_range"]
    assert vr["x"] == (-0.2, 0.2)
    assert vr["y"] == (-0.2, 0.2)

    assert actor["base_ang_vel"].noise.n_min == -0.06
    assert actor["base_ang_vel"].noise.n_max == 0.06
    assert actor["projected_gravity"].noise.n_min == -0.03
    assert actor["projected_gravity"].noise.n_max == 0.03
    assert actor["joint_pos"].noise.n_min == -0.01
    assert actor["joint_pos"].noise.n_max == 0.01

    assert cfg.rewards["pose"].weight == 1.3
    assert cfg.rewards["pose"].params["std_standing"][r".*hip_yaw.*"] == 0.05
    assert cfg.rewards["pose"].params["std_walking"][r".*hip_yaw.*"] == 0.15
    assert cfg.rewards["pose"].params["std_running"][r".*hip_yaw.*"] == 0.25


def test_xgoduck_vin_min_is_7():
    from mjlab_microduck.robot.microduck_constants import actuators as md
    from mjlab_microduck.robot.xgoduck_constants import actuators as xg

    assert xg.vin_min == 7.0
    assert md.vin_min == 6.0


def test_microduck_velocity_keeps_original_walk_recipe():
    cfg = make_microduck_velocity_env_cfg()
    actor = cfg.observations["actor"].terms

    assert cfg.events["foot_friction"].params["ranges"] == (0.7, 1.3)
    vr = cfg.events["push_robot"].params["velocity_range"]
    assert vr["x"] == (-0.3, 0.3)
    assert actor["base_ang_vel"].noise.n_max == 0.03
    assert actor["projected_gravity"].noise.n_max == 0.01
    assert actor["joint_pos"].noise.n_max == 0.001
    assert cfg.rewards["pose"].weight == 1.0
    assert cfg.rewards["pose"].params["std_walking"][r".*hip_yaw.*"] == 0.3


def test_walk_keeps_original_recipe_except_explicit_overrides():
    """Walk tuning is isolated from the upstream MicroDuck reward recipe."""
    import pytest

    for kwargs in ({}, {"play": True}, {"rough": True}, {"rough": True, "play": True}):
        original = make_microduck_velocity_env_cfg(**kwargs)
        walk = _xgoduck_velocity_cfg(**kwargs)
        assert set(walk.rewards) == set(original.rewards) | {"hip_yaw_home"}
        for name in original.rewards:
            if name not in ("pose", "track_linear_velocity", "foot_swing_height"):
                assert walk.rewards[name].weight == pytest.approx(original.rewards[name].weight)
        assert walk.rewards["track_linear_velocity"].weight == pytest.approx(3.0)
        assert walk.rewards["track_linear_velocity"].params == {
            **original.rewards["track_linear_velocity"].params, "std": 0.25,
        }
        assert original.rewards["track_linear_velocity"].weight == pytest.approx(2.0)
        assert original.rewards["track_linear_velocity"].params["std"] == pytest.approx(0.1 ** 0.5)
        assert walk.rewards["foot_swing_height"].weight == pytest.approx(-5.0)
        assert original.rewards["foot_swing_height"].weight == pytest.approx(-0.25)
        for name in ("air_time", "foot_clearance", "foot_swing_height"):
            assert walk.rewards[name].params == original.rewards[name].params
        if kwargs.get("play"):
            assert set(walk.curriculum) == set(original.curriculum) - {"head_pose_range"}
            assert walk.commands["head_pose"].ranges == ((0.0, 0.0),) * 4
        else:
            assert walk.curriculum.keys() == original.curriculum.keys()
            assert walk.curriculum["head_pose_range"].params == original.curriculum["head_pose_range"].params
            assert walk.commands["head_pose"].ranges == original.commands["head_pose"].ranges
        assert walk.commands["twist"].rel_backward_envs == (0.0 if kwargs.get("play") else 0.15)
        assert walk.sim.mujoco.timestep == original.sim.mujoco.timestep == 0.005
        assert walk.decimation == original.decimation == 4
        assert ("randomize_mass_inertia" in walk.events) == ("randomize_mass_inertia" in original.events)


def test_walk_pose_and_delay_are_isolated_from_other_tasks():
    """Restoring Walk must not mutate the shared roller/trick configuration."""
    import math
    import pytest
    from mjlab_microduck.robot.xgoduck_constants import (
        XGODUCK_HOME_FRAME, XGODUCK_STANDUP_ROBOT_CFG, actuators,
    )
    from mjlab_microduck.tasks.xgoduck_velocity_swizzle_env_cfg import (
        make_xgoduck_velocity_swizzle_env_cfg,
    )

    walk = _xgoduck_velocity_cfg()
    robot = walk.scene.entities["robot"]
    assert robot.init_state.pos[2] == pytest.approx(0.139439)
    assert robot.init_state.joint_pos[r".*left_hip_pitch.*"] == pytest.approx(-math.radians(24))
    servo = robot.articulation.actuators[0]
    assert (servo.delay_min_lag, servo.delay_max_lag) == (3, 6)
    robot.init_state.joint_pos[r".*left_hip_pitch.*"] = 0.0
    servo.delay_min_lag = 0

    fresh = _xgoduck_velocity_cfg().scene.entities["robot"]
    assert fresh.init_state.joint_pos[r".*left_hip_pitch.*"] == pytest.approx(-math.radians(24))
    assert fresh.articulation.actuators[0].delay_min_lag == 3
    assert XGODUCK_HOME_FRAME.joint_pos[r".*left_hip_pitch.*"] == pytest.approx(-math.radians(22))
    assert XGODUCK_STANDUP_ROBOT_CFG.init_state is XGODUCK_HOME_FRAME
    assert (actuators.delay_min_lag, actuators.delay_max_lag) == (1, 1)
    skating = make_xgoduck_velocity_swizzle_env_cfg()
    assert skating.scene.entities["robot"].init_state.joint_pos["left_hip_pitch"] == pytest.approx(-math.radians(10))
    assert skating.sim.mujoco.timestep == 0.004 and skating.decimation == 5
    assert "randomize_mass_inertia" not in skating.events


def test_swizzle_reset_starts_at_grounded_home():
    # Reset offsets must not add the standing height a second time.
    import re
    import mujoco
    from mjlab_microduck.tasks.xgoduck_velocity_swizzle_env_cfg import (
        make_xgoduck_velocity_swizzle_env_cfg,
    )

    for play in (False, True):
        cfg = make_xgoduck_velocity_swizzle_env_cfg(play=play)
        robot = cfg.scene.entities["robot"]
        reset = cfg.events["reset_base"].params
        joints = cfg.events["reset_robot_joints"].params
        assert reset["velocity_range"] == {}
        assert joints["position_range"] == joints["velocity_range"] == (0.0, 0.0)
        assert reset["pose_range"].get("roll", (0.0, 0.0)) == (0.0, 0.0)
        assert reset["pose_range"].get("pitch", (0.0, 0.0)) == (0.0, 0.0)
        model = robot.spec_fn().compile()
        data = mujoco.MjData(model)
        for z_offset in reset["pose_range"]["z"]:
            data.qpos[:3] = robot.init_state.pos
            data.qpos[2] += z_offset
            data.qpos[3:7] = robot.init_state.rot
            for i in range(model.njnt):
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                    continue
                name = model.joint(i).name
                for pattern, value in robot.init_state.joint_pos.items():
                    if re.fullmatch(pattern, name):
                        data.qpos[model.jnt_qposadr[i]] = value
            mujoco.mj_forward(model, data)
            bottoms = []
            for i in range(model.ngeom):
                if not re.fullmatch(r"(left|right)_(front|rear)_tire_collision", model.geom(i).name):
                    continue
                mesh = model.geom_dataid[i]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
                world = vertices @ data.geom_xmat[i].reshape(3, 3).T + data.geom_xpos[i]
                bottoms.append(world[:, 2].min())
            assert len(bottoms) == 4
            # CAD left/right tolerances are sub-millimetre; no drop or deep overlap.
            assert abs(min(bottoms)) < 5e-5
            assert max(bottoms) < 5e-4


def test_swizzle_uses_historical_fall_termination():
    import math
    import pytest
    from mjlab_microduck.tasks.xgoduck_velocity_swizzle_env_cfg import (
        make_xgoduck_velocity_swizzle_env_cfg,
    )

    for play in (False, True):
        cfg = make_xgoduck_velocity_swizzle_env_cfg(play=play)
        assert cfg.terminations["fell_over"].params["limit_angle"] == pytest.approx(math.radians(70))
        assert "body_ground_contact" not in cfg.terminations
        assert "termination_penalty" not in cfg.rewards
