"""Regression coverage for the historical push-based XgoDuck Swizzle task."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_velocity_swizzle_env_cfg import (
    make_microduck_velocity_swizzle_env_cfg,
)
from mjlab_microduck.tasks.xgoduck_velocity_swizzle_env_cfg import (
    XgoduckSwizzleCommand,
    make_xgoduck_velocity_swizzle_env_cfg,
)


def test_signed_push_rewards_acceleration_instead_of_target_speed():
    commands = torch.tensor([[0.3, 0, 0], [0.3, 0, 0], [-0.3, 0, 0],
                             [-0.3, 0, 0], [0, 0, 0]])
    velocity = torch.tensor([[0.3, 0, 0], [0.6, 0, 0], [-0.6, 0, 0],
                             [0.6, 0, 0], [0.6, 0, 0]])
    env = SimpleNamespace(
        command_manager=SimpleNamespace(get_command=lambda _: commands),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_b=velocity))},
    )
    term = make_xgoduck_velocity_swizzle_env_cfg().rewards["body_speed"]
    reward = term.func(env, **term.params)
    assert reward[1] > reward[0] > 0
    assert reward[2] == reward[1]
    assert reward[3] == reward[4] == 0


@pytest.mark.parametrize("play", [False, True])
def test_retargeted_task_does_not_mutate_microduck(play):
    cfg = make_xgoduck_velocity_swizzle_env_cfg(play=play)
    original = make_microduck_velocity_swizzle_env_cfg(play=play)
    assert cfg.commands["twist"].ranges.lin_vel_x == (-0.6, 0.6)
    assert cfg.commands["head_pose"].ranges == ((0.0, 0.0),) * 4
    assert "velocity_command_ranges" not in cfg.curriculum
    assert "wheel_friction" not in cfg.curriculum
    assert "body_speed" not in original.rewards
    assert "wheel_friction" in original.curriculum
    assert original.rewards["wheel_speed"].weight == 10.0
    assert original.events["randomize_wheel_friction"].params["ranges"] != cfg.events["randomize_wheel_friction"].params["ranges"]
    assert "randomize_mass_inertia" not in cfg.events
    assert "track_linear_velocity" not in cfg.rewards
    assert "four_wheels_grounded" in cfg.rewards
    assert "four_wheels_grounded" not in original.rewards


def test_roller_model_matches_full_contact_training_inertials():
    model = make_xgoduck_velocity_swizzle_env_cfg().scene.entities["robot"].spec_fn().compile()
    assert model.body_mass.sum() == pytest.approx(0.8740209967010302)
    for name in ("left_front_tire", "left_rear_tire", "right_front_tire", "right_rear_tire"):
        assert model.body(name).mass[0] == pytest.approx(0.02)


def test_gui_labels_push_strength_and_disabled_lateral_axis():
    sliders = {}

    def add_slider(label, **kwargs):
        value = SimpleNamespace(value=kwargs["initial_value"], **kwargs)
        sliders[label] = value
        return value

    server = SimpleNamespace(gui=SimpleNamespace(
        add_folder=lambda _: nullcontext(),
        add_checkbox=lambda _, **kw: SimpleNamespace(value=kw["initial_value"]),
        add_slider=add_slider,
        add_button=lambda _: SimpleNamespace(on_click=lambda callback: callback),
    ))
    command = SimpleNamespace(cfg=make_xgoduck_velocity_swizzle_env_cfg(play=True).commands["twist"])
    XgoduckSwizzleCommand.create_gui(command, "twist", server, lambda: 0)
    assert sliders["Push strength"].min == -0.6
    assert sliders["Push strength"].max == 0.6
    assert sliders["Lateral (fixed zero)"].disabled
    assert "vx (m/s)" not in sliders


def test_over_limit_penalty_resolves_servo_indices_across_passive_wheels():
    hard = torch.tensor([[[-0.22, 0.45], [0., 0.], [-0.4, 0.4]]])
    action = SimpleNamespace(raw_action=torch.tensor([[0.9, -0.8]]),
                             scale=1.0, offset=0.0, target_ids=torch.tensor([0, 2]))
    env = SimpleNamespace(
        action_manager=SimpleNamespace(get_term=lambda _: action),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(joint_pos_limits=hard))},
    )
    assert mdp.action_over_limit_penalty(env).item() == pytest.approx(0.25)

@pytest.mark.parametrize("step,fraction", [(0,0.0),(12000,0.25),(24000,0.5),(48000,1.0),(72000,1.0)])
def test_push_curriculum_updates_live_event_and_caps_all_world_axes(step, fraction):
    import math
    event = SimpleNamespace(params={"velocity_range":{}})
    env = SimpleNamespace(common_step_counter=step,device="cpu",
                          event_manager=SimpleNamespace(get_term_cfg=lambda _:event))
    maximum = {"x":(-.35,.35),"y":(-.35,.35),
               "roll":(-math.radians(20),math.radians(20)),
               "pitch":(-math.radians(20),math.radians(20)),
               "yaw":(-math.radians(20),math.radians(20))}
    scale = mdp.push_velocity_linear_curriculum(env,None,"push_robot",maximum,48000)
    assert scale.item() == pytest.approx(fraction)
    for axis,(lo,hi) in maximum.items():
        assert event.params["velocity_range"][axis] == pytest.approx((lo*fraction,hi*fraction))
    assert maximum["x"] == (-.35,.35)


def test_four_wheel_reward_requires_every_wheel_to_touch():
    found = torch.tensor([[1,1,1,1],[1,1,1,0],[1,0,1,0],[0,0,0,0]])
    env = SimpleNamespace(scene={"wheel_ground_contact":SimpleNamespace(data=SimpleNamespace(found=found))})
    torch.testing.assert_close(mdp.four_wheels_grounded_reward(env,"wheel_ground_contact"),
                               torch.tensor([1.,0.,0.,0.]))


def test_play_disables_training_disturbances_and_uses_final_heading_weights():
    cfg = make_xgoduck_velocity_swizzle_env_cfg(play=True)
    assert not cfg.observations["actor"].enable_corruption
    assert "push_robot" not in cfg.events
    assert not cfg.curriculum
    assert cfg.rewards["heading_hold"].weight == 0
    assert cfg.rewards["heading_tracking"].weight == 1.5


def test_roller_servo_parameters_are_independent_of_walk():
    from pathlib import Path
    from mjlab_microduck.robot.xgoduck_constants import actuators
    from mjlab_microduck.robot.xgoduck_roller_constants import XGODUCK_ROLLER_ROBOT_CFG
    servo = XGODUCK_ROLLER_ROBOT_CFG.articulation.actuators[0]
    assert servo is not actuators
    assert Path(servo.json_path).parent.name == "xgoduck_roller"
    assert Path(servo.json_path).resolve() == Path(servo._resolved_json_path).resolve()
    assert servo.kp_fw == 5
    assert (servo.delay_min_lag,servo.delay_max_lag) == (3,6)


def test_tire_contact_parameters_and_wheel_directions():
    import mujoco
    import numpy as np
    model = make_xgoduck_velocity_swizzle_env_cfg().scene.entities["robot"].spec_fn().compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model,data,0)
    mujoco.mj_forward(model,data)
    for name in ["left_front_tire","left_rear_tire","right_front_tire","right_rear_tire"]:
        g = model.geom(name+"_collision").id
        np.testing.assert_allclose(model.geom_solref[g],[.012,1])
        np.testing.assert_allclose(model.geom_solimp[g],[.9,.95,.001,.5,2])
        assert model.geom_priority[g] == 1
    for name in ["passive_LF_wheel","passive_LR_wheel","passive_RF_wheel","passive_RR_wheel"]:
        j = model.joint(name).id
        # The bottom surface moves backward for positive wheel spin, so the
        # robot rolls forward; right local axes need not match left local axes.
        assert np.cross(data.xaxis[j],[0,0,-.015])[0] < -.01
