"""Regression coverage for grounded, single-shot XgoDuck forward rolls."""

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab_microduck.tasks import mdp
from test_xgoduck_tricks import home_model


def rotation_fixture():
    support = SimpleNamespace(data=SimpleNamespace(found=torch.ones(1, 1)))
    head = SimpleNamespace(data=SimpleNamespace(found=torch.zeros(1, 1)))
    feet = SimpleNamespace(data=SimpleNamespace(found=torch.ones(1, 2)))
    asset = SimpleNamespace(data=SimpleNamespace(
        root_link_quat_w=torch.tensor([[1., 0., 0., 0.]]),
        root_link_ang_vel_b=torch.zeros(1, 3),
        root_link_lin_vel_w=torch.zeros(1, 3),
        root_link_pos_w=torch.tensor([[0., 0., .139158]]),
        body_link_quat_w=torch.tensor([[[1., 0., 0., 0.]]]),
    ))
    sensors = {"robot_ground_contact": support, "head_ground_contact": head,
               "feet_ground_contact": feet}

    class Scene(dict):
        pass

    scene = Scene(robot=asset)
    scene.sensors = sensors
    scene.terrain = SimpleNamespace(env_origins=torch.zeros(1, 3))
    env = SimpleNamespace(num_envs=1, device="cpu", step_dt=.02,
                          common_step_counter=0, scene=scene,
                          _roulade_head_body_id=0,
                          _roulade_prev_pitch=torch.zeros(1))
    mdp._roulade_state(env)
    env._roulade_head_latch[:] = True
    return env, asset, support, feet


def turn(env, asset, support, start, end, contact):
    support.data.found.fill_(contact)
    steps = max(1, round(abs(end - start) / math.radians(3)))
    for i in range(1, steps + 1):
        pitch = start + (end - start) * i / steps
        asset.data.root_link_quat_w[0] = torch.tensor(
            [math.cos(pitch / 2), 0., math.sin(pitch / 2), 0.])
        env.common_step_counter += 1
        mdp._update_roulade_accum(env, asset)


def test_airborne_return_cannot_farm_roll_completion():
    env, asset, support, _ = rotation_fixture()
    for _ in range(3):
        turn(env, asset, support, 0, math.radians(120), 1)
        turn(env, asset, support, math.radians(120), 0, 0)
    assert env._roulade_accum.item() == pytest.approx(0, abs=1e-5)
    assert env._roulade_max.item() == pytest.approx(math.radians(120), abs=1e-5)
    assert mdp._roulade_completion_gate(env, math.radians(260), math.radians(330), True).item() == 0


@pytest.mark.parametrize("contact,expected", [(0, 0.), (1, 2 * math.pi)])
def test_only_supported_full_roll_earns_completion(contact, expected):
    env, asset, support, _ = rotation_fixture()
    turn(env, asset, support, 0, 2 * math.pi, contact)
    assert env._roulade_accum.item() == pytest.approx(2 * math.pi, abs=2e-5)
    assert env._roulade_max.item() == pytest.approx(expected, abs=2e-5)


def test_quaternion_sign_and_angular_velocity_do_not_fake_progress():
    env, asset, _, _ = rotation_fixture()
    asset.data.root_link_ang_vel_b[:, 1] = 100.
    for _ in range(5):
        asset.data.root_link_quat_w *= -1
        env.common_step_counter += 1
        mdp._update_roulade_accum(env, asset)
    assert env._roulade_max.item() == pytest.approx(0, abs=1e-6)


def test_closed_yaw_roll_loops_cannot_accumulate_forward_rotation():
    env, asset, _, _ = rotation_fixture()
    for _ in range(12):
        for ya, yb, ra, rb in [(0, -90, 0, 0), (-90, -90, 0, 20),
                               (-90, 0, 20, 20), (0, 0, 20, 0)]:
            for i in range(1, 31):
                yaw = math.radians(ya + (yb - ya) * i / 30)
                roll = math.radians(ra + (rb - ra) * i / 30)
                cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
                cr, sr = math.cos(roll / 2), math.sin(roll / 2)
                asset.data.root_link_quat_w[0] = torch.tensor([cy*cr, cy*sr, sy*sr, sy*cr])
                env.common_step_counter += 1
                mdp._update_roulade_accum(env, asset)
    assert env._roulade_accum.item() == pytest.approx(0., abs=1e-5)
    assert env._roulade_max.item() < 1e-5


def test_angle_observation_is_signed_unclipped_and_does_not_mutate_reward_state():
    env, asset, support, feet = rotation_fixture()
    assert mdp.roulade_angle_observation(env).item() == 0
    turn(env, asset, support, 0, 4 * math.pi, 1)
    angle = mdp.roulade_angle_observation(env)
    assert angle.shape == (1, 1)
    assert angle.item() == pytest.approx(4 * math.pi, abs=4e-5)
    angle.add_(math.radians(30))
    assert env._roulade_accum.item() == pytest.approx(4 * math.pi, abs=4e-5)
    turn(env, asset, support, 4 * math.pi, -math.pi, 1)
    assert mdp.roulade_angle_observation(env).item() == pytest.approx(-math.pi, abs=5e-5)


def test_baseline_keeps_original_observations_and_angle_variant_adds_only_one_value():
    baseline = load_env_cfg("Mjlab-Roulade-Flat-XgoDuck")
    angle = load_env_cfg("Mjlab-Roulade-Angle-Flat-XgoDuck")
    original = load_env_cfg("Mjlab-Roulade-Flat-MicroDuck")
    for group in ("actor", "critic"):
        plain = baseline.observations[group].terms
        extra = angle.observations[group].terms
        assert list(plain) == list(original.observations[group].terms)
        assert plain["body_command"].params["dim"] == 6
        assert "roulade_angle" in extra and extra["body_command"].params["dim"] == 5
        assert "roulade_phase" not in extra and "roulade_state" not in extra
    noise = angle.observations["actor"].terms["roulade_angle"].noise
    assert (noise.n_min, noise.n_max) == pytest.approx((-math.pi / 6, math.pi / 6))
    assert angle.observations["critic"].terms["roulade_angle"].noise is None


def test_ablation_rewards_are_identical_and_both_rolls_start_standing():
    baseline = load_env_cfg("Mjlab-Roulade-Flat-XgoDuck")
    angle = load_env_cfg("Mjlab-Roulade-Angle-Flat-XgoDuck")
    assert list(baseline.rewards) == list(angle.rewards)
    assert baseline.rewards["roulade_progress"].weight == 8.0
    for name, term in baseline.rewards.items():
        other = angle.rewards[name]
        assert term.func is other.func and term.weight == other.weight
        assert term.params == other.params
    for cfg in (baseline, angle):
        assert cfg.episode_length_s == 7.0
        assert "roulade_success" not in cfg.terminations
        assert "roulade_spawn_mix" not in cfg.curriculum
        spawn = cfg.events["set_roulade_state"].params
        assert spawn["standing_prob"] == 1.0 and spawn["midroll_prob"] == 0.0


def test_sharp_landing_reward_requires_both_feet():
    env, asset, support, feet = rotation_fixture()
    turn(env, asset, support, 0, 2 * math.pi, 1)
    assert mdp.roulade_landing_sharp(env, .139158, require_feet=True).item() > .99
    feet.data.found[0, 1] = 0
    assert mdp.roulade_landing_sharp(env, .139158, require_feet=True).item() == 0
    # Upright/height bootstrapping remains available without requiring feet.
    assert mdp.roulade_landing_sharp(env, .139158).item() > .99


def test_full_collision_matches_microduck_and_home_has_no_self_penetration():
    cfg = load_env_cfg("Mjlab-Roulade-Flat-XgoDuck")
    model, data = home_model(cfg)
    active = [g for g in range(model.ngeom) if model.geom(g).name.endswith("_collision")]
    assert len(active) == 15
    for g in active:
        assert model.geom_contype[g] == model.geom_conaffinity[g] == 1
        assert model.geom_condim[g] == (3 if "foot_collision" in model.geom(g).name else 1)
    assert model.nexclude == 3
    assert data.ncon == 0
    # A genuine nonadjacent pair is still collision eligible.
    a, b = model.geom("left_shin_collision").id, model.geom("head_collision").id
    assert model.geom_contype[a] & model.geom_conaffinity[b]


def test_roulade_randomization_targets_and_nominal_play():
    train = load_env_cfg("Mjlab-Roulade-Flat-XgoDuck")
    play = load_env_cfg("Mjlab-Roulade-Flat-XgoDuck", play=True)
    assert "bearing_roll" not in train.events["randomize_head_com"].params["asset_cfg"].body_names
    assert list(train.events).index("ground_roulade_reset") > list(train.events).index("set_roulade_state")
    assert "resolve_roulade_reset_collisions" not in train.events
    assert train.events["reset_robot_joints"].params["position_range"] == (0.0, 0.0)
    assert not play.observations["actor"].enable_corruption
    assert not play.curriculum
    assert "randomize_head_com" not in play.events
    assert "randomize_joint_friction" not in play.events
    servo = play.scene.entities["robot"].articulation.actuators[0]
    assert servo.vin_range[0] == servo.vin_range[1]
    assert servo.vin_drop_gain_range[0] == servo.vin_drop_gain_range[1]
    for cfg in (train, play):
        assert cfg.commands["twist"].ranges.lin_vel_x == (0., 0.)
    original = load_env_cfg("Mjlab-Roulade-Flat-MicroDuck")
    assert "roulade_success" not in original.terminations
    assert "ground_roulade_reset" not in original.events
