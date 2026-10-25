"""Walk reset geometry, command sampling and train/play configuration isolation."""

import importlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mjlab_microduck.tasks import _xgoduck_velocity_cfg, mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    MICRODUCK_ROUGH_TERRAINS_CFG,
)
from test_xgoduck_tricks import home_model


@pytest.mark.parametrize("kwargs", [{}, {"play": True}, {"rough": True},
                                   {"rough": True, "play": True}])
@pytest.mark.parametrize("ids", [[0, 1], [1]])
def test_actual_reset_function_adds_home_height_once(kwargs, ids):
    cfg = _xgoduck_velocity_cfg(**kwargs)
    pose = cfg.scene.entities["robot"].init_state
    default = torch.tensor([list(pose.pos) + list(pose.rot) + [0.] * 6] * 2)
    captured = {}

    def write(state, env_ids):
        captured["state"] = state

    asset = SimpleNamespace(is_fixed_base=False,
                            data=SimpleNamespace(default_root_state=default),
                            write_root_link_pose_to_sim=write,
                            write_root_link_velocity_to_sim=lambda velocity, env_ids:None)

    class Scene(dict):
        pass

    scene = Scene(robot=asset)
    scene.env_origins = torch.tensor([[0., 0., 0.], [2., 0., .4]])
    env = SimpleNamespace(scene=scene, num_envs=2, device="cpu")
    selected = torch.tensor(ids)
    reset = cfg.events["reset_base"]
    reset.func(env, selected, **reset.params)
    expected = default[selected, 2] + scene.env_origins[selected, 2]
    torch.testing.assert_close(captured["state"][:, 2], expected)


def test_flat_walk_home_soles_have_small_positive_ground_clearance():
    cfg = _xgoduck_velocity_cfg()
    model, data = home_model(cfg)
    bottoms = []
    for side in ("left", "right"):
        gid = model.geom(side + "_foot_collision").id
        mesh = model.geom_dataid[gid]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count]
        bottoms.append((vertices @ data.geom_xmat[gid].reshape(3, 3)[2]
                        + data.geom_xpos[gid, 2]).min())
    assert np.all((np.array(bottoms) > 0) & (np.array(bottoms) < .002))


@pytest.mark.parametrize("module, factory", [
    ("microduck_velocity_env_cfg", "make_microduck_velocity_env_cfg"),
    ("microduck_ground_pick_env_cfg", "make_microduck_ground_pick_env_cfg"),
    ("microduck_sitstand_env_cfg", "make_microduck_sitstand_env_cfg"),
    ("microduck_standup_env_cfg", "make_microduck_standup_env_cfg"),
])
def test_rough_play_does_not_mutate_training_or_shared_template(module, factory):
    make = getattr(importlib.import_module("mjlab_microduck.tasks." + module), factory)
    train = make(rough=True).scene.terrain.terrain_generator
    play = make(rough=True, play=True).scene.terrain.terrain_generator
    assert train is not play and train is not MICRODUCK_ROUGH_TERRAINS_CFG
    assert (train.num_rows, train.num_cols) == (10, 20)
    assert (play.num_rows, play.num_cols) == (5, 5)
    assert (MICRODUCK_ROUGH_TERRAINS_CFG.num_rows,
            MICRODUCK_ROUGH_TERRAINS_CFG.num_cols) == (10, 20)
    assert train.sub_terrains["random_grid"] is not play.sub_terrains["random_grid"]


def test_walk_play_head_and_rough_curriculum():
    train = _xgoduck_velocity_cfg(rough=True)
    play = _xgoduck_velocity_cfg(rough=True, play=True)
    assert train.scene.terrain.terrain_generator.curriculum
    assert not play.scene.terrain.terrain_generator.curriculum
    assert train.commands["head_pose"].ranges != ((0., 0.),) * 4
    assert play.commands["head_pose"].ranges == ((0., 0.),) * 4
    assert play.commands["head_pose"].zero_command_prob == 1.0
    assert "head_pose_range" not in play.curriculum


@pytest.mark.parametrize("play", [False, True])
def test_head_com_randomization_can_be_disabled(monkeypatch, play):
    module = importlib.import_module("mjlab_microduck.tasks.microduck_velocity_env_cfg")
    monkeypatch.setattr(module, "ENABLE_HEAD_COM_RANDOMIZATION", False)
    cfg = _xgoduck_velocity_cfg(play=play)
    assert "randomize_head_com" not in cfg.events
    assert "head_com_range" not in cfg.curriculum


def test_turn_in_place_preserves_standing_samples_and_its_own_command_flags(monkeypatch):
    count, standing = 4096, 1024
    term = mdp.VelocityCommandCommandOnly.__new__(mdp.VelocityCommandCommandOnly)
    term.cfg = replace(_xgoduck_velocity_cfg().commands["twist"], rel_backward_envs=0.)
    term._env = SimpleNamespace(device="cpu", num_envs=count)
    term.vel_command_b = torch.full((count, 3), .3)
    term.vel_command_b[:standing] = 0
    term.vel_command_w = term.vel_command_b.clone()
    term.is_standing_env = torch.arange(count) < standing
    for name in ["is_heading_env", "is_world_env", "is_forward_env"]:
        setattr(term, name, torch.ones(count, dtype=torch.bool))
    monkeypatch.setattr(mdp.UniformVelocityCommand, "_resample_command", lambda self, ids:None)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        term._resample_command(torch.arange(count))
    assert int(term.is_standing_env.sum()) == standing
    assert torch.count_nonzero(term.vel_command_b[:standing]) == 0
    turning = (term.vel_command_b[:, :2] == 0).all(dim=-1) & (term.vel_command_b[:, 2] != 0)
    assert .12 < turning.sum().item() / (count - standing) < .18
    assert not term.is_heading_env[turning].any()
    assert not term.is_world_env[turning].any()
    assert not term.is_forward_env[turning].any()
