#!/usr/bin/env python3
"""Measure servo load while a policy walks or stands XgoDuck in sim.

Runs N parallel envs per fixed-command scenario and reports, per joint:
  - applied output torque (what MuJoCo applied, after the force limit): RMS, p95,
    peak, and the share of samples above 30% / 50% of stall;
  - copper loss I^2*R, with I from BAM's motor torque before the force limit
    (a backdriven servo draws more current than its applied torque shows);
  - bracing: the share of samples resting on a hard stop (within 0.02 rad,
    nearly still, torque pushed INTO the stop) and the torque there, with plain
    proximity to a stop reported separately.
Plus the share of samples where any leg joint is above 50% of stall, falls and
timeouts per env-minute, the upright share (for standup), and achieved vs
commanded velocity. The HD-1910's continuous rating is unpublished, so loads
are compared against stall torque (vin * kt / R). Bracing against a stop is
free stiffness in sim but a stalled servo pressing on a printed stop on the robot.

Evaluation conditions are frozen: pushes, every randomization event, encoder
bias, observation noise, IMU misalignment and every curriculum are switched off,
head/body pose commands are held at zero, observation lags are fixed at their
maximum, and the actuator's battery voltage, sag and command delay are fixed at
mid-range values. load_env_cfg returns a deep copy, so nothing shared is mutated. Loading a checkpoint restores the training step
counter, so without this a checkpoint would face its curricula's final stage
(head swings to +-1.1 rad, wide CoM randomization) while an ONNX policy faced
stage 0. Statistics are taken at every physics substep (200 Hz), after the
scene update and before resets. Envs with a non-finite sample are excluded from
that sample and counted.

  # a training checkpoint
  uv run python scripts/servo_load.py Mjlab-Velocity-Flat-XgoDuck \
      --checkpoint logs/rsl_rl/xgoduck_velocity/<run>/model_3000.pt
  # an exported runtime policy (batch-1 ONNX, e.g. xgoduck_runtime_arduino/python/xgoduck_walk.onnx)
  uv run python scripts/servo_load.py Mjlab-Velocity-Flat-XgoDuck --onnx xgoduck_walk.onnx
  # standup: measured in the "stand" scenario (drop, get up, stand)
  uv run python scripts/servo_load.py Mjlab-StandUp-Flat-XgoDuck --onnx xgoduck_getup.onnx --scenarios stand
"""

import argparse
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

import mjlab_microduck
import mjlab_microduck.tasks  # registers the tasks

DEFAULT_PARAMS = Path(mjlab_microduck.__file__).parent / "robot/xgoduck/params/1910_m6.json"
VIN_NOMINAL = 7.7  # middle of the actuator cfg's vin_range (7.4, 8.0)
STOP_MARGIN = 0.02  # rad from a joint limit that counts as at the stop
STILL_VEL = 0.2  # rad/s; slower than this at a stop counts as resting on it
UPRIGHT_GZ = -0.8  # projected gravity z below this counts as upright
FALLEN_GZ = -0.3  # above this counts as fallen; upright -> fallen is a fall
SETTLE_S = 3.0  # s after a reset in which a fall is not counted (standup drops the robot)
EVAL_VIN = 7.7  # V, middle of the actuator's vin_range (7.4, 8.0)
EVAL_VIN_DROP_GAIN = 0.1  # V/Nm, middle of vin_drop_gain_range (0.0, 0.2)
EVAL_DELAY_LAG = 4  # physics substeps, inside delay lag range (3, 6)
HIST_MAX_NM = 1.5  # torque histogram range for p95
HIST_BINS = 300

# Events that perturb or randomize the robot. Evaluation keeps only the BAM
# friction-field expansion and the resets.
EVAL_DISABLED_EVENTS = (
    "push_robot",
    "base_com",
    "encoder_bias",
    "foot_friction",
    "randomize_armature",
    "randomize_com",
    "randomize_head_com",
    "randomize_joint_friction",
    "randomize_mass_inertia",
)

# name: (vx [m/s], vy [m/s], wz [rad/s])
SCENARIOS = {
    "stand": (0.0, 0.0, 0.0),
    "walk": (0.15, 0.0, 0.0),
    "walk_fast": (0.30, 0.0, 0.0),
    "turn": (0.0, 0.0, 0.8),
    "walk_turn": (0.15, 0.0, 0.5),
}


class ScenarioFailed(Exception):
    """One scenario could not be measured; the others still run."""


def freeze_eval_cfg(env_cfg, vx: float, vy: float, wz: float) -> None:
    """Fixed twist, zero head/body pose commands, no disturbances, no curricula."""
    cmd = env_cfg.commands["twist"]
    cmd.ranges.lin_vel_x = (vx, vx)
    cmd.ranges.lin_vel_y = (vy, vy)
    cmd.ranges.ang_vel_z = (wz, wz)
    cmd.heading_command = False
    cmd.ranges.heading = None
    cmd.rel_heading_envs = 0.0
    cmd.rel_world_envs = 0.0
    cmd.rel_forward_envs = 0.0  # forward envs clamp vx up to >= 0.3
    cmd.rel_standing_envs = 1.0 if (vx, vy, wz) == (0.0, 0.0, 0.0) else 0.0
    if hasattr(cmd, "rel_turn_in_place_envs"):
        cmd.rel_turn_in_place_envs = 0.0
    for name in ("head_pose", "body_pose"):
        pose = env_cfg.commands.get(name)
        if pose is not None:
            pose.ranges = tuple((0.0, 0.0) for _ in pose.ranges)
    for name in EVAL_DISABLED_EVENTS:
        env_cfg.events.pop(name, None)
    if env_cfg.curriculum:
        env_cfg.curriculum.clear()
    for group in env_cfg.observations.values():
        if hasattr(group, "enable_corruption"):
            group.enable_corruption = False
        for term in getattr(group, "terms", {}).values():
            if term is None:
                continue
            if term.params and "max_angle_deg" in term.params:
                term.params = {**term.params, "max_angle_deg": 0.0}  # IMU misalignment
            if getattr(term, "delay_max_lag", 0) != getattr(term, "delay_min_lag", 0):
                term.delay_min_lag = term.delay_max_lag  # one fixed observation lag
    # Fixed battery voltage, sag and command delay, on copies of the robot cfg.
    robot = env_cfg.scene.entities["robot"]
    acts = tuple(
        replace(a, vin_range=(EVAL_VIN, EVAL_VIN),
                vin_drop_gain_range=(EVAL_VIN_DROP_GAIN, EVAL_VIN_DROP_GAIN),
                delay_min_lag=EVAL_DELAY_LAG, delay_max_lag=EVAL_DELAY_LAG)
        if hasattr(a, "vin_range") else a
        for a in robot.articulation.actuators
    )
    env_cfg.scene.entities = {
        **env_cfg.scene.entities,
        "robot": replace(robot, articulation=replace(robot.articulation, actuators=acts)),
    }


class OnnxPolicy:
    """Batch-1 ONNX policy from the XGO runtime, run batched on CPU.

    The shipped policies carry their own default pose in metadata. When it
    differs from the env's home pose, joint_pos and last-action observations
    and the returned actions are shifted so targets land where the policy
    meant them.
    """

    JOINT_POS = slice(6, 20)
    LAST_ACTION = slice(34, 48)

    def __init__(self, path: str, robot):
        import onnx
        import onnxruntime as ort

        model = onnx.load(path)
        meta = {p.key: p.value for p in model.metadata_props}
        for t in list(model.graph.input) + list(model.graph.output):
            t.type.tensor_type.shape.dim[0].dim_param = "N"
        self.sess = ort.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        names = meta["joint_names"].split(",")
        env_names = list(robot.actuator_names)
        assert names == env_names, f"joint order differs:\n{names}\n{env_names}"
        onnx_default = np.array([float(v) for v in meta["default_joint_pos"].split(",")])
        joint_ids = [robot.joint_names.index(n) for n in names]
        env_default = robot.data.default_joint_pos[0, joint_ids].cpu().numpy()
        self.delta = (env_default - onnx_default).astype(np.float32)
        print(f"[onnx] default-pose delta (deg): {np.round(np.degrees(self.delta), 2).tolist()}")

    def __call__(self, obs) -> torch.Tensor:
        actor = obs["actor"]
        x = actor.detach().cpu().numpy().astype(np.float32)
        assert x.shape[1] == 61, f"expected 61-d actor obs, got {x.shape[1]}"
        x[:, self.JOINT_POS] += self.delta
        x[:, self.LAST_ACTION] += self.delta
        (a,) = self.sess.run(None, {self.sess.get_inputs()[0].name: x})
        return torch.as_tensor(a - self.delta, device=actor.device)


class SubstepStats:
    """Running per-joint statistics, accumulated on the env's device each substep."""

    def __init__(self, robot, bam, kt: float, R: float, device, physics_dt: float):
        self.robot, self.bam, self.kt, self.R = robot, bam, kt, R
        self.physics_dt = physics_dt
        self.names = list(robot.actuator_names)
        self.joint_ids = [robot.joint_names.index(n) for n in self.names]
        lim = robot.data.joint_pos_limits[0, self.joint_ids]  # (J, 2)
        self.lo, self.hi = lim[:, 0], lim[:, 1]
        self.stall = VIN_NOMINAL * kt / R
        self.leg = torch.tensor(
            [not n.startswith(("neck", "head")) for n in self.names], device=device
        )
        J = len(self.names)
        z = lambda: torch.zeros(J, device=device, dtype=torch.float64)
        self.n = 0  # valid env-substeps recorded
        self.n_all = 0  # all env-substeps recorded, for termination rates
        self.tau_sq, self.cur_sq, self.mech = z(), z(), z()
        self.over30, self.over50, self.brace, self.near = z(), z(), z(), z()
        self.brace_tau = z()
        self.peak = torch.zeros(J, device=device)
        self.hist = torch.zeros(J, HIST_BINS, device=device, dtype=torch.float64)
        self.any_leg_over50 = 0.0
        self.upright = 0.0
        self.invalid = 0
        self.falls_detected = 0
        self.was_upright = None  # (N,) bool, armed after standing upright
        self.ep_time = None  # (N,) s since the env's last reset
        self.speed = 0.0
        self.yaw = 0.0
        self.cmd = torch.zeros(3, device=device, dtype=torch.float64)

    def forget(self, env_ids) -> None:
        """Envs that just reset start over: not armed, settle clock at zero."""
        if self.was_upright is not None:
            self.was_upright[env_ids] = False
            self.ep_time[env_ids] = 0.0

    def step(self, env, recording: bool) -> None:
        """One physics substep. Fall tracking always runs; statistics only while recording."""
        d = self.robot.data
        gz_all = d.projected_gravity_b[:, 2].float()
        tau = d.actuator_force.float()  # (N, J), applied after the force limit
        motor = self.bam._prev_motor_torque.float()  # (N, J), before the limit
        q = d.joint_pos[:, self.joint_ids].float()
        qd = d.joint_vel[:, self.joint_ids].float()
        lin = d.root_link_lin_vel_b[:, 0].float()
        yaw = d.root_link_ang_vel_b[:, 2].float()
        valid = (
            torch.isfinite(tau).all(1) & torch.isfinite(motor).all(1)
            & torch.isfinite(q).all(1) & torch.isfinite(qd).all(1)
            & torch.isfinite(gz_all) & torch.isfinite(lin) & torch.isfinite(yaw)
        )
        if self.was_upright is None:
            self.was_upright = torch.zeros_like(valid)
            self.ep_time = torch.zeros(valid.shape[0], device=valid.device)
        self.ep_time += self.physics_dt
        # A fall is upright -> fallen after the settle window; an invalid sample
        # leaves an env's state as it was. Resets call forget().
        down = valid & (gz_all > FALLEN_GZ)
        up = valid & (gz_all < UPRIGHT_GZ)
        fell = self.was_upright & down & (self.ep_time >= SETTLE_S)
        self.was_upright = (self.was_upright & ~down) | up
        if not recording:
            return
        self.n_all += valid.shape[0]
        self.falls_detected += int(fell.sum())
        self.invalid += int((~valid).sum())
        if not bool(valid.any()):
            return
        tau, motor, q, qd = tau[valid], motor[valid], q[valid], qd[valid]
        gz, lin, yaw = gz_all[valid], lin[valid], yaw[valid]
        cmd = env.command_manager.get_command("twist")[valid, :3]
        a = tau.abs()
        N = tau.shape[0]
        self.n += N
        self.tau_sq += (a.double() ** 2).sum(0)
        self.cur_sq += ((motor.double() / self.kt) ** 2).sum(0)
        self.mech += (tau * qd).clamp(min=0).double().sum(0)
        self.over30 += (a > 0.3 * self.stall).double().sum(0)
        over50 = a > 0.5 * self.stall
        self.over50 += over50.double().sum(0)
        self.any_leg_over50 += float((over50 & self.leg).any(dim=1).sum())
        self.peak = torch.maximum(self.peak, a.max(0).values)
        b = (a / HIST_MAX_NM * HIST_BINS).long().clamp(max=HIST_BINS - 1)
        self.hist.scatter_add_(1, b.T, torch.ones_like(b.T, dtype=torch.float64))
        at_lo = q < self.lo + STOP_MARGIN
        at_hi = q > self.hi - STOP_MARGIN
        still = qd.abs() < STILL_VEL
        brace = still & ((at_lo & (tau < 0)) | (at_hi & (tau > 0)))
        self.near += (at_lo | at_hi).double().sum(0)
        self.brace += brace.double().sum(0)
        self.brace_tau += (a * brace).double().sum(0)
        self.upright += float((gz < UPRIGHT_GZ).sum())
        self.speed += float(lin.sum())
        self.yaw += float(yaw.sum())
        self.cmd += cmd.double().sum(0)

    def joints(self) -> dict:
        n = self.n
        cdf = self.hist.cumsum(1) / self.hist.sum(1, keepdim=True).clamp(min=1)
        # Upper edge of the bin holding the 95th percentile, never above the peak.
        p95 = ((cdf < 0.95).sum(1).double() + 1) * HIST_MAX_NM / HIST_BINS
        p95 = torch.minimum(p95, self.peak.double())
        out = {}
        for j, name in enumerate(self.names):
            br = float(self.brace[j])
            out[name] = {
                "rms_nm": float((self.tau_sq[j] / n).sqrt()),
                "p95_nm": float(p95[j]),
                "peak_nm": float(self.peak[j]),
                "pct_over_30_stall": float(100 * self.over30[j] / n),
                "pct_over_50_stall": float(100 * self.over50[j] / n),
                "copper_w": float(self.cur_sq[j] / n * self.R),
                "mech_w": float(self.mech[j] / n),
                "pct_at_stop": float(100 * br / n),
                "tau_at_stop_nm": float(self.brace_tau[j] / br) if br else 0.0,
                "pct_near_stop": float(100 * self.near[j] / n),
            }
        return out


def run_scenario(task, scenario, args, kt, R):
    vx, vy, wz = SCENARIOS[scenario]
    env_cfg = load_env_cfg(task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    freeze_eval_cfg(env_cfg, vx, vy, wz)
    step_dt = env_cfg.sim.mujoco.timestep * env_cfg.decimation
    steps = int(args.seconds / step_dt)
    warmup = int(args.warmup / step_dt)
    if steps <= warmup:
        raise ScenarioFailed(f"{steps} steps leave nothing after {warmup} warm-up steps")
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    try:
        return _measure(env, task, scenario, args, kt, R, agent_cfg, steps, warmup)
    finally:
        env.close()


def _measure(env, task, scenario, args, kt, R, agent_cfg, steps, warmup):
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    robot = env.scene["robot"]
    bams = [a for a in robot.actuators if hasattr(a, "_prev_motor_torque")]
    assert len(bams) == 1, f"expected one BAM actuator group, got {len(bams)}"
    bam = bams[0]
    assert list(bam._target_names) == list(robot.actuator_names), "BAM target order differs"

    if args.onnx:
        policy = OnnxPolicy(args.onnx, robot)
    else:
        runner_cls = load_runner_cls(task) or MjlabOnPolicyRunner
        runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(args.checkpoint, load_cfg={"actor": True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)

    physics_dt = env.step_dt / env.cfg.decimation
    stats = SubstepStats(robot, bam, kt, R, env.device, physics_dt)
    recording = False
    scene_update = env.scene.update

    def update_and_record(*a, **kw):
        scene_update(*a, **kw)
        stats.step(env, recording)

    # Called once per physics substep, after sim.step(), before terminations and resets.
    env.scene.update = update_and_record

    falls = timeouts = 0
    obs = wrapped.get_observations()
    with torch.inference_mode():
        for i in range(steps):
            recording = i >= warmup
            obs, _, dones, extras = wrapped.step(policy(obs))
            stats.forget(dones.bool())
            if not recording:
                continue
            time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
            falls += int((dones.bool() & ~time_outs).sum())
            timeouts += int(time_outs.sum())
    if stats.n == 0:
        raise ScenarioFailed(f"no valid samples recorded ({stats.invalid} invalid)")
    # Exposure from recorded samples: all of them for terminations, valid ones for the rest.
    minutes_all = stats.n_all * physics_dt / 60.0
    minutes_valid = stats.n * physics_dt / 60.0

    n = stats.n
    joints = stats.joints()
    return {
        "scenario": scenario,
        "command_mean": (stats.cmd / n).tolist(),
        "forward_speed_mean": stats.speed / n,
        "yaw_rate_mean": stats.yaw / n,
        "falls_per_env_minute": falls / minutes_all,
        "falls_detected_per_env_minute": stats.falls_detected / minutes_valid,
        "invalid_samples": stats.invalid,
        "timeouts_per_env_minute": timeouts / minutes_all,
        "pct_upright": 100 * stats.upright / n,
        "pct_any_leg_over_50_stall": 100 * stats.any_leg_over50 / n,
        "stall_nm": stats.stall,
        "total_copper_w": float(sum(v["copper_w"] for v in joints.values())),
        "samples": stats.n,
        "joints": joints,
    }


def print_report(res):
    print(
        f"\n== {res['scenario']}: cmd={np.round(res['command_mean'], 3).tolist()} "
        f"vx={res['forward_speed_mean']:.3f} m/s wz={res['yaw_rate_mean']:.3f} rad/s "
        f"fell/env-min={res['falls_detected_per_env_minute']:.3f} terminations/env-min={res['falls_per_env_minute']:.3f} "
        f"upright={res['pct_upright']:.1f}% invalid={res['invalid_samples']} "
        f"any-leg>50%={res['pct_any_leg_over_50_stall']:.1f}% "
        f"copper={res['total_copper_w']:.2f} W (stall {res['stall_nm']:.3f} Nm)"
    )
    print(
        f"{'joint':16} {'rms':>6} {'p95':>6} {'peak':>6} {'>30%':>6} {'>50%':>6} "
        f"{'Cu W':>6} {'brace%':>7} {'braceNm':>7} {'near%':>6}"
    )
    for name, v in res["joints"].items():
        print(
            f"{name:16} {v['rms_nm']:6.3f} {v['p95_nm']:6.3f} {v['peak_nm']:6.3f} "
            f"{v['pct_over_30_stall']:6.1f} {v['pct_over_50_stall']:6.1f} {v['copper_w']:6.2f} "
            f"{v['pct_at_stop']:7.1f} {v['tau_at_stop_nm']:7.3f} {v['pct_near_stop']:6.1f}"
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="rsl_rl model_*.pt")
    src.add_argument("--onnx", help="batch-1 ONNX policy from the XGO runtime")
    ap.add_argument("--scenarios", default=",".join(SCENARIOS))
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--warmup", type=float, default=2.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--params", default=str(DEFAULT_PARAMS), help="BAM actuator JSON (kt, R) used for stall and copper loss")
    ap.add_argument("--out", help="JSON path (default logs/servo_load/<task>_<time>.json)")
    args = ap.parse_args()
    if (args.num_envs < 1 or not math.isfinite(args.seconds) or not math.isfinite(args.warmup)
            or args.seconds <= 0 or args.warmup < 0 or args.warmup >= args.seconds):
        ap.error("need num-envs >= 1, finite seconds > 0 and 0 <= warmup < seconds")

    params = json.loads(Path(args.params).read_text())
    kt, R = params["kt"], params["R"]
    out = Path(args.out) if args.out else Path("logs/servo_load") / (
        f"{args.task}_{time.strftime('%Y%m%d-%H%M%S')}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    results, failed = [], []
    for scenario in args.scenarios.split(","):
        try:
            res = run_scenario(args.task, scenario, args, kt, R)
        except ScenarioFailed as e:
            print(f"\n== {scenario}: FAILED: {e}")
            failed.append({"scenario": scenario, "error": str(e)})
            _write(out, args, kt, R, results, failed)
            continue
        results.append(res)
        _write(out, args, kt, R, results, failed)  # after each scenario, so a crash keeps what finished
        print_report(res)
    _write(out, args, kt, R, results, failed)
    print(f"\nwrote {out}")
    if failed:
        raise SystemExit(f"{len(failed)} scenario(s) failed: {', '.join(f['scenario'] for f in failed)}")


def _write(out, args, kt, R, results, failed):
    """Write the JSON to a unique sibling temp file, then replace: an interruption keeps the last good file."""
    fd, tmp_name = tempfile.mkstemp(dir=out.parent, prefix=out.name + ".", suffix=".tmp")
    os.close(fd)
    tmp = Path(tmp_name)
    tmp.write_text(json.dumps({
        "task": args.task,
        "policy": args.onnx or args.checkpoint,
        "num_envs": args.num_envs,
        "seconds": args.seconds,
        "warmup": args.warmup,
        "kt": kt,
        "R": R,
        "vin_nominal": VIN_NOMINAL,
        "eval": "frozen v3: no pushes/randomization/curricula/IMU misalignment, zero pose commands, "
                f"obs lags at max, vin {EVAL_VIN} V, sag {EVAL_VIN_DROP_GAIN} V/Nm, delay {EVAL_DELAY_LAG} substeps, "
                "substep stats after scene update, invalid samples excluded",
        "results": results,
        "failed": failed,
    }, indent=2))
    tmp.replace(out)


if __name__ == "__main__":
    main()
