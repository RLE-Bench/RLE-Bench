"""The task07 episode loop: 20 Hz policy control over 500 Hz physics.

Drives any object exposing the Policy interface (in-process test policies,
the privileged golden picker, or a sandboxed submission) through one seeded
episode, tracking clears / floor drops / damage from HARNESS state only.
Cleared parts are pinned in the graveyard every tick ("the belt carried it
away"), keeping contact costs bounded and the latch irreversible.

The magnet end-effector is simulated here: while ctrl[7] commands the head
energized, parts touching the magnet face weld to it at their current
relative pose; de-energizing releases everything (MagnetEngine).

A policy fault (exception, bad shape, non-finite values, sandbox death)
never crashes the episode: the last valid ctrl is held, the fault is
counted, and physics runs to the budget. Faults surface through the gate.
"""
from __future__ import annotations

import hashlib
import pathlib
from dataclasses import dataclass, field

import mujoco
import numpy as np

from rlebench.core.media import MujocoCamera, Recorder, VideoWriter

from . import config, episodes, metrics, parts, scene, sensor, spec

PRE_SETTLE_T = 0.2   # relax pile contacts in the full cell before t = 0


class CameraView:
    """One camera's view of the cell as an rgb frame, for a Recorder."""

    def __init__(self, model, camera: str = "overview",
                 width: int = 960, height: int = 540):
        self.camera = MujocoCamera(model, camera, size=(width, height))

    def frame(self, data, hud: dict | None = None) -> np.ndarray:
        return self.camera.render(data)

    def close(self) -> None:
        self.camera.close()


class DashboardView:
    """Composite dev frame: the third-person view plus everything the
    policy senses — overhead RGB, wrist RGB, overhead depth, and a
    scrolling wrist force-torque trace with episode state. 1280x720."""

    W, H = 1280, 720
    MAIN_W, MAIN_H = 960, 540
    TILE_W, TILE_H = 320, 240
    STRIP_H = 180
    FT_WINDOW = 400          # ticks of force history in the strip (20 s)
    FT_FMAX = 3000.0         # sqrt-scale ceiling of the force axis

    def __init__(self, model):
        from PIL import Image, ImageDraw, ImageFont   # dev-only dependency
        self._pil = (Image, ImageDraw)
        try:
            import matplotlib
            from matplotlib import cm
            font = (pathlib.Path(matplotlib.get_data_path())
                    / "fonts" / "ttf" / "DejaVuSans.ttf")
            self.font = ImageFont.truetype(str(font), 15)
            self.font_small = ImageFont.truetype(str(font), 12)
            self._turbo = cm.turbo
        except Exception:
            self.font = self.font_small = ImageFont.load_default()
            self._turbo = None
        self.model = model
        self.r_main = mujoco.Renderer(model, self.MAIN_H, self.MAIN_W)
        self.r_tile = mujoco.Renderer(model, self.TILE_H, self.TILE_W)
        self.r_depth = mujoco.Renderer(model, self.TILE_H, self.TILE_W)
        self.r_depth.enable_depth_rendering()
        self.ft_hist: list[float] = []

    def _tile(self, data, camera: str) -> np.ndarray:
        self.r_tile.update_scene(data, camera=camera)
        return self.r_tile.render()

    def _depth_tile(self, data) -> np.ndarray:
        self.r_depth.update_scene(data, camera="overhead")
        x = np.clip((self.r_depth.render() - 0.45) / 0.5, 0.0, 1.0)
        if self._turbo is not None:
            return (self._turbo(x)[..., :3] * 255).astype(np.uint8)
        g = ((1.0 - x) * 255).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)

    def _label(self, img: np.ndarray, text: str) -> np.ndarray:
        Image, ImageDraw = self._pil
        im = Image.fromarray(img)
        ImageDraw.Draw(im).text((6, 4), text, fill=(255, 255, 255),
                                font=self.font_small,
                                stroke_width=1, stroke_fill=(0, 0, 0))
        return np.asarray(im)

    def _strip(self, ft: np.ndarray, hud: dict) -> np.ndarray:
        """Scrolling |force| trace on a sqrt scale, plus live readouts."""
        Image, ImageDraw = self._pil
        f_mag = float(np.linalg.norm(ft[:3]))
        t_mag = float(np.linalg.norm(ft[3:]))
        self.ft_hist.append(f_mag)
        hist = self.ft_hist[-self.FT_WINDOW:]

        im = Image.new("RGB", (self.MAIN_W, self.STRIP_H), (18, 20, 19))
        dr = ImageDraw.Draw(im)
        plot_h = self.STRIP_H - 34

        def y_of(f):
            frac = min(np.sqrt(max(f, 0.0) / self.FT_FMAX), 1.0)
            return self.STRIP_H - 6 - frac * plot_h

        for ref in (10, 100, 1000):
            y = y_of(ref)
            dr.line([(0, y), (self.MAIN_W, y)], fill=(52, 56, 53))
            dr.text((self.MAIN_W - 52, y - 15), f"{ref} N",
                    fill=(120, 126, 121), font=self.font_small)
        if len(hist) > 1:
            x0 = self.MAIN_W - len(hist) * 2
            dr.line([(x0 + 2 * i, y_of(f)) for i, f in enumerate(hist)],
                    fill=(240, 180, 60), width=2)
        dr.text(
            (8, 6),
            f"wrist F/T   |F| {f_mag:7.1f} N   |T| {t_mag:5.2f} Nm      "
            f"t {hud.get('t', 0.0):6.1f} s    held {hud.get('held', 0)}    "
            f"cleared {hud.get('cleared', 0)}/{hud.get('n', 0)}",
            fill=(235, 238, 235), font=self.font)
        return np.asarray(im)

    def frame(self, data, hud: dict | None = None) -> np.ndarray:
        hud = hud or {}
        frame = np.full((self.H, self.W, 3), 24, dtype=np.uint8)
        self.r_main.update_scene(data, camera="overview")
        frame[:self.MAIN_H, :self.MAIN_W] = self.r_main.render()
        frame[0:240, 960:1280] = self._label(
            self._tile(data, "overhead"), "overhead rgb")
        frame[240:480, 960:1280] = self._label(
            self._tile(data, "arm_wrist"), "wrist rgb")
        frame[480:720, 960:1280] = self._label(
            self._depth_tile(data), "overhead depth")
        ft = scene.ft_reading(self.model, data)
        frame[self.MAIN_H:, :self.MAIN_W] = self._strip(ft, hud)
        return frame

    def close(self) -> None:
        for r in (self.r_main, self.r_tile, self.r_depth):
            r.close()


def _open_view(video, model, mode, camera, size):
    """The frame producer for a recording, or None when there is nothing to
    record or no GL to render with (noted on a borrowed writer)."""
    if not video:
        return None
    try:
        if mode == "dashboard":
            return DashboardView(model)
        return CameraView(model, camera, width=size[0], height=size[1])
    except Exception as exc:  # noqa: BLE001
        if isinstance(video, VideoWriter):
            video.skipped = f"recorder: {exc}"
        return None


class MagnetEngine:
    """Contact-electromagnet: welds parts that touch the energized face.

    Welding freezes the part's CURRENT pose relative to the head (nothing
    is attracted at a distance); several parts may be held at once. All
    welds drop the moment the command de-energizes."""

    def __init__(self, model):
        self.model = model
        n = scene.n_parts(model)
        self.eq = {i: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY,
                                        f"mag_weld_{i}") for i in range(n)}
        self.head_geom = model.geom("arm_magnet_head").id
        self.head_body = model.body("arm_magnet").id
        self.part_of_body = {model.body(parts.part_name(i)).id: i
                             for i in range(n)}

    def held(self, data) -> set[int]:
        return {i for i, e in self.eq.items() if data.eq_active[e]}

    def _weld(self, data, i: int) -> None:
        b1, b2 = self.head_body, self.model.body(parts.part_name(i)).id
        R1 = data.xmat[b1].reshape(3, 3)
        rel_pos = R1.T @ (data.xpos[b2] - data.xpos[b1])
        q1n = np.zeros(4)
        rel_quat = np.zeros(4)
        mujoco.mju_negQuat(q1n, data.xquat[b1])
        mujoco.mju_mulQuat(rel_quat, q1n, data.xquat[b2])
        e = self.eq[i]
        self.model.eq_data[e, 0:3] = 0.0
        self.model.eq_data[e, 3:6] = rel_pos
        self.model.eq_data[e, 6:10] = rel_quat
        self.model.eq_data[e, 10] = 1.0
        data.eq_active[e] = 1

    def update(self, data, energized: bool,
               exclude: set[int] | None = None) -> set[int]:
        """Advance the magnet state at a control tick; returns held parts."""
        exclude = exclude or set()
        if not energized:
            for e in self.eq.values():
                data.eq_active[e] = 0
            return set()
        held = self.held(data)
        for i in held & exclude:
            data.eq_active[self.eq[i]] = 0
        held -= exclude
        for c in range(data.ncon):
            con = data.contact[c]
            if con.dist > spec.MAG_TOUCH:
                continue
            g1, g2 = int(con.geom1), int(con.geom2)
            if g1 == self.head_geom:
                other = g2
            elif g2 == self.head_geom:
                other = g1
            else:
                continue
            i = self.part_of_body.get(int(self.model.geom_bodyid[other]))
            if i is None or i in held or i in exclude:
                continue
            self._weld(data, i)
            held.add(i)
        return held


@dataclass
class EpisodeLog:
    seed: int
    n_parts: int = 0
    clear_times: dict = field(default_factory=dict)
    floor_drops: int = 0
    damage_events: int = 0
    bin_hits: int = 0             # tool-strike events on the bin
    peak_force: float = 0.0
    peak_sustained: float = 0.0   # max 2-tick-sustained force on any part
    peak_tool_bin: float = 0.0    # max 2-tick-sustained tool-bin force
    penalty_events: list = field(default_factory=list)
    ticks: int = 0
    policy_faults: int = 0
    fault_reasons: list = field(default_factory=list)
    aborted: str | None = None      # sim_unstable, wall_budget, policy_startup
    digest: str = ""
    final_positions: np.ndarray | None = None

    def metrics(self, budget_t: float = spec.EPISODE_T) -> dict:
        m = metrics.episode_metrics(self.n_parts, self.clear_times,
                                    self.floor_drops, self.damage_events,
                                    budget_t)
        m.update(ticks=self.ticks, policy_faults=self.policy_faults,
                 fault_reasons=list(self.fault_reasons),
                 aborted=self.aborted, bin_hits=self.bin_hits,
                 peak_force=round(self.peak_force, 2),
                 peak_sustained=round(self.peak_sustained, 2),
                 peak_tool_bin=round(self.peak_tool_bin, 2),
                 penalty_events=list(self.penalty_events),
                 digest=self.digest)
        return m


class NullPolicy:
    """Holds the home pose, magnet off. The do-nothing reward floor."""

    def reset(self, cell_spec: dict, seed: int) -> None:
        pass

    def act(self, obs: dict):
        return np.array([*spec.ARM_HOME, 0.0])


def _policy_wall_s(policy) -> float:
    """Parent-metered submission time, when the policy adapter exposes it."""
    return float(getattr(policy, "policy_wall_s", 0.0))


def _digest(log: EpisodeLog, data) -> str:
    h = hashlib.sha256()
    h.update(np.round(np.asarray(data.qpos, dtype=np.float64), 9).tobytes())
    for i, t in sorted(log.clear_times.items()):
        h.update(f"{i}:{t:.6f};".encode())
    return h.hexdigest()[:16]


def _rounded_vector(values, digits: int = 6) -> list[float]:
    return [round(float(x), digits) for x in np.asarray(values).reshape(-1)]


def _part_state(model, data, i: int) -> dict:
    joint = data.joint(parts.part_name(i))
    linear, angular = scene.part_velocity(model, data, i)
    return {
        "part_id": int(i),
        "part_body": parts.part_name(i),
        "part_pos_world_m": _rounded_vector(joint.qpos[:3]),
        "part_quat_world_wxyz": _rounded_vector(joint.qpos[3:7]),
        "linear_velocity_world_mps": _rounded_vector(linear),
        "linear_speed_mps": round(float(np.linalg.norm(linear)), 6),
        "angular_velocity_world_radps": _rounded_vector(angular),
    }


def _rounded_contact(contact: dict | None) -> dict | None:
    if contact is None:
        return None
    out = dict(contact)
    out["pos_world_m"] = _rounded_vector(out["pos_world_m"])
    out["normal_world"] = _rounded_vector(out["normal_world"])
    out["force_n"] = round(float(out["force_n"]), 2)
    return out


def run_episode(policy, seed: int, budget_t: float = spec.EPISODE_T,
                render: bool = True, wall_budget_s: float | None = None,
                cell_spec: dict | None = None,
                policy_seed: int | None = None,
                video=None,
                video_mode: str = "dashboard",
                video_camera: str = "overview",
                video_every: int = 1,
                video_size: tuple = (960, 540)) -> EpisodeLog:
    """One seeded episode. ``policy`` implements reset/act (see spec).

    ``seed`` is private environment randomness: it constructs the pile and
    sensor realizations. ``policy_seed`` is the independent RNG seed exposed
    through ``policy.reset``. Callers that omit it get zero, never ``seed``,
    so public pile-generation code cannot be replayed from policy input.

    ``render=False`` skips cameras (frame keys absent from obs) — for
    scripted harness-side policies that don't look at pixels.
    ``wall_budget_s`` applies only to parent-metered submission ``reset`` and
    ``act`` waits. Physics, sensor rendering, and diagnostic video are excluded.
    ``video`` records an mp4: a path, or an open rlebench.core.media
    VideoWriter the caller finishes. ``video_mode="dashboard"`` composites
    the third-person view with every sensor stream (overhead RGB + depth,
    wrist RGB, wrist F/T trace); ``"camera"`` records just ``video_camera``.
    Recording is a pure observer: it reads state the loop already computed,
    never steps physics, and any failure (no ffmpeg, no GL) only ends the
    recording.
    """
    model, data = episodes.build_episode_scene(seed)
    n = scene.n_parts(model)
    log = EpisodeLog(seed=seed, n_parts=n)

    scene.set_arm(model, data, spec.ARM_HOME)
    scene.apply_ctrl(model, data, [*spec.ARM_HOME, 0.0])
    scene.settle(model, data, PRE_SETTLE_T)

    clear = metrics.ClearanceTracker(n)
    floor = metrics.FloorTracker(n)
    damage = metrics.DamageTracker(n, config.DAMAGE_FORCE_N,
                                   config.DMG_TICKS)
    bin_hit = metrics.BinHitTracker(config.BIN_HIT_FORCE_N,
                                    config.BIN_HIT_TICKS,
                                    config.BIN_HIT_REARM_TICKS)
    magnet = MagnetEngine(model)
    held: set[int] = set()

    # Harness-side privileged reference policies get direct sim handles.
    # Sandboxed submissions can never reach this: their parent-side adapter
    # has no ``privileged`` attribute.
    if getattr(policy, "privileged", False):
        policy.bind(model, data)
    if cell_spec is None:
        cell_spec = spec.cell_spec()
        cell_spec["damage_force_n"] = config.DAMAGE_FORCE_N
        cell_spec["damage_ticks"] = config.DMG_TICKS
        cell_spec["bin_hit_force_n"] = config.BIN_HIT_FORCE_N
    policy_wall_start = _policy_wall_s(policy)
    policy.reset(cell_spec, 0 if policy_seed is None else int(policy_seed))
    if getattr(policy, "dead", None) is not None:
        log.aborted = "policy_startup"
        log.policy_faults = 1
        log.fault_reasons.append(f"policy reset failed: {policy.dead}")
        return log

    rig = sensor.CameraRig(model) if render else None
    view = _open_view(video, model, video_mode, video_camera, video_size)
    recorder = (Recorder(video, fps=spec.CTRL_HZ / video_every, every=video_every)
                if view is not None else None)
    held_ctrl = np.array([*spec.ARM_HOME, 0.0])
    prev_forces = np.zeros(n)
    last_release: dict[int, dict] = {}
    ft_sigma = np.array([spec.FT_NOISE_FORCE_N] * 3
                        + [spec.FT_NOISE_TORQUE_NM] * 3)
    total_ticks = int(round(budget_t * spec.CTRL_HZ))
    frame_idx = 0
    try:
        for tick in range(total_ticks):
            if (wall_budget_s is not None
                    and _policy_wall_s(policy) - policy_wall_start
                    > wall_budget_s):
                log.aborted = "wall_budget"
                break
            t = tick / spec.CTRL_HZ
            ft_rng = np.random.default_rng([seed, spec.STREAM_FT, tick])
            obs = {
                "qpos": np.array(scene.get_arm(data)),
                "qvel": np.array([data.joint(f"arm_joint{i}").qvel[0]
                                  for i in range(1, 8)]),
                "tau": scene.arm_tau(data),
                "ft": scene.ft_reading(model, data)
                + ft_rng.normal(0.0, 1.0, 6) * ft_sigma,
                "mag_on": 1.0 if held_ctrl[7] >= spec.MAG_ON else 0.0,
                "t": t,
            }
            if (rig is not None
                    and getattr(policy, "dead", None) is None
                    and tick % spec.FRAME_EVERY_TICKS == 0):
                obs.update(sensor.frame_obs(rig, model, data, seed,
                                            frame_idx))
                frame_idx += 1

            ctrl, fault = None, None
            try:
                ctrl = policy.act(obs)
            except Exception as e:
                fault = f"act raised: {e!r}"
            # Stop before applying an action that crossed the submission-only
            # budget. Rendering and simulation work above are not charged.
            if (wall_budget_s is not None
                    and _policy_wall_s(policy) - policy_wall_start
                    > wall_budget_s):
                log.aborted = "wall_budget"
                break
            if ctrl is not None:
                ctrl = np.asarray(ctrl, dtype=float).reshape(-1)
                if ctrl.shape != (spec.N_CTRL,) \
                        or not np.isfinite(ctrl).all():
                    fault = f"bad ctrl (shape {ctrl.shape})"
                    ctrl = None
            if ctrl is None:
                log.policy_faults += 1
                if fault and len(log.fault_reasons) < 10:
                    log.fault_reasons.append(f"t={t:.2f} {fault}")
                ctrl = held_ctrl
            held_ctrl = scene.apply_ctrl(model, data, ctrl)
            held_before = set(held)
            held = magnet.update(data, held_ctrl[7] >= spec.MAG_ON,
                                 exclude=set(clear.clear_time))
            for i in sorted(held_before - held - set(clear.clear_time)):
                last_release[i] = {
                    "t": round(float(t), 3),
                    **_part_state(model, data, i),
                }

            for _ in range(spec.CTRL_DECIMATION):
                mujoco.mj_step(model, data)
            log.ticks = tick + 1
            if recorder is not None:
                recorder.capture(lambda: view.frame(data, dict(
                    t=(tick + 1) / spec.CTRL_HZ, held=len(held),
                    cleared=clear.cleared, n=n)))

            if data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0 \
                    or not np.isfinite(data.qpos).all():
                log.aborted = "sim_unstable"
                break

            t_next = (tick + 1) / spec.CTRL_HZ
            positions = scene.part_positions(data, n)
            speeds = scene.part_speeds(data, n)
            forces, force_contacts = scene.part_contact_force_details(
                model, data, n)
            if n:
                log.peak_force = max(log.peak_force, float(forces.max()))
                log.peak_sustained = max(
                    log.peak_sustained,
                    float(np.minimum(forces, prev_forces).max()))
            newly_damaged = damage.update(forces)
            for i in newly_damaged:
                contact = scene.part_contact_detail(
                    model, data, i, int(force_contacts[i]))
                log.penalty_events.append({
                    "type": "damage",
                    "t": round(float(t_next), 3),
                    "tick": int(tick + 1),
                    **_part_state(model, data, i),
                    "held_by_magnet": bool(i in held),
                    "magnet_on": bool(held_ctrl[7] >= spec.MAG_ON),
                    "contact": _rounded_contact(contact),
                    "force_n": round(float(forces[i]), 2),
                    "previous_force_n": round(float(prev_forces[i]), 2),
                    "sustained_force_n": round(
                        float(min(forces[i], prev_forces[i])), 2),
                    "part_peak_force_n": round(float(damage.peak[i]), 2),
                    "threshold_n": float(config.DAMAGE_FORCE_N),
                    "streak_ticks": int(config.DMG_TICKS),
                })
            prev_forces = forces
            bin_hit.update(scene.tool_bin_force(model, data))
            newly_dropped = floor.update(
                positions, exclude=set(clear.clear_time))
            for i in newly_dropped:
                contacts = [_rounded_contact(x)
                            for x in scene.part_contacts(model, data, i)]
                log.penalty_events.append({
                    "type": "floor_drop",
                    "t": round(float(t_next), 3),
                    "tick": int(tick + 1),
                    **_part_state(model, data, i),
                    "held_by_magnet": bool(i in held),
                    "magnet_on": bool(held_ctrl[7] >= spec.MAG_ON),
                    "floor_z_threshold_m": float(spec.FLOOR_Z),
                    "current_contacts": contacts,
                    "last_release": last_release.get(i),
                })
            newly = clear.update(t_next, positions, speeds, exclude=held)
            for i in newly:
                data.eq_active[magnet.eq[i]] = 0   # never weld a cleared part
                held.discard(i)
            for k, i in enumerate(sorted(clear.clear_time)):
                scene.set_part_pose(
                    data, i,
                    [spec.GRAVEYARD[0] + k * spec.GRAVEYARD_DX,
                     spec.GRAVEYARD[1], spec.GRAVEYARD[2]],
                    [1, 0, 0, 0])
            if newly and clear.cleared == n:
                break   # bin is empty — episode over
    finally:
        if rig is not None:
            rig.close()
        if recorder is not None:
            recorder.close()
            try:
                view.close()
            except Exception:  # noqa: BLE001
                pass

    log.clear_times = dict(clear.clear_time)
    log.floor_drops = len(floor.dropped)
    log.damage_events = len(damage.damaged)
    log.bin_hits = bin_hit.hits
    log.peak_tool_bin = bin_hit.peak_sustained
    log.final_positions = scene.part_positions(data, n) if n else None
    log.digest = _digest(log, data)
    return log
