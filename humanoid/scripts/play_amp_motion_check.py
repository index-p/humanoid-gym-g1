import argparse
import json
import os
import sys
import time

import numpy as np
from isaacgym import gymapi, gymtorch, gymutil
import torch
from humanoid.amp_motion_utils import canonicalize_motion_dict, finite_difference, load_motion_file
from humanoid.envs import *  # noqa: F401,F403
from humanoid.utils import get_args, task_registry


def parse_custom_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--input",
        required=True,
        help="Motion file to visualize: motion_visualization/motion_amp_expert .npz/.pkl or TienKung-style .txt",
    )
    parser.add_argument("--task", default="g1_walk_amp", help="Registered Isaac Gym task")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop the motion after the final frame")
    parser.add_argument("--start_frame", type=int, default=0, help="Frame to start playback from")
    parser.add_argument("--max_frames", type=int, default=-1, help="Maximum frames to play, -1 means all")
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for quick scanning")
    parser.add_argument(
        "--root",
        choices=("follow", "fixed"),
        default="follow",
        help="Follow dataset root motion, or keep root xy fixed while replaying joints",
    )
    parser.add_argument(
        "--draw_feet",
        action="store_true",
        help="Draw dataset foot points when feet_pos_world exists in the motion file",
    )
    parser.add_argument(
        "--z_offset",
        type=float,
        default=0.0,
        help="Extra root z offset in meters. Positive values raise the robot.",
    )
    parser.add_argument(
        "--ground_align",
        action="store_true",
        help="Auto-raise the motion so the first replayed rigid body is above the ground.",
    )
    parser.add_argument(
        "--ground_clearance",
        type=float,
        default=0.02,
        help="Clearance used by --ground_align.",
    )
    parser.add_argument(
        "--no_sim_step",
        action="store_true",
        help="Do not step Isaac Gym after writing each motion frame. Useful only for debugging raw state writes.",
    )
    return parser.parse_known_args(argv)


def euler_xyz_to_quat_xyzw(euler_xyz):
    roll = euler_xyz[:, 0]
    pitch = euler_xyz[:, 1]
    yaw = euler_xyz[:, 2]
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    quat = np.stack(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ),
        axis=-1,
    )
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True).clip(min=1e-8)
    return quat.astype(np.float32)


def load_tienkung_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    frames = np.asarray(payload["Frames"], dtype=np.float32)
    if frames.ndim != 2:
        raise ValueError(f"{path}: Frames must be a rank-2 array, got {frames.shape}")
    if (frames.shape[1] - 12) % 2 != 0:
        raise ValueError(
            f"{path}: frame width {frames.shape[1]} is incompatible with "
            "root_pos(3)+root_euler(3)+dof_pos(N)+root_lin_vel(3)+root_ang_vel(3)+dof_vel(N)"
        )

    num_dof = (frames.shape[1] - 12) // 2
    dt = float(payload.get("FrameDuration", 1.0 / 30.0))
    return {
        "dof_pos": frames[:, 6 : 6 + num_dof],
        "dof_vel": frames[:, 12 + num_dof : 12 + 2 * num_dof],
        "root_pos": frames[:, 0:3],
        "root_quat": euler_xyz_to_quat_xyzw(frames[:, 3:6]),
        "root_lin_vel": frames[:, 6 + num_dof : 9 + num_dof],
        "root_ang_vel": frames[:, 9 + num_dof : 12 + num_dof],
        "feet_pos_world": None,
        "hand_pos_world": None,
        "joint_names": None,
        "dt": dt,
        "fps": 1.0 / dt,
        "source_format": "tienkung_txt",
    }


def load_motion(path, target_joint_names):
    if path.endswith(".txt"):
        motion = load_tienkung_txt(path)
        if motion["dof_pos"].shape[1] != len(target_joint_names):
            raise ValueError(
                f"{path}: TienKung txt has {motion['dof_pos'].shape[1]} dofs, "
                f"but env has {len(target_joint_names)} dofs. Txt files do not carry joint names, "
                "so automatic reordering is not possible."
            )
        motion["joint_names"] = list(target_joint_names)
        return motion

    raw_data = load_motion_file(path)
    motion = canonicalize_motion_dict(raw_data, target_joint_names=target_joint_names)
    if motion["dof_vel"] is None:
        motion["dof_vel"] = finite_difference(motion["dof_pos"], motion["dt"])
    motion["root_lin_vel"] = finite_difference(motion["root_pos"], motion["dt"])
    motion["root_ang_vel"] = np.zeros((motion["dof_pos"].shape[0], 3), dtype=np.float32)
    motion["source_format"] = os.path.splitext(path)[1].lstrip(".")
    return motion


def validate_motion(motion, env):
    dof_pos = np.asarray(motion["dof_pos"], dtype=np.float32)
    dof_vel = np.asarray(motion["dof_vel"], dtype=np.float32)
    root_pos = np.asarray(motion["root_pos"], dtype=np.float32)
    root_quat = np.asarray(motion["root_quat"], dtype=np.float32)

    num_frames = dof_pos.shape[0]
    checks = {
        "dof_pos": dof_pos.shape == (num_frames, env.num_dof),
        "dof_vel": dof_vel.shape == (num_frames, env.num_dof),
        "root_pos": root_pos.shape == (num_frames, 3),
        "root_quat": root_quat.shape == (num_frames, 4),
        "finite": np.isfinite(dof_pos).all()
        and np.isfinite(dof_vel).all()
        and np.isfinite(root_pos).all()
        and np.isfinite(root_quat).all(),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"Motion validation failed: {failed}")

    quat_norm_err = np.max(np.abs(np.linalg.norm(root_quat, axis=1) - 1.0))
    if quat_norm_err > 5e-2:
        print(f"WARNING: root_quat max |norm-1| = {quat_norm_err:.4f}")

    return dof_pos, dof_vel, root_pos, root_quat


def make_env(args):
    env_cfg, _ = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = 1
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.noise.add_noise = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)
    return env


def draw_motion_feet(env, feet_pos_world, frame_idx, env_origin, z_offset):
    if env.viewer is None or feet_pos_world is None:
        return
    feet = feet_pos_world[frame_idx].reshape(-1, 3)
    colors = [(0.1, 0.6, 1.0), (1.0, 0.35, 0.1)]
    for idx, foot in enumerate(feet[:2]):
        color = colors[idx % len(colors)]
        sphere = gymutil.WireframeSphereGeometry(0.035, 8, 8, None, color=color)
        pose = gymapi.Transform()
        pos = torch.as_tensor(foot, device=env.device, dtype=torch.float32) + env_origin
        pos[2] += z_offset
        pose.p = gymapi.Vec3(float(pos[0]), float(pos[1]), float(pos[2]))
        gymutil.draw_lines(sphere, env.gym, env.viewer, env.envs[0], pose)


def apply_motion_frame(
    env,
    dof_pos,
    dof_vel,
    root_pos,
    root_quat,
    root_lin_vel,
    root_ang_vel,
    frame_idx,
    env_origin,
    root_mode,
    z_offset,
):
    device = env.device
    env.dof_pos[0] = torch.as_tensor(dof_pos[frame_idx], device=device)
    env.dof_vel[0] = torch.as_tensor(dof_vel[frame_idx], device=device)
    if root_mode == "follow":
        env.root_states[0, :3] = torch.as_tensor(root_pos[frame_idx], device=device) + env_origin
    else:
        env.root_states[0, 2] = float(root_pos[frame_idx, 2]) + float(env_origin[2])
    env.root_states[0, 2] += z_offset
    env.root_states[0, 3:7] = torch.as_tensor(root_quat[frame_idx], device=device)
    env.root_states[0, 7:10] = torch.as_tensor(root_lin_vel[frame_idx], device=device)
    env.root_states[0, 10:13] = torch.as_tensor(root_ang_vel[frame_idx], device=device)

    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    env.gym.set_actor_root_state_tensor(env.sim, gymtorch.unwrap_tensor(env.root_states))
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)


def refresh_visual_frame(env, no_sim_step):
    if not no_sim_step:
        env.gym.simulate(env.sim)
        env.gym.fetch_results(env.sim, True)
    env.gym.refresh_dof_state_tensor(env.sim)
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)


def compute_ground_align_offset(
    env,
    dof_pos,
    dof_vel,
    root_pos,
    root_quat,
    root_lin_vel,
    root_ang_vel,
    frame_idx,
    env_origin,
    root_mode,
    requested_z_offset,
    ground_clearance,
):
    apply_motion_frame(
        env,
        dof_pos,
        dof_vel,
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        frame_idx,
        env_origin,
        root_mode,
        requested_z_offset,
    )
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    min_body_z = float(env.rigid_state[0, :, 2].min().item())
    extra_offset = max(0.0, ground_clearance - min_body_z)
    if extra_offset > 0.0:
        print(
            f"ground_align: min_body_z={min_body_z:.4f}, "
            f"adding z_offset={extra_offset:.4f}"
        )
    else:
        print(f"ground_align: min_body_z={min_body_z:.4f}, no extra z offset needed")
    return requested_z_offset + extra_offset


def main():
    custom_args, remaining = parse_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task = custom_args.task
    args.headless = False

    env = make_env(args)
    motion = load_motion(custom_args.input, env.dof_names)
    dof_pos, dof_vel, root_pos, root_quat = validate_motion(motion, env)
    root_lin_vel = np.asarray(motion["root_lin_vel"], dtype=np.float32)
    root_ang_vel = np.asarray(motion["root_ang_vel"], dtype=np.float32)
    feet_pos_world = motion.get("feet_pos_world")
    if feet_pos_world is not None:
        feet_pos_world = np.asarray(feet_pos_world, dtype=np.float32)

    dt = float(motion["dt"])
    start_frame = max(0, min(custom_args.start_frame, dof_pos.shape[0] - 1))
    end_frame = dof_pos.shape[0] if custom_args.max_frames < 0 else min(
        dof_pos.shape[0], start_frame + custom_args.max_frames
    )
    stride = max(custom_args.stride, 1)
    env_origin = env.env_origins[0].clone()
    z_offset = custom_args.z_offset
    if custom_args.ground_align:
        z_offset = compute_ground_align_offset(
            env,
            dof_pos,
            dof_vel,
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            start_frame,
            env_origin,
            custom_args.root,
            z_offset,
            custom_args.ground_clearance,
        )

    print(f"Loaded motion: {custom_args.input}")
    print(
        f"source={motion.get('source_format')} frames={dof_pos.shape[0]} "
        f"dof={dof_pos.shape[1]} fps={float(motion['fps']):.3f} dt={dt:.5f}"
    )
    root_xy_delta = float(np.linalg.norm(root_pos[-1, :2] - root_pos[0, :2]))
    max_joint_span = float(np.max(np.ptp(dof_pos, axis=0)))
    print(
        f"root_z_range=[{root_pos[:, 2].min():.3f}, {root_pos[:, 2].max():.3f}] "
        f"root_xy_delta={root_xy_delta:.3f} max_joint_span={max_joint_span:.3f} z_offset={z_offset:.3f}"
    )
    if root_xy_delta < 0.05 and max_joint_span < 0.8:
        print("WARNING: this motion is nearly static; use a Walk_*.npz file to inspect gait movement.")
    print(f"env dof_names: {env.dof_names}")
    print("Controls: ESC quits, V toggles viewer sync")

    frame_idx = start_frame
    while True:
        frame_start = time.perf_counter()
        if env.viewer is not None:
            env.gym.clear_lines(env.viewer)

        apply_motion_frame(
            env,
            dof_pos,
            dof_vel,
            root_pos,
            root_quat,
            root_lin_vel,
            root_ang_vel,
            frame_idx,
            env_origin,
            custom_args.root,
            z_offset,
        )
        refresh_visual_frame(env, custom_args.no_sim_step)
        if custom_args.draw_feet:
            draw_motion_feet(env, feet_pos_world, frame_idx, env_origin, z_offset)

        if frame_idx % (100 * stride) == start_frame % max(100 * stride, 1):
            print(
                f"frame={frame_idx:5d} root_z={env.root_states[0, 2].item():.3f} "
                f"first_dof={env.dof_pos[0, 0].item():+.3f}"
            )

        env.render(sync_frame_time=False)
        frame_idx += stride
        if frame_idx >= end_frame:
            if not custom_args.loop:
                break
            frame_idx = start_frame

        target_dt = dt * stride / max(custom_args.speed, 1e-6)
        elapsed = time.perf_counter() - frame_start
        if elapsed < target_dt:
            time.sleep(target_dt - elapsed)


if __name__ == "__main__":
    main()
