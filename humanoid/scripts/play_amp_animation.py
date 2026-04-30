import argparse
import os

import numpy as np

from humanoid.amp_motion_utils import (
    build_amp_observations_from_motion,
    canonicalize_motion_dict,
    finite_difference,
    load_motion_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight preview tool for motion_visualization files before AMP export."
    )
    parser.add_argument("--input", required=True, help="Path to a motion_visualization .npz file")
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame index to print in detail for quick inspection.",
    )
    parser.add_argument(
        "--save_path",
        help="Optional path to export AMP expert data after previewing the motion.",
    )
    parser.add_argument(
        "--absolute_dof_pos",
        action="store_true",
        help="Deprecated compatibility flag. TienKung-style AMP export always uses absolute joint positions.",
    )
    return parser.parse_args()


def summarize_motion(motion, frame_idx):
    num_frames, num_dof = motion["dof_pos"].shape
    duration = num_frames * motion["dt"]
    frame_idx = max(0, min(frame_idx, num_frames - 1))

    print(f"frames: {num_frames}")
    print(f"dof: {num_dof}")
    print(f"fps: {motion['fps']:.3f}")
    print(f"duration_s: {duration:.3f}")
    print(f"joint_names: {motion['joint_names']}")
    print(f"root_height_range: [{motion['root_pos'][:, 2].min():.4f}, {motion['root_pos'][:, 2].max():.4f}]")

    if motion["feet_pos_world"] is not None and motion["feet_pos_world"].size > 0:
        feet = motion["feet_pos_world"].reshape(num_frames, -1, 3)
        print(f"num_foot_effectors: {feet.shape[1]}")
        print(f"feet_height_range: [{feet[:, :, 2].min():.4f}, {feet[:, :, 2].max():.4f}]")
    else:
        print("num_foot_effectors: 0")
    if motion["hand_pos_world"] is not None and motion["hand_pos_world"].size > 0:
        hands = motion["hand_pos_world"].reshape(num_frames, -1, 3)
        print(f"num_hand_effectors: {hands.shape[1]}")
    else:
        print("num_hand_effectors: 0")

    print(f"preview_frame: {frame_idx}")
    print(f"root_pos[{frame_idx}]: {motion['root_pos'][frame_idx]}")
    print(f"root_quat[{frame_idx}]: {motion['root_quat'][frame_idx]}")
    print(f"dof_pos[{frame_idx}]: {motion['dof_pos'][frame_idx]}")
    if motion["dof_vel"] is not None:
        print(f"dof_vel[{frame_idx}]: {motion['dof_vel'][frame_idx]}")


def export_amp_expert(motion, save_path, absolute_dof_pos=False):
    dof_vel = motion["dof_vel"]
    if dof_vel is None:
        dof_vel = finite_difference(motion["dof_pos"], motion["dt"])

    feet_pos_world = motion["feet_pos_world"]
    if feet_pos_world is None or feet_pos_world.size == 0:
        raise ValueError(
            "Cannot export AMP expert data because the motion does not contain feet/end-effector positions."
        )
    hand_pos_world = motion["hand_pos_world"]
    if hand_pos_world is None or hand_pos_world.size == 0:
        raise ValueError(
            "Cannot export AMP expert data because the motion does not contain hand end-effector positions."
        )

    amp_obs = build_amp_observations_from_motion(
        motion["dof_pos"],
        dof_vel,
        feet_pos_world,
        hand_pos_world,
        motion["root_pos"],
        motion["root_quat"],
        joint_names=motion["joint_names"],
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez_compressed(
        save_path,
        amp_obs=amp_obs,
        dof_pos=motion["dof_pos"],
        dof_vel=dof_vel,
        root_pos=motion["root_pos"],
        root_quat=motion["root_quat"],
        hand_pos_world=hand_pos_world,
        feet_pos_world=feet_pos_world,
        joint_names=np.asarray(motion["joint_names"], dtype="<U64"),
        fps=np.asarray(motion["fps"], dtype=np.float32),
        dt=np.asarray(motion["dt"], dtype=np.float32),
    )
    print(f"Saved AMP expert motion to {save_path}")
    print(f"frames={amp_obs.shape[0]} amp_obs_dim={amp_obs.shape[1]}")


def main():
    args = parse_args()
    raw_data = load_motion_file(args.input)
    motion = canonicalize_motion_dict(raw_data)
    summarize_motion(motion, args.frame)
    if args.save_path:
        export_amp_expert(motion, args.save_path, absolute_dof_pos=args.absolute_dof_pos)


if __name__ == "__main__":
    main()
