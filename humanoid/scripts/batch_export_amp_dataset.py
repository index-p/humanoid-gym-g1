import argparse
import glob
import os

import numpy as np

from humanoid.amp_motion_utils import (
    G1_DEFAULT_DOF_POS,
    G1_DEFAULT_JOINT_NAMES,
    build_amp_observations_from_motion,
    canonicalize_motion_dict,
    finite_difference,
    load_motion_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-convert retargeted G1 motion files into motion_visualization and motion_amp_expert datasets."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing source .pkl files")
    parser.add_argument("--visualization_dir", required=True, help="Output directory for motion_visualization .npz files")
    parser.add_argument("--expert_dir", required=True, help="Output directory for motion_amp_expert .npz files")
    parser.add_argument(
        "--pattern",
        default="**/*.pkl",
        help="Glob pattern relative to input_dir used to match source motion files.",
    )
    parser.add_argument(
        "--use_default_joint_order",
        action="store_true",
        help="Reorder motions to the repository's default G1 joint order.",
    )
    parser.add_argument("--fps", type=float, help="Override frames per second")
    parser.add_argument("--dt", type=float, help="Override frame timestep")
    parser.add_argument(
        "--absolute_dof_pos",
        action="store_true",
        help="Store absolute dof positions in amp_obs instead of centering by the default pose.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip files whose visualization and expert outputs already exist.",
    )
    parser.add_argument("--limit", type=int, help="Optional maximum number of files to process.")
    return parser.parse_args()


def collect_inputs(input_dir, pattern):
    search_pattern = os.path.join(os.path.abspath(input_dir), pattern)
    return sorted(glob.glob(search_pattern, recursive=True))


def save_visualization_motion(motion, save_path):
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez_compressed(
        save_path,
        dof_pos=motion["dof_pos"],
        dof_vel=motion["dof_vel"] if motion["dof_vel"] is not None else np.empty((0,), dtype=np.float32),
        root_pos=motion["root_pos"],
        root_quat=motion["root_quat"],
        feet_pos_world=motion["feet_pos_world"] if motion["feet_pos_world"] is not None else np.empty((0,), dtype=np.float32),
        joint_names=np.asarray(motion["joint_names"], dtype="<U64"),
        fps=np.asarray(motion["fps"], dtype=np.float32),
        dt=np.asarray(motion["dt"], dtype=np.float32),
    )


def save_amp_expert_motion(motion, save_path, absolute_dof_pos=False):
    dof_vel = motion["dof_vel"]
    if dof_vel is None:
        dof_vel = finite_difference(motion["dof_pos"], motion["dt"])

    feet_pos_world = motion["feet_pos_world"]
    if feet_pos_world is None or feet_pos_world.size == 0:
        raise ValueError("Motion does not contain feet/end-effector positions required for AMP export")

    amp_obs = build_amp_observations_from_motion(
        motion["dof_pos"],
        dof_vel,
        feet_pos_world,
        motion["root_pos"],
        motion["root_quat"],
        default_dof_pos=G1_DEFAULT_DOF_POS,
        subtract_default=not absolute_dof_pos,
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.savez_compressed(
        save_path,
        amp_obs=amp_obs,
        dof_pos=motion["dof_pos"],
        dof_vel=dof_vel,
        feet_pos_world=feet_pos_world,
        joint_names=np.asarray(motion["joint_names"], dtype="<U64"),
        fps=np.asarray(motion["fps"], dtype=np.float32),
        dt=np.asarray(motion["dt"], dtype=np.float32),
    )
    return amp_obs.shape[1]


def build_output_paths(input_path, input_dir, visualization_dir, expert_dir):
    rel_path = os.path.relpath(input_path, os.path.abspath(input_dir))
    rel_base, _ = os.path.splitext(rel_path)
    visualization_path = os.path.join(os.path.abspath(visualization_dir), rel_base + ".npz")
    expert_path = os.path.join(os.path.abspath(expert_dir), rel_base + ".npz")
    return visualization_path, expert_path


def main():
    args = parse_args()
    input_paths = collect_inputs(args.input_dir, args.pattern)
    if args.limit is not None:
        input_paths = input_paths[: args.limit]
    if not input_paths:
        raise ValueError(f"No source motion files matched pattern '{args.pattern}' under {args.input_dir}")

    processed = 0
    skipped = 0
    for input_path in input_paths:
        visualization_path, expert_path = build_output_paths(
            input_path,
            args.input_dir,
            args.visualization_dir,
            args.expert_dir,
        )
        if args.skip_existing and os.path.exists(visualization_path) and os.path.exists(expert_path):
            skipped += 1
            print(f"[skip] {input_path}")
            continue

        raw_data = load_motion_file(input_path)
        motion = canonicalize_motion_dict(
            raw_data,
            target_joint_names=G1_DEFAULT_JOINT_NAMES if args.use_default_joint_order else None,
            fps=args.fps,
            dt=args.dt,
        )
        save_visualization_motion(motion, visualization_path)
        amp_obs_dim = save_amp_expert_motion(motion, expert_path, absolute_dof_pos=args.absolute_dof_pos)
        processed += 1
        print(
            f"[ok] {input_path} -> {visualization_path} / {expert_path} "
            f"(frames={motion['dof_pos'].shape[0]} dof={motion['dof_pos'].shape[1]} amp_obs_dim={amp_obs_dim})"
        )

    print(f"Finished batch export: processed={processed} skipped={skipped} total={len(input_paths)}")


if __name__ == "__main__":
    main()
