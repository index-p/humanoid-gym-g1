import argparse
import os

import numpy as np

from humanoid.amp_motion_utils import G1_DEFAULT_JOINT_NAMES, canonicalize_motion_dict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize retargeted motion data into the project's motion_visualization schema."
    )
    parser.add_argument("--input", required=True, help="Path to a source .npz motion file")
    parser.add_argument("--output", required=True, help="Path to the normalized motion_visualization .npz")
    parser.add_argument(
        "--use_default_joint_order",
        action="store_true",
        help="Reorder the input dof arrays to the Unitree G1 joint order used by this repository.",
    )
    parser.add_argument("--fps", type=float, help="Override frames per second")
    parser.add_argument("--dt", type=float, help="Override frame timestep")
    return parser.parse_args()


def main():
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as raw_data:
        motion = canonicalize_motion_dict(
            raw_data,
            target_joint_names=G1_DEFAULT_JOINT_NAMES if args.use_default_joint_order else None,
            fps=args.fps,
            dt=args.dt,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        dof_pos=motion["dof_pos"],
        dof_vel=motion["dof_vel"] if motion["dof_vel"] is not None else np.empty((0,), dtype=np.float32),
        root_pos=motion["root_pos"],
        root_quat=motion["root_quat"],
        hand_pos_world=motion["hand_pos_world"] if motion["hand_pos_world"] is not None else np.empty((0,), dtype=np.float32),
        feet_pos_world=motion["feet_pos_world"] if motion["feet_pos_world"] is not None else np.empty((0,), dtype=np.float32),
        joint_names=np.asarray(motion["joint_names"], dtype="<U64"),
        fps=np.asarray(motion["fps"], dtype=np.float32),
        dt=np.asarray(motion["dt"], dtype=np.float32),
    )
    print(f"Saved normalized visualization motion to {args.output}")
    print(
        f"frames={motion['dof_pos'].shape[0]} dof={motion['dof_pos'].shape[1]} "
        f"fps={motion['fps']:.3f} hand_features="
        f"{0 if motion['hand_pos_world'] is None else motion['hand_pos_world'].shape[1]} "
        f"feet_features={0 if motion['feet_pos_world'] is None else motion['feet_pos_world'].shape[1]}"
    )


if __name__ == "__main__":
    main()
