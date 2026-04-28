import argparse
import os

import numpy as np

from humanoid.amp_motion_utils import (
    G1_DEFAULT_DOF_POS,
    build_amp_observations_from_motion,
    canonicalize_motion_dict,
    finite_difference,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export AMP expert motion features from normalized motion_visualization files."
    )
    parser.add_argument("--input", required=True, help="Path to a motion_visualization .npz file")
    parser.add_argument("--output", required=True, help="Path to the motion_amp_expert .npz file")
    parser.add_argument(
        "--absolute_dof_pos",
        action="store_true",
        help="Store absolute dof positions in amp_obs instead of centering them by the default pose.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with np.load(args.input, allow_pickle=True) as raw_data:
        motion = canonicalize_motion_dict(raw_data)

    dof_vel = motion["dof_vel"]
    if dof_vel is None:
        dof_vel = finite_difference(motion["dof_pos"], motion["dt"])

    feet_pos_world = motion["feet_pos_world"]
    if feet_pos_world is None or feet_pos_world.size == 0:
        raise ValueError(
            "The visualization motion file does not contain feet/end-effector positions. "
            "Please carry them through during conversion before exporting AMP expert data."
        )

    amp_obs = build_amp_observations_from_motion(
        motion["dof_pos"],
        dof_vel,
        feet_pos_world,
        motion["root_pos"],
        motion["root_quat"],
        default_dof_pos=G1_DEFAULT_DOF_POS,
        subtract_default=not args.absolute_dof_pos,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        amp_obs=amp_obs,
        dof_pos=motion["dof_pos"],
        dof_vel=dof_vel,
        feet_pos_world=feet_pos_world,
        joint_names=np.asarray(motion["joint_names"], dtype="<U64"),
        fps=np.asarray(motion["fps"], dtype=np.float32),
        dt=np.asarray(motion["dt"], dtype=np.float32),
    )
    print(f"Saved AMP expert motion to {args.output}")
    print(f"frames={amp_obs.shape[0]} amp_obs_dim={amp_obs.shape[1]}")


if __name__ == "__main__":
    main()
