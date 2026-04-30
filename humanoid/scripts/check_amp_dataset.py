import argparse
import glob
import os

import numpy as np

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.amp_motion_utils import canonicalize_motion_dict, load_motion_file
from humanoid.envs import *
from humanoid.utils import task_registry


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect AMP visualization and expert datasets configured for a task."
    )
    parser.add_argument("--task", default="g1_walk_amp", help="Registered task name to inspect.")
    parser.add_argument("--max_files", type=int, default=3, help="Maximum files to summarize per dataset type.")
    return parser.parse_args()


def resolve_patterns(patterns):
    resolved_patterns = [
        pattern.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        for pattern in patterns
    ]
    matched = []
    for pattern in resolved_patterns:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            matched.extend(matches)
        elif os.path.isfile(pattern):
            matched.append(pattern)
    return resolved_patterns, sorted(dict.fromkeys(matched))


def summarize_visualization_file(path):
    raw_data = load_motion_file(path)
    motion = canonicalize_motion_dict(raw_data)
    hand_dim = 0 if motion["hand_pos_world"] is None else motion["hand_pos_world"].shape[1]
    feet_dim = 0 if motion["feet_pos_world"] is None else motion["feet_pos_world"].shape[1]
    print(
        f"  visualization: {path}\n"
        f"    frames={motion['dof_pos'].shape[0]} dof={motion['dof_pos'].shape[1]} "
        f"fps={motion['fps']:.3f} hand_features={hand_dim} feet_features={feet_dim}"
    )


def summarize_expert_file(path):
    with np.load(path, allow_pickle=True) as data:
        amp_obs = np.asarray(data["amp_obs"], dtype=np.float32)
        dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
        joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
        print(
            f"  expert: {path}\n"
            f"    frames={amp_obs.shape[0]} amp_obs_dim={amp_obs.shape[1]} "
            f"dof={dof_pos.shape[1]} joints={len(joint_names)}"
        )


def main():
    cli_args = parse_args()
    _, train_cfg = task_registry.get_cfgs(name=cli_args.task)
    amp_cfg = train_cfg.amp

    for label, patterns in (
        ("motion_files_display", amp_cfg.motion_files_display),
        ("motion_files", amp_cfg.motion_files),
    ):
        resolved_patterns, matched_files = resolve_patterns(patterns)
        print(f"{label}: patterns={len(resolved_patterns)} matched_files={len(matched_files)}")
        for pattern in resolved_patterns:
            print(f"  pattern: {pattern}")
        for path in matched_files[: cli_args.max_files]:
            if label == "motion_files_display":
                summarize_visualization_file(path)
            else:
                summarize_expert_file(path)
        if len(matched_files) > cli_args.max_files:
            print(f"  ... and {len(matched_files) - cli_args.max_files} more")


if __name__ == "__main__":
    main()
