import glob
import os

import numpy as np
import torch

from humanoid.amp_motion_utils import build_amp_observations_from_motion


class AMPMotionLoader:
    """Strict AMP expert-data loader modeled after TienKung-style fixed schemas."""

    _REQUIRED_METADATA = ("joint_names", "fps", "dt")
    _AMP_OBS_KEYS = ("amp_obs",)
    _END_EFFECTOR_KEYS = ("feet_pos_world", "end_effector_pos")
    _HAND_KEYS = ("hand_pos_world", "hands_pos_world")

    def __init__(self, motion_files, amp_obs_dim=None, num_preload_transitions=None, device="cpu"):
        self.device = device
        self.amp_obs_dim = amp_obs_dim
        self.num_preload_transitions = (
            None if num_preload_transitions is None else int(num_preload_transitions)
        )
        self.motion_files = self._resolve_files(motion_files)
        self.amp_obs = torch.empty(0, amp_obs_dim or 0, device=device)
        self.next_amp_obs = torch.empty(0, amp_obs_dim or 0, device=device)
        self.motion_info = []
        self.dof_dim = None
        self.total_frames = 0
        self.total_transitions = 0
        if self.motion_files:
            self._load()

    def _resolve_files(self, motion_files):
        resolved = []
        for pattern in motion_files:
            matches = sorted(glob.glob(os.path.expanduser(pattern)))
            if matches:
                resolved.extend(matches)
            elif os.path.isfile(pattern):
                resolved.append(pattern)
        return sorted(dict.fromkeys(resolved))

    def _load(self):
        amp_sequences = []
        for motion_file in self.motion_files:
            with np.load(motion_file, allow_pickle=True) as data:
                motion_record = self._validate_and_extract(data, motion_file)
                if motion_record["amp_obs"].shape[0] < 2:
                    print(f"AMPMotionLoader: skipping {motion_file} because it has fewer than 2 frames.")
                    continue
                amp_sequences.append(motion_record["amp_obs"])
                self.motion_info.append(motion_record["info"])

        if not amp_sequences:
            self.amp_obs = torch.empty(0, self.amp_obs_dim or 0, device=self.device)
            self.next_amp_obs = torch.empty(0, self.amp_obs_dim or 0, device=self.device)
            return

        current = np.concatenate([seq[:-1] for seq in amp_sequences], axis=0)
        nxt = np.concatenate([seq[1:] for seq in amp_sequences], axis=0)
        if self.num_preload_transitions is not None and self.num_preload_transitions > 0:
            preload_count = min(self.num_preload_transitions, current.shape[0])
            indices = np.random.permutation(current.shape[0])[:preload_count]
            current = current[indices]
            nxt = nxt[indices]
        self.amp_obs = torch.tensor(current, dtype=torch.float32, device=self.device)
        self.next_amp_obs = torch.tensor(nxt, dtype=torch.float32, device=self.device)
        self.total_transitions = int(self.amp_obs.shape[0])
        self._print_summary()

    def _validate_and_extract(self, data, motion_file):
        self._require_metadata(data, motion_file)

        joint_names = [str(name) for name in np.asarray(data["joint_names"]).tolist()]
        dof_pos = self._require_matrix(data, motion_file, "dof_pos")
        dof_vel = self._require_matrix(data, motion_file, "dof_vel")
        if dof_pos.shape != dof_vel.shape:
            raise ValueError(
                f"{motion_file}: 'dof_pos' shape {dof_pos.shape} must match 'dof_vel' shape {dof_vel.shape}"
            )
        if len(joint_names) != dof_pos.shape[1]:
            raise ValueError(
                f"{motion_file}: joint_names length {len(joint_names)} does not match dof dimension {dof_pos.shape[1]}"
            )

        feet_pos = self._require_end_effector_positions(data, motion_file, dof_pos.shape[0])
        hand_pos = self._optional_matrix(data, self._HAND_KEYS, dof_pos.shape[0])
        amp_obs = self._extract_amp_obs(
            data,
            motion_file,
            joint_names,
            dof_pos,
            dof_vel,
            feet_pos,
            hand_pos,
        )

        if self.dof_dim is None:
            self.dof_dim = int(dof_pos.shape[1])
        elif self.dof_dim != int(dof_pos.shape[1]):
            raise ValueError(
                f"{motion_file}: dof dimension {dof_pos.shape[1]} does not match previous motions {self.dof_dim}"
            )

        fps = float(np.asarray(data["fps"]).reshape(()))
        dt = float(np.asarray(data["dt"]).reshape(()))
        if fps <= 0.0 or dt <= 0.0:
            raise ValueError(f"{motion_file}: fps ({fps}) and dt ({dt}) must both be positive")

        num_frames = int(amp_obs.shape[0])
        self.total_frames += num_frames
        info = {
            "path": motion_file,
            "num_frames": num_frames,
            "num_transitions": max(0, num_frames - 1),
            "dof_dim": int(dof_pos.shape[1]),
            "amp_obs_dim": int(amp_obs.shape[1]),
            "fps": fps,
            "dt": dt,
        }
        return {"amp_obs": amp_obs, "info": info}

    def _require_metadata(self, data, motion_file):
        missing = [key for key in self._REQUIRED_METADATA if key not in data]
        if missing:
            raise KeyError(
                f"{motion_file}: missing required AMP metadata fields {missing}. "
                "Expected exported motion_amp_expert data."
            )

    def _require_matrix(self, data, motion_file, key):
        if key not in data:
            raise KeyError(f"{motion_file}: missing required field '{key}'")
        value = np.asarray(data[key], dtype=np.float32)
        if value.ndim != 2:
            raise ValueError(f"{motion_file}: field '{key}' must be rank-2, got shape {value.shape}")
        return value

    def _optional_matrix(self, data, keys, num_frames):
        for key in keys:
            if key not in data:
                continue
            value = np.asarray(data[key], dtype=np.float32).reshape(num_frames, -1)
            if value.size == 0:
                return None
            return value
        return None

    def _require_end_effector_positions(self, data, motion_file, num_frames):
        feet_key = None
        for candidate in self._END_EFFECTOR_KEYS:
            if candidate in data:
                feet_key = candidate
                break
        if feet_key is None:
            raise KeyError(
                f"{motion_file}: missing end-effector positions. Expected one of {self._END_EFFECTOR_KEYS}"
            )
        feet_pos = np.asarray(data[feet_key], dtype=np.float32).reshape(num_frames, -1)
        if feet_pos.shape[0] != num_frames:
            raise ValueError(
                f"{motion_file}: end-effector frame count {feet_pos.shape[0]} does not match dof frame count {num_frames}"
            )
        return feet_pos

    def _extract_amp_obs(self, data, motion_file, joint_names, dof_pos, dof_vel, feet_pos, hand_pos):
        amp_obs = None
        for key in self._AMP_OBS_KEYS:
            if key in data:
                amp_obs = np.asarray(data[key], dtype=np.float32)
                break
        if amp_obs is None:
            if (
                hand_pos is not None
                and "root_pos" in data
                and "root_quat" in data
            ):
                amp_obs = build_amp_observations_from_motion(
                    dof_pos,
                    dof_vel,
                    feet_pos,
                    hand_pos,
                    np.asarray(data["root_pos"], dtype=np.float32),
                    np.asarray(data["root_quat"], dtype=np.float32),
                    joint_names=joint_names,
                )
            else:
                amp_obs = np.concatenate((dof_pos, dof_vel, feet_pos), axis=-1)
        if amp_obs.ndim != 2:
            raise ValueError(f"{motion_file}: 'amp_obs' must be rank-2, got shape {amp_obs.shape}")
        if amp_obs.shape[0] != dof_pos.shape[0]:
            raise ValueError(
                f"{motion_file}: amp_obs frame count {amp_obs.shape[0]} does not match dof frame count {dof_pos.shape[0]}"
            )
        if self.amp_obs_dim is not None and amp_obs.shape[1] != self.amp_obs_dim:
            raise ValueError(
                f"{motion_file}: AMP feature dim mismatch, expected {self.amp_obs_dim}, got {amp_obs.shape[1]}"
            )
        return amp_obs.astype(np.float32)

    def _print_summary(self):
        if not self.motion_info:
            return
        frame_counts = [item["num_frames"] for item in self.motion_info]
        amp_dims = sorted({item["amp_obs_dim"] for item in self.motion_info})
        dof_dims = sorted({item["dof_dim"] for item in self.motion_info})
        print(
            "AMPMotionLoader: loaded "
            f"{len(self.motion_info)} motions, total_frames={self.total_frames}, "
            f"total_transitions={self.total_transitions}, "
            f"frame_range=[{min(frame_counts)}, {max(frame_counts)}], "
            f"dof_dims={dof_dims}, amp_obs_dims={amp_dims}"
        )

    def sample(self, batch_size):
        if len(self) == 0:
            raise ValueError("No AMP motion files were loaded")
        indices = torch.randint(0, len(self), (batch_size,), device=self.device)
        return self.amp_obs[indices], self.next_amp_obs[indices]

    def __len__(self):
        return self.amp_obs.shape[0]
