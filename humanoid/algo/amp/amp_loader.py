import glob
import os

import numpy as np
import torch


class AMPMotionLoader:
    def __init__(self, motion_files, amp_obs_dim=None, device="cpu"):
        self.device = device
        self.amp_obs_dim = amp_obs_dim
        self.motion_files = self._resolve_files(motion_files)
        self.amp_obs = torch.empty(0, amp_obs_dim or 0, device=device)
        self.next_amp_obs = torch.empty(0, amp_obs_dim or 0, device=device)
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
        return resolved

    def _load(self):
        amp_sequences = []
        for motion_file in self.motion_files:
            with np.load(motion_file, allow_pickle=True) as data:
                amp_sequence = self._sequence_to_amp_obs(data)
                if amp_sequence.shape[0] < 2:
                    continue
                amp_sequences.append(amp_sequence)

        if not amp_sequences:
            self.amp_obs = torch.empty(0, self.amp_obs_dim or 0, device=self.device)
            self.next_amp_obs = torch.empty(0, self.amp_obs_dim or 0, device=self.device)
            return

        current = np.concatenate([seq[:-1] for seq in amp_sequences], axis=0)
        nxt = np.concatenate([seq[1:] for seq in amp_sequences], axis=0)
        self.amp_obs = torch.tensor(current, dtype=torch.float32, device=self.device)
        self.next_amp_obs = torch.tensor(nxt, dtype=torch.float32, device=self.device)

    def _sequence_to_amp_obs(self, data):
        if "amp_obs" in data:
            amp_obs = np.asarray(data["amp_obs"], dtype=np.float32)
        elif "amp_observations" in data:
            amp_obs = np.asarray(data["amp_observations"], dtype=np.float32)
        else:
            dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)
            dof_vel = np.asarray(data["dof_vel"], dtype=np.float32)
            foot_key = None
            for candidate in ("feet_pos", "end_effector_pos", "ee_pos"):
                if candidate in data:
                    foot_key = candidate
                    break
            if foot_key is None:
                raise KeyError(
                    "AMP motion files must contain 'amp_obs' or the trio "
                    "'dof_pos', 'dof_vel', and one of 'feet_pos'/'end_effector_pos'"
                )
            feet_pos = np.asarray(data[foot_key], dtype=np.float32).reshape(dof_pos.shape[0], -1)
            amp_obs = np.concatenate((dof_pos, dof_vel, feet_pos), axis=-1)

        if self.amp_obs_dim is not None and amp_obs.shape[-1] != self.amp_obs_dim:
            raise ValueError(
                f"AMP motion feature dim mismatch for motion data: expected {self.amp_obs_dim}, got {amp_obs.shape[-1]}"
            )
        return amp_obs

    def sample(self, batch_size):
        if len(self) == 0:
            raise ValueError("No AMP motion files were loaded")
        indices = torch.randint(0, len(self), (batch_size,), device=self.device)
        return self.amp_obs[indices], self.next_amp_obs[indices]

    def __len__(self):
        return self.amp_obs.shape[0]
