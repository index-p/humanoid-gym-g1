# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaacgym.torch_utils import normalize, quat_rotate_inverse

from humanoid.amp_motion_utils import (
    G1_LEFT_FOOT_LINK_CANDIDATES,
    G1_LEFT_HAND_LINK_CANDIDATES,
    G1_RIGHT_FOOT_LINK_CANDIDATES,
    G1_RIGHT_HAND_LINK_CANDIDATES,
    G1_TIENKUNG_LEFT_ARM_JOINT_NAMES,
    G1_TIENKUNG_LEFT_LEG_JOINT_NAMES,
    G1_TIENKUNG_RIGHT_ARM_JOINT_NAMES,
    G1_TIENKUNG_RIGHT_LEG_JOINT_NAMES,
)
from humanoid.envs.custom.g1_env import G1FreeEnv


class G1AMPFreeEnv(G1FreeEnv):
    """G1 PPO task with TienKung-style AMP observations."""

    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        self.command_obs_dim = 5
        self._terminal_amp_obs = torch.empty(0)
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        self._init_amp_indices()
        self.num_amp_obs = self.get_amp_observations().shape[-1]

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros(self.cfg.env.num_single_obs, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales

        cursor = 0
        noise_vec[cursor: cursor + self.command_obs_dim] = 0.0
        cursor += self.command_obs_dim
        noise_vec[cursor: cursor + self.num_actions] = (
            noise_scales.dof_pos * self.obs_scales.dof_pos
        )
        cursor += self.num_actions
        noise_vec[cursor: cursor + self.num_actions] = (
            noise_scales.dof_vel * self.obs_scales.dof_vel
        )
        cursor += self.num_actions
        noise_vec[cursor: cursor + self.num_actions] = 0.0
        cursor += self.num_actions
        noise_vec[cursor: cursor + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
        cursor += 3
        noise_vec[cursor: cursor + 3] = noise_scales.quat * self.obs_scales.quat
        return noise_vec

    def _build_amp_joint_index_tensor(self, joint_names):
        return torch.tensor(
            [self.dof_names.index(joint_name) for joint_name in joint_names],
            dtype=torch.long,
            device=self.device,
        )

    def _find_amp_body_index(self, body_name_candidates):
        for body_name in body_name_candidates:
            body_index = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], body_name
            )
            if body_index != -1:
                return int(body_index)
        raise ValueError(
            f"Unable to find AMP body handle for any of {body_name_candidates} in the G1 asset."
        )

    def _init_amp_indices(self):
        self.right_arm_amp_indices = self._build_amp_joint_index_tensor(
            G1_TIENKUNG_RIGHT_ARM_JOINT_NAMES
        )
        self.left_arm_amp_indices = self._build_amp_joint_index_tensor(
            G1_TIENKUNG_LEFT_ARM_JOINT_NAMES
        )
        self.right_leg_amp_indices = self._build_amp_joint_index_tensor(
            G1_TIENKUNG_RIGHT_LEG_JOINT_NAMES
        )
        self.left_leg_amp_indices = self._build_amp_joint_index_tensor(
            G1_TIENKUNG_LEFT_LEG_JOINT_NAMES
        )
        self.left_hand_amp_index = self._find_amp_body_index(
            G1_LEFT_HAND_LINK_CANDIDATES
        )
        self.right_hand_amp_index = self._find_amp_body_index(
            G1_RIGHT_HAND_LINK_CANDIDATES
        )
        self.left_foot_amp_index = self._find_amp_body_index(
            G1_LEFT_FOOT_LINK_CANDIDATES
        )
        self.right_foot_amp_index = self._find_amp_body_index(
            G1_RIGHT_FOOT_LINK_CANDIDATES
        )
        self.hand_amp_indices = torch.tensor(
            [self.left_hand_amp_index, self.right_hand_amp_index],
            dtype=torch.long,
            device=self.device,
        )
        self.foot_amp_indices = torch.tensor(
            [self.left_foot_amp_index, self.right_foot_amp_index],
            dtype=torch.long,
            device=self.device,
        )

    def _cache_terminal_amp_observations(self, env_ids):
        if len(env_ids) == 0:
            self._terminal_amp_obs = torch.empty(0, self.num_amp_obs, device=self.device)
            return
        self._terminal_amp_obs = self._build_amp_observations(env_ids)

    def _build_amp_observations(self, env_ids=None):
        if env_ids is None:
            dof_pos = self.dof_pos
            dof_vel = self.dof_vel
            root_pos = self.root_states[:, :3]
            base_quat = self.base_quat
            rigid_state = self.rigid_state
        else:
            dof_pos = self.dof_pos[env_ids]
            dof_vel = self.dof_vel[env_ids]
            root_pos = self.root_states[env_ids, :3]
            base_quat = self.base_quat[env_ids]
            rigid_state = self.rigid_state[env_ids]

        yaw_quat = base_quat.clone()
        yaw_quat[:, :2] = 0.0
        yaw_quat = normalize(yaw_quat)

        hand_world = rigid_state[:, self.hand_amp_indices, :3] - root_pos.unsqueeze(1)
        hand_quat = yaw_quat.repeat_interleave(self.hand_amp_indices.shape[0], dim=0)
        hand_local = quat_rotate_inverse(
            hand_quat, hand_world.reshape(-1, 3)
        ).reshape(dof_pos.shape[0], -1)

        foot_world = rigid_state[:, self.foot_amp_indices, :3] - root_pos.unsqueeze(1)
        foot_quat = yaw_quat.repeat_interleave(self.foot_amp_indices.shape[0], dim=0)
        foot_local = quat_rotate_inverse(
            foot_quat, foot_world.reshape(-1, 3)
        ).reshape(dof_pos.shape[0], -1)

        return torch.cat(
            (
                dof_pos[:, self.right_arm_amp_indices],
                dof_pos[:, self.left_arm_amp_indices],
                dof_pos[:, self.right_leg_amp_indices],
                dof_pos[:, self.left_leg_amp_indices],
                dof_vel[:, self.right_arm_amp_indices],
                dof_vel[:, self.left_arm_amp_indices],
                dof_vel[:, self.right_leg_amp_indices],
                dof_vel[:, self.left_leg_amp_indices],
                hand_local,
                foot_local,
            ),
            dim=-1,
        )

    def get_amp_observations(self):
        return self._build_amp_observations()

    def get_terminal_amp_observations(self, env_ids):
        if len(env_ids) == 0:
            return torch.empty(0, self.num_amp_obs, device=self.device)
        if self._terminal_amp_obs.numel() == 0:
            return self._build_amp_observations(env_ids)
        return self._terminal_amp_obs
