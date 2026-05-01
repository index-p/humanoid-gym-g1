# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

import os
from datetime import datetime

import cv2
import numpy as np
from isaacgym import gymapi
from isaacgym.torch_utils import *
from tqdm import tqdm
import torch
from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.algo.ppo.actor_critic import ActorCritic
from humanoid.envs import *
from humanoid.utils import (
    class_to_dict,
    export_policy_as_jit,
    get_args,
    get_load_path,
    Logger,
    task_registry,
)

POLICY_CLASS_MAP = {
    "ActorCritic": ActorCritic,
}


def load_inference_policy(env, train_cfg, args):
    if args.experiment_name is not None:
        train_cfg.runner.experiment_name = args.experiment_name
    if args.load_run is not None:
        train_cfg.runner.load_run = args.load_run
    if args.checkpoint is not None:
        train_cfg.runner.checkpoint = args.checkpoint

    log_root = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    resume_path = get_load_path(
        log_root,
        load_run=train_cfg.runner.load_run,
        checkpoint=train_cfg.runner.checkpoint,
    )
    print(f"Loading model from: {resume_path}")

    if env.num_privileged_obs is not None:
        num_critic_obs = env.num_privileged_obs
    else:
        num_critic_obs = env.num_obs

    policy_class_name = train_cfg.runner.policy_class_name
    if policy_class_name not in POLICY_CLASS_MAP:
        raise ValueError(f"Unsupported policy class for inference: {policy_class_name}")
    actor_critic_class = POLICY_CLASS_MAP[policy_class_name]
    policy_cfg = class_to_dict(train_cfg.policy)
    actor_critic = actor_critic_class(
        env.num_obs,
        num_critic_obs,
        env.num_actions,
        **policy_cfg,
    ).to(env.device)

    loaded_dict = torch.load(resume_path, map_location=env.device)
    actor_critic.load_state_dict(loaded_dict["model_state_dict"])
    actor_critic.eval()
    return actor_critic.act_inference, actor_critic


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    # Playback runs a single env and uses a safer contact-pair budget than the original play.py.
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    if hasattr(env_cfg.sim, "physx") and hasattr(env_cfg.sim.physx, "max_gpu_contact_pairs"):
        env_cfg.sim.physx.max_gpu_contact_pairs = max(
            env_cfg.sim.physx.max_gpu_contact_pairs, 2**20
        )
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.joint_angle_noise = 0.0
    env_cfg.noise.curriculum = False
    env_cfg.noise.noise_level = 0.5

    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    if not args.headless:
        env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)

    obs = env.get_observations()
    policy, actor_critic = load_inference_policy(env, train_cfg, args)

    if EXPORT_POLICY:
        export_dir = os.path.join(
            LEGGED_GYM_ROOT_DIR,
            "logs",
            train_cfg.runner.experiment_name,
            "exported",
            "policies",
        )
        export_policy_as_jit(actor_critic, export_dir)
        print("Exported policy as jit script to:", export_dir)

    output_dir = os.path.join(LEGGED_GYM_ROOT_DIR, "videos", train_cfg.runner.experiment_name)
    output_stem = datetime.now().strftime("%b%d_%H-%M-%S") + args.run_name
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, output_stem + "_states.png")

    logger = Logger(env.dt)
    robot_index = 0
    joint_index = 1
    stop_state_log = 1200

    video = None
    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        camera_handle = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(-0.3, 0.2, 1), np.deg2rad(135)
        )
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            camera_handle,
            env.envs[0],
            body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_path = os.path.join(output_dir, output_stem + ".mp4")
        video = cv2.VideoWriter(video_path, fourcc, 50.0, (1920, 1080))

    for _ in tqdm(range(stop_state_log)):
        actions = policy(obs.detach())

        if FIX_COMMAND:
            env.commands[:, 0] = 0.0
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0

        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER and video is not None:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], camera_handle, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        logger.log_states(
            {
                "dof_pos_target": actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                "dof_pos": env.dof_pos[robot_index, joint_index].item(),
                "dof_vel": env.dof_vel[robot_index, joint_index].item(),
                "dof_torque": env.torques[robot_index, joint_index].item(),
                "command_x": env.commands[robot_index, 0].item(),
                "command_y": env.commands[robot_index, 1].item(),
                "command_yaw": env.commands[robot_index, 2].item(),
                "base_vel_x": env.base_lin_vel[robot_index, 0].item(),
                "base_vel_y": env.base_lin_vel[robot_index, 1].item(),
                "base_vel_z": env.base_lin_vel[robot_index, 2].item(),
                "base_vel_yaw": env.base_ang_vel[robot_index, 2].item(),
                "contact_forces_z": env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
            }
        )

        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes > 0:
                logger.log_rewards(infos["episode"], num_episodes)

    logger.print_rewards()
    logger.plot_states(save_path=plot_path)

    if video is not None:
        video.release()


if __name__ == "__main__":
    EXPORT_POLICY = True
    RENDER = True
    FIX_COMMAND = True
    args = get_args()
    play(args)
