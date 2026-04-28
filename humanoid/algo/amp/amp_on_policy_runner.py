import os
import statistics
import time
from collections import deque
from datetime import datetime

import torch
import wandb
from torch.utils.tensorboard import SummaryWriter

from humanoid import LEGGED_GYM_ROOT_DIR
from humanoid.algo.ppo.actor_critic import ActorCritic

from .amp_loader import AMPMotionLoader
from .amp_ppo import AMPPPO


class AMPOnPolicyRunner:
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.amp_cfg = train_cfg.get("amp", {})
        self.all_cfg = train_cfg
        self.wandb_run_name = (
            datetime.now().strftime("%b%d_%H-%M-%S")
            + "_"
            + train_cfg["runner"]["experiment_name"]
            + "_"
            + train_cfg["runner"]["run_name"]
        )
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs
        else:
            num_critic_obs = self.env.num_obs

        actor_critic_class = eval(self.cfg["policy_class_name"])
        actor_critic = actor_critic_class(
            self.env.num_obs, num_critic_obs, self.env.num_actions, **self.policy_cfg
        ).to(self.device)
        self.alg = AMPPPO(
            actor_critic,
            amp_obs_dim=self.env.num_amp_obs,
            device=self.device,
            **self.alg_cfg,
            **self.amp_cfg,
        )
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
        )
        self.alg.set_motion_loader(self._build_motion_loader())

        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0

        _, _ = self.env.reset()

    def _build_motion_loader(self):
        motion_files = self.amp_cfg.get("motion_files", [])
        resolved_files = [
            motion_file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
            for motion_file in motion_files
        ]
        if not resolved_files:
            print("AMPOnPolicyRunner: no motion files configured, discriminator training will be skipped.")
            return None
        loader = AMPMotionLoader(
            resolved_files, amp_obs_dim=self.env.num_amp_obs, device=self.device
        )
        if len(loader) == 0:
            print("AMPOnPolicyRunner: motion files resolved but no AMP transitions were loaded.")
        return loader

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if self.log_dir is not None and self.writer is None:
            wandb.init(
                project="XBot",
                sync_tensorboard=True,
                name=self.wandb_run_name,
                config=self.all_cfg,
            )
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        amp_obs = self.env.get_amp_observations()
        obs = obs.to(self.device)
        critic_obs = critic_obs.to(self.device)
        amp_obs = amp_obs.to(self.device)
        self.alg.actor_critic.train()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    next_amp_obs = self.env.get_amp_observations()
                    reset_env_ids = self.env.reset_env_ids
                    if len(reset_env_ids) > 0:
                        terminal_amp_obs = self.env.get_terminal_amp_observations(reset_env_ids)
                        next_amp_obs = next_amp_obs.clone()
                        next_amp_obs[reset_env_ids] = terminal_amp_obs

                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs = obs.to(self.device)
                    critic_obs = critic_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    next_amp_obs = next_amp_obs.to(self.device)

                    self.alg.process_env_step(rewards, dones, infos, amp_obs, next_amp_obs)
                    amp_obs = next_amp_obs

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop
                self.alg.compute_returns(critic_obs)

            metrics = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals(), metrics)
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def log(self, locs, metrics, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]

        ep_string = ""
        if locs["ep_infos"]:
            for key in locs["ep_infos"][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs["ep_infos"]:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar("Episode/" + key, value, locs["it"])
                ep_string += f"{f'Mean episode {key}:':>{pad}} {value:.4f}\n"

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs["collection_time"] + locs["learn_time"]))

        self.writer.add_scalar("Loss/value_function", metrics["value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", metrics["surrogate_loss"], locs["it"])
        self.writer.add_scalar("Loss/discriminator", metrics["discriminator_loss"], locs["it"])
        self.writer.add_scalar("Train/amp_reward", metrics["amp_reward"], locs["it"])
        self.writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        self.writer.add_scalar("Perf/total_fps", fps, locs["it"])
        self.writer.add_scalar("Perf/collection time", locs["collection_time"], locs["it"])
        self.writer.add_scalar("Perf/learning_time", locs["learn_time"], locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])

        header = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "
        log_string = (
            f"{'#' * width}\n"
            f"{header.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
            f"{'Value function loss:':>{pad}} {metrics['value_loss']:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {metrics['surrogate_loss']:.4f}\n"
            f"{'Discriminator loss:':>{pad}} {metrics['discriminator_loss']:.4f}\n"
            f"{'Mean AMP reward:':>{pad}} {metrics['amp_reward']:.4f}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
        )
        if len(locs["rewbuffer"]) > 0:
            log_string += (
                f"{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"
                f"{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"
            )
        log_string += ep_string
        log_string += (
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"
        )
        print(log_string)

    def save(self, path, infos=None):
        torch.save(
            {
                "model_state_dict": self.alg.actor_critic.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
                "discriminator_state_dict": self.alg.discriminator.state_dict(),
                "discriminator_optimizer_state_dict": self.alg.discriminator_optimizer.state_dict(),
                "iter": self.current_learning_iteration,
                "infos": infos,
            },
            path,
        )

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            if "discriminator_optimizer_state_dict" in loaded_dict:
                self.alg.discriminator_optimizer.load_state_dict(
                    loaded_dict["discriminator_optimizer_state_dict"]
                )
        if "discriminator_state_dict" in loaded_dict:
            self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
