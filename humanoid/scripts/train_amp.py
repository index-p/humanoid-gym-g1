# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2021 ETH Zurich, Nikita Rudin
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.

from humanoid.envs import *
from humanoid.utils import get_args, task_registry


def train(args):
    env, _ = task_registry.make_env(name=args.task, args=args)
    runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    if args.task == "XBotL_free":
        args.task = "g1_walk_amp"
    train(args)
