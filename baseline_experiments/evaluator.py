# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import os

import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack
from tensorboardX import SummaryWriter

from mazeexplorer import MazeExplorer


def load_stable_baselines_env(cfg_path, vector_length, mp, n_stack, number_maps, action_frame_repeat,
                              scaled_resolution):
    env_fn = lambda: MazeExplorer.load_vizdoom_env(cfg_path, number_maps, action_frame_repeat, scaled_resolution)

    if mp:
        env = SubprocVecEnv([env_fn for _ in range(vector_length)])
    else:
        env = DummyVecEnv([env_fn for _ in range(vector_length)])

    if n_stack > 0:
        env = VecFrameStack(env, n_stack=n_stack)

    return env


class Evaluator:
    def __init__(self, mazes_path, tensorboard_dir, vector_length, mp, n_stack, action_frame_repeat=4,
                 scaled_resolution=(42, 42)):

        self.tensorboard_dir = tensorboard_dir

        mazes_folders = [(x, os.path.join(mazes_path, x)) for x in os.listdir(mazes_path)]
        get_cfg = lambda x: os.path.join(x, [cfg for cfg in sorted(os.listdir(x)) if cfg.endswith('.cfg')][0])
        self.eval_cfgs = [(x[0], get_cfg(x[1])) for x in mazes_folders]

        if not len(self.eval_cfgs):
            raise FileNotFoundError("No eval cfgs found")

        number_maps = 1  # number of maps inside each eval map path

        self.eval_envs = [(name, load_stable_baselines_env(cfg_path, vector_length, mp, n_stack, number_maps,
                                                           action_frame_repeat, scaled_resolution))
                          for name, cfg_path in self.eval_cfgs]

        self.vector_length = vector_length
        self.mp = mp
        self.n_stack = n_stack

        self.eval_summary_writer = SummaryWriter(tensorboard_dir)

    def evaluate(self, model, n_training_steps, desired_episode_count, save=False):
        eval_means = []
        eval_scores = []

        for eval_name, eval_env in self.eval_envs:
            eval_episode_rewards = []
            total_rewards = [0] * self.vector_length

            episode_count = 0

            obs = eval_env.reset()

            while episode_count < desired_episode_count:
                action, _states = model.predict(obs)
                obs, rewards, dones, info = eval_env.step(action)
                total_rewards += rewards

                for idx, done in enumerate(dones):
                    if done:
                        eval_episode_rewards.append(total_rewards[idx])
                        total_rewards[idx] = 0
                        episode_count += 1

            mean = sum(eval_episode_rewards) / len(eval_episode_rewards)
            self.eval_summary_writer.add_scalar('eval/' + eval_name + '_reward', mean, n_training_steps)
            eval_means.append(mean)
            eval_scores.append(eval_episode_rewards)

        self.eval_summary_writer.add_scalar('eval/' + 'mean_reward', sum(eval_means) / len(eval_means),
                                            n_training_steps)

        if save:
            np.save(os.path.join(self.tensorboard_dir, "eval_" + str(n_training_steps)), eval_scores)
