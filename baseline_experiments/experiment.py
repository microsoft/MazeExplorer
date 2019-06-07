# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import argparse
import datetime
import os
import time

from evaluator import Evaluator
from stable_baselines import PPO2, A2C
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from mazeexplorer import MazeExplorer

parser = argparse.ArgumentParser(description='Stable Baseline MazeExplorer Experiment Specification')

parser.add_argument('--algorithm', type=str, default="ppo",
                    help='name of algorithm to be run')
parser.add_argument('--number_maps', type=int,
                    help='number of maps to train on', default=1)
parser.add_argument('--random_spawn', type=int,
                    help='whether or not to have agent respawn randomly every new episode', default=0)
parser.add_argument('--random_textures', type=int,
                    help='whether or not floor, wall, ceilling textures should be randomly sampled', default=0)
parser.add_argument('--experiment_name', type=str,
                    help='name of the experiment', default=None)
parser.add_argument('--lstm', type=int,
                    help='whether to use an lstm with the algorithm specified', default=1)
parser.add_argument('--random_keys', type=int,
                    help='whether or not to fix keys within a particular generated map', default=0)
parser.add_argument('--keys', type=int,
                    help='number of keys to place in each map', default=9)
parser.add_argument('--x', type=int,
                    help='x dimension of map', default=10)
parser.add_argument('--y', type=int,
                    help='y dimension of map', default=10)
parser.add_argument('--cpu', type=int,
                    help='number of cpus', default=20)
parser.add_argument('--steps', type=int,
                    help='number of steps to train agent', default=10000000)
parser.add_argument('--eval_occurrence', type=int,
                    help='number of steps training until running eval', default=200000)
parser.add_argument('--eval_episodes', type=int,
                    help='number of steps to train agent', default=100)
parser.add_argument('--eval', type=int,
                    help='whether to use the eval loop', default=1)
parser.add_argument('--n_stack', type=int,
                    help='number of frames to stack (use 0 for no frame stack)', default=4)
parser.add_argument('--clip', type=int,
                    help='whether or not to clip environment reward', default=1)
parser.add_argument('--mp', type=int,
                    help='whether or not to use multiprocessing', default=0)
parser.add_argument('--env_seed', type=int,
                    help='the seed to use for map generation and spawns', default=None)
parser.add_argument('--alg_seed', type=int,
                    help='the seed to use for learning initialisation', default=None)
parser.add_argument('--complexity', type=float,
                    help='complexity for maze generation', default=.7)
parser.add_argument('--density', type=float,
                    help='density for maze generation', default=.7)
parser.add_argument('--gpu_id', type=float,
                    help='specify which gpu to use', default=0)
parser.add_argument('--episode_timeout', type=int,
                    help='maximum epsiode length', default=2100)

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def stable_baseline_training(algorithm, steps, number_maps, random_spawn, random_textures, lstm,
                             random_keys, keys, dimensions, num_cpus, n_stack, clip, complexity, density,
                             mp, eval_occurrence, eval_episodes, eval, experiment_name, env_seed, alg_seed,
                             episode_timeout):
    """
    Runs OpenAI stable baselines implementation of specified algorithm on specified environment with specified training configurations.
    Note: For scenarios not using MazeExplorer but using the vizdoom wrapper, the .T transpose on the array being fed into the process image method needs to be removed.
    Note: Ensure relevant maps are in specified paths under mazes folder for simpler and manual scenarios.

    :param algorithm: which algorithm to run (currently support for PPO and A2C)
    :param steps: number of steps to run training
    :param number_maps: number of maps to generate and train on
    :param random_spawn: whether or not to randomise the spawn position of the agent
    :param random_textures: whether or not to randomise textures in generated maps
    :param lstm: whether or not to add an lstm to the network
    :param random_keys: whether to randmise key placement upon each new training episode in a given map
    :param keys: number of keys to place in generated maps
    :param dimensions: x, y dimensions of maps to be generated
    :param num_cpus: number of environments in which to train
    :param n_stack: number of frames to stack to feed as a state to the agent
    :param clip: whether or not to clip rewards from the environment
    :param complexity: float between 0 and 1 describing the complexity of the generated mazes
    :param density: float between 0 and 1 describing the density of the generated mazes
    :param mp: whether or not to use multiprocessing for workers
    :param eval_occurrence: parameter specifying period of running evaluation 
    :param eval_episodes: number of times to perform episode rollout during evaluation 
    :param eval: whether or not to use evaluation during training
    :param experiment_name: name of experiment for use in logging and file saving
    :param env_seed: seed to be used for environment generation
    :param alg_seed: seed to be used for stable-baseline algorithms
    :param episode_timeout: number of steps after which to terminate episode
    """

    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')

    # generate a file in log directory containing training configuration information.
    experiment_name = experiment_name + "/" if experiment_name else ""
    OUTPUT_PATH = os.path.join(DIR_PATH, 'results', experiment_name, timestamp)
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    with open(os.path.join(OUTPUT_PATH, 'params.txt'), 'w+') as f:
        f.write(str({'algorithm': algorithm, 'number_maps': number_maps, 'random_spawn': random_spawn,
                     'random_textures': random_textures, 'lstm': lstm, 'random_keys': random_keys,
                     'keys': keys, 'dimensions': dimensions, 'num_cpus': num_cpus,
                     'clip': clip, 'mp': mp, 'n_stack': n_stack, 'env_seed': env_seed, 'alg_seed': alg_seed,
                     'experiment_name': experiment_name,
                     "eval_occurrence": eval_occurrence, "eval_episodes": eval_episodes,
                     "episode_timeout": episode_timeout}))

    if clip:
        clip_range = (-1, 1)
    else:
        clip_range = False

    mazeexplorer_env = MazeExplorer(number_maps=number_maps, random_spawn=random_spawn, random_textures=random_textures,
                                    random_key_positions=random_keys, keys=keys, size=dimensions, clip=clip_range,
                                    seed=env_seed,
                                    complexity=complexity, density=density)

    if mp:
        env = SubprocVecEnv([mazeexplorer_env.create_env() for _ in range(num_cpus)])
    else:
        env = DummyVecEnv([mazeexplorer_env.create_env() for _ in range(num_cpus)])  # vectorise env

    if n_stack > 0:
        env = VecFrameStack(env, n_stack=n_stack)

    if algorithm == 'ppo':
        algo = PPO2
    elif algorithm == 'a2c':
        algo = A2C
    else:
        raise NotImplementedError("Only supports PPO and A2C")

    if lstm:
        model = algo(CnnLstmPolicy, env, verbose=1, tensorboard_log=OUTPUT_PATH)
    else:
        model = algo(CnnPolicy, env, verbose=1, tensorboard_log=OUTPUT_PATH)

    if eval:
        evaluator = Evaluator(os.path.join(DIR_PATH, "eval_maps"), OUTPUT_PATH, num_cpus, mp, n_stack)

        steps_taken = 0

        print("Training started...")

        while steps_taken < steps:
            print("Training...")
            model.learn(total_timesteps=min(eval_occurrence, (steps - steps_taken)), reset_num_timesteps=False,
                        seed=alg_seed)

            steps_taken += eval_occurrence

            print("Evaluating...")

            evaluator.evaluate(model, steps_taken, eval_episodes, save=True)  # do 100 rollouts and save scores

        print("Training completed.")

    else:
        model.learn(total_timesteps=steps, seed=alg_seed)


if __name__ == "__main__":
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    stable_baseline_training(algorithm=args.algorithm, steps=args.steps,
                             number_maps=args.number_maps, random_spawn=bool(args.random_spawn),
                             random_textures=bool(args.random_textures), lstm=bool(args.lstm),
                             random_keys=bool(args.random_keys),
                             keys=args.keys, dimensions=(args.x, args.y), num_cpus=args.cpu, n_stack=args.n_stack,
                             clip=bool(args.clip), mp=bool(args.mp), eval_occurrence=args.eval_occurrence,
                             eval_episodes=args.eval_episodes, eval=bool(args.eval),
                             experiment_name=args.experiment_name,
                             env_seed=args.env_seed, alg_seed=args.alg_seed, complexity=args.complexity,
                             density=args.density,
                             episode_timeout=args.episode_timeout)
