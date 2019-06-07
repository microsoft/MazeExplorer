import os
from pathlib import Path

import numpy as np
from vizdoom import GameVariable

from mazeexplorer import MazeExplorer

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_create_mazes(tmpdir):
    env = MazeExplorer(mazes_path=tmpdir.strpath)

    files = os.listdir(env.mazes_path)

    required_files = ["10x10.cfg", "10x10.wad", "10x10_MAP01.txt", "outputs"]

    assert set(files) == set(required_files)

    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        observation, *_ = env.step(action)
        assert observation.shape == (42, 42, 3)


def test_save_load(tmpdir):
    env = MazeExplorer(mazes_path=tmpdir.mkdir("maze").strpath)

    saved_mazes_destination = os.path.join(tmpdir, "test_mazes")

    env.save(saved_mazes_destination)

    required_files = ["10x10.cfg", "10x10.wad", "10x10_MAP01.txt", "outputs"]

    assert set(required_files) == set(os.listdir(saved_mazes_destination))

    env = MazeExplorer.load_vizdoom_env(saved_mazes_destination, 1)

    for _ in range(10):
        action = env.action_space.sample()
        observation, *_ = env.step(action)
        assert observation.shape == (42, 42, 3)


def test_step_record(tmpdir):
    env = MazeExplorer(random_textures=True, mazes_path=tmpdir.strpath)
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        observation, *_ = env.step_record(action, tmpdir)
        assert observation.shape == (42, 42, 3)
    assert len(list(Path(tmpdir).glob("*.png"))) == 6


def test_generate_video(tmpdir):
    record_path = tmpdir.mkdir("record")

    env = MazeExplorer(mazes_path=tmpdir.strpath)
    env.reset()
    for _ in range(10):
        action = env.action_space.sample()
        env.step_record(action, record_path=record_path)

    video_destination = os.path.join(record_path, "movie.avi")
    MazeExplorer.generate_video(record_path, video_destination)

    assert os.path.isfile(video_destination)
    assert os.path.getsize(video_destination) > 0


def test_generate_with_seed(tmpdir):
    env = MazeExplorer(10, seed=5, mazes_path=tmpdir.mkdir("maze_1").strpath)
    assert len(set(env.mazes)) == 10

    same_env = MazeExplorer(10, seed=5, mazes_path=tmpdir.mkdir("maze_2").strpath)
    assert set(env.mazes) == set(same_env.mazes)

    different_env = MazeExplorer(10, seed=42, mazes_path=tmpdir.mkdir("maze_3").strpath)
    assert len(set(env.mazes) - set(different_env.mazes)) == 10


def test_generate_with_seed_step(tmpdir):
    env = MazeExplorer(10, seed=5, mazes_path=tmpdir.mkdir("maze_1").strpath)
    env.reset()
    for _ in range(5):
        env.step(env.action_space.sample())


def test_fixed_keys_step(tmpdir):
    env = MazeExplorer(random_key_positions=False, mazes_path=tmpdir.strpath)
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, *_ = env.step(action)
        assert observation.shape == (42, 42, 3)


def test_generate_multiple_mazes(tmpdir):
    """
    This function should test whether new episodes sample from the selection of map levels in the wad.
    The assertion could be that every map is used at least once but given the stochasticity, it becomes
    a trade off between how many episodes to sample and how certain one can be that the test will pass.
    For now this test is implemented with the weaker condition that more than one map is sampled.
    This ensures at least that the map level is not being fixed.
    """
    for number_mazes in [1, 5, 10]:
        maps = set()
        env = MazeExplorer(number_maps=number_mazes, mazes_path=tmpdir.strpath)
        for _ in range(5000):
            *_, done, _ = env.step(env.action_space.sample())
            if done:
                env.reset()
                map_id = int(env.env.get_game_variable(GameVariable.USER4))
                maps.add(map_id)
        if number_mazes == 1:
            assert len(maps) == 1
        else:
            assert len(maps) > 1


def test_reward_signal(tmpdir):
    env = MazeExplorer(seed=42, mazes_path=tmpdir.strpath)
    rews = []
    rewards = []
    for _ in range(1000):
        _, reward, done, _ = env.step(env.action_space.sample())
        rewards.append(reward)
        if done:
            env.reset()

    assert sum(rewards) > 1

    return rews, rewards
