import os
from pathlib import Path

from mazeexplorer.vizdoom_gym import VizDoom

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_vizdoom_gym():
    test_mazes = os.path.join(dir_path, "mazes", "test.cfg")
    env = VizDoom(test_mazes, number_maps=1)
    env.reset()
    for _ in range(1000):
        action = env.action_space.sample()
        observation, *_ = env.step(action)
        assert observation.shape == (42, 42, 3)


def test_vizdoom_gym_step_record(tmpdir):
    test_mazes = os.path.join(dir_path, "mazes", "test.cfg")
    env = VizDoom(test_mazes, number_maps=1)
    env.reset()
    for _ in range(3):
        action = env.action_space.sample()
        observation, *_ = env.step_record(action, tmpdir)
        assert observation.shape == (42, 42, 3)
    assert len(list(Path(tmpdir).glob("*.png"))) == 6
