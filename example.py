# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from mazeexplorer import MazeExplorer

env_train = MazeExplorer(number_maps=10, keys=9, size=(10, 10), random_spawn=True, random_textures=True, seed=42)
env_test = MazeExplorer(number_maps=10, keys=9, size=(10, 10), random_spawn=True, random_textures=True, seed=43)
