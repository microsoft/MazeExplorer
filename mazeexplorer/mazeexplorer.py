# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import datetime
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

import cv2

from .maze import generate_mazes
from .script_manipulator import write_config, write_acs
from .vizdoom_gym import VizDoom
from .wad import generate_wads
from .compile_acs import compile_acs

dir_path = os.path.dirname(os.path.realpath(__file__))


class MazeExplorer(VizDoom):
    def __init__(self, unique_maps=False, number_maps=1, keys=9, size=(10, 10), random_spawn=False, random_textures=False,
                 random_key_positions=False, seed=None, clip=(-1, 1),
                 floor_texture="CEIL5_2", ceilling_texture="CEIL5_1", wall_texture="STONE2",
                 action_frame_repeat=4, actions="MOVE_FORWARD TURN_LEFT TURN_RIGHT", scaled_resolution=(42, 42),
                 episode_timeout=1500, complexity=.7, density=.7, data_augmentation=False, mazes_path=None):
        """
        Gym environment where the goal is to collect a preset number of keys within a procedurally generated maze.

        MazeExplorer is a customisable 3D benchmark for assessing generalisation in Reinforcement Learning.

        :params unique_maps: if set, every map will only be seen once. cfg files will be recreated after all its maps have been seen.
        :param number_maps: number of maps which are contained within the cfg file. If unique maps is set, this acts like a cache of maps
        :param keys: number of keys which need to be collected for each episode
        :param size: the size of generated mazes in the format (width, height)
        :param random_spawn: whether to randomise the spawn each time the environment is reset
        :param random_textures: whether to randomise the textures for each map each time the environment is reset
        :param random_key_positions: whether to randomise the position of the keys each time the environment is reset
        :param seed: seed for random, used to determine the other that the doom maps should be shown.
        :param clip: how much the reward returned on each step should be clipped to
        :param floor_texture: the texture to use for the floor, options are in mazeexplorer/content/doom_textures.acs
        Only used when random_textures=False
        :param ceilling_texture: the texture to use for the ceiling, options are in mazeexplorer/content/doom_textures.acs
        Only used when random_textures=False
        :param wall_texture: the texture to use for the walls, options are in mazeexplorer/content/doom_textures.acs
        Only used when random_textures=False
        :param action_frame_repeat: how many game tics should an action be active
        :param actions: the actions which can be performed by the agent
        :param scaled_resolution: resolution (height, width) of the observation to be returned with each step
        :param episode_timeout: the number of ticks in the environment before it time's out
        :param complexity: float between 0 and 1 describing the complexity of the generated mazes
        :param density: float between 0 and 1 describing the density of the generated mazes
        :param data_augmentation: bool to determine whether or not to use data augmentation
            (adding randomly colored, randomly sized boxes to observation)
        :type mazes_path: path to where to save the mazes
        """
        self.unique_maps = unique_maps
        self.number_maps = number_maps
        self.keys = keys
        self.size = size
        self.random_spawn = random_spawn
        self.random_textures = random_textures
        self.random_key_positions = random_key_positions
        self.seed = seed
        self.clip = clip
        self.actions = actions
        self.mazes = None
        self.action_frame_repeat = action_frame_repeat
        self.scaled_resolution = scaled_resolution
        self.episode_timeout = episode_timeout
        self.complexity = complexity
        self.density = density
        self.data_augmentation = data_augmentation

        # The mazeexplorer textures to use if random textures is set to False
        self.wall_texture = wall_texture
        self.floor_texture = floor_texture
        self.ceilling_texture = ceilling_texture

        self.mazes_path = mazes_path if mazes_path is not None else tempfile.mkdtemp()
        # create new maps and corresponding config
        shutil.rmtree(self.mazes_path, ignore_errors=True)
        os.mkdir(self.mazes_path)

        self.cfg_path = self.generate_mazes()

        # start map with -1 since it will always be reseted one time.
        self.current_map = -1
        super().__init__(self.cfg_path, number_maps=self.number_maps, scaled_resolution=self.scaled_resolution,
                         action_frame_repeat=self.action_frame_repeat, seed=seed,
                         data_augmentation=self.data_augmentation)

    def generate_mazes(self):
        """
        Generate the maze cfgs and wads and place them in self.mazes_path

        :return: path to the maze_cfg
        """
        # edit base acs template to reflect user specification
        write_acs(self.keys, self.random_spawn, self.random_textures, self.random_key_positions, map_size=self.size,
                  number_maps=self.number_maps, floor_texture=self.floor_texture,
                  ceilling_texture=self.ceilling_texture,
                  wall_texture=self.wall_texture, seed=self.seed)

        compile_acs(self.mazes_path)

        # generate .txt maze files
        self.mazes = generate_mazes(self.mazes_path + "/" + str(self.size[0]) + "x" + str(self.size[1]),
                                    self.number_maps, self.size[0],
                                    self.size[1],
                                    seed=self.seed, complexity=self.complexity, density=self.density)

        outputs = os.path.join(self.mazes_path, "outputs/")

        # convert .txt mazes to wads and link acs scripts
        try:
            generate_wads(self.mazes_path + "/" + str(self.size[0]) + "x" + str(self.size[1]),
                          self.mazes_path + "/" + str(self.size[0]) + "x" + str(self.size[1]) + ".wad",
                          outputs + "maze.o")
        except FileNotFoundError as e:
            raise FileNotFoundError(e.strerror + "\n"
                                                 "Have you pulled the required submodules?\n"
                                                 "If not, use the line:\n\n\t"
                                                 "git submodule update --init --recursive")
        cfg = write_config(self.mazes_path + "/" + str(self.size[0]) + "x" + str(self.size[1]),
                           self.actions, episode_timeout=self.episode_timeout)

        return cfg

    def reset(self):
        """Resets environment to start a new mission.

        If `unique_maps` is set and and all cached maps have been seen, it wil also generate
        new maps using the ACC script. Otherwise if there is more than one maze 
        it will randomly select a new maze for the list.

        :return: initial observation of the environment as an rgb array in the format (rows, columns, channels) """
        if not self.unique_maps:
            return super().reset()
        else:
            self.current_map += 1
            if self.current_map > self.number_maps:
                print("Generating new maps")

                if self.seed is not None:
                    np.random.seed(self.seed)

                self.seed = np.random.randint(np.iinfo(np.int32).max)

                # create new maps and corresponding config
                shutil.rmtree(self.mazes_path)

                os.mkdir(self.mazes_path)
                self.cfg_path = self.generate_mazes()

                # reset the underlying DoomGame class
                self.env.load_config(self.cfg_path)
                self.env.init()
                self.current_map = 0

            self.doom_map = "map" + str(self.current_map).zfill(2)
            self.env.set_doom_map(self.doom_map)
            self.env.new_episode()
            self._rgb_array = self.env.get_state().screen_buffer
            observation = self._process_image()
            return observation
            


    def save(self, destination_dir):
        """
        Save the maze files to a directory
        :param destination_dir: the path of where to save the maze files
        """
        shutil.copytree(self.mazes_path, destination_dir)

    @staticmethod
    def load_vizdoom_env(mazes_path, number_maps, action_frame_repeat=4, scaled_resolution=(42, 42)):
        """
        Takes the path to a maze cfg or a folder of mazes created by mazeexplorer.save() and returns a vizdoom environment
        using those mazes
        :param mazes_path: path to a .cfg file or a folder containg the cfg file
        :param number_maps: number of maps contained within the wad file
        :param action_frame_repeat: how many game tics should an action be active
        :param scaled_resolution: resolution (height, width) of the observation to be returned with each step
        :return: VizDoom gym env
        """
        if str(mazes_path).endswith(".cfg"):
            return VizDoom(mazes_path, number_maps=number_maps, scaled_resolution=scaled_resolution,
                           action_frame_repeat=action_frame_repeat)
        else:
            cfg_paths = list(Path(mazes_path).glob("*.cfg"))
            if len(cfg_paths) != 1:
                raise ValueError("Invalid number of cfgs within the mazes path: ", len(cfg_paths))
            return VizDoom(cfg_paths[0], number_maps=number_maps, scaled_resolution=scaled_resolution,
                           action_frame_repeat=action_frame_repeat)

    @staticmethod
    def generate_video(images_path, movie_path):
        """
        Generates a video of the agent from the images saved using record_path

        Example:
        ```python
        images_path = "path/to/save_images_dir"
        movie_path = "path/to/movie.ai"

        env = MazeNavigator(record_path=images_path)
        env.reset()
        for _ in range(100):
            env.step_record(env.action_space.sample(), record_path=images_path)
        MazeNavigator.generate_video(images_path, movie_path)
        ```

        :param images_path: path of the folder containg the generated images
        :param movie_path: file path ending with .avi to where the movie should be outputted to.
        """

        if not movie_path.endswith(".avi"):
            raise ValueError("movie_path must end with .avi")

        images = sorted([img for img in os.listdir(images_path) if img.endswith(".png")])

        if not len(images):
            raise FileNotFoundError("Not png images found within the images path")

        frame = cv2.imread(os.path.join(images_path, images[0]))
        height, width, _ = frame.shape

        video = cv2.VideoWriter(movie_path, 0, 30, (width, height))

        for image in images:
            video.write(cv2.imread(os.path.join(images_path, image)))

        cv2.destroyAllWindows()
        video.release()

