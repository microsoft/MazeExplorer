# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

# This script uses acc (from https://github.com/rheit/acc) to compile the acs scripts.
import os
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))


def compile_acs(mazes_path):
    os.makedirs(os.path.join(mazes_path, "outputs", "sources"))
    os.makedirs(os.path.join(mazes_path, "outputs", "images"))

    acc_path = os.path.join(dir_path, "acc/acc")

    if not os.path.isfile(acc_path):
        print("Compiling ACC as File not does exist: ", acc_path, "")
        subprocess.call(["make", "-C", os.path.join(dir_path, "acc")])

    maze_acs_path = os.path.join(dir_path, "content", "maze.acs")
    output_file_path = os.path.join(mazes_path, "outputs", "maze.o")
    subprocess.call([acc_path, "-i", "./acc", maze_acs_path, output_file_path])
