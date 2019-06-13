# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

import setuptools
import os

if not os.path.isdir("mazeexplorer/acc") or not os.listdir("mazeexplorer/acc"):
    raise FileNotFoundError("\nacc files not found.\n"
                            "Have you pulled the required submodules?\n"
                            "If not, use the line:\n\n\t"
                            "git submodule update --init --recursive\n")

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mazeexplorer",
    version="1.0.2",
    author="Luke Harries, Sebastian Lee, Jaroslaw Rzepecki, Katya Hofmann, Sam Devlin",
    author_email="sam.devlin@microsoft.com",
    description="Customisable 3D benchmark for assessing generalisation in Reinforcement Learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    package_data={'mazeexplorer': ['content/*', 'acc/*', 'config_template.txt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "vizdoom",
        "gym",
        "omgifol",
        "opencv-python",
        "imageio",
        "numpy",
        "tensorboardX",
        "pytest"
    ]
)
