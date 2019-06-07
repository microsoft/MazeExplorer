import os

from mazeexplorer.compile_acs import compile_acs

dir_path = os.path.dirname(os.path.realpath(__file__))


def test_compile_acs(tmpdir):
    compile_acs(tmpdir.strpath)
    assert os.path.isfile(os.path.join(tmpdir, "outputs", "maze.o"))
    assert os.path.getsize(os.path.join(tmpdir, "outputs", "maze.o")) > 0
    assert os.path.isdir(os.path.join(tmpdir, "outputs", "sources"))
    assert os.path.isdir(os.path.join(tmpdir, "outputs", "images"))
