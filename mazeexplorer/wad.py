# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License

from omg import *


def build_wall(maze, BLOCK_SIZE):
    things = []
    linedefs = []
    vertexes = []
    v_indexes = {}

    max_w = len(maze[0]) - 1
    max_h = len(maze) - 1

    def __is_edge(w, h):
        return w in (0, max_w) or h in (0, max_h)

    def __add_start(w, h):
        x, y = w * BLOCK_SIZE, h * BLOCK_SIZE
        x += int(BLOCK_SIZE / 2)
        y += int(BLOCK_SIZE / 2)
        things.append(ZThing(*[len(things) + 1000, x, y, 0, 0, 9001, 22279]))

    def __add_vertex(w, h):
        if (w, h) in v_indexes:
            return

        x, y = w * BLOCK_SIZE, h * BLOCK_SIZE
        x += int(BLOCK_SIZE / 2)
        y += int(BLOCK_SIZE / 2)
        v_indexes[w, h] = len(vertexes)
        vertexes.append(Vertex(x, y))

    def __add_line(start, end, edge=False):
        assert start in v_indexes
        assert end in v_indexes

        mask = 1
        left = right = 0
        if __is_edge(*start) and __is_edge(*end):
            if not edge:
                return
            else:
                # Changed the right side (one towards outside the map)
                # to be -1 (65535 for Doom)
                right = 65535
                mask = 15

        # Flipped end and start vertices to make lines "point" at right direction (mostly to see if it works)
        line_properties = [v_indexes[end], v_indexes[start], mask
                           ] + [0] * 6 + [left, right]
        line = ZLinedef(*line_properties)
        linedefs.append(line)

    for h, row in enumerate(maze):
        for w, block in enumerate(row.strip()):
            if block == 'X':
                __add_vertex(w, h)
            else:
                pass

    corners = [(0, 0), (max_w, 0), (max_w, max_h), (0, max_h)]
    for v in corners:
        __add_vertex(*v)

    for i in range(len(corners)):
        if i != len(corners) - 1:
            __add_line(corners[i], corners[i + 1], True)
        else:
            __add_line(corners[i], corners[0], True)

    # Now connect the walls
    for h, row in enumerate(maze):

        for w, _ in enumerate(row):
            if (w, h) not in v_indexes:
                __add_start(w, h)
                continue

            if (w + 1, h) in v_indexes:
                __add_line((w, h), (w + 1, h))

            if (w, h + 1) in v_indexes:
                __add_line((w, h), (w, h + 1))

    return things, vertexes, linedefs


def generate_wads(prefix, wad, behavior, BLOCK_SIZE=96, script=None):
    """
    args: 

    prefix:
    wad: 
    behavior: path to compiled lump containing map behavior (default: None)
    script: path to script source lump containing map behavior (optional)
    """

    new_wad = WAD()

    for map_index, file_name in enumerate(
            glob.glob('{}_*.txt'.format(prefix))):
        with open(file_name) as maze_source:
            maze = [line.strip() for line in maze_source.readlines()]
            maze = [line for line in maze if line]

        new_map = MapEditor()
        new_map.Linedef = ZLinedef
        new_map.Thing = ZThing
        new_map.behavior = Lump(from_file=behavior or None)
        new_map.scripts = Lump(from_file=script or None)
        things, vertexes, linedefs = build_wall(maze, BLOCK_SIZE)
        new_map.things = things + [ZThing(0, 0, 0, 0, 0, 1, 7)]
        new_map.vertexes = vertexes
        new_map.linedefs = linedefs
        new_map.sectors = [Sector(0, 128, 'CEIL5_2', 'CEIL5_1', 240, 0, 0)]
        new_map.sidedefs = [
            Sidedef(0, 0, '-', '-', 'STONE2', 0),
            Sidedef(0, 0, '-', '-', '-', 0)
        ]
        new_wad.maps['MAP{:02d}'.format(map_index)] = new_map.to_lumps()

    new_wad.to_file(wad)
