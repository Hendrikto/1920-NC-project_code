import enum

import numpy as np


class Game():
    class State(enum.Enum):
        ACTIVE = enum.auto()
        LOST = enum.auto()
        WON = enum.auto()

    def __init__(self):
        self.walls = np.array((
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0),
            (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0),
            (0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0),
            (1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0),
            (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0),
            (0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0),
            (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
        ), dtype=np.bool8)
        self.board_size = np.array(self.walls.shape, dtype=np.int8)

        self.reset()

    def __str__(self):
        shape = self.board_size + (2, 3)  # add size for frame and newlines

        characters = np.chararray(shape, unicode=True)
        characters[1:-1, 1:-2] = ' '
        characters[:, -1] = '\n'

        # frame
        characters[[0, -1], 1:-2] = '─'
        characters[1:-1, [0, -2]] = '│'
        characters[0, 0] = '┌'
        characters[0, -2] = '┐'
        characters[-1, 0] = '└'
        characters[-1, -2] = '┘'

        # objects
        characters[1:-1, 1:-2][self.walls] = '█'
        characters[tuple(self.pacman + 1)] = 'P'

        return ''.join(characters.flat)

    def reset(self):
        self.pacman = np.array((5, 5), dtype=np.int8)