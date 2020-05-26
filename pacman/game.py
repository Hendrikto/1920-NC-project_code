import enum

import numpy as np

from .util import ones_at


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
        characters[1:-1, 1:-2][ones_at(self.board_size, self.food, dtype=np.bool8)] = '•'
        characters[tuple(self.pacman + 1)] = 'P'

        return ''.join(characters.flat)

    def move(self, position, direction):
        new_position = position + direction.value
        if not self.position_blocked(new_position):
            position[:] = new_position

    def position_blocked(self, position):
        position = np.asarray(position)
        return (
            # out of bounds
            any((position < 0) | (position >= self.board_size))
            # blocked by wall
            or self.walls[tuple(position)]
        )

    def reset(self):
        self.food = set(zip(*(~self.walls).nonzero()))
        self.pacman = np.array((5, 5), dtype=np.int8)
        self.state = Game.State.ACTIVE

    def step(self, direction):
        self.move(self.pacman, direction)

        if tuple(self.pacman) in self.food:
            self.food.remove(tuple(self.pacman))

        if not self.food:
            self.state = Game.State.WON
