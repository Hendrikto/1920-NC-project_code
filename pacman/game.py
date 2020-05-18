import enum
from itertools import chain

import numpy as np

from .ghosts import (
    ChasingGhost,
    RandomGhost,
)
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
        for ghost in self.ghosts:
            characters[tuple(ghost.position + 1)] = 'G'
        characters[tuple(self.pacman + 1)] = 'P'

        return ''.join(chain(characters.flat, f'score: {self.score}'))

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
        self.ghosts = (
            ChasingGhost(self, (10, 10)),
            RandomGhost(self, (0, 0)),
        )
        self.pacman = np.array((5, 5), dtype=np.int8)
        self.score = 0
        self.state = Game.State.ACTIVE

    def step(self, direction):
        self.move(self.pacman, direction)

        if tuple(self.pacman) in self.food:
            self.food.remove(tuple(self.pacman))
            self.score += 1

        for ghost in self.ghosts:
            self.move(ghost.position, ghost.choose_direction())
            if all(self.pacman == ghost.position):
                self.state = Game.State.LOST

        if not self.food:
            self.state = Game.State.WON
