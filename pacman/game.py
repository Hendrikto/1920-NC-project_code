import enum
from itertools import chain

import numpy as np
import pandas as pd

from .ghosts import (
    ChasingGhost,
    Ghost,
    RandomGhost,
)
from .util import ones_at


class Game():
    class State(enum.Enum):
        ACTIVE = enum.auto()
        LOST = enum.auto()
        WON = enum.auto()

    reward_scores = pd.Series({
        'food': 10,
        'powerup': 50,
        'ghost': 200,
    })

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
        characters[1:-1, 1:-2][ones_at(self.board_size, self.powerups, dtype=np.bool8)] = '◆'
        for ghost in self.ghosts:
            if ghost.state is Ghost.State.ALIVE:
                characters[tuple(ghost.position + 1)] = 'G'
        characters[tuple(self.pacman + 1)] = 'P'

        return ''.join(chain(characters.flat, ' | '.join((
            f'score: {self.score}',
            f'empowered: {self.empowered}',
        ))))

    @property
    def array(self):
        return np.stack((
            self.walls.view(np.int8),  # coerce type
            ones_at(self.board_size, (self.pacman,), dtype=np.int8),
            *(
                ones_at(self.board_size, (ghost.position,), dtype=np.int8)
                * (-self.empowered if self.empowered else 1)
                * (0 if ghost.state is Ghost.State.DEAD else 1)
                for ghost in self.ghosts
            ),
            ones_at(self.board_size, self.food, dtype=np.int8),
            ones_at(self.board_size, self.powerups, dtype=np.int8),
        ))

    def consume(self, consumable):
        position = tuple(self.pacman)
        success = position in consumable
        consumable.discard(position)
        return success

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
        self.empowered = 0
        self.powerups = {(0, 10), (10, 0)}
        self.food = set(zip(*(~self.walls).nonzero())) - self.powerups
        self.ghosts = (
            ChasingGhost(self, (10, 10)),
            RandomGhost(self, (0, 0)),
        )
        self.pacman = np.array((5, 5), dtype=np.int8)
        self.score = 0
        self.state = Game.State.ACTIVE

    def step(self, direction):
        rewards = pd.Series(False, index=self.reward_scores.index, dtype=np.bool8)

        self.move(self.pacman, direction)

        if self.consume(self.food):
            rewards.food = True

        if self.consume(self.powerups):
            self.empowered = 12
            rewards.powerup = True
        elif self.empowered > 0:
            self.empowered -= 1

        for ghost in self.ghosts:
            if ghost.state is Ghost.State.DEAD:
                continue

            self.move(ghost.position, ghost.choose_direction())
            if all(self.pacman == ghost.position):
                if self.empowered:
                    ghost.kill()
                    rewards.ghost = True
                else:
                    self.state = Game.State.LOST

        if not self.food and not self.powerups:
            self.state = Game.State.WON

        self.score += sum(rewards * self.reward_scores)

        return rewards
