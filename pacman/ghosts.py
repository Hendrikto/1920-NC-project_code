import enum

import numpy as np

from .direction import Direction


class Ghost():
    class State(enum.Enum):
        ALIVE = enum.auto()
        DEAD = enum.auto()

    def __init__(self, game, position):
        self.game = game
        self.position = np.array(position, dtype=np.int8)
        self.state = Ghost.State.ALIVE

    def kill(self):
        self.state = Ghost.State.DEAD


class ChasingGhost(Ghost):
    def choose_direction(self):
        direction = self.game.pacman - self.position
        direction[np.abs(direction).argmin()] = 0
        return Direction(tuple(np.sign(direction)))


class RandomGhost(Ghost):
    def choose_direction(self):
        return np.random.choice(tuple(
            direction for direction in Direction
            if not self.game.position_blocked(self.position + direction.value)
        ))
