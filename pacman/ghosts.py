import numpy as np

from .direction import Direction


class Ghost():
    def __init__(self, game, position):
        self.game = game
        self.position = np.array(position, dtype=np.int8)


class RandomGhost(Ghost):
    def choose_direction(self):
        return np.random.choice(tuple(
            direction for direction in Direction
            if not self.game.position_blocked(self.position + direction.value)
        ))
