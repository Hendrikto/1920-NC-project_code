from functools import partial

import numpy as np

from .ghosts import (
    ChasingGhost,
    RandomGhost,
)


class Level():
    def __init__(
        self,
        walls,
        pacman,
        food=None,
        ghosts=(),
        powerups=(),
    ):
        self.ghosts = ghosts
        self.pacman = pacman
        self.powerups = powerups
        self.walls = walls

        if food is None:
            self.food = {
                position
                for y in range(len(walls))
                for x in range(len(walls[0]))
                if (position := (y, x)) not in powerups and not walls[y][x]
            }
        else:
            self.food = ()


level1 = Level(
    ghosts=(
        partial(ChasingGhost, position=(10, 10)),
        partial(RandomGhost, position=(0, 0)),
    ),
    pacman=(5, 5),
    powerups={(0, 10), (10, 0)},
    walls=(
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
    ),
)

tutorial_food = Level(
    pacman=(3, 3),
    walls=np.zeros((7, 7), dtype=np.int8),
)

tutorial_ghost = Level(
    ghosts=(
        partial(RandomGhost, position=(0, 0)),
    ),
    pacman=(3, 3),
    walls=np.zeros((7, 7), dtype=np.int8),
)

tutorial_powerup = Level(
    ghosts=(
        partial(RandomGhost, position=(0, 0)),
    ),
    pacman=(3, 3),
    powerups={(6, 6)},
    walls=np.zeros((7, 7), dtype=np.int8),
)
