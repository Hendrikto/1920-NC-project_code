import numpy as np

from pacman.direction import Direction
from pacman.game import Game


class PacMan:
    def __init__(self, num_frames):
        self.game = Game()
        self.direction_dict = dict(enumerate(Direction))

        channels, height, width = self.game.array.shape
        self.state_shape = (channels * num_frames, height, width)
        self.num_frames = num_frames
        self.num_channels = self.game.array.shape[0]
        self.num_actions = len(Direction)
        self.frames = []

    @property
    def score(self):
        return self.game.score

    @property
    def won(self):
        return self.game.state is Game.State.WON

    def reset(self):
        self.game.reset()

        self.frames = self.game.array.astype(np.float32).tolist()
        for _ in range(self.num_frames - 1):
            self.game.step(self.direction_dict[0])
            self.frames += self.game.array.astype(np.float32).tolist()

        return self.frames

    def reward(self, rewards):
        reward = 10.0 if rewards.food else 0.0
        reward += rewards.powerup * 50.0 + rewards.ghost * 200.0
        reward += (self.game.state is Game.State.LOST) * -250.0

        return reward

    def step(self, action):
        direction = self.direction_dict[action]

        rewards = self.game.step(direction)

        end = self.game.state in (Game.State.WON, Game.State.LOST)
        self.frames[:self.num_channels] = []
        self.frames += self.game.array.astype(np.float32).tolist()

        return end, self.frames, self.reward(rewards)


class EnsemblePacMan(PacMan):
    def reward(self, rewards):
        ensemble_rewards = [0]*4
        ensemble_rewards[0] = rewards.food * 10.0
        ensemble_rewards[1] += rewards.ghost * 200.0 - (self.game.state is Game.State.LOST) * 200.0
        ensemble_rewards[1] += (self.game.state is Game.State.LOST) * -200.0
        ensemble_rewards[2] += rewards.ghost * 200.0
        ensemble_rewards[2] += (self.game.state is Game.State.LOST) * -200.0
        ensemble_rewards[3] = rewards.powerup * 50.0

        return np.array(ensemble_rewards, dtype=np.float32)
