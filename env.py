import gym
import numpy as np

from pacman.direction import Direction
from pacman.game import Game
from pacman.levels import tutorial_powerup


class PacMan:
    def __init__(self, num_frames, radius):
        self.game = Game(tutorial_powerup, radius)
        self.direction_dict = dict(enumerate(Direction))

        self.num_channels, height, width = self.game.array.shape
        self.state_shape = (self.num_channels * num_frames, height, width)
        self.num_frames = num_frames
        self.num_actions = len(Direction)
        self.frames = []
        self.radius = radius

    @property
    def score(self):
        return self.game.score

    @property
    def won(self):
        return self.game.state is Game.State.WON

    def frame(self):
        y, x = self.game.pacman

        frame = self.game.array
        frame = frame[:, y - self.radius:y + self.radius + 1]
        frame = frame[:, :, x - self.radius:x + self.radius + 1]
        frame = frame.astype(np.float32).tolist()

        return frame

    def render(self):
        print(self.game)

    def reset(self):
        self.game.reset(self.radius)

        self.frames = self.frame()
        for _ in range(self.num_frames - 1):
            self.game.step(self.direction_dict[0])
            self.frames += self.frame()

        return self.frames

    def reward(self, rewards):
        return (
            10.0 if rewards.food else -2.25
            + 2.5 * (self.game.state is Game.State.ACTIVE)
            + 22.5 * rewards.powerup
            + 75.0 * rewards.ghost
            - 65.0 * (self.game.state is Game.State.LOST)
        )

    def step(self, action):
        direction = self.direction_dict[action]

        rewards = self.game.step(direction)

        end = self.game.state in (Game.State.WON, Game.State.LOST)
        self.frames[:self.num_channels] = []
        self.frames += self.frame()
        rewards = self.reward(rewards)

        return end, self.frames, rewards


class EnsemblePacMan(PacMan):
    def reward(self, rewards):
        return np.array((
            40 if rewards.food else -9,
            -250 if self.game.state is Game.State.LOST else 10,
            300 * rewards.ghost,
            90 * rewards.powerup,
        ), dtype=np.float32)


class CartPole:
    def __init__(self):
        self.game = gym.make("CartPole-v0")

        self.state_shape = self.game.observation_space.shape
        self.num_actions = self.game.action_space.n
        self.steps = 0

    def render(self):
        pass

    def reset(self):
        state = self.game.reset()
        self.steps = 0

        self.game.render()

        return state.astype(np.float32)

    @property
    def won(self):
        return self.steps == 200

    @property
    def score(self):
        return self.steps

    @staticmethod
    def reward(end):
        reward = 1 if not end else -1

        return np.array(reward, dtype=np.float32)

    def step(self, action):
        state, _, end, _ = self.game.step(action)
        self.steps += 1

        self.game.render()

        return end, state.astype(np.float32), self.reward(end)


class EnsembleCartPole(CartPole):
    def __init__(self, num_agents):
        super(EnsembleCartPole, self).__init__()

        self.num_agents = num_agents

    def reward(self, end):
        reward = 1 if not end else -1

        return np.full(self.num_agents, reward, dtype=np.float32)


def environment(num_agents, num_frames, radius, cartpole):
    if cartpole:
        if num_agents == 1:
            return CartPole()

        return EnsembleCartPole(num_agents)

    if num_agents == 1:
        return PacMan(num_frames, radius)

    return EnsemblePacMan(num_frames, radius)
