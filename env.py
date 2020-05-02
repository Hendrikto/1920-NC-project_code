import gym
import numpy as np

from pacman.direction import Direction
from pacman.game import Game


class PacMan:

    def __init__(self, num_frames):
        self.game = Game()
        self.direction_dict = dict(zip(range(len(Direction)), Direction))

        self.num_frames = num_frames
        self.num_channels = self.game.array.shape[0]
        self.num_actions = len(Direction)
        self.frames = []

        self.reward_fn = self._reward

    def _reward(self, rewards):
        reward = 10.0 if rewards.food else 0.0
        reward += rewards.powerup * 50.0 + rewards.ghost * 200.0
        reward += (self.game.state is Game.State.LOST) * -250.0

        return reward

    def reset(self):
        self.game.reset()

        self.frames = self.game.array.astype(np.float32).tolist()
        for _ in range(self.num_frames - 1):
            self.game.step(self.direction_dict[0])
            self.frames += self.game.array.astype(np.float32).tolist()

        return self.frames

    def step(self, action):
        direction = self.direction_dict[action]

        rewards = self.game.step(direction)

        end = self.game.state in (Game.State.WON, Game.State.LOST)
        self.frames[self.num_channels:] = []
        self.frames += self.game.array.astype(np.float32).tolist()
        rewards = self.reward_fn(rewards)

        return end, self.frames, rewards


class EnsemblePacMan(PacMan):

    def __init__(self, num_frames):
        super(EnsemblePacMan, self).__init__(num_frames)

        self.reward_fn = self._rewards

    def _rewards(self, rewards):
        ensemble_rewards = [0]*4
        ensemble_rewards[0] = rewards.food * 10.0
        ensemble_rewards[1] += rewards.ghost * 200.0 - (self.game.state is Game.State.LOST) * 200.0
        ensemble_rewards[1] += (self.game.state is Game.State.LOST) * -200.0
        ensemble_rewards[2] += rewards.ghost * 200.0
        ensemble_rewards[2] += (self.game.state is Game.State.LOST) * -200.0
        ensemble_rewards[3] = rewards.powerup * 50.0

        return ensemble_rewards


class CartPole:

    def __init__(self):
        self.game = gym.make('CartPole-v1')

        self.state_shape = self.game.observation_space.shape
        self.num_actions = self.game.action_space.n

    def reset(self):
        state = self.game.reset()
        self.game.render()

        return state.astype(np.float32)

    def step(self, action):
        state, reward, end, _ = self.game.step(action)
        self.game.render()

        state = state.astype(np.float32)
        reward = reward if not end else -reward

        return end, state, reward


class EnsembleCartPole:

    def __init__(self, num_agents):
        self.game = gym.make('CartPole-v1')

        self.num_agents = num_agents
        self.state_shape = self.game.observation_space.shape
        self.num_actions = self.game.action_space.n

    def reset(self):
        state = self.game.reset()
        self.game.render()

        return state.astype(np.float32)

    def step(self, action):
        state, reward, end, _ = self.game.step(action)
        self.game.render()

        state = state.astype(np.float32)
        rewards = np.full(self.num_agents, reward, dtype=np.float32)
        rewards = rewards if not end else -rewards

        return end, state, rewards
