import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .networks import CategoricalLoss


class QAgentBase:
    """
    Q-agent base.

    Attributes:
        policy_net = [nn.Module] model used for action selection
        target_net = [nn.Module] model used for retrieving target Q-values
        optimizer = [Optimizer] optimizer used to train policy network
        q_indices = [[torch.Tensor]*2] Q-value index arrays to index faster
        memory = [MemoryBase] replay memory to sample state transitions
        batch_size = [int] batch size per training step
        atoms = [torch.Tensor] Q-value positions of all the atoms
        criterion = [nn.Module] loss module as weighted cross-entropy
        train_onset = [int] number of replay memories before training starts
        num_update = [int] period to update the target network
        num_steps = [int] number of times policy network has been updated
    """

    def __init__(
        self,
        model_factory,
        lr,
        memory,
        batch_size,
        num_atoms,
        v_min, v_max,
        gamma,
        n,
        train_onset,
        num_update,
        device,
    ):
        """
        Initialize base Q-agent.

        Args:
            model_factory = [fn] function that initializes model
            lr = [float] learning rate of Adam optimizer
            memory = [MemoryBase] replay memory to sample state transitions
            batch_size = [int] batch size per training step
            num_atoms = [int] number of atoms in each Q-value distribution
            v_min = [float] Q-value position of first atom
            v_max = [float] Q-value position of last atom
            gamma = [float] discount factor for future rewards
            n = [int] number of transitions used for N-step DQN
            train_onset = [int] number of replay memories before training starts
            num_update = [int] period to update the target network
            device = [torch.device] device to put the models and data on
        """
        # setup the policy and target neural networks
        self.policy_net = model_factory()
        self.target_net = model_factory()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # setup an optimizer
        self.optimizer = Adam(self.policy_net.parameters(),
                              lr=lr, weight_decay=1e-5)

        # q_indices is initialized by inheriting classes
        self.q_indices = None

        # set parameters for the replay memory
        self.memory = memory
        self.batch_size = batch_size

        # set parameters for categorical DQN
        self.atoms = torch.linspace(v_min, v_max, steps=num_atoms).to(device)
        self.criterion = CategoricalLoss(self.atoms, gamma, n, device)

        # set training parameters
        self.train_onset = train_onset
        self.num_update = num_update
        self.num_steps = 0

    def _q_distributions(self, actions, old_states, new_states):
        """
        Compute predicted and target Q-value distributions with double DQN.

        Args [[torch.Tensor]*3]:
            actions = actions in old_states of shape (batch_size, agents)
            old_states = oldest states of shape (batch_size, agents, *)
            new_states = newest states of shape (batch_size, agents, *)

        Returns [[torch.Tensor]*2]:
            q_pred = predicted Q-value distributions on oldest states
                This output has shape (batch_size, agents, atoms). The actions
                are selected with the parameter actions.
            q_target = target Q-value distributions on newest states
                This output has shape (batch_size, agents, atoms). The actions
                are selected by the policy network given the newest states.
        """
        # get predicted Q distributions of actions performed in oldest states
        q_pred = self.policy_net(old_states)[self.q_indices + (actions,)]

        with torch.no_grad():
            # get actions performed in newest states from policy network
            q_distr = F.softmax(self.policy_net(new_states), dim=-1)
            q_values = torch.sum(self.atoms * q_distr, dim=-1)
            actions = q_values.argmax(dim=-1)

            # get target Q distributions of actions performed in newest states
            q_target = self.target_net(new_states)[self.q_indices + (actions,)]
            q_target = F.softmax(q_target, dim=-1)

        return q_pred, q_target

    def _update_networks(self, loss):
        """
        Update policy and target networks.

        The policy network is updated according to the provided loss and the
        target network is given the same weights as the policy network every
        self.num_update training steps.

        Args:
            loss [torch.Tensor] loss of the policy network in the current step
        """
        # update policy network parameters
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10)
        self.optimizer.step()

        # synchronize policy and target networks periodically
        self.num_steps += 1
        if self.num_steps % self.num_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        """Train on previous transitions."""
        # start training after train_onset memories have been accumulated
        if len(self.memory) < self.train_onset:
            return

        # sample state transitions and importance sampling weights
        batch = self.memory.sample(self.batch_size)
        ends, actions, old_states, returns, new_states, is_weights = batch

        # get predicted and target Q-value distributions from networks
        q_pred, q_target = self._q_distributions(actions, old_states, new_states)

        # compute loss as weighted cross-entropy between Q-value distributions
        feedback = self.criterion(q_pred, q_target, ends, returns, is_weights)
        magnitudes, loss = feedback

        # update the priorities of the sampled replay memories
        self.memory.update_priorities(magnitudes)

        # update policy and target networks given the loss
        self._update_networks(loss)


class QAgent(QAgentBase):
    """
    Simple Q-agent.

    Attributes:
        policy_net = [nn.Module] model used for action selection
        target_net = [nn.Module] model used for retrieving target Q-values
        optimizer = [Optimizer] optimizer used to train policy network
        q_indices = [[torch.Tensor]*2] Q-value index arrays to index faster
        memory = [MemoryBase] replay memory to sample state transitions
        batch_size = [int] batch size per training step
        atoms = [torch.Tensor] Q-value positions of all the atoms
        criterion = [nn.Module] loss module as weighted cross-entropy
        train_onset = [int] number of replay memories before training starts
        num_update = [int] period to update the target network
        num_steps = [int] number of times policy network has been updated
    """

    def __init__(
        self,
        model_factory,
        lr,
        memory,
        batch_size,
        num_atoms,
        v_min, v_max,
        gamma,
        n,
        train_onset,
        num_update,
        device,
    ):
        """Initialize the Q-agent.

        Args:
            model_factory = [fn] sizes of dimensions of an environment state
            lr = [float] learning rate of Adam optimizer
            memory = [MemoryBase] replay memory to sample state transitions
            batch_size = [int] batch size per training step
            num_atoms = [int] number of atoms in each Q-value distribution
            v_min = [float] Q-value position of first atom
            v_max = [float] Q-value position of last atom
            gamma = [float] discount factor for future rewards
            n = [int] number of transitions used for N-step DQN
            train_onset = [int] number of replay memories before training starts
            num_update = [int] period to update the target network
            device = [torch.device] device to put the models and data on
        """
        super(QAgent, self).__init__(
            model_factory,
            lr,
            memory,
            batch_size,
            num_atoms,
            v_min, v_max,
            gamma,
            n,
            train_onset,
            num_update,
            device,
        )

        # set Q-value index arrays beforehand for faster indexing
        self.q_indices = (torch.arange(self.batch_size),)

    def step(self, state, epsilon):
        """
        Select an action for the current state, using the policy network.

        Args:
            state = [list] current state of the environment

        Returns [int]:
            Selected action encoded as number in the range [0, num_actions).
        """
        # apply Q-learning neural network to get Q-value distributions
        with torch.no_grad():
            state = torch.tensor(state)
            q_distr = F.softmax(self.policy_net(state), dim=-1)

        # compute the expected Q-value for each action
        q_values = torch.sum(self.atoms * q_distr, dim=-1)

        # choose an action by greedily picking from Q table
        action = q_values.argmax()
        if np.random.rand() < epsilon:
            return np.random.randint(5, dtype=np.int64)

        return int(action)


class EnsembleQAgent(QAgentBase):
    """
    Ensemble of Q-agents.

    Attributes:
        policy_net = [nn.Module] model used for action selection
        target_net = [nn.Module] model used for retrieving target Q-values
        optimizer = [Optimizer] optimizer used to train policy network
        q_indices = [[torch.Tensor]*2] Q-value index arrays to index faster
        memory = [MemoryBase] replay memory to sample state transitions
        batch_size = [int] batch size per training step
        atoms = [torch.Tensor] Q-value positions of all the atoms
        criterion = [nn.Module] loss module as weighted cross-entropy
        train_onset = [int] number of replay memories before training starts
        num_update = [int] period to update the target network
        num_steps = [int] number of times policy network has been updated
        combiner = [Combiner|MLPCombiner] combines Q-value distributions
    """

    def __init__(
        self,
        num_agents,
        model_factory,
        lr,
        memory,
        batch_size,
        combiner,
        num_atoms,
        v_min, v_max,
        gamma,
        n,
        train_onset,
        num_update,
        device,
    ):
        """
        Initialize the ensemble of Q-agents.

        Args:
            num_agents = [int] number of ensemble members
            model_factory = [fn] sizes of dimensions of an environment state
            lr = [float] learning rate of Adam optimizer
            memory = [MemoryBase] replay memory to sample state transitions
            batch_size = [int] batch size per training step
            combiner = [Combiner|MLPCombiner] combines Q-value distributions
            num_atoms = [int] number of atoms in each Q-value distribution
            v_min = [float] Q-value position of first atom
            v_max = [float] Q-value position of last atom
            gamma = [float] discount factor for future rewards
            n = [int] number of transitions used for N-step DQN
            train_onset = [int] number of replay memories before training starts
            num_update = [int] period to update the target network
            device = [torch.device] device to put the models and data on
        """
        super(EnsembleQAgent, self).__init__(
            model_factory,
            lr,
            memory,
            batch_size,
            num_atoms,
            v_min, v_max,
            gamma,
            n,
            train_onset,
            num_update,
            device,
        )

        # set Q-value index arrays beforehand for faster indexing
        self.q_indices = (
            torch.arange(self.batch_size).view(-1, 1),  # batch indices
            torch.arange(num_agents),  # agent indices
        )

        # set combiner attribute
        self.combiner = combiner

    def step(self, state, epsilon):
        """
        Select an action for the current state, using the policy network.

        The Q-value distributions are combined with self.combiner to one
        Q-value per action. The selected action is the argmax of these values.

        Args:
            state = [np.ndarray] current state of the environment

        Returns [int]:
            Selected action encoded as number in the range [0, num_actions).
        """
        # apply Q-learning neural network to get Q-value distributions

        with torch.no_grad():
            state = torch.tensor(state)
            q_distribution = F.softmax(self.policy_net(state), dim=-1)

        # combine Q-value distributions to one Q-value for each action
        q_values = self.combiner.step(q_distribution)

        # choose an action by greedily picking from Q table
        action = q_values.argmax()
        if np.random.rand() < epsilon:
            return np.random.randint(5, dtype=np.int64)
        return int(action)
