import numpy as np
import torch
import torch.nn.functional as F


class MemoryBase:
    """Base class which replay memories will extend.

    Attributes:
        capacity    = [int] maximum number of replay memories
        memories    = [list] replay memories stored in a list
        position    = [int] index in memories list for a new replay memory
        counter     = [int] counter to store the number of pushed memories
        n           = [int] number of transitions used for N-step DQN
        transitions = [list] list of at most n transitions for N-step DQN
        discounts   = [np.ndarray] discount factors of future rewards
        priorities  = [torch.Tensor] priorities for PER
        indices     = [torch.Tensor] indices of last sampled replay memories
        b           = [float] exponent of importance sampling weights
        b_increase  = [float] increase to b after each episode
        e           = [float] small number to prevent zero priority
        a           = [float] exponent of priorities
    """

    def __init__(self, capacity, n, b, b_increase, e, a):
        """Initializes the Replay Memory base.

        Args:
            capacity   = [int] maximum number of replay memories
            n          = [int] number of transitions used for N-step DQN
            b          = [float] importance sampling weighting exponent
            b_increase = [float] increase to b after each episode
            e          = [float] small number to prevent zero priority
            a          = [float] controls randomness of sampling in range [0, 1]
        """
        # set parameters for DQN
        self.capacity = capacity
        self.memories = [[] for _ in range(self.capacity)]
        self.position = 0
        self.counter = 0

        # set parameters for N-step DQN
        self.n = n
        self.transitions = []
        self.discounts = None

        # set parameters for prioritized experience replay
        self.priorities = None
        self.indices = None
        self.b = b
        self.b_increase = b_increase
        self.e = e
        self.a = a

    def reset(self, state):
        """Resets replay memory at start of each episode.

        Args:
            state = [list] first state of the environment
        """
        # reset transition list and increase b at start of episode
        self.transitions = [state]
        self.b = min(self.b + self.b_increase, 1)

    def push(self, end, action, rewards, state):
        """Stores new transition(s) in replay buffer.

        Aggregates state transitions until self.n transitions have been
        aggregated in the self.transitions list. Then it adds a transition
        list to the memory each time step. If the end of the episode has been
        reached, then it adds self.n transition lists, where the ending
        transition replaces the next latest transition for each list. So if we
        have transitions 5, 6, 7, and 8, where 8 is the transition where the
        episode has ended, then it adds the lists [5, 6, 7, 8], [6, 7, 8, 8],
        [7, 8, 8, 8], and [8, 8, 8, 8]. If the episode has not yet ended, then
        it only adds the [5, 6, 7, 8] list.

        Args:
            end     = [bool] whether the episode has finished
            action  = [int] performed action in range [0, num_actions)
            rewards = [float|np.ndarray] reward(s) after performing action
            state   = [list] current state of the environment
        """
        if len(self.transitions) < (self.n - 1) * 4:
            # add new transition to self.transitions
            self.transitions += [end, action, rewards, state]
        else:
            for i in range(1 + end * (self.n - 1)):
                # add transition with zero rewards if episode has ended
                self.transitions += [end, action, rewards * (not i), state]

                # add latest transition list to memories
                self._add_memory()

                # remove oldest transition from transition list
                self.transitions[:4] = []

    def _add_memory(self):
        """Adds current transition list to replay memories.

        Only the necessary data needed for Q-learning is stored. These are:
        - whether the episode has finished;
        - the action performed in the oldest state;
        - the oldest environment state;
        - the discounted return for each ensemble member;
        - the newest environment state.
        Memories are overridden if the replay buffer is full and
        the priority of new memories is set to the current maximum value.
        """
        # add only necessary data from transition list
        self.memories[self.position] = [
            self.transitions[-4],
            self.transitions[2],
            self.transitions[0],
            np.sum(self.discounts * self.transitions[3::4], axis=0),
            self.transitions[-1]
        ]

        # set priority of new memory to currently maximum priority
        max_priority, _ = torch.max(self.priorities, dim=0)
        self.priorities[self.position] = max_priority

        # bookkeeping
        self.position = (self.position + 1) % self.capacity
        self.counter += 1

    def __len__(self):
        """Returns current number of replay memories."""
        return min(self.counter, self.capacity)


class ReplayMemory(MemoryBase):
    """Implements prioritized N-step experience replay.

    Attributes:
        capacity    = [int] maximum number of replay memories
        memories    = [list] replay memories stored in a list
        position    = [int] index in memories list for a new replay memory
        counter     = [int] counter to store the number of pushed memories
        n           = [int] number of transitions used for N-step DQN
        transitions = [list] list of at most n transitions for N-step DQN
        discounts   = [np.ndarray] discount factors of future rewards
        priorities  = [torch.Tensor] priorities for PER
        indices     = [torch.Tensor] indices of last sampled replay memories
        b           = [float] exponent of importance sampling weights
        b_increase  = [float] increase to b after each episode
        e           = [float] small number to prevent zero priority
        a           = [float] exponent of priorities
    """

    def __init__(self, capacity, gamma, n, b, b_increase, e, a):
        """Initializes memory with a given capacity.

        Args:
            capacity   = [int] maximum number of replay memories
            gamma      = [float] discounting factor for future rewards
            n          = [int] number of transitions used for N-step DQN
            b          = [float] exponent of importance sampling weights
            b_increase = [float] increase to b after each episode
            e          = [float] small number to prevent zero priority
            a          = [float] exponent of priorities
        """
        super(ReplayMemory, self).__init__(capacity, n, b, b_increase, e, a)

        # set parameter for N-step DQN
        self.discounts = gamma**np.arange(n, dtype=np.float32)

        # set parameter for prioritized experience replay
        self.priorities = torch.zeros(self.capacity)
        self.priorities[0] = 1

    def sample(self, batch_size):
        """Take a prioritized sample of the available replay memories.

        Args:
            batch_size = [int] number of replay memories to be sampled

        Returns [[torch.Tensor]*6]:
            ends       = whether the episode has ended of shape (batch_size,)
            actions    = action performed in old state of shape (batch_size,)
            old_states = oldest state of transitions of shape (batch_size, *)
            returns    = discounted return of shape (batch_size,)
            new_states = newest state of transitions of shape (batch_size, *)
            is_weights = importance sampling weight of shape (batch_size,)
        """
        # sample indices given priorities (with replacement)
        self.indices = torch.multinomial(self.priorities, batch_size, True)

        # decompose memories into tensors of shape (batch_size, *)
        memories = [self.memories[i] for i in self.indices]
        memories = map(torch.tensor, zip(*memories))
        ends, actions, old_states, returns, new_states = memories

        # compute normalized information sampling weights
        probs = self.priorities[self.indices] / torch.sum(self.priorities)
        is_weights = (self.capacity * probs)**-self.b
        is_weights = is_weights / torch.max(is_weights)

        return ends, actions, old_states, returns, new_states, is_weights

    def update_priorities(self, magnitudes):
        """Sets new priorities for last sampled replay memories.

        Args:
            magnitudes = [torch.Tensor] loss on each sampled replay memory
        """
        priorities = (magnitudes.detach().cpu() + self.e)**self.a
        self.priorities[self.indices] = priorities


class EnsembleMemory(MemoryBase):
    """Implements prioritized N-step experience replay for Ensemble Q-Learning.

    Attributes:
        capacity    = [int] maximum number of replay memories
        memories    = [list] replay memories stored in a list
        position    = [int] index in memories list for a new replay memory
        counter     = [int] counter to store the number of pushed memories
        agent_idx   = [torch.Tensor] pre-computed index array to index faster
        n           = [int] number of transitions used for N-step DQN
        transitions = [list] list of at most n transitions for N-step DQN
        discounts   = [np.ndarray] discount factors of future rewards
        priorities  = [torch.Tensor] priorities for PER
        indices     = [torch.Tensor] indices of last sampled replay memories
        b           = [float] exponent of importance sampling weights
        b_increase  = [float] increase to b after each episode
        e           = [float] small number to prevent zero priority
        a           = [float] exponent of priorities
    """

    def __init__(self, num_agents, capacity, gamma, n, b, b_increase, e, a):
        """Initializes replay memory with a given capacity.

        Args:
            num_agents = [int] number of Q-learning agents in the ensemble
            capacity   = [int] maximum number of replay memories
            gamma      = [float] discounting factor for future rewards
            n          = [int] number of transitions used for N-step DQN
            b          = [float] exponent of importance sampling weights
            b_increase = [float] increase to b after each episode
            e          = [float] small number to prevent zero priority
            a          = [float] exponent of priorities
        """
        super(EnsembleMemory, self).__init__(capacity, n, b, b_increase, e, a)

        # set parameter for Ensemble Q-Learning
        self.agent_idx = torch.arange(num_agents)

        # set parameter for N-step DQN
        self.discounts = gamma**np.arange(n, dtype=np.float32).reshape(n, 1)

        # set parameters for prioritized experience replay
        self.priorities = torch.zeros(self.capacity, num_agents)
        self.priorities[0] = 1

    def sample(self, batch_size):
        """Take a prioritized sample of the available replay memories.

        Args:
            batch_size = [int] number of replay memories to be sampled

        Returns [[torch.Tensor]*6]:
            ends       = whether episode ended of shape (batch_size, agents)
            actions    = action done in old state of shape (batch_size, agents)
            old_states = oldest state of shape (batch_size, agents, *)
            returns    = discounted return of shape (batch_size, agents)
            new_states = newest state of shape (batch_size, agents, *)
            is_weights = importance sampling w of shape (batch_size, agents)
        """
        # sample indices given priorities (with replacement)
        self.indices = torch.multinomial(self.priorities.T, batch_size, True).T

        # decompose memories into tensors of shape (batch_size, agents, *)
        memories = [[self.memories[i] for i in idx] for idx in self.indices]
        memories = map(torch.tensor, zip(*[zip(*m) for m in memories]))
        ends, actions, old_states, returns, new_states = memories

        # select discounted returns belonging to each agent
        returns = returns[:, self.agent_idx, self.agent_idx]

        # compute normalized information sampling weights
        priorities = self.priorities[self.indices, self.agent_idx]
        probs = priorities / torch.sum(self.priorities, dim=0)
        is_weights = (self.capacity * probs)**-self.b
        is_weights = is_weights / torch.max(is_weights, dim=0)[0]

        return ends, actions, old_states, returns, new_states, is_weights

    def update_priorities(self, magnitudes):
        """Sets new priorities for last sampled replay memories.

        Args:
            magnitudes = [torch.Tensor] loss on each sampled replay memory
        """
        priorities = (magnitudes.detach().cpu() + self.e)**self.a
        self.priorities[self.indices, self.agent_idx] = priorities


class MetaMemory:
    """Houses replay memories for Ensemble Q-Learning and MLP combiner function.

    Attributes:
        ensemble_memory = [EnsembleMemory] replay memory for Ensemble Q-Learning
        combiner_memory = [ReplayMemory] replay memory for MLP combiner function
        policy_net      = [nn.Module] model to get last q_distr of episode
        end             = [bool] whether the episode has finished
        action          = [int] performed action in range [0, num_actions)
        reward          = [float] mean of rewards received after doing action
    """

    def __init__(self, num_agents, capacity, gamma, n, b, b_increase, e, a):
        """Initializes replay memories with a given capacity.

        Args:
            num_agents = [int] number of Q-learning agents in the ensemble
            capacity   = [int] maximum number of replay memories
            gamma      = [float] discounting factor for future rewards
            n          = [int] number of transitions used for N-step DQN
            b          = [float] exponent of importance sampling weights
            b_increase = [float] increase to b after each episode
            e          = [float] small number to prevent zero priority
            a          = [float] exponent of priorities
        """
        # initialize replay memories
        self.ensemble_memory = EnsembleMemory(
            num_agents,
            capacity,
            gamma, n,
            b, b_increase,
            e, a
        )
        self.combiner_memory = ReplayMemory(
            capacity,
            gamma, n,
            b, b_increase,
            e, a
        )

        # set parameters for replay memory for MLP combiner function
        self.policy_net = None
        self.end = False
        self.action = 0
        self.reward = 0.0

    def reset(self, state):
        """Resets replay memories at start of each episode.

        Args:
            state = [np.ndarray] first state of the environment
        """
        self.ensemble_memory.reset(state)
        self.combiner_memory.reset(None)

    def step(self, q_distr):
        """Pushes new transition to replay memory for MLP combiner function.

        At the start of an episode, q_distr replaces the None distribution
        after setting it in reset(). At other times, a transition is pushed
        with self.end, self.action, and self.reward.

        Args:
            q_distr = [torch.Tensor] Q-value distribution of each agent
                They are normalized with shape (agents, num_actions, atoms).
        """
        # turn Q-value distributions into NumPy ndarray
        q_distr = q_distr.cpu().numpy()

        if self.combiner_memory.transitions[0] is None:
            # replace None q_distr at start of episode
            self.combiner_memory.transitions = [q_distr]
        else:
            # add new transition to replay memory given saved data
            transition = self.end, self.action, self.reward, q_distr
            self.combiner_memory.push(*transition)

    def push(self, end, action, rewards, state):
        """Pushes new transition to replay memory for Ensemble Q-Learning.

        The end, action, and rewards arguments are saved to push a transition
        to the replay memory for the MLP combiner function after calling
        step(). At the end of an episode, the Q-value distribution is computed
        for the last environment state and added to the replay memory of the
        MLP combiner function.

        Args:
            end     = [bool] whether the episode has finished
            action  = [int] performed action in range [0, num_actions)
            rewards = [np.ndarray] rewards after performing action
            state   = [np.ndarray] current state of the environment
        """
        self.ensemble_memory.push(end, action, rewards, state)

        self.end = end
        self.action = action
        self.reward = np.mean(rewards)

        if end:
            # apply Q-learning neural network to get Q-value distributions
            with torch.no_grad():
                state = torch.tensor(state)
                q_distr = F.softmax(self.policy_net(state), dim=-1)

            # add last Q-value distributions to replay memory of combiner
            self.step(q_distr)
