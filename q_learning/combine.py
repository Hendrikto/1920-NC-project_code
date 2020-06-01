import torch
import torch.nn.functional as F

from .agent import QAgent


class Combiner:
    """
    Combines Q-value distributions arithmetically or with voting.

    Attributes:
        mode = [str] type of combiner function from {sum, min, max, avg, vote}
        atoms = [torch.Tensor] Q-value positions of all the atoms
    """

    def __init__(
        self,
        mode,
        num_atoms,
        v_min, v_max,
        device,
    ):
        self.mode = mode
        self.atoms = torch.linspace(v_min, v_max, steps=num_atoms).to(device)

    def step(self, q_distr):
        """
        Combine Q-value distributions.

        Args:
            q_distr = [torch.Tensor] Q-value distribution from each agent
                They are normalized with shape (agents, num_actions, atoms).

        Returns [torch.Tensor]:
            Combined Q-values of shape (num_actions,).
        """
        q_values = torch.sum(self.atoms * q_distr, dim=-1)

        combiner_function = getattr(self, self.mode)
        return combiner_function(q_values)

    @staticmethod
    def sum(q_values):
        return q_values.sum(dim=0)

    @staticmethod
    def min(q_values):
        return q_values.min(dim=0)[0]

    @staticmethod
    def max(q_values):
        return q_values.max(dim=0)[0]

    @staticmethod
    def avg(q_values):
        return q_values.mean(dim=0)

    @staticmethod
    def vote(q_values):
        """
        Combine expected Q-values through majority voting.

        Some random uniform noise is added to the votes. If there are actions
        that get the same number of maximum votes, then this will ensure that
        each such action is uniformly selected, instead of always the last one.

        Args:
            q_values = [torch.Tensor] expected Q-values from each agent
                The shape is (agents, num_actions).

        Returns [torch.Tensor]:
            Combined Q-values of shape (num_actions,).
        """
        indices = q_values.argmax(dim=1)
        votes = torch.bincount(indices, minlength=q_values.shape[1])
        votes = votes.float() + torch.rand_like(q_values[0])

        return votes


class MLPCombiner(QAgent):
    """
    Combine Q-value distributions with a multilayer perceptron.

    Attributes:
        policy_net = [MLP] model used for action selection
        target_net = [MLP] model used for retrieving target Q-values
        optimizer = [Optimizer] optimizer used to train policy network
        q_indices = [[torch.Tensor]*2] Q-value index arrays to index faster
        meta_memory = [MetaMemory] agent and combiner replay memories
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
        # initialize Q-agent parameters
        super(MLPCombiner, self).__init__(
            model_factory,
            lr,
            memory.combiner_memory,
            batch_size,
            num_atoms,
            v_min, v_max,
            gamma,
            n,
            train_onset,
            num_update,
            device,
        )

        # set MetaMemory object
        self.meta_memory = memory

    def step(self, q_distribution):
        """
        Combine Q-value distributions with a multilayer perceptron.

        Args:
            q_distribution = [torch.Tensor] Q-value distribution from each agent
                They are normalized with shape (agents, num_actions, atoms).

        Returns [torch.Tensor]:
            Combined Q-values of shape (num_actions,).
        """
        # add q_distribution to replay memory
        self.meta_memory.step(q_distribution)

        # train MLP each step
        self.train()

        # apply Q-Learning MLP to get combined Q-value distributions
        with torch.no_grad():
            q_distribution = F.softmax(self.policy_net(q_distribution), dim=-1)

        # compute the expected Q-value for each action
        q_values = torch.sum(self.atoms * q_distribution, dim=-1)

        return q_values
