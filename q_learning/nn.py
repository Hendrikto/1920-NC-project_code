from functools import reduce
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CategoricalLoss(nn.Module):
    """Implements weighted N-step cross-entropy loss over Q-value distributions.

    Attributes:
        num_atoms = [int] number of atoms in each Q-value distribution
        v_min     = [float] Q-value position of first atom
        delta_z   = [float] Q-value position difference between adjacent atoms
        atoms     = [torch.Tensor] discounted Q-value position of each atom
        device    = [torch.device] device to compute loss on
    """

    def __init__(self, atoms, gamma, n, device):
        """Initializes loss module.

        Args:
            atoms  = [torch.Tensor] Q-value positions of all the atoms
            gamma  = [float] discount factor for future rewards
            n      = [int] number of transitions used for N-step DQN
            device = [torch.device] device to compute loss on
        """
        super(CategoricalLoss, self).__init__()

        # set parameters for categorical DQN
        self.num_atoms = len(atoms)
        self.v_min = atoms[0]
        self.delta_z = (atoms[-1] - self.v_min) / (self.num_atoms - 1)
        self.atoms = gamma**n * atoms

        # set parameter to compute loss on correct device
        self.device = device

    def forward(self, q_pred, q_target, ends, returns, is_weights):
        """Computes magnitudes and loss.

        More specifically, the magnitudes are the N-step cross entropies for
        each Q-value distribution and the loss is a weighted sum over the
        magnitudes weighted by the given importance sampling weights.

        Args [[torch.Tensor]*5]:
            q_pred     = predicted Q-value distributions on oldest states
                They are not normalized with shape (batch_size, *, atoms).
            q_target   = target Q-value distributions on newest states
                They are normalized with shape (batch_size, *, atoms).
            ends       = whether the episode has ended of shape (batch_size, *)
            returns    = discounted returns of shape (batch_size, *)
            is_weights = importance sampling weights of shape (batch_size, *)

        Returns [[torch.tensor]*2]:
            magnitudes = N-step cross-entropies of shape (batch_size, *)
            loss       = weighted N-step cross-entropy loss
        """
        # expand atom dimension and put data on correct device
        ends = ends.unsqueeze(-1).to(self.device)
        returns = returns.unsqueeze(-1).to(self.device)
        is_weights = is_weights.to(self.device)

        # compute Q-value positions of misaligned target distribution atoms
        tau = returns + ends * self.atoms

        # compute upper and lower atom indices of aligned target distribution
        b = (tau - self.v_min) / self.delta_z
        b = torch.clamp(b, min=0, max=self.num_atoms - 1)
        l = b.floor().long()
        u = b.ceil().long()

        # compute index offsets of distributions in flattened q_target
        offset = torch.arange(0, b.numel(), step=self.num_atoms)
        offset = offset.view(ends.shape).to(self.device)

        # compute aligned target Q-value distribution
        m = torch.zeros_like(q_target)
        m.put_(l + offset, q_target * (u - b), accumulate=True)
        m.put_(u + offset, q_target * (b - l), accumulate=True)
        m.put_(l + offset, q_target * (l == u), accumulate=True)

        # loss as weighted N-step cross-entropy over Q-value distributions
        magnitudes = -torch.sum(m * F.log_softmax(q_pred, dim=-1), dim=-1)
        loss = torch.sum(is_weights * magnitudes) / magnitudes.shape[0]

        return magnitudes, loss


class NoisyLinear(nn.Module):
    """Implements a noisy linear layer with factorized Gaussian noise.

    Attributes:
        num_in       = [int] total number of input units
        num_out      = [int] total number of output units
        weight_mu    = [nn.Parameter] learnable weight mean parameters
        weight_sigma = [nn.Parameter] learnable weight stdev parameters
        bias_mu      = [nn.Parameter] learnable bias mean parameters
        bias_sigma   = [nn.Parameter] learnable bias stdev parameters
        weight_idx   = [[torch.Tensor]*2] weight index arrays
            These are used to set inter-agent weights and weight gradients to
            zero to ensure independence between each agent's network.
    """

    def __init__(self, in_features, out_features, num_agents=1):
        """Initializes noisy linear layer.

        Args:
            in_features  = [int] number of input units per agent
            out_features = [int] number of output units per agent
            num_agents   = [int] number of independent sub-layers
        """
        super(NoisyLinear, self).__init__()

        # determine total number of inputs and outputs
        self.num_in = in_features * num_agents
        self.num_out = out_features * num_agents

        # initialize parameters
        self.weight_mu = nn.Parameter(torch.empty(self.num_out, self.num_in))
        self.weight_sigma = nn.Parameter(torch.empty(self.num_out, self.num_in))
        self.bias_mu = nn.Parameter(torch.empty(self.num_out))
        self.bias_sigma = nn.Parameter(torch.empty(self.num_out))
        self.reset_parameters(in_features)

        # set inter-agent weights to zero
        self.weight_idx = self._weight_indices(in_features, out_features)
        with torch.no_grad():
            self.weight_mu[self.weight_idx] = 0
            self.weight_sigma[self.weight_idx] = 0

        # hooks to set inter-agent weight gradients to zero
        self.weight_mu.register_hook(self._hook)
        self.weight_sigma.register_hook(self._hook)

    def reset_parameters(self, in_features):
        """Initialize mean and standard deviation parameters."""
        bound = 1 / math.sqrt(in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)

        val = 0.5 / math.sqrt(in_features)
        nn.init.constant_(self.weight_sigma, val)
        nn.init.constant_(self.bias_sigma, val)

    def _weight_indices(self, in_features, out_features):
        """Computes index arrays of inter-agent weights of a linear layer.

        Args:
            in_features  = [int] number of input units per agent
            out_features = [int] number of output units per agent

        Returns [[torch.Tensor]*2]:
            Row and column inter-agent weight index arrays of shapes
            (num_out, 1) and (num_out, num_in * (agents - 1) / agents).
        """
        row_indices = torch.arange(self.num_out).view(-1, 1)

        col_indices = []
        for i in range(0, self.num_in, in_features):
            col_idx = torch.cat((
                torch.arange(0, i),
                torch.arange(i + in_features, self.num_in)
            ))
            col_indices.append(col_idx)
        col_indices = torch.stack(col_indices)
        col_indices = col_indices.repeat_interleave(out_features, dim=0)

        return row_indices, col_indices

    def _hook(self, grad):
        """Backward hook to set inter-agent weight gradients to zero."""
        grad = grad.clone()
        grad[self.weight_idx] = 0

        return grad

    def forward(self, x):
        """Forward pass of noisy linear layer.

        Args:
            x = [torch.Tensor] input of shape (*, self.num_in)

        Returns [torch.Tensor]:
            Output of shape (*, self.num_out).
        """
        rand = torch.randn(self.num_in + self.num_out).to(x.device)
        rand = torch.sign(rand) * torch.sqrt(abs(rand))
        rand_in, rand_out = rand.split([self.num_in, self.num_out])

        weight_rand = torch.einsum('i,j->ji', rand_in, rand_out)
        weight = self.weight_sigma * weight_rand + self.weight_mu
        bias = self.bias_sigma * rand_out + self.bias_mu

        return F.linear(x, weight, bias)


class LinearModel(nn.Module):
    """Implements network as sequence of noisy linear layers.

    Attributes:
        fc1        = [NoisyLinear] first linear layer
        fc2a       = [NoisyLinear] second linear layer of advantage pathway
        fc2v       = [NoisyLinear] second linear layer of state value pathway
        fc3a       = [NoisyLinear] third linear layer of advantage pathway
        fc3v       = [NoisyLinear] last linear layer of state value pathway
        state_ndim = [int] number of dimensions in an environment state
        num_atoms  = [int] number of atoms in each Q-value distribution
        device     = [torch.device] device to put the model and data on
    """

    def __init__(self,
                 state_shape, num_actions,
                 num_hidden, num_atoms,
                 num_agents,
                 device):
        """Initializes the linear model.

        Args:
            state_shape = [tuple] sizes of dimensions of an environment state
            num_actions = [int] number of possible actions in environment
            num_hidden  = [int] number of hidden units per agent
            num_atoms   = [int] number of atoms in each Q-value distribution
            num_agents  = [int] number of independent sub-models
            device      = [torch.device] device to put the model and data on
        """
        super(LinearModel, self).__init__()

        # initialize noisy linear layers
        in_features = reduce(lambda x, y: x * y, state_shape)
        self.fc1 = NoisyLinear(in_features, num_hidden, num_agents)

        self.fc2a = NoisyLinear(num_hidden, num_hidden, num_agents)
        self.fc2v = NoisyLinear(num_hidden, num_hidden, num_agents)

        self.fc3a = NoisyLinear(num_hidden, num_actions * num_atoms, num_agents)
        self.fc3v = NoisyLinear(num_hidden, num_atoms, num_agents)

        # save number of state dimensions and number of atoms
        self.state_ndim = len(state_shape)
        self.num_atoms = num_atoms

        # put model on correct device
        self.device = device
        self.to(self.device)

    def forward(self, input):
        """Forward pass of the linear model.

        Args:
            input = [torch.Tensor] a single state or a batch of states
                In agent.step(), the input has shape state_shape.
                In agent.train(), the input has shape
                (batch_size,) + state_shape.
        """
        # get input to correct number of dimensions and device
        if input.ndim == self.state_ndim:
            x = input.flatten()
        else:
            x = input.flatten(start_dim=-self.state_ndim)
        x = x.to(self.device)

        # run input through noisy linear layers
        h = F.relu(self.fc1(x))

        ha = F.relu(self.fc2a(h))
        hv = F.relu(self.fc2v(h))

        a = self.fc3a(ha)
        v = self.fc3v(hv)

        # get output to correct number of dimensions
        if input.ndim == self.state_ndim:
            a = a.view(-1, self.num_atoms)
            v = v.view(1, self.num_atoms)
        else:
            a = a.view(input.shape[0], -1, self.num_atoms)
            v = v.view(input.shape[0], 1, self.num_atoms)

        # dueling DQN, mean of advantage is forced to be zero
        return v + a - a.mean(dim=-2, keepdim=True)


class EnsembleLinearModel(LinearModel):
    """Implements network as sequence of noisy linear layers.

    Attributes:
        fc1        = [NoisyLinear] first linear layer
        fc2a       = [NoisyLinear] second linear layer of advantage pathway
        fc2v       = [NoisyLinear] second linear layer of state value pathway
        fc3a       = [NoisyLinear] third linear layer of advantage pathway
        fc3v       = [NoisyLinear] last linear layer of state value pathway
        state_ndim = [int] number of dimensions in an environment state
        num_atoms  = [int] number of atoms in each Q-value distribution
        num_agents = [int] number of independent sub-models
        device     = [torch.device] device to put the model and data on
    """

    def __init__(self,
                 state_shape, num_actions,
                 num_hidden, num_atoms,
                 num_agents,
                 device):
        """Initializes the linear model.

        Args:
            state_shape = [tuple] sizes of dimensions of an environment state
            num_actions = [int] number of possible actions in environment
            num_hidden  = [int] number of hidden units per agent
            num_atoms   = [int] number of atoms in each Q-value distribution
            num_agents  = [int] number of independent sub-models
            device      = [torch.device] device to put the model and data on
        """
        super(EnsembleLinearModel, self).__init__(
            state_shape, num_actions,
            num_hidden, num_atoms,
            num_agents,
            device
        )

        # save number of agents
        self.num_agents = num_agents

    def forward(self, input):
        """Forward pass of the linear model.

        Args:
            input = [torch.Tensor] a single state or a batch of states
                In agent.step(), the input has shape state_shape.
                In agent.train(), the input has shape
                (batch_size, agents) + state_shape.
        """
        # get input to correct number of dimensions and device
        if input.ndim == self.state_ndim:
            x = input.flatten().repeat(self.num_agents)
        else:
            x = input.flatten(start_dim=-self.state_ndim - 1)
        x = x.to(self.device)

        # run input through noisy linear layers
        h = F.relu(self.fc1(x))

        ha = F.relu(self.fc2a(h))
        hv = F.relu(self.fc2v(h))

        a = self.fc3a(ha)
        v = self.fc3v(hv)

        # get output to correct number of dimensions
        if input.ndim == self.state_ndim:
            a = a.view(self.num_agents, -1, self.num_atoms)
            v = v.view(self.num_agents, 1, self.num_atoms)
        else:
            a = a.view(input.shape[0], self.num_agents, -1, self.num_atoms)
            v = v.view(input.shape[0], self.num_agents, 1, self.num_atoms)

        # dueling DQN, mean of advantage is forced to be zero
        return v + a - a.mean(dim=-2, keepdim=True)


class MLP(nn.Module):
    """Implements multilayer perceptron.

    Attributes:
        fc1     = [nn.Module] first linear layer
        fc2     = [nn.Module] second linear layer
        device  = [torch.device] device to put the model on
    """

    def __init__(self,
                 num_agents, num_actions,
                 num_hidden, num_atoms,
                 device):
        """Initializes the MLP.

        Args:
            num_agents  = [int] number of independent sub-models
            num_actions = [int] number of possible actions in environment
            num_hidden  = [int] number of hidden units per agent
            num_atoms   = [int] number of atoms in each Q-value distribution
            device      = [torch.device] device to put the model on
        """
        super(MLP, self).__init__()

        # initialize linear layers
        self.fc1 = NoisyLinear(
            in_features=num_agents * num_actions * num_atoms,
            out_features=num_agents * num_hidden
        )
        self.fc2 = NoisyLinear(
            in_features=num_agents * num_hidden,
            out_features=num_actions * num_atoms
        )

        self.device = device
        self.to(self.device)

    def forward(self, input):
        """Forward pass of the MLP combiner function.

        Args:
            input = [torch.Tensor] input of shape (*, agents, actions, atoms)

        Returns [torch.Tensor]:
            Combined Q-value distributions of shape (*, actions, atoms).
        """
        x = input.flatten(start_dim=-3).to(self.device)

        h = F.relu(self.fc1(x))
        y = self.fc2(h)

        return y.view(input.shape[:-3] + input.shape[-2:])
