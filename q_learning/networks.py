from functools import reduce
import math
from operator import mul

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
        floor = b.floor().long()
        ceil = b.ceil().long()

        # compute index offsets of distributions in flattened q_target
        offset = torch.arange(0, b.numel(), step=self.num_atoms)
        offset = offset.view(ends.shape).to(self.device)

        # compute aligned target Q-value distribution
        m = torch.zeros_like(q_target)
        m.put_(floor + offset, q_target * (ceil - b), accumulate=True)
        m.put_(ceil + offset, q_target * (b - floor), accumulate=True)
        m.put_(floor + offset, q_target * (floor == ceil), accumulate=True)

        # loss as weighted N-step cross-entropy over Q-value distributions
        magnitudes = -torch.sum(m * F.log_softmax(q_pred, dim=-1), dim=-1)
        loss = torch.sum(is_weights * magnitudes) / magnitudes.shape[0]

        return magnitudes, loss


def _weight_indices(self, in_features, out_features):
    """Computes index arrays of inter-agent weights of a linear layer.

    Args:
        self         = [nn.Module] network module to compute index arrays for
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
            torch.arange(i + in_features, self.num_in),
        ))
        col_indices.append(col_idx)
    col_indices = torch.stack(col_indices)
    col_indices = col_indices.repeat_interleave(out_features, dim=0)

    return row_indices, col_indices


class Linear(nn.Module):
    """Implements a standard linear layer.

    Attributes:
        num_in     = [int] total number of input units
        num_out    = [int] total number of output units
        weight_idx = [[torch.Tensor]*2] weight indices to set inter-agent
            weights to zero to ensure independence between each agent's network
        linear     = [nn.Linear] linear layer module
    """

    def __init__(self, in_features, out_features, num_agents=1):
        """Initializes linear layer.

        Args:
            in_features  = [int] number of input units per agent
            out_features = [int] number of output units per agent
            num_agents   = [int] number of independent sub-layers
        """
        super(Linear, self).__init__()

        self.num_in = in_features * num_agents
        self.num_out = out_features * num_agents
        self.weight_idx = _weight_indices(self, in_features, out_features)

        self.linear = nn.Linear(self.num_in, self.num_out)
        if num_agents > 1:
            self.linear.weight.register_hook(self._hook)
            with torch.no_grad():
                self.linear.weight[self.weight_idx] = 0

    def _hook(self, grad):
        """Backward hook to set inter-agent weight gradients to zero."""
        grad = grad.clone()
        grad[self.weight_idx] = 0

        return grad

    def forward(self, x):
        """Forward pass of linear layer.

        Args:
            x = [torch.Tensor] input of shape (*, self.num_in)

        Returns [torch.Tensor]:
            Output of shape (*, self.num_out).
        """
        y = self.linear(x)

        return y


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
        self.weight_idx = _weight_indices(self, in_features, out_features)
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
        fc1        = [Linear] first linear layer
        fc2a       = [Linear] second linear layer of advantage pathway
        fc2v       = [Linear] second linear layer of state value pathway
        fc3a       = [Linear] third linear layer of advantage pathway
        fc3v       = [Linear] last linear layer of state value pathway
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
        in_features = reduce(mul, state_shape)
        self.fc1 = Linear(in_features, num_hidden, num_agents)

        self.fc2a = Linear(num_hidden, num_hidden, num_agents)
        self.fc2v = Linear(num_hidden, num_hidden, num_agents)

        self.fc3a = Linear(num_hidden, num_actions * num_atoms, num_agents)
        self.fc3v = Linear(num_hidden, num_atoms, num_agents)

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
        fc1        = [Linear] first linear layer
        fc2a       = [Linear] second linear layer of advantage pathway
        fc2v       = [Linear] second linear layer of state value pathway
        fc3a       = [Linear] third linear layer of advantage pathway
        fc3v       = [Linear] last linear layer of state value pathway
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
            device,
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


class DDQN(nn.Module):
    """Implements a noisy dueling deep Q network.

    Attributes:
        conv1 = [nn.Module] first convolutional layer
        conv2 = [nn.Module] second convolutional layer
        conv3 = [nn.Module] third convolutional layer
        conv4 = [nn.Module] fourth convolutional layer
        fc1a = [NoisyLinear] hidden advantage noisy linear layer
        fc1v = [NoisyLinear] hidden value noisy linear layer
        fc2a = [NoisyLinear] output advantage noisy linear layer
        fc2v = [NoisyLinear] output value noisy linear layer
        num_atoms = [int] number of atoms in each Q-value distribution
        num_agents = [int] number of independent sub-models
        device = [torch.device] device to put the model on
    """

    def __init__(
        self,
        state_shape,
        num_actions,
        num_atoms,
        num_agents,
        device
    ):
        """Initializes the DDQN.

        Args:
            state_shape = [tuple] sizes of dimensions of an environment state
            num_actions = [int] number of possible actions in environment
            num_atoms = [int] number of atoms in each Q-value distribution
            num_agents = [int] number of independent sub-models
            device = [torch.device] device to put the model and data on
        """
        super(DDQN, self).__init__()

        # initialize convolutional layers
        num_channels, height, width = state_shape
        self.conv1 = nn.Conv2d(
            in_channels=num_channels * num_agents,
            out_channels=num_channels*2 * num_agents,
            kernel_size=1,
            stride=1,
            groups=num_agents,
        )
        self.conv2 = nn.Conv2d(
            in_channels=num_channels*2 * num_agents,
            out_channels=num_channels*4 * num_agents,
            kernel_size=3,
            stride=1,
            groups=num_agents,
        )
        self.conv3 = nn.Conv2d(
            in_channels=num_channels*4 * num_agents,
            out_channels=num_channels*8 * num_agents,
            kernel_size=3,
            stride=1,
            groups=num_agents,
        )
        self.conv4 = nn.Conv2d(
            in_channels=num_channels*8 * num_agents,
            out_channels=num_channels*12 * num_agents,
            kernel_size=3,
            stride=1,
            groups=num_agents,
        )

        # initialize noisy linear layers
        num_features = num_channels*12
        self.fc1a = NoisyLinear(num_features, num_features, num_agents)
        self.fc1v = NoisyLinear(num_features, num_features, num_agents)
        self.fc2a = NoisyLinear(num_features, num_actions * num_atoms, num_agents)
        self.fc2v = NoisyLinear(num_features, num_atoms, num_agents)

        # put model on correct device
        self.num_atoms = num_atoms
        self.num_agents = num_agents
        self.device = device
        self.to(self.device)

    def forward(self, input):
        """Forward pass of the DDQN.

        In QAgent.step():
        Args:
            input = [torch.Tensor] state of shape (channels, height, width)

        Returns [torch.Tensor]:
            Q-value distributions of shape (actions, atoms).

        In QAgent.train():
        Args:
            input = [torch.Tensor] states of shape (batch_size, channels,
                height, width)

        Returns [torch.Tensor]:
            Q-value distributions of shape (batch_size, actions, atoms).
        """
        # add batch dimension in QAgent.step()
        x = input.unsqueeze(0) if input.ndim == 3 else input
        x = x.to(self.device)

        # run input through convolutional layers
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        # run latent vectors through noisy linear layers
        h = h.flatten(start_dim=1)
        ha = F.relu(self.fc1a(h))
        hv = F.relu(self.fc1v(h))
        a = self.fc2a(ha)
        v = self.fc2v(hv)

        # separate action and atom dimensions
        a = a.view(v.shape[0], -1, self.num_atoms)
        v = v.view(v.shape[0], 1, self.num_atoms)

        # dueling DQN
        y = v + a - a.mean(dim=1, keepdim=True)

        return y.squeeze(0) if input.ndim == 3 else y


class EnsembleDDQN(DDQN):
    """Implements ensemble of noisy dueling deep Q networks.

    Attributes:
        conv1 = [nn.Module] first convolutional layer
        conv2 = [nn.Module] second convolutional layer
        conv3 = [nn.Module] third convolutional layer
        conv4 = [nn.Module] fourth convolutional layer
        fc1a = [NoisyLinear] hidden advantage noisy linear layer
        fc1v = [NoisyLinear] hidden value noisy linear layer
        fc2a = [NoisyLinear] output advantage noisy linear layer
        fc2v = [NoisyLinear] output value noisy linear layer
        num_atoms = [int] number of atoms in each Q-value distribution
        num_agents = [int] number of independent sub-models
        channel_idx = [torch.Tensor] index array of channels per agent
        agent_idx = [torch.Tensor] index array in the agent dimension
        device = [torch.device] device to put the model on
    """

    def __init__(
        self,
        state_shape,
        num_frames,
        channel_idx,
        num_actions,
        num_atoms,
        num_agents,
        device,
    ):
        """Initializes the ensemble DDQN.

        Args:
            state_shape = [tuple] sizes of dimensions of an environment state
            num_frames = [int] number of frames in an environment state
            channel_idx = [list] indices of channels per agent
                The number of channels per agent must be constant to be able to
                implement the Ensemble DDQN in one neural network.
                For example: [[0, 1, 2], [0, 2, -1], [0, 3, -1]]; the first
                agent gets channels 0, 1, and 2 from the state, the second
                agent channels 0, 2, and the last channel, and so on.
            num_actions = [int] number of possible actions in environment
            num_atoms = [int] number of atoms in each Q-value distribution
            num_agents = [int] number of independent sub-models
            device = [torch.device] device to put the model on
        """
        # number of channels of the total state given as state to each agent
        num_channels, height, width = state_shape
        channel_idx = torch.tensor(channel_idx)
        num_channels_per_agent = channel_idx.shape[1] * num_frames

        # initialize network layers
        super(EnsembleDDQN, self).__init__(
            (num_channels_per_agent, height, width),
            num_actions,
            num_atoms,
            num_agents,
            device,
        )

        # determine indices of first channels of frames in environment state
        num_channels_per_frame = num_channels // num_frames
        frame_idx = torch.arange(0, num_channels, step=num_channels_per_frame)

        # compute the channel indices for each agent and frame
        self.channel_idx = channel_idx.unsqueeze(1) + frame_idx.unsqueeze(1)
        self.channel_idx = self.channel_idx.flatten()

        # set agent index array beforehand for faster indexing
        self.agent_idx = torch.arange(num_agents)
        self.agent_idx = self.agent_idx.repeat_interleave(num_channels_per_agent)

    def forward(self, input):
        """Forward pass of the ensemble of DDQNs.

        In EnsembleQAgent.step():
        Args:
            input = [torch.Tensor] state of shape (channels, height, width)

        Returns [torch.Tensor]:
            Q-value distributions of shape (agents, actions, atoms).

        In EnsembleQAgent.train():
        Args:
            input = [torch.Tensor] states of shape (batch_size, agents,
                channels, height, width)

        Returns [torch.Tensor]:
            Q-value distributions of shape (batch_size, agents, actions, atoms).
        """
        if input.ndim == 3:  # in EnsembleQAgent.step()
            x = input[None, self.channel_idx]
        else:  # in EnsembleQAgent.train()
            x = input[:, self.agent_idx, self.channel_idx]
        x = x.to(self.device)

        # run input through convolutional layers
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))

        # run latent vectors through noisy linear layers
        h = h.flatten(start_dim=1)
        ha = F.relu(self.fc1a(h))
        hv = F.relu(self.fc1v(h))
        a = self.fc2a(ha)
        v = self.fc2v(hv)

        # separate agent, action, and atom dimensions
        a = a.view(v.shape[0], self.num_agents, -1, self.num_atoms)
        v = v.view(v.shape[0], self.num_agents, 1, self.num_atoms)

        # dueling DQN
        y = v + a - a.mean(dim=2, keepdim=True)

        return y.squeeze(0) if input.ndim == 3 else y


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
        self.fc1 = Linear(
            in_features=num_agents * num_actions * num_atoms,
            out_features=num_agents * num_hidden,
        )
        self.fc2 = Linear(
            in_features=num_agents * num_hidden,
            out_features=num_actions * num_atoms,
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
