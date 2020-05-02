from functools import partial

from .agent import EnsembleQAgent, QAgent
from .data import EnsembleMemory, MetaMemory, ReplayMemory
from .nn import EnsembleLinearModel, LinearModel, MLP
from .combine import Combiner, MLPCombiner


def q_agent(env, args, device):
    # make function that initializes model
    model_fn = partial(
        LinearModel,
        env.state_shape, env.num_actions,
        args.num_hidden, args.num_atoms,
        args.num_agents,
        device
    )

    # initialize replay memory
    memory = ReplayMemory(
        args.capacity,
        args.gamma, args.num_transitions,
        args.b, args.b_increase,
        args.e, args.a
    )

    # initialize agent
    agent = QAgent(
        model_fn, args.learning_rate,
        memory, args.batch_size,
        args.num_atoms, args.v_min, args.v_max,
        args.gamma, args.num_transitions,
        args.train_onset, args.num_update,
        device
    )

    return memory, agent


def ensemble_q_agent(env, args, device):
    combiner = Combiner(
        args.combine_mode,
        args.num_atoms, args.v_min, args.v_max,
        device
    )

    memory = EnsembleMemory(
        args.num_agents,
        args.capacity,
        args.gamma, args.num_transitions,
        args.b, args.b_increase,
        args.e, args.a
    )

    # make function that initializes model
    model_fn = partial(
        EnsembleLinearModel,
        env.state_shape, env.num_actions,
        args.num_hidden, args.num_atoms,
        args.num_agents,
        device
    )

    # initialize agent
    agent = EnsembleQAgent(
        args.num_agents,
        model_fn, args.learning_rate,
        memory, args.batch_size,
        combiner,
        args.num_atoms, args.v_min, args.v_max,
        args.gamma, args.num_transitions,
        args.train_onset, args.num_update,
        device
    )

    return memory, agent


def q_agent_mlp_combiner(env, args, device):
    # make function initializes MLP
    mlp_fn = partial(
        MLP,
        args.num_agents, env.num_actions,
        args.num_hidden, args.num_atoms,
        device
    )

    # initialize replay memory
    memory = MetaMemory(
        args.num_agents,
        args.capacity,
        args.gamma, args.num_transitions,
        args.b, args.b_increase,
        args.e, args.a
    )

    combiner = MLPCombiner(
        mlp_fn, args.learning_rate,
        memory, args.batch_size,
        args.num_atoms, args.v_min, args.v_max,
        args.gamma, args.num_transitions,
        args.train_onset, args.num_update,
        device
    )

    # make function that initializes model
    model_fn = partial(
        EnsembleLinearModel,
        env.state_shape, env.num_actions,
        args.num_hidden, args.num_atoms,
        args.num_agents,
        device
    )

    # initialize agent
    agent = EnsembleQAgent(
        args.num_agents,
        model_fn, args.learning_rate,
        memory.ensemble_memory, args.batch_size,
        combiner,
        args.num_atoms, args.v_min, args.v_max,
        args.gamma, args.num_transitions,
        args.train_onset, args.num_update,
        device
    )

    # set policy net to compute Q-value distributions at end of episode
    memory.policy_net = agent.policy_net

    return memory, agent
