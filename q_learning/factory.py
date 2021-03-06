from functools import partial

from env import CartPole
from .agent import (
    EnsembleQAgent,
    QAgent,
)
from .combine import (
    Combiner,
    MLPCombiner,
)
from .data import (
    EnsembleMemory,
    MetaMemory,
    ReplayMemory,
)
from .networks import (
    MLP,
    EnsembleDDQN,
    EnsembleLinearModel,
    DDQN,
    LinearModel,
)


def q_agent(env, args, device):
    # make function that initializes model
    if isinstance(env, CartPole):
        model_fn = partial(
            LinearModel,
            env.state_shape,
            env.num_actions,
            args.num_hidden,
            args.num_atoms,
            args.num_agents,
            device,
        )
    else:
        model_fn = partial(
            DDQN,
            env.state_shape, env.num_actions,
            args.num_atoms,
            args.num_agents,
            device,
        )

    # initialize replay memory
    memory = ReplayMemory(
        args.capacity,
        args.gamma, args.num_transitions,
        args.b, args.b_increase,
        args.e, args.a,
    )

    # initialize agent
    agent = QAgent(
        model_fn, args.learning_rate,
        memory, args.batch_size,
        args.num_atoms, args.v_min, args.v_max,
        args.gamma, args.num_transitions,
        args.train_onset, args.num_update,
        device,
    )

    return memory, agent


def ensemble_network(env, args, device):
    if isinstance(env, CartPole):
        return partial(
            EnsembleLinearModel,
            env.state_shape, env.num_actions,
            args.num_hidden, args.num_atoms,
            args.num_agents,
            device
        )
    else:
        return partial(
            EnsembleDDQN,
            env.state_shape,
            env.num_frames,
            args.channel_indices,
            env.num_actions,
            args.num_atoms,
            args.num_agents,
            device,
        )


def ensemble_q_agent(env, args, device):
    combiner = Combiner(
        args.combine_mode,
        args.num_atoms, args.v_min, args.v_max,
        device,
    )

    memory = EnsembleMemory(
        args.num_agents,
        args.capacity,
        args.gamma, args.num_transitions,
        args.b, args.b_increase,
        args.e, args.a,
    )

    # make function that initializes model
    model_fn = ensemble_network(env, args, device)

    # initialize agent
    agent = EnsembleQAgent(
        args.num_agents,
        model_fn, args.learning_rate,
        memory, args.batch_size,
        combiner,
        args.num_atoms, args.v_min, args.v_max,
        args.gamma, args.num_transitions,
        args.train_onset, args.num_update,
        device,
    )

    return memory, agent


def q_agent_mlp_combiner(env, args, device):
    # make function initializes MLP
    mlp_fn = partial(
        MLP,
        args.num_agents, env.num_actions,
        args.num_hidden, args.num_atoms,
        device,
    )

    # initialize replay memory
    memory = MetaMemory(
        args.num_agents,
        args.capacity,
        args.gamma, args.num_transitions,
        args.b, args.b_increase,
        args.e, args.a,
    )

    combiner = MLPCombiner(
        mlp_fn, args.learning_rate,
        memory, args.batch_size,
        args.num_atoms, args.v_min, args.v_max,
        args.gamma, args.num_transitions,
        args.train_onset, args.num_update,
        device,
    )

    # make function that initializes model
    model_fn = ensemble_network(env, args, device)

    # initialize agent
    agent = EnsembleQAgent(
        args.num_agents,
        model_fn, args.learning_rate,
        memory.ensemble_memory, args.batch_size,
        combiner,
        args.num_atoms, args.v_min, args.v_max,
        args.gamma, args.num_transitions,
        args.train_onset, args.num_update,
        device,
    )

    # set policy net to compute Q-value distributions at end of episode
    memory.policy_net = agent.policy_net

    return memory, agent


def memory_agent(env, args, device):
    if args.num_agents == 1:
        # initialize memory and Q agent
        memory, agent = q_agent(env, args, device)
    else:
        if args.combine_mode == 'mlp':
            # initialize memory and ensemble Q agent with MLP combiner
            memory, agent = q_agent_mlp_combiner(env, args, device)
        else:
            # initialize memory and ensemble Q agent
            memory, agent = ensemble_q_agent(env, args, device)

    return memory, agent
