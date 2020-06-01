import argparse
from itertools import count

import numpy as np
import pandas as pd
import torch

from env import (
    EnsemblePacMan,
    PacMan,
)
from q_learning.factory import (
    ensemble_q_agent,
    q_agent,
    q_agent_mlp_combiner,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--a',
        type=float,
        default=0.6,
        help='exponent of priorities for PER',
    )
    parser.add_argument(
        '--b',
        type=float,
        default=0.4,
        help='initial exponent of IS weights',
    )
    parser.add_argument(
        '--b_increase',
        type=float,
        default=0.001,
        help='increase to b at end of episode',
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=20,
        help='batch size per training step',
    )
    parser.add_argument(
        '-c', '--capacity',
        type=int,
        default=10_000,
        help='size of replay memory',
    )
    parser.add_argument(
        '-z', '--combine_mode',
        type=str,
        default='sum',
        choices=('avg', 'max', 'min', 'mlp', 'sum', 'vote'),
        help='mode of combiner function',
    )
    parser.add_argument(
        '-d', '--cuda',
        action='store_true',
        help='enable GPU acceleration',
    )
    parser.add_argument(
        '--e',
        type=float,
        default=0.001,
        help='number to prevent zero priority',
    )
    parser.add_argument(
        '-y', '--gamma',
        type=float,
        default=0.99,
        help='discount factor γ',
    )
    parser.add_argument(
        '-e', '--learning_rate',
        type=float,
        default=0.01,
        help='learning rate of Adam optimizer',
    )
    parser.add_argument(
        '-p', '--model_path',
        type=str,
        default='model.pt',
        help='path to save trained model',
    )
    parser.add_argument(
        '-m', '--num_agents',
        type=int,
        default=1,
        help='number of ensemble members',
    )
    parser.add_argument(
        '-a', '--num_atoms',
        type=int,
        default=51,
        help='number of atoms in Q-value distribution',
    )
    parser.add_argument(
        '-g', '--num_episodes',
        type=int,
        default=1000,
        help='number of training episodes',
    )
    parser.add_argument(
        '-l', '--num_hidden',
        type=int,
        default=50,
        help='number of hidden units per agent',
    )
    parser.add_argument(
        '-n', '--num_transitions',
        type=int,
        default=8,
        help='number of transitions for N-step DQN',
    )
    parser.add_argument(
        '-u', '--num_update',
        type=int,
        default=64,
        help='period to update target network',
    )
    parser.add_argument(
        '-f', '--results_file',
        type=str,
        default='results.csv',
        help='name of CSV results file',
    )
    parser.add_argument(
        '-r', '--run',
        action='store_true',
        help='run a trained agent',
    )
    parser.add_argument(
        '-o', '--train_onset',
        type=int,
        default=40,
        help='number of memories before training',
    )
    parser.add_argument(
        '--v_max',
        type=int,
        default=250.0,
        help='highest x-value of Q-value distribution',
    )
    parser.add_argument(
        '--v_min',
        type=int,
        default=-250.0,
        help='lowest x-value of Q-value distribution',
    )

    return parser.parse_args()


def print_arguments(arguments, width=80):
    print(' Arguments '.center(width, '='))
    for argument_name, argument in vars(args).items():
        print(f'{argument_name} = {argument}')
    print('=' * width)


def run_agent(env, memory, agent, num_episodes, train, results_file_name):
    episode_wins, episode_steps, episode_scores = [[] for _ in range(3)]
    for i_episode in range(num_episodes):
        state = env.reset()
        memory.reset(state)
        for step in count(1):
            action = agent.step(state)

            end, state, rewards = env.step(action)
            memory.push(end, action, rewards, state)

            if train:
                agent.train()

            if end or step >= 500:
                break

        # determine whether the agent lost or won
        episode_wins.append(env.won)
        episode_steps.append(step)
        episode_scores.append(env.score)
        print(f'Episode {i_episode + 1} of {num_episodes}')
        print('--- WINNER ---' if episode_wins[i_episode] else '--- LOSER ---')
        print(f'Number of steps: {episode_steps[i_episode]}')
        print(f'Reward: {episode_scores[i_episode]}')

    # save wins, number of steps, and rewards of all episodes to CSV file
    stats = pd.DataFrame({
        'wins': np.asarray(episode_wins),
        'num_steps': np.asarray(episode_steps),
        'rewards': np.asarray(episode_scores),
    })
    stats.to_csv(results_file_name, index=False)


if __name__ == '__main__':
    # handle console arguments
    args = parse_arguments()
    print_arguments(args)

    # make CPU or GPU based on console argument
    device = torch.device('cuda' if args.cuda else 'cpu')

    if args.num_agents == 1:
        # initialize environment
        env = PacMan(4)

        # initialize memory and Q agent
        memory, agent = q_agent(env, args, device)
    else:
        # initialize environment
        env = EnsemblePacMan(args.num_agents)

        if args.combine_mode == 'mlp':
            # initialize memory and ensemble Q agent with MLP combiner
            memory, agent = q_agent_mlp_combiner(env, args, device)
        else:
            # initialize memory and ensemble Q agent
            memory, agent = ensemble_q_agent(env, args, device)

    # run or train the agent
    if args.run:
        agent.policy_net.load_state_dict(torch.load(args.model_path))
        agent.policy_net.eval()
        run_agent(env, memory, agent,
                  args.num_episodes, False, args.results_file)
    else:
        run_agent(env, memory, agent,
                  args.num_episodes, True, args.results_file)
        torch.save(agent.policy_net.state_dict(), args.model_path)
