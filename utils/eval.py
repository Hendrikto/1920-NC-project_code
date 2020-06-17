import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# settings for pyplot
matplotlib.use('tkagg')
plt.ion()

# empty figures for location prediction
fig_locations = None
ax_locations = None
state_img = None
red_location = None
blue_location = None

# empty figures for rewards
fig_rewards = None
ax_rewards = None
rewards_plot = None


def plot_rewards(rewards, scores, window=10):
    """Plot the given rewards as the average reward in a window of episodes.

    Args:
        rewards = [[float]] list of sum of rewards for each episode
        window  = [int] number of episode rewards to take the average of
    """
    global fig_rewards
    global ax_rewards
    global rewards_plot
    global ax_scores
    global scores_plot

    if len(rewards) < window:
        return

    rewards_smoothed = []
    for i in range(len(rewards) - window + 1):
        rewards_smoothed.append(np.mean(rewards[i:i + window]))

    scores_smoothed = []
    for i in range(len(scores) - window + 1):
        scores_smoothed.append(np.mean(scores[i:i + window]))

    if not fig_rewards:
        fig_rewards = plt.figure()
        ax_rewards = fig_rewards.add_subplot(2, 1, 1)
        ax_rewards.set_title('Smoothed episode rewards')
        ax_rewards.set_xlabel('Episode')
        ax_rewards.set_ylabel(f'Reward')
        rewards_plot = ax_rewards.plot(0)[0]
        ax_scores = fig_rewards.add_subplot(2, 1, 2)
        ax_scores.set_title('Smoothed episode scores')
        ax_scores.set_xlabel('Episode')
        ax_scores.set_ylabel(f'Score')
        scores_plot = ax_scores.plot(0)[0]

    ax_rewards.set_xlim(1, len(rewards))
    ax_rewards.set_ylim(min(rewards), max(rewards))
    ax_scores.set_xlim(1, len(scores))
    ax_scores.set_ylim(min(scores), max(scores))
    rewards_plot.set_data(np.arange(window, len(rewards) + 1), rewards_smoothed)
    scores_plot.set_data(np.arange(window, len(scores) + 1), scores_smoothed)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)
