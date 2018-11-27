import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import gym
import numpy as np

number_of_states = 40
episodes = 1000
initial_learning_rate = 1.0
minimum_learning_rate = 0.003
discount_factor = 0.99
epsilon = 0.02
# Needed to plot the iterations needed to converge per episode
iterations_needed = []

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,  # MountainCar-v0 uses 200
)

env = gym.make('MountainCarMyEasyVersion-v0')

# Initiate q values with zero values
q_learning_values = np.zeros((number_of_states, number_of_states, env.action_space.n))


def calculate_q_values():
    for episode in range(episodes):

        observation = env.reset()

        current_learning_rate = max(minimum_learning_rate, initial_learning_rate * (0.85 ** (episode // 100)))
        done = False
        iterations_needed_episode = 0
        while not done:
            # Get current state
            state = get_state_from_observation(observation)

            # Choose an action
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_learning_values[state])
            # Perform chosen action
            observation, reward, done, info = env.step(action)

            # Discover new state
            new_state = get_state_from_observation(observation)

            # Update Q Learning Values
            q_learning_values[state][action] = q_learning_values[state][action] + current_learning_rate * (
                    reward + discount_factor * np.max(q_learning_values[new_state]) - q_learning_values[state][action])
            iterations_needed_episode += 1
        iterations_needed.append(iterations_needed_episode)
        print('Episode #{} achieved goal after {} iterations'.format(episode + 1, iterations_needed_episode))


def get_state_from_observation(observation):
    difference = (env.observation_space.high - env.observation_space.low) / number_of_states
    a = int((observation[0] - env.observation_space.low[0]) / difference[0])
    b = int((observation[1] - env.observation_space.low[1]) / difference[1])
    return a, b


if __name__ == '__main__':
    calculate_q_values()
    optimal_policy = np.argmax(q_learning_values, axis=2)
    print(optimal_policy)

    # In order to plot the q values of the policy take the max value in the last dimension of the q_learning_table

    # q_values_to_plot = np.max(q_learning_values, axis=2)
    # plt.plot(range(len(iterations_needed)), iterations_needed)

    state_values = pd.DataFrame(index=np.round(np.arange(-1.2, 0.6, 0.045), 2), columns=np.round(np.arange(-0.07, 0.07, 0.0035), 3),
                                data=np.max(q_learning_values, axis=2))

    sns.heatmap(state_values, linewidths=0)
    plt.show()

    # Perform a demonstration
    observation = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.0015)
        state = get_state_from_observation(observation)
        action = optimal_policy[state]
        observation, reward, done, info = env.step(action)

    env.close()
