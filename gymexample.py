import time
import matplotlib.pyplot as plt
import seaborn as sns
import gym
import numpy as np

number_of_states = 40
episodes = 1000
initial_learning_rate = 1.0
minimum_learning_rate = 0.003
discount_factor = 1.0
epsilon = 0.02

gym.envs.register(
    id='MountainCarMyEasyVersion-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=100000,      # MountainCar-v0 uses 200
)

env = gym.make('MountainCarMyEasyVersion-v0')

# Initiate q values with zero values
q_learning_values = np.zeros((number_of_states, number_of_states, env.action_space.n))


def calculate_q_values():
    for episode in range(episodes):

        observation = env.reset()

        eta = max(minimum_learning_rate, initial_learning_rate * (0.85 ** (episode // 100)))
        done = False
        while not done:
            # Get current state
            state = obs_to_state(observation)

            # Choose an action
            if np.random.uniform(0, 1) < epsilon:
                action = np.random.choice(env.action_space.n)
            else:
                action = np.argmax(q_learning_values[state])
            # Perform chosen action
            observation, reward, done, info = env.step(action)

            # Discover new state
            new_state = obs_to_state(observation)

            # Update Q Learning Values
            q_learning_values[state + (action,)] = q_learning_values[state + (action,)] + eta * (
                        reward + discount_factor * np.max(q_learning_values[new_state]) - q_learning_values[
                    state + (action,)])

        if episode % 100 == 0:
            print('Episode #{}'.format(episode+1))


def obs_to_state(obs):
    """ Maps an observation to state """
    env_dx = (env.observation_space.high - env.observation_space.low) / number_of_states
    a = int((obs[0] - env.observation_space.low[0])/env_dx[0])
    b = int((obs[1] - env.observation_space.low[1])/env_dx[1])
    return a, b


if __name__ == '__main__':
    calculate_q_values()
    optimal_policy = np.argmax(q_learning_values, axis=2)
    print(optimal_policy)

    # In order to plot the q values of the policy take the max value in the last dimension of the q_learning_table
    q_values_to_plot = np.max(q_learning_values, axis=2)
    graph = sns.heatmap(q_values_to_plot, linewidths=0.5)
    plt.show()


    # Perform a demonstration
    observation = env.reset()
    done = False
    while not done:
        env.render()
        time.sleep(0.015)
        state = obs_to_state(observation)
        action = optimal_policy[state]
        observation, reward, done, info = env.step(action)

    env.close()
