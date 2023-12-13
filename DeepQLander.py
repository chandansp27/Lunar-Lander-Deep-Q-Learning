# importing packages
import time
import numpy as np
import PIL.Image
from pyvirtualdisplay import Display

# deque for storing the lander memory buffer
# namedtuple for storing the experienced tuples
from collections import deque, namedtuple

# openai RL env
import gym
import gymnasium as gym
from gym.envs import box2d

# tensorflow libraries
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers import Adam

# helper functions
import utils

# setup display to display the lander env
env = gym.make('LunarLander-v2', render_mode='human')
observation, info = env.reset()

# for i in range(100):
#     action = env.action_space.sample()
#     observation, reward, terminated, turncated, info = env.step(action)
#     if terminated or turncated:
#         observation, info = env.reset()
# env.close()

# hyperparameters
tf.random.set_seed = 0
MEMORY_SIZE = 100_000
GAMMA = 0.995
ALPHA = 1e-3
NUM_STEPS_FOR_UPDATE = 4

# loading env
env = gym.make('LunarLander-v2')
initial_state = env.reset()

action = 0
print(env.step(action))
next_state, reward, done, _ = env.step(action)[:4]
state_size = env.observation_space.shape  # (8,)
num_actions = env.action_space.n  # 4

# create 2 neural networks for the experience replay Q-network and the target network Q*

q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])

target_q_network = Sequential([
    Input(shape=state_size),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])

optimizer = Adam(learning_rate=ALPHA)

# storing experiences as named tuples
experience = namedtuple("Experience", field_names=['state', 'action', 'reward', 'next_state', 'done'])


@tf.function
def agent_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = utils.compute_loss(experiences, gamma, q_network, target_q_network)
        gradients = tape.gradient(loss, q_network.trainable_variables)
        optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))
        utils.update_target_network(q_network, target_q_network)


# training the agent
start = time.time()

num_episodes = 20000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging.
epsilon = 1.0     # initial ε value for ε-greedy policy.

# memory buffer D
memory_buffer = deque(maxlen=MEMORY_SIZE)

target_q_network.set_weights(q_network.get_weights())

for i in range(num_episodes):

    # Reset the environment to the initial state and get the initial state.
    state = env.reset()
    total_points = 0

    for t in range(max_num_timesteps):

        # Take action A and receive reward R and the next state S'.
        next_state, reward, done, _ = env.step(action)[:4]

        # Ensure the next_state is a NumPy array.
        state = np.array(next_state)

        # Store experience tuple (S,A,R,S') in the memory buffer, including the state before the expansion.
        memory_buffer.append(experience(state, action, reward, next_state, done))

        # Only update the network every NUM_STEPS_FOR_UPDATE time steps.
        update = utils.check_update_conditions(t, NUM_STEPS_FOR_UPDATE, memory_buffer)

        if update:
            # Sample random mini-batch of experience tuples (S,A,R,S') from D.
            experiences = utils.get_experiences(memory_buffer)

            # Set the y targets, perform a gradient descent step, and update the network weights.
            agent_learn(experiences, GAMMA)

        total_points += reward

        if done:
            break
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])

    # Update the ε value.
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i + 1) % num_p_av == 0:
        print(f"\rEpisode {i + 1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 200 points in the last 100 episodes.
    if av_latest_points >= 200.0:
        print(f"\n\nEnvironment solved in {i + 1} episodes!")
        q_network.save('lunar_lander_model.h5')
        break

tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time / 60):.2f} min)")

tf.keras.backend.clear_session()
