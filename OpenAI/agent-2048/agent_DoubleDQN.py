import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import gym_2048
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt
from time import perf_counter
from collections import deque
import random
import os


class ExperienceBuffer:
    def __init__(self, buffer_size=50000):
        self.buffer = deque(iterable=[], maxlen=buffer_size)
        self.buffer_size = buffer_size

    def add(self, experience):
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


class QNetwork:
    def __init__(self, state_size, num_actions, hidden_size, num_hidden=3):

        # Create the feed-forward network
        self.state_in = tf.placeholder(shape=[None, state_size],
                                       dtype=tf.float32)

        in_tensor = self.state_in
        for _ in range(num_hidden):
            in_tensor = slim.fully_connected(in_tensor,
                                             hidden_size,
                                             biases_initializer=None,
                                             activation_fn=tf.nn.relu)

        self.stream_advantage, self.stream_value = tf.split(in_tensor, 2, axis=1)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.advantage_weights = tf.Variable(xavier_init([int(hidden_size/2), num_actions]))
        self.value_weights = tf.Variable(xavier_init([int(hidden_size/2), 1]))
        self.advantage = tf.matmul(self.stream_advantage, self.advantage_weights)
        self.value = tf.matmul(self.stream_value, self.value_weights)

        self.Q_out = self.value + tf.subtract(self.advantage, tf.reduce_mean(self.advantage, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Q_out, 1)

        # Feed-back network
        self.target_Q = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, num_actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Q_out, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.target_Q - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)  # TODO: Use RMSPropOptimizer
        self.updateModel = self.trainer.minimize(self.loss)


def transform_state(new_observation, state_history):
    new_observation = np.log2(new_observation.ravel())
    new_observation[new_observation == -np.inf] = 0
    n = len(state_history) - len(new_observation)
    new_state_history = np.concatenate([new_observation, state_history[:n]])
    return new_state_history


def update_target_graph(tf_vars, tau):
    total_vars = len(tf_vars)
    op_holder = []
    for i, var in enumerate(tf_vars[0: total_vars//2]):
        idx = i + total_vars//2
        # This looks like a low pass filter for smoothing transitions
        op_holder.append(tf_vars[idx].assign((var.value() * tau) + ((1 - tau) * tf_vars[idx].value())))
    return op_holder


def update_target(op_holder, sess):
    for op in op_holder:
        sess.run(op)


if __name__ == '__main__':
    batch_size = 32     # How many experiences to use for each training step
    update_freq = 4     # How often to perform a training step
    y = 0.99            # Discount factor on the Q-values
    start_e = 1         # Starting chance of random action
    end_e = 0.1         # Final chance of random action
    annealing_steps = 250000     # How many steps of training to reduce start_e to end_e
    num_episodes = 500000        # How many episodes of game environment to train network with
    pre_train_steps = 250000     # How many steps of random action before training begins
    max_ep_length = 10000        # The max allowed length of our episode
    load_model = False          # Whether to load a saved model
    model_path = './model_dqn'    # The path to save our model
    hidden_size = 64                    # The hidden layer sizes
    tau = 0.001                         # Rate to update target network toward primary network
    max_same_states = 50
    num_state_history = 4
    state_size = 16
    dqn_input_size = state_size * num_state_history

    input_map = {0: 'w',
                 1: 'd',
                 2: 's',
                 3: 'a'}

    action_map = {0: 'UP',
                  1: 'RIGHT',
                  2: 'DOWN',
                  3: 'LEFT'}

    env = gym.make('My2048-v0')

    tf.reset_default_graph()
    main_Q_network = QNetwork(state_size=dqn_input_size,
                              num_actions=4,
                              hidden_size=hidden_size,
                              num_hidden=3)
    target_Q_network = QNetwork(state_size=dqn_input_size,
                                num_actions=4,
                                hidden_size=hidden_size,
                                num_hidden=3)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    trainables = tf.trainable_variables()

    target_ops = update_target_graph(trainables, tau)

    outer_experience_buffer = ExperienceBuffer()

    epsilon = start_e
    step_drop = (start_e - end_e) / annealing_steps

    rewards_list = []
    steps_list = []
    total_steps = 0

    # Make a path for our model to be saved in.
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with tf.Session() as sess:
        sess.run(init)

        if load_model:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(model_path)
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(num_episodes):
            episode_buffer = ExperienceBuffer()

            # We want a memory of 4 of the last moves. Since this is the first move we just pad it out
            state = env.reset()
            state = transform_state(state, np.zeros(16 * num_state_history))
            done = False
            reward_all = 0
            num_same_states = 0
            for j in range(max_ep_length):
                # Choose action by greedily (with epsilon chance of random action) from the Q-network
                if np.random.rand(1) < epsilon or total_steps < pre_train_steps:
                    action = np.random.randint(0, 4)
                else:
                    action = sess.run(main_Q_network.predict,
                                      feed_dict={main_Q_network.state_in:[state]})
                    action = action[0]

                # Run the action
                new_state, reward, done, _ = env.step(input_map[action])
                new_state = transform_state(new_state, state)
                if np.array_equal(state, new_state):
                    num_same_states += 1

                if num_same_states == max_same_states:
                    done = True

                # Clip rewards
                #reward = min(reward, 1)

                total_steps += 1

                # Add the experience to the episode buffer
                episode_buffer.add(np.reshape(np.array([state, action, reward, new_state, done]), [1, 5]))

                # Reduce epsilon
                if total_steps > pre_train_steps:
                    if epsilon > end_e:
                        epsilon -= step_drop

                    if total_steps % update_freq == 0:
                        # Get a random batch of experiences
                        train_batch = outer_experience_buffer.sample(batch_size)

                        # Perform Double-QDN update to the target q-values
                        Q1 = sess.run(main_Q_network.predict,
                                      feed_dict={main_Q_network.state_in: np.vstack(train_batch[:, 3])})
                        Q2 = sess.run(target_Q_network.Q_out,
                                      feed_dict={target_Q_network.state_in: np.vstack(train_batch[:, 3])})

                        #print(train_batch[:, 4])
                        #end_multiplier = train_batch[:, 4].astype(int)
                        end_multiplier = -(train_batch[:, 4] - 1)
                        #print(end_multiplier)

                        double_Q = Q2[range(batch_size), Q1]
                        target_Q = train_batch[:, 2] + (y * double_Q * end_multiplier)

                        # Update the network with our target values
                        _ = sess.run(main_Q_network.updateModel,
                                     feed_dict={main_Q_network.state_in: np.vstack(train_batch[:, 0]),
                                                main_Q_network.target_Q: target_Q,
                                                main_Q_network.actions: train_batch[:, 1]})

                        # Update the target network toward the primary network
                        update_target(target_ops, sess)

                reward_all += reward
                state = new_state

                if done:
                    break

            outer_experience_buffer.add(episode_buffer.buffer)
            steps_list.append(j)
            rewards_list.append(reward_all)

            # Periodically save the model
            if i % 1000 == 0:
                saver.save(sess, model_path + '/model-' + str(i) + '.ckpt')
                print('Saved Model')

            if len(rewards_list) % 250 == 0:
                print(i, np.mean(rewards_list[-250:]), epsilon)

            report_period = 250
            if i % report_period == 0 and i != 0:
                print(i)
                print('Mean reward of last {0}: {1}'.format(report_period, np.mean(rewards_list[-report_period:])))
                print('Mean ep length of last {0}: {1}'.format(report_period, np.mean(steps_list[-report_period:])))

            if i % 1000 == 0 and i != 0:
                fig, ax = plt.subplots()
                pd.Series(rewards_list).rolling(report_period).mean().plot(ax=ax)
                ax2 = ax.twinx()
                pd.Series(steps_list).rolling(report_period).mean().plot(ax=ax2, color='red')
                fig.savefig('agent_' + str(i) + '.png')

        saver.save(sess, model_path + '/model-' + str(i) + '.ckpt')
