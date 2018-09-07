import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import gym
import gym_2048
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, learning_rate, state_size, action_size, hidden_size):

        # Feed-forward network
        self.state_in = tf.placeholder(shape=[None, state_size],
                                       dtype=tf.float32)

        hidden = slim.fully_connected(self.state_in,
                                      hidden_size,
                                      biases_initializer=None,
                                      activation_fn=tf.nn.relu)

        self.output = slim.fully_connected(hidden,
                                           action_size,
                                           biases_initializer=None,
                                           activation_fn=tf.nn.softmax)
        self.chosen_action = tf.argmax(self.output, 1)

        # Feed-back network
        self.reward_holder = tf.placeholder(shape=[None], dtype=tf.float32)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32)

        self.indexes = tf.range(0, tf.shape(self.output)[0]) * tf.shape(self.output)[1] + self.action_holder
        self.responsible_outputs = tf.gather(tf.reshape(self.output, [-1]), self.indexes)

        self.loss = -tf.reduce_mean(tf.log(self.responsible_outputs) * self.reward_holder)

        tvars = tf.trainable_variables()
        self.gradient_holders = []
        for idx, var in enumerate(tvars):
            placeholder = tf.placeholder(tf.float32, name=str(idx) + '_holder')
            self.gradient_holders.append(placeholder)

        self.gradients = tf.gradients(self.loss, tvars)

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        self.update_batch = optimizer.apply_gradients(zip(self.gradient_holders, tvars))


if __name__ == '__main__':
    # TODO: Agent can get stuck making the same move over and over again that doesn't changed the game state
    def transform_state(original_state):
        return original_state.ravel()

    def discount_rewards(rewards):
        gamma = 0.99
        discounted_rewards = np.zeros_like(rewards)
        running_sum = 0
        for i in reversed(range(0, rewards.size)):
            running_sum = running_sum * gamma + rewards[i]
            discounted_rewards[i] = running_sum

        return discounted_rewards

    input_map = {0: 'w',
                 1: 'd',
                 2: 's',
                 3: 'a'}

    action_map =  {0: 'UP',
                   1: 'RIGHT',
                   2: 'DOWN',
                   3: 'LEFT'}

    env = gym.make('My2048-v0')

    tf.reset_default_graph()

    agent = Agent(learning_rate=1e-2,
                  state_size=16,
                  action_size=4,
                  hidden_size=64)

    total_episodes = 50000
    max_ep = 1000
    update_frequency = 50
    max_unchanged_states = 10

    report_period = 250

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        i = 0
        total_reward = []
        total_length = []

        gradient_buffer = sess.run(tf.trainable_variables())
        for idx, grad in enumerate(gradient_buffer):
            gradient_buffer[idx] = grad * 0

        for i in range(total_episodes):
            state = env.reset()
            state = transform_state(state)
            running_reward = 0
            episode_history = []
            num_state_unchanged = 0
            for j in range(max_ep):

                action_dist = sess.run(agent.output,
                                       feed_dict={agent.state_in: [state]})
                action = np.random.choice(action_dist[0], p=action_dist[0])
                action = np.argmax(action_dist == action)

                new_state, reward, done, _ = env.step(input_map[action])
                new_state = transform_state(new_state)
                if np.array_equal(new_state, state):
                    num_state_unchanged += 1
                else:
                    num_state_unchanged = 0

                if num_state_unchanged == max_unchanged_states:
                    done = True

                episode_history.append([state, action, reward, new_state])
                state = new_state
                running_reward += reward

                if done or j == max_ep - 1:
                    #print("done:", i)
                    episode_history = np.array(episode_history)
                    episode_history[:, 2] = discount_rewards(episode_history[:, 2])
                    #pprint(episode_history[:, 2])
                    feed_dict = {agent.reward_holder: episode_history[:, 2],
                                 agent.action_holder: episode_history[:, 1],
                                 agent.state_in: np.vstack(episode_history[:, 0])}

                    gradients = sess.run(agent.gradients,
                                         feed_dict=feed_dict)
                    for idx, grad in enumerate(gradients):
                        gradient_buffer[idx] += grad

                    if i % update_frequency == 0 and i != 0:
                        #print('foo')

                        #pprint(gradient_buffer)
                        feed_dict = dict(zip(agent.gradient_holders, gradient_buffer))
                        sess.run(agent.update_batch, feed_dict=feed_dict)

                        for idx, grad in enumerate(gradient_buffer):
                            gradient_buffer[idx] = grad * 0

                    #print(running_reward)
                    total_reward.append(running_reward)
                    total_length.append(j)
                    break

            if i % report_period == 0 and i != 0:
                print(i)
                counts = pd.Series([action_map[x] for x in episode_history[:, 1]]).value_counts().sort_index()
                for x in counts.iteritems():
                    print(x[0], ':', x[1])
                #pprint(episode_history[:, 1])
                #pprint(episode_history[:, 2])
                #pprint(episode_history[:, 0])
                #print('foo')
                print('Mean reward of last {0}: {1}'.format(report_period, np.mean(total_reward[-report_period:])))
                print('Mean ep length of last {0}: {1}'.format(report_period, np.mean(total_length[-report_period:])))

            if i % 5000 == 0 and i != 0:
                fig, ax = plt.subplots()
                pd.Series(total_reward).rolling(report_period).mean().plot(ax=ax)
                ax2 = ax.twinx()
                pd.Series(total_length).rolling(report_period).mean().plot(ax=ax2, color='red')
                fig.savefig('agent_'+str(i)+'.png')

        #save_inputs = {'state_in': agent.state_in}
        #save_outputs = {'output': agent.output}
        #tf.saved_model.simple_save(sess, 'saved_model', save_inputs, save_outputs)



