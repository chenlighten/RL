import gym
import random
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class Estimator:
    def __init__(self, scope="estimator", num_states=2, num_actions=3, batch_size=32):
        self.scope = scope
        self.num_states = num_states
        self.num_actions = num_actions
        # create network for specific scope
        with tf.variable_scope(scope):
            self.input_states = tf.placeholder(shape=[None, num_states], dtype=tf.float64, name="input_states")
            # the real q value observed, used for training
            self.real_q = tf.placeholder(shape=[None, num_actions], dtype=tf.float64, name="real_q")
            # network layers
            lay1 = tf.layers.dense(self.input_states, 64, activation=tf.nn.relu)
            lay2 = tf.layers.dense(lay1, 64, activation=tf.nn.relu)
            lay3 = tf.layers.dense(lay2, 64, activation=tf.nn.relu)
            # output
            self.output_q = tf.layers.dense(lay3, num_actions)
            # loss
            loss = tf.reduce_mean(tf.squared_difference(self.output_q, self.real_q))
            self.optimizer = tf.train.AdamOptimizer().minimize(loss)
            self.var_init = tf.global_variables_initializer()
            
    def initialize(self, sess):
        sess.run(self.var_init)

    def predict_one(self, sess, states):
        return sess.run(self.output_q, feed_dict={self.input_states:states.reshape(1, self.num_states)})

    def predict_all(self, sess, states):
        return sess.run(self.output_q, feed_dict={self.input_states:states})

    def optimize(self, sess, states_batch, q_batch):
        sess.run(self.optimizer, feed_dict={self.input_states:states_batch, self.real_q:q_batch})

    def copy(self, sess, another):
        e1_params = [t for t in tf.trainable_variables() if t.name.startswith(self.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if t.name.startswith(another.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)
        copy_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e1_v.assign(e2_v)
            copy_ops.append(op)
        sess.run(copy_ops)
        snakeValue

class Memory:
    def __init__(self, max_memory_size=65536):
        self.max_memory_size = max_memory_size
        self.samples = list()

    def add_sample(self, sample):
        self.samples.append(sample)
        if len(self.samples) > self.max_memory_size:
            self.samples.pop(0)

    def get_sample(self, num_samples):
        num_samples = min(len(self.samples), num_samples)
        return random.sample(self.samples, num_samples)

class Agent:
    def __init__(self,
                 env,
                 num_episodes,
                 batch_size,
                 num_states,
                 num_actions,
                 max_eps,
                 min_eps,
                 update_target_network_every, 
                 render=True,
                 plot_every=1):
        self.env = env
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.num_states = num_states
        self.num_actions = num_actions
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.eps = max_eps
        self.render = render
        self.plot_every = plot_every
        self.q_estimator = Estimator(scope="q_estimator")
        self.target_estimator = Estimator(scope="target_estimator")
        self.memory = Memory()
        self.sess = tf.Session()
        self.gamma = 0.99
        self.tot_steps = 0
        self.reward_store = list()
        self.max_x_store = list()
        self.tot_episodes = 0
        self.update_target_network_every = update_target_network_every

    def choose_action(self, state):
        if random.random() < self.eps:
            return random.randint(0, self.num_actions - 1)
        else:
            return np.argmax(self.q_estimator.predict_one(self.sess, state)

    def replay(self):
        batch = self.memory.get_sample(self.batch_size)
        states = np.array([val[0] for val in batch])
        next_states = np.array([(np.zeros(self.num_states)) if val[3] is None else val[3] for val in batch])
        q = self.target_estimator.predict_all(self.sess, states)
        q_d = self.target_estimator.predict_all(self.sess, next_states)
        x = np.zeros((len(batch), self.num_states))
        y = np.zeros((len(batch), self.num_actions))
        for i, b in enumerate(batch):
            state, action, reward, next_state = b[0], b[1], b[2], b[3]
            current_q = q[i]
            if next_state is None:
                current_q[action] = reward
            else:
                current_q[action] = reward + self.gamma * np.amax(q_d[i])
            x[i] = state
            y[i] = current_q
        self.q_estimator.optimize(self.sess, x, y)
        if self.tot_steps % self.update_target_network_every == 0:
            self.target_estimator.copy(self.sess, self.q_estimator)
        return

    def run_once(self):
        state = self.env.reset()
        tot_reward = 0
        max_x = -1000
        this_steps = 0
        
        while True:
            # print the image
            if self.render: env.render()
            
            # choose an action using epsilon-greedy
            action = self.choose_action(state)
            # take it and observe
            next_state, reward, done, info = self.env.step(action)

            # change reward
            #  if next_state[0] >= 0.5:
                #  reward += 10000
            #  elif next_state[0] >= 0.25:
                #  reward += 50
            #  elif next_state[0] >= 0.1:
                #  reward += 10

            # update max_x
            max_x = max(next_state[0], max_x)
            # if completed this game, change next state
            if done: next_state = None

            # put the result into memory as a turple
            self.memory.add_sample((state, action, reward, next_state))
            # using data in memory to train the nexwork
            self.replay()

            # update epsilon
            self.tot_steps += 1
            self.eps = self.min_eps + (self.max_eps - self.min_eps) \
                                * math.exp(-0.0001 * self.tot_steps)

            state = next_state
            tot_reward += reward

            if done:
                self.tot_episodes += 1
                self.reward_store.append(tot_reward)
                self.max_x_store.append(max_x)
                print("Total reward: {}, Eps: {}".format(tot_reward, self.eps))
                return max_x >= 0.5

    def run_all(self):
        self.q_estimator.initialize(self.sess)
        self.target_estimator.initialize(self.sess)
        plt.ion()
        success_time = 0
        while self.tot_episodes < self.num_episodes:
            if self.run_once(): success_time += 1
            else: success_time = 0

            if self.tot_episodes % 10 == 1:
                print("Episode {} of {}, success_time {}".format(self.tot_episodes, self.num_episodes, success_time))
            if self.tot_episodes % self.plot_every == 0: 
                plt.clf()
                plt.plot(self.max_x_store)
                plt.show()
                plt.pause(0.001)
        input()

if __name__ == '__main__':
    env = gym.make("MountainCar-v0")
    agt = Agent(env=env,
                num_episodes=10000,
                batch_size=32,
                num_states=2,
                num_actions=3,
                max_eps=1.0,
                min_eps=0.1,
                update_target_network_every=500,
                render=True,
                plot_every=1)
    agt.run_all()

            
        
        

            

