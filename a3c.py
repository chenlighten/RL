import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
import threading
import os
os.environ["OMP_NUM_THREADS"] = '1'

GAME_NAME = 'Pendulum-v0'
MAX_EPISODES = 10000
MAX_STEPS = 200
UPDATE_GLOBAL_EVERY = 10
ACTION_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
AGT_NUM = 4
GAMMA = 0.9
RENDER = False
PLOT = False

ENV = gym.make(GAME_NAME)
ENV.reset()
STATES_NUM = ENV.observation_space.shape[0]
ACTIONS_NUM = ENV.action_space.shape[0]
ACTION_LOWER = ENV.action_space.low
ACTION_UPPER = ENV.action_space.high

class Estimator:
    def __init__(self, sess, scope, is_global=False, global_estimator=None):
        self.sess = sess
        self.scope = scope
        self.is_global = is_global    

        if is_global:
            with tf.variable_scope(scope):
                self.states_input = tf.placeholder(tf.float32, [None, STATES_NUM], "states_input")
                self.action_params, self.critic_params = self.create_network(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.states_input = tf.placeholder(tf.float32, [None, STATES_NUM], "states_input")
                self.actions_input = tf.placeholder(tf.float32, [None, ACTIONS_NUM], "actions_input")
                self.values_input = tf.placeholder(tf.float32, [None, 1], "values_input")
                mu, sigma, self.values_predict, self.action_params, self.critic_params = self.create_network(scope)

                difference = tf.subtract(self.values_input, self.values_predict, name='error')
                self.critic_loss = tf.reduce_mean(tf.square(difference))

                mu, sigma = mu * ACTION_UPPER, sigma + 1e-4

                normal_distribution = tf.contrib.distributions.Normal(mu, sigma)

                log_pi = normal_distribution.log_prob(self.actions_input)
                expected_value = log_pi * difference
                entropy = normal_distribution.entropy()  # encourage exploration
                self.expedctd_value = 0.01 * entropy + expected_value
                self.action_loss = tf.reduce_mean(-self.expedctd_value)
                # self.action_loss = tf.reduce_mean(-log_pi*difference)

                self.action_choose = tf.clip_by_value(tf.squeeze(normal_distribution.sample(1), axis=0), ACTION_LOWER, ACTION_UPPER)

                self.action_gradiants = tf.gradients(self.action_loss, self.action_params)
                self.critic_gradiants = tf.gradients(self.critic_loss, self.critic_params)

                self.actions_optimizer = tf.train.RMSPropOptimizer(ACTION_LEARNING_RATE)
                self.critic_optimizer = tf.train.RMSPropOptimizer(CRITIC_LEARNING_RATE)  

                self.action_update_local = [local_param.assign(global_param) for local_param, global_param in zip(self.action_params, global_estimator.action_params)]
                self.critic_update_local = [local_param.assign(global_param) for local_param, global_param in zip(self.critic_params, global_estimator.critic_params)]

                self.action_update_global = self.actions_optimizer.apply_gradients(zip(self.action_gradiants, global_estimator.action_params))
                self.critic_update_global = self.critic_optimizer.apply_gradients(zip(self.critic_gradiants, global_estimator.critic_params))
    
    def create_network(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope("action"):
            action_layer1 = tf.layers.dense(self.states_input, 200, tf.nn.relu6, kernel_initializer=w_init, name='action_layer1')
            mu = tf.layers.dense(action_layer1, ACTIONS_NUM, tf.nn.tanh, kernel_initializer=w_init, name='mu')
            sigma = tf.layers.dense(action_layer1, ACTIONS_NUM, tf.nn.softplus, kernel_initializer=w_init, name='sigma')
        with tf.variable_scope("critic"):
            critic_layer1 = tf.layers.dense(self.states_input, 100, tf.nn.relu6, kernel_initializer=w_init, name='critic_layer1')
            value_pridict = tf.layers.dense(critic_layer1, 1, kernel_initializer=w_init, name='value_predict')
        action_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/action')
        critic_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mu, sigma, value_pridict, action_params, critic_params

    def update_local(self):
        self.sess.run([self.action_update_local, self.critic_update_local])

    def update_global(self, states, actions, Rs):
        self.sess.run([self.action_update_global, self.critic_update_global],feed_dict={self.states_input: states, self.actions_input: actions, self.values_input: Rs})

    def choose_action(self, states):
        states = states[np.newaxis, :]
        return self.sess.run(self.action_choose, {self.states_input: states})[0]

class Memory:
    def __init__(self):
        self.states_memory = []
        self.actions_memory = []
        self.rewards_memory = []
        self.R_memory = []
        self.length = 0
    
    def add(self, state, action, reward):
        self.states_memory.append(state)
        self.actions_memory.append(action)
        self.rewards_memory.append(reward)
        self.R_memory.append(0)
        self.length += 1

    def reset(self):
        self.states_memory.clear()
        self.actions_memory.clear()
        self.rewards_memory.clear()
        self.R_memory.clear()
        self.length = 0

class Agent:
    def __init__(self, sess, global_esitimator, index):
        self.sess = sess
        self.env = gym.make(GAME_NAME).unwrapped
        self.global_esitimator = global_esitimator
        self.index = index
        self.all_steps = 0

        global global_rewards
        global global_episodes
        
        self.esitimator = Estimator(
            sess=self.sess, 
            scope="Agent%i"%index,
            is_global=False,
            global_estimator=global_esitimator)

        self.memory = Memory()

    def run_once(self):
        global global_rewards
        global global_episodes
        #self.esitimator.update_local()
        self.memory.reset()
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False
        while not done:
            if RENDER and self.index == 0:
                self.env.render()

            action = self.esitimator.choose_action(state)
            state_next, reward, done, info = self.env.step(action)
            self.memory.add(state, action, (reward + 8)/8)
            episode_reward += reward
            episode_steps += 1
            self.all_steps += 1
            state = state_next
            done = True if episode_steps == MAX_STEPS - 1 else False

            if self.all_steps % UPDATE_GLOBAL_EVERY == 0 or done:
                if done:
                    R = 0
                else:
                    R = self.sess.run(self.esitimator.values_predict, feed_dict={self.esitimator.states_input: state_next[np.newaxis, :]})[0, 0]
                for i in range(self.memory.length - 1, -1, -1):
                    R = self.memory.rewards_memory[i] + GAMMA*R
                    self.memory.R_memory[i] = R
                self.esitimator.update_global(np.vstack(self.memory.states_memory).copy(), np.vstack(self.memory.actions_memory).copy(), np.vstack(self.memory.R_memory).copy())
                self.esitimator.update_local()

            # if self.all_steps % self.update_local_every == 0:
            #     self.esitimator.update_local()

        # if done:
        #     R = 0
        # else:
        #     R = self.sess.run(self.esitimator.values_predict, feed_dict={self.esitimator.states_input: state_next[np.newaxis, :]})[0, 0]
        # for i in range(self.memory.length - 1, -1, -1):
        #     R = self.memory.rewards_memory[i] + self.gamma*R
        #     self.memory.R_memory[i] = R
        # self.esitimator.update_global(np.vstack(self.memory.states_memory), np.vstack(self.memory.actions_memory), np.vstack(self.memory.R_memory))
        # self.esitimator.update_local()

        global_rewards.append(episode_reward)
        if global_episodes > 10:
            global_rewards[-1] = np.mean(global_rewards[-10:])
        #global_average.append((global_average[-1]*global_episodes + episode_reward)/(global_episodes + 1))
        global_episodes += 1
        print("Agent %i get reward %i in episode %i"%(self.index, global_rewards[-1], global_episodes))

        

    def run_all(self):
        global global_rewards
        global global_episodes
        global global_average
        plt.ion()
        while not coord.should_stop() and global_episodes < MAX_EPISODES:
            if (PLOT and self.index == 0):
                plt.clf()
                plt.plot(global_rewards)
                plt.xlabel('episode')
                plt.ylabel('reward')
                plt.show()
                plt.pause(0.001)
            self.run_once()

        if PLOT and self.index == 0:
            input()



if __name__ == '__main__':
    global_rewards = []
    global_episodes = 0
    global_average = [0]

    sess = tf.Session()

    with tf.device("/cpu:0"):
        global_estimator = Estimator(sess=sess, scope='global', is_global=True)
        agents = []
        for i in range(0, AGT_NUM):
            agents.append(Agent(sess=sess, global_esitimator=global_estimator,index=i))

    coord = tf.train.Coordinator()
    sess.run(tf.global_variables_initializer())

    threads = []
    for agent in agents: #start workers
        job = lambda: agent.run_all()
        t = threading.Thread(target=job)
        t.start()
        threads.append(t)
    coord.join(threads) 
    

    plt.plot(np.arange(len(global_rewards)), global_rewards) # plot rewards
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.show()
    input()



        

