import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt
import random
import virtualTB

A_LR = 0.001
C_LR = 0.002
GAMMA = 0.9
TAU = 0.01
EXPLORATION_DECAY = 0.9995

ACTOR_HIDDEN_UNITS = [32]
CRITIC_HIDDNE_UNITS = [16, 32]

ENV_NAME = 'VirtualTB-v0'
ENV = gym.make(ENV_NAME)
S_DIM = ENV.observation_space.shape[0]
A_DIM = ENV.action_space.shape[0]
A_UPPER = ENV.action_space.high[0]
A_LOWER = ENV.action_space.low[0]

TOTAL_EPISODES = 2000
STEPS_PER_EPISODE = 200
MEMORY_SIZE = 65536
BATCH_SIZE = 32

PLOT = False
Render = False

class Estimator():
    def __init__(self):
        self.s = tf.placeholder(tf.float32, [None, S_DIM])
        self.s_ = tf.placeholder(tf.float32, [None, S_DIM])
        self.r = tf.placeholder(tf.float32, [None, 1])

        self.a = self._build_actor(self.s, 'actor')
        self.a_ = self._build_actor(self.s_, 'target_actor')
        # 更新actor网络的时候self.a接actor网络, 更新Q网络的时候self.a接外部输入的action
        self.q = self._build_critic(self.s, self.a, 'critic')
        self.q_ = self._build_critic(self.s_, self.a_, 'target_critic')

        self.exploration_var = 3

        self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'actor')
        self.target_a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_actor')
        self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'critic')
        self.target_c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'target_critic')

        self.target_init = [[tf.assign(ta, a), tf.assign(tc, c)] 
            for ta, a, tc, c in zip(self.target_a_params, self.a_params, self.target_c_params, self.c_params)]
        self.target_update = [[tf.assign(ta, (1 - TAU)*ta + TAU*a), tf.assign(tc, (1 - TAU)*tc + TAU*c)]
            for ta, a, tc, c in zip(self.target_a_params, self.a_params, self.target_c_params, self.c_params)]

        actor_loss = -tf.reduce_mean(self.q)
        self.train_actor = tf.train.AdamOptimizer(A_LR).minimize(actor_loss, var_list=self.a_params)

        y = self.r + GAMMA*self.q_
        critic_loss = tf.reduce_mean(tf.square(y - self.q))
        self.train_critic = tf.train.AdamOptimizer(C_LR).minimize(critic_loss, var_list=self.c_params)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.target_init)
        
    def _build_actor(self, s, scope):
        init = tf.random_normal_initializer()
        with tf.variable_scope(scope):
            h = s
            for unit in ACTOR_HIDDEN_UNITS:
                h = tf.layers.dense(h, unit, tf.nn.relu, kernel_initializer=init)
            action = tf.layers.dense(h, A_DIM, tf.nn.tanh, kernel_initializer=init)
        return action*A_UPPER

    def _build_critic(self, s, a, scope):
        # init = tf.random_normal_initializer()
        # with tf.variable_scope(scope):
        #     unit = CRITIC_HIDDNE_UNITS[0]
        #     h_s = tf.layers.dense(s, unit, kernel_initializer=init)
        #     h_a = tf.layers.dense(a, unit, kernel_initializer=init)
        #     h = h_s + h_a
        #     for unit in CRITIC_HIDDNE_UNITS[1:]:
        #         h = tf.layers.dense(h, unit, tf.nn.relu, kernel_initializer=init)
        #     q_value = tf.layers.dense(h, 1, kernel_initializer=init)
        # return q_value
        with tf.variable_scope(scope):
            w1 = tf.get_variable('w1', [S_DIM, 32])
            w2 = tf.get_variable('w2', [A_DIM, 32])
            b1 = tf.get_variable('b1', [1, 32])
            net = tf.nn.relu(tf.matmul(s, w1) + tf.matmul(a, w2) + b1)
            return tf.layers.dense(net, 1)

    def get_action(self, s):
        s = np.array(s)
        single = False
        if len(s.shape) == 1:
            single = True
            s = np.reshape(s, [-1, S_DIM])
        a = self.sess.run(self.a, {self.s: s})
        if single == True:
            a = np.reshape(a, [A_DIM])
        # a = np.clip(np.random.normal(a, self.exploration_var), A_LOWER, A_UPPER)
        a = np.clip(a, A_LOWER, A_UPPER)
        return a

    def learn(self, s, a, r, s_):
        self.exploration_var *= EXPLORATION_DECAY
        self.sess.run(self.train_critic, {self.s: s, self.a: a, self.s_: s_, self.r: r})
        self.sess.run(self.train_actor, {self.s: s})
        self.sess.run(self.target_update)

class Memory():
    def __init__(self):
        self.memory = []
        self.len = 0
    
    def add(self, s, a, r, s_):
        if self.len >= MEMORY_SIZE:
            self.memory.pop(0)
            self.len -= 1
        self.memory.append((s, a, r, s_))
        self.len += 1

    def get_batch(self, batch_size):
        if self.len < batch_size:
            batch = random.sample(self.memory, self.len)
        else:
            batch = random.sample(self.memory, batch_size)
        
        s = np.array([_[0] for _ in batch])
        a = np.array([_[1] for _ in batch])
        r = np.array([[_[2]] for _ in batch])
        s_ = np.array([_[3] for _ in batch])
        return s, a, r, s_

class Agent():
    def __init__(self):
        self.env = gym.make(ENV_NAME)
        self.estimator = Estimator()
        self.memory = Memory()
        self.rewards = []
    
    def run(self):
        plt.ion()
        for eps in range(TOTAL_EPISODES):
            s = self.env.reset()
            reward = 0
            for steps in range(STEPS_PER_EPISODE):
                if Render: self.env.render()

                a = self.estimator.get_action(s)
                s_, r, done, info = self.env.step(a)
                reward += r
                self.memory.add(s, a, r/10, s_)
                s = s_

                s_batch, a_batch, r_batch, s__batch = self.memory.get_batch(BATCH_SIZE)
                self.estimator.learn(s_batch, a_batch, r_batch, s__batch)

                if done: break

            self.rewards.append(reward)
            print("reward %i in episode %i"%(reward, eps + 1))

            if PLOT:
                    plt.clf()
                    plt.plot(self.rewards) # plot rewards
                    plt.xlabel('episode')
                    plt.ylabel('reward')
                    plt.show()
                    plt.pause(0.001)

        plt.clf()
        plt.plot(self.rewards) # plot rewards
        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.show()
        input()
        

if __name__ == '__main__':
    agt = Agent()
    agt.run()
            
