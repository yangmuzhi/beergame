#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

dqn 训练cartpole

"""

import gym
import sys
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from dqn.deepqn import DQN
from utils.net import simple_net

eps = int(sys.argv[1])
batch_size = 128

dqn = DQN(state_shape=4, n_action=2, net=simple_net)
dqn.agent.q_eval_net.summary()
env = gym.make("CartPole-v0")

dqn.train(env, eps, batch_size)

#用训后模型测试
def play(N=200):
    r = []
    tqdm_e = tqdm(range(N))
    for i in tqdm_e:
        state = env.reset()
        cum_r = 0
        done = False
        while not done:
            state = state.reshape(-1,4)
            action = np.argmax(dqn.agent.q_eval(state))
            state, reward, done, _ = env.step(action)
            cum_r += reward
        r.append(cum_r)
    plt.plot(range(len(r)), np.array(r))
play(200)
