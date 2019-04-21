#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

函数：玩游戏。
render显示环境时 速度会放慢，让人眼看清
N 玩的次数
最后绘制累计reward
"""
import time
import matplotlib.pyplot as plt
import numpy as np

def play(env, choose_action, N=100, render=False):
    """
    args
        choose_action: 选择action的函数
        N: 玩的次数
        render: 显示环境时 速度会放慢，让人眼看清

    """
    r = []
    for i in range(N):
        state = env.reset()

        cum_r = 0
        done = False
        while not done:
            if render:
                env.render()
                time.sleep(0.1)
            state = state[np.newaxis,:]
            action = np.argmax(choose_action(state))
            state, reward, done, _ = env.step(action)
            cum_r += reward
        r.append(cum_r)
    plt.plot(range(len(r)), np.array(r))
