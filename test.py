from Beer_game.beer_game_env import BeerGameEnv
from dqn.deepqn import DQN
from utils.net import simple_net
import numpy as np



eps = 10
batch_size = 32

def demand():
    while True:
        yield np.random.uniform(10)
demand_gen = demand()
env = BeerGameEnv(demand_gen, lag=2)


api = env.start_play()
agents = []

state, r, d = next(api)
r
d
shape = np.array(state).flatten().shape
shape
state

for i in range(4):
    agents.append(DQN(state_shape=shape, n_action=10, net=simple_net))

from Beer_game.wrapper import chain_wrapper

bg = chain_wrapper(agents, env)


import matplotlib.pyplot as plt


bg.play_4dqn(episode=10000)


x = np.array([bg.agents[0].cum_r, bg.agents[1].cum_r,
bg.agents[2].cum_r,bg.agents[3].cum_r]).sum(axis=0)

plt.plot(x[x>=-5000]

)
plt.show()


plt.plot(np.array(bg.agents[0].cum_r))
plt.show()

plt.plot(np.array(bg.agents[1].cum_r))
plt.show()

plt.plot(np.array(bg.agents[2].cum_r))
plt.show()

plt.plot(np.array(bg.agents[3].cum_r))
plt.show()

import pickle
file = open("./data/savedata.pickle", "rb")
data_dict = pickle.load(file)
file.close()
