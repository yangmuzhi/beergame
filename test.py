from Beer_game.beer_game_env import BeerGameEnv
from dqn.deepqn import DQN
from utils.net import simple_net
import numpy as np
from Beer_game.wrapper import chain_wrapper
import matplotlib.pyplot as plt
import pickle



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



bg = chain_wrapper(agents, env)


bg.play_4dqn(episode=10000)


with open("./data/4dqn_10000.pickle", "rb") as f:
    data = pickle.load(f)

plt.plot(x)
plt.show()


plt.plot(np.array(data[0]))
plt.show()

plt.plot(np.array(data[1]))
plt.show()

plt.plot(np.array(data[2]))
plt.show()

plt.plot(np.array(data[3]))
plt.show()


data = {0:np.array(bg.agents[0].cum_r),
        1:np.array(bg.agents[1].cum_r),
        2:np.array(bg.agents[2].cum_r),
        3:np.array(bg.agents[3].cum_r)}
import pickle
with open("./data/savedata_4dqn_10000.pickle", "wb+") as f:
    pickle.dump(data, f)
