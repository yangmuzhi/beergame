from Beer_game.utils import chain_wrapper, Agent
from Beer_game.beer_game_env import BeerGameEnv
import numpy as np
import matplotlib.pyplot as plt
from dqn.deepqn import DQN
from utils.net import simple_net
import pickle

seed = 1234
np.random.seed(seed)
def demand():
    while True:
        d = int(np.random.normal(10, 5))
        if d < 0:
            d = 0
        yield d
demand_gen = demand()

env_exp4 = BeerGameEnv(demand_gen, lag=2)
agents_exp4 = []

for i in range(4):
    agents_exp4.append(DQN(state_shape=(8,), n_action=25, net=simple_net, model_path='models/dqn4'))
bg_exp4 = chain_wrapper(agents_exp4, env_exp4)
bg_exp4.play(episode=1)


plt.plot(np.array(bg_exp4.agents[0].cum_r) + np.array(bg_exp4.agents[1].cum_r)
        + np.array(bg_exp4.agents[2].cum_r) + np.array(bg_exp4.agents[3].cum_r))
plt.savefig('models/cum_r.png')
data_dict = {"exp4": [bg_exp4.agents[0].cum_r, bg_exp4.agents[1].cum_r,
                bg_exp4.agents[2].cum_r,bg_exp4.agents[3].cum_r]}
print('saving data ... ')
with open("data/dqn4.pickle", "wb") as f:
    pickle.dump(data_dict, f)
print('saving data done ... ')
