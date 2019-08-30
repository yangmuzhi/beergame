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
#

env_exp5 = BeerGameEnv(demand_gen, lag=2)

api = env_exp5.start_play()
state, r, d = next(api)
shape = np.array(state).flatten().shape

agents_exp5 = [[Agent(), Agent(), Agent(), Agent()],
               [Agent(policy='ar'), Agent(), Agent(), Agent()],
               [Agent(policy='ar'), Agent(), DQN(state_shape=shape, n_action=25, net=simple_net), Agent()],
               []]

bg_exp5 = chain_wrapper(agents_list=agents_exp5, env=env_exp5)
bg_exp5.play_4_step(episode=1000)

plt.plot(np.array(bg_exp4.agents[0].cum_r) + np.array(bg_exp4.agents[1].cum_r)
        + np.array(bg_exp4.agents[2].cum_r) + np.array(bg_exp4.agents[3].cum_r))
plt.show()
