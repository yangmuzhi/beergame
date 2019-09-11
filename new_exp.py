from Beer_game.utils import chain_wrapper, Agent
from Beer_game.beer_game_env import BeerGameEnv
import numpy as np
import matplotlib.pyplot as plt
from dqn.deepqn import DQN
from utils.net import simple_net
import pickle
from keras.models import load_model

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
env_exp3 = BeerGameEnv(demand_gen, lag=2)
api = env_exp3.start_play()
state, r, d = next(api)
shape = np.array(state).flatten().shape
#
# agents_exp3 = [Agent(policy="ar")]
# agents_exp3.append(Agent())
# agents_exp3.append(DQN(state_shape=shape, n_action=25, net=simple_net))
# agents_exp3.append(Agent())
#
# bg_exp3 = chain_wrapper(agents_exp3, env_exp3)
# bg_exp3.play_mixed(episode=1000)
#
# plt.plot(np.array(bg_exp3.agents[0].cum_r) + np.array(bg_exp3.agents[1].cum_r)
#         + np.array(bg_exp3.agents[2].cum_r) + np.array(bg_exp3.agents[3].cum_r))
# plt.show()
#
# plt.plot(np.array(bg_exp3.agents[2].cum_r))
# plt.show()


# load model policy3 for policy4
model_path = 'models/p3.h5'
dqn_agent = load_model(model_path)
dqn_agent

#
agents_list = []
agents_list.append([Agent(policy='ar',step4=True),Agent(,step4=True),
                    Agent(,step4=True),Agent(,step4=True)])
agents_list.append([Agent()])


bg_exp4 = chain_wrapper(env=BeerGameEnv(demand_gen, lag=2),
                        agents_list=agents_list)
