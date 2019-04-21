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

env = BeerGameEnv(demand_gen, lag=1)

state = env.reset()

agents = []
shape = np.array(state[0]).shape
for i in range(4):
    agents.append(DQN(state_shape=(4,), n_action=10, net=simple_net))


for ep in range(eps):
    state = env.reset()

    d = False
    while not d:
        a = []
        for i in range(4):
            a.append(agents[i].agent.e_greedy_action(np.squeeze(np.array(state[i])).reshape(4,)))
        state, cost, d = env.step(a)
        
        
        
    





