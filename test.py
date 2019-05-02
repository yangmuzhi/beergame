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

env = BeerGameEnv(demand_gen, lag=5)

api = env.start_play()
agents = []

state, r, d = next(api)

shape = np.array(state).flatten().shape


for i in range(4):
    agents.append(DQN(state_shape=shape, n_action=10, net=simple_net))



from Beer_game.utils import chain_wrapper

bg = chain_wrapper(agents, env)
bg.play(episode=100)




while not d:
    
    for i in range(4):
        a = agents[i].agent.e_greedy_action(state=np.array(state).flatten()[np.newaxis,:])
        state,r,d = api.send(a)

