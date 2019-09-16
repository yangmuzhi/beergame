"""
生产厂家会出现雪崩，建议厂家使用保守的策略
"""


from Beer_game.beer_game_env import BeerGameEnv
from dqn.deepqn import DQN
from utils.net import simple_net
import numpy as np
from Beer_game.wrapper import chain_wrapper, Agent
import matplotlib.pyplot as plt
eps = 10
batch_size = 32

def demand():
    while True:
        yield np.random.uniform(10)
demand_gen = demand()
env = BeerGameEnv(demand_gen, lag=2)
api = env.start_play()

state, r, d = next(api)
shape = np.array(state).flatten().shape


# 4 policy
agents = []
for i in range(4):
    agents.append(Agent())

bg = chain_wrapper(agents, env)
bg.play_4policy(1000)

# 3policy 2 ar1
agents = []
agents.append(Agent(policy='ar'))
agents.append(Agent())
agents.append(Agent())
agents.append(Agent())
bg = chain_wrapper(agents, env)
bg.play_4policy(1000)


plt.plot(
np.array([bg.agents[0].cum_r, bg.agents[1].cum_r,
bg.agents[2].cum_r,bg.agents[3].cum_r]).sum(axis=0)

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




# 3 policy 1 dqn
agents = []
agents.append(Agent(policy='ar'))
agents.append(Agent())
agents.append(DQN(state_shape=shape, n_action=10, net=simple_net))
agents.append(Agent())
bg = chain_wrapper(agents, env)
bg.play_1dqn_3policy(5000)

# 3 dqn 1 ar1
agents = []
agents.append(DQN(state_shape=shape, n_action=10, net=simple_net))
agents.append(DQN(state_shape=shape, n_action=10, net=simple_net))
agents.append(DQN(state_shape=shape, n_action=10, net=simple_net))
agents.append(Agent(policy='ar'))
bg = chain_wrapper(agents, env)

bg.play_3dqn(1000)



plt.plot(
np.array([bg.agents[0].cum_r, bg.agents[1].cum_r,
bg.agents[2].cum_r,bg.agents[3].cum_r]).sum(axis=0)

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
