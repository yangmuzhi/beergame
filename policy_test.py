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

data = {0:np.array(bg.agents[0].cum_r),
        1:np.array(bg.agents[1].cum_r),
        2:np.array(bg.agents[2].cum_r),
        3:np.array(bg.agents[3].cum_r)}
import pickle
with open("./data/4simple_policy_1000.pickle", "wb+") as f:
    pickle.dump(data, f)


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



# 3policy 2 ar1
agents = []
agents.append(Agent(policy='ar'))
agents.append(Agent())
agents.append(Agent())
agents.append(Agent())
bg = chain_wrapper(agents, env)
bg.play_4policy(1000)

np.array(bg.agents[2].cum_r).mean()

data = {0:np.array(bg.agents[0].cum_r),
        1:np.array(bg.agents[1].cum_r),
        2:np.array(bg.agents[2].cum_r),
        3:np.array(bg.agents[3].cum_r)}
import pickle
with open("./data/3simple_policy_1ar_1000.pickle", "wb+") as f:
    pickle.dump(data, f)



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
bg.play_1dqn_3policy(1000)


data = {0:np.array(bg.agents[0].cum_r),
        1:np.array(bg.agents[1].cum_r),
        2:np.array(bg.agents[2].cum_r),
        3:np.array(bg.agents[3].cum_r)}
import pickle
with open("./data/2simple_policy_1ar_1dqn_1000.pickle", "wb+") as f:
    pickle.dump(data, f)
np.array(bg.agents[2].cum_r).mean()

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

# 4 step
agents_list = [[],[],[]]
agents_list[0].append(Agent(policy='ar'))
for i in range(3):
    agents_list[0].append(Agent())
agents_list[1].append(DQN(state_shape=shape, n_action=10, net=simple_net))
for i in range(4):
    agents_list[2].append(DQN(state_shape=shape, n_action=10, net=simple_net))

bg = chain_wrapper(env=env, agents_list=agents_list)
bg.play_4step(1000)


plt.plot(np.array(bg.r_0[-5:]).sum(axis=0))
plt.show()
plt.plot(np.array(bg.r_1[-5:]).sum(axis=0))
plt.show()
plt.plot(np.array(bg.r_2[-5:]).sum(axis=0))
plt.show()
plt.plot(np.array(bg.r_3[-5:]).sum(axis=0))
plt.show()



plt.plot(np.array(bg.cum_r_0[-5:]).sum(axis=0))
plt.show()
plt.plot(np.array(bg.cum_r_1[-5:]).sum(axis=0))
plt.show()
plt.plot(np.array(bg.cum_r_2[-5:]).sum(axis=0))
plt.show()
plt.plot(np.array(bg.cum_r_3[-5:]).sum(axis=0))
plt.show()

data = {0:np.array(bg.cum_r_0),
        1:np.array(bg.cum_r_1),
        2:np.array(bg.cum_r_2),
        3:np.array(bg.cum_r_3),
        "0":np.array(bg.r_0),
        "1":np.array(bg.r_1),
        "2":np.array(bg.r_2),
        "3":np.array(bg.r_3),}


import pickle
with open("./data/4step.pickle", "wb+") as f:
    pickle.dump(data, f)
