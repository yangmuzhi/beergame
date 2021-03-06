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
# env = BeerGameEnv(demand_gen, lag=2)

# exp 1
env_exp1 = BeerGameEnv(demand_gen, lag=2)
agents_exp1 = []
for i in range(4):
    agents_exp1.append(Agent())
bg_exp1 = chain_wrapper(agents_exp1, env_exp1)
bg_exp1.play_policy(episode=1000)


# bg.agents[0].demand_his
plt.plot(np.array(bg_exp1.agents[0].cum_r) + np.array(bg_exp1.agents[1].cum_r)
        + np.array(bg_exp1.agents[2].cum_r) + np.array(bg_exp1.agents[3].cum_r))
plt.show()

# exp 2
env_exp2 = BeerGameEnv(demand_gen, lag=2)
agents_exp2 = []
agents_exp2.append(Agent(policy='ar'))
for i in range(3):
    agents_exp2.append(Agent())
bg_exp2 = chain_wrapper(agents_exp2, env_exp2)
bg_exp2.play_policy(episode=1000)
# bg.agents[0].demand_his
plt.plot(np.array(bg_exp2.agents[0].cum_r) + np.array(bg_exp2.agents[1].cum_r)
        + np.array(bg_exp2.agents[2].cum_r) + np.array(bg_exp2.agents[3].cum_r))
plt.show()


# policy 3
env_exp3 = BeerGameEnv(demand_gen, lag=2)
api = env_exp3.start_play()
state, r, d = next(api)
shape = np.array(state).flatten().shape

agents_exp3 = [Agent(policy="ar")]
agents_exp3.append(Agent())
agents_exp3.append(DQN(state_shape=shape, n_action=25, net=simple_net))
agents_exp3.append(Agent())

bg_exp3 = chain_wrapper(agents_exp3, env_exp3)
bg_exp3.play_mixed(episode=1000)

plt.plot(np.array(bg_exp3.agents[0].cum_r) + np.array(bg_exp3.agents[1].cum_r)
        + np.array(bg_exp3.agents[2].cum_r) + np.array(bg_exp3.agents[3].cum_r))
plt.show()

plt.plot(np.array(bg_exp1.agents[2].cum_r))
plt.show()
plt.plot(np.array(bg_exp2.agents[2].cum_r))
plt.show()
plt.plot(np.array(bg_exp3.agents[2].cum_r))
plt.show()
# bg_exp3.agents[2].agent.q_eval_net.save(f'models/p3.h5')
#
# # policy 4
# env_exp4 = BeerGameEnv(demand_gen, lag=2)
# agents_exp4 = []
# agents_exp4.append(DQN(state_shape=(8,), n_action=25, net=simple_net))
# for i in range(4):
#     agents_exp4.append(DQN(state_shape=(8,), n_action=25, net=simple_net))
# bg_exp4 = chain_wrapper(agents_exp4, env_exp4)
# bg_exp4.play(episode=10000)
#
# plt.plot(np.array(bg_exp4.agents[0].cum_r) + np.array(bg_exp4.agents[1].cum_r)
#         + np.array(bg_exp4.agents[2].cum_r) + np.array(bg_exp4.agents[3].cum_r))
# plt.show()
#
#
# ### save as pickle
# data_dict = {"exp1": [bg_exp1.agents[0].cum_r, bg_exp1.agents[1].cum_r, bg_exp1.agents[2].cum_r,bg_exp1.agents[3].cum_r],
#             "exp2": [bg_exp2.agents[0].cum_r, bg_exp2.agents[1].cum_r,bg_exp2.agents[2].cum_r,bg_exp2.agents[3].cum_r],
#             "exp3": [bg_exp3.agents[0].cum_r, bg_exp3.agents[1].cum_r,bg_exp3.agents[2].cum_r,bg_exp3.agents[3].cum_r],}
#
# file = open("./data/savedata_exp_123.pickle", "wb")
# pickle.dump(data_dict , file)
# file.close()
#
# file = open("./data/savedata.pickle", "rb")
# pickle.load(file)
# file.close()

# ---------

agents_list = []
agents_list.append([Agent(policy='ar',step4=True),Agent(step4=True),
                    Agent(step4=True),Agent(step4=True)])
agents_list.append([bg_exp3.agents[2]])
agents_list.append([bg_exp3.agents[2], bg_exp3.agents[2],
                    bg_exp3.agents[2], bg_exp3.agents[2]])

bg_exp5 = chain_wrapper(env=BeerGameEnv(demand_gen, lag=2),
                        agents_list=agents_list)
bg_exp5.play_4_policy(1000)

plt.plot(np.array(bg_exp5.r_2[999]))
plt.show()
len(bg_exp5.r_0[9])
np.array(bg_exp5.r_0).shape


# plt.plot(np.array(bg_exp5.r_0[0][0]))
# plt.show()
# plt.plot(np.array(bg_exp5.r_0[1][1]))
# plt.show()
# plt.plot(np.array(bg_exp5.r_0[2]))
# plt.show()
# plt.plot(np.array(bg_exp5.r_0[3][2]))
# plt.show()
#
# new_policy_3 = {
# 'exp3': [bg_exp3.agents[0].cum_r, bg_exp3.agents[1].cum_r,
#         bg_exp3.agents[2].cum_r,bg_exp3.agents[3].cum_r]}
#
# with open('data/new_policy_3.pickle', 'wb') as f:
#     pickle.dump(new_policy_3, f)
#
# data_dict = {"r_0": bg_exp5.r_0,
#     "r_1": bg_exp5.r_1,
#     "r_2": bg_exp5.r_2,
#     "r_3": bg_exp5.r_3}
#
# with open('data/step4policy.pickle', 'wb') as f:
#     pickle.dump(data_dict, f)
