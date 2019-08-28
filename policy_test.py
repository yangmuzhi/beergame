"""

需求一个分布：
规则 缺货时增加a%

demand miu’50 sigma20 正太

1）
写死的规则：
不够卖 增加a%  建议10%

2）修改策略

零售商 ：AR one 模型，其他的 不变（希望可以零售商降低，其他人的变高）

3）加入dqn
4）全部dqn（加入通信）


# miu 50
# miu 100
# 正态分布0.9 正态分布0.1

"""

from Beer_game.beer_game_env import BeerGameEnv
from Beer_game.utils import chain_wrapper, Agent
import numpy as np
from sklearn.linear_model import LinearRegression


# int(np.random.normal(10))
seed = 1234
np.random.seed(seed)

def demand():
    while True:
        d = int(np.random.normal(10, 5))
        if d < 0:
            d = 0
        yield d
demand_gen = demand()
next(demand_gen)
env = BeerGameEnv(demand_gen, lag=2)
# ---------------------------- policy ----------------------------

# 针对零售商 ar1 回归模型,
# 预测一个滞后期的数据
#

agents = []
for i in range(4):
    agents.append(Agent())

# agent = Agent(policy=simple_policy)
agents = []
agents.append(Agent(policy='ar'))
for i in range(3):
    agents.append(Agent())

bg = chain_wrapper(agents, env)
bg.play_policy()
bg.agents[0].cum_r
bg.agents[0].action_his[6]
env.shangyou_order

list(range(1,4))
