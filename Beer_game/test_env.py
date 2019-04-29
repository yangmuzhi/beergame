from Beer_game.beer_game_env import BeerGameEnv
import numpy as np
import time
import tqdm
def demand():
    while True:
        yield np.random.randint(10)
demand_gen = demand()

env = BeerGameEnv(demand_gen, lag=2)
api = env.start_play()
next(api)
env.on_order
env.on_order
env.trans[-env.lag]
env.lag
env.arr_order
env.arr_order
api.send(0)
env.trans
env.cost
env.week














state = env.reset()

action = np.array([0,0,0,0])
env.stock
env.trans
state, cost, done = env.step(action)


state
cost
done

env.cost_his
env.reset()
def test_time(N = 1000, period = 60):
    start = time.time()
    for ep in range(N):
        env.reset()
        for i in range(period):
            action = np.random.uniform(0,10,4)
            env.step(action)
    end = time.time()
    t = (end - start ) / N

    return t, env.cost_his

N = 10
e = tqdm.tqdm(range(N))
t_sum = []
cost_his = []
for i in e:
    t, cost = test_time()
    t_sum.append(t)
    cost_his.append(cost)
t_mean = np.array(t_sum).mean()
24 * 60 * 60 / t_mean
1 / t_mean


# 515.3
env.cost_his
cost_his = np.array(cost_his).reshape(-1,4)
np.mean(cost_his, axis=0)
