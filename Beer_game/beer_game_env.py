"""
Beer game 啤酒游戏
四个agent
action: 每周向上游订单数量
player: F:制造商；D：经销商; W：批发商；R：零售商
demand: 是一个 generator
"""
import numpy as np

class BeerGameEnv:

    def __init__(self, demand, lag=1):
        self.delay_time = {}
        self.delay_time['slove order'] = 1
        self.delay_time['trans'] = 2
        self.demand_gen = demand
        self.feature_num = 4
        self.lag = lag

    def _init_chain(self):
        """初始化库存"""
        self.stock = np.zeros(4)
        self.order = np.zeros(4)
        self.backorder = np.zeros(4)
        self.trans = [np.zeros(4) for _ in range(self.delay_time['trans'])] # several weeks delay
        self.cost = np.zeros(4)
        self.cost_his = []
        self.week = 0
        self.max_week = 60


    @property
    def done(self):
        d = False
        if self.week >= self.max_week:
            d = True
        return d

    def _deal_after_each_week(self):
        """每周结算, 返回 state，r, d"""
        self.cost_his.append(self.cost.copy())
        for i in range(4):
            self.state[i].pop(-1)
        new_state = []
        for _ in range(4):
            new_state.append(np.zeros(self.feature_num))
        for i in range(4):
            new_state[i][0] = self.stock[i]
            new_state[i][1] = self.backorder[i]
            new_state[i][2] = self.trans[0][i]
            new_state[i][3] = self.order[i]
        for i in range(4):
            self.state[i].append(new_state[i])

        return self.state, self.cost, self.done

    def _init_state(self, m=1):
        """初始化 state """
        self.state = []
        demand = next(self.demand_gen)
        for i in range(4):
            state = []
            for j in range(m):
                state.append(np.zeros(self.feature_num))
            self.state.append(state)
        # print(self.state)
        return self.state[0], self.state[1], self.state[2], self.state[3]

    def _check_stock_and_trans(self, a):
        trans = np.zeros(4)
        self.backorder = np.zeros(4)
        trans[0] = a
        for i in range(4):
            # 库存足够，全部发出,接受库存惩罚
            if self.stock[i] >= self.order[i]:
                if i is not 0:
                    trans[i] = self.stock[i]
                self.stock[i] -= self.order[i]
                assert self.stock[i] >= 0
                self.cost[i] = - self.stock[i]

            else: # 库存不够,把库存全发出，然后接受惩罚
                if i is not 0:
                    trans[i] = self.stock[i]
                self.backorder[i] = - (self.stock[i] - self.order[i])
                assert (self.stock[i] - self.order[i]) < 0
                self.cost[i] = 2 * (self.stock[i] - self.order[i])
                self.stock[i] = 0
        self.trans.append(trans)

    def reset(self):
        self._init_chain()
        state = self._init_state(m=self.lag)
        return state

    def step(self, action):
        """同时决策、结算"""
        demand = next(self.demand_gen)
        self.order[-1] = demand
        self.order[:-1] = action[1:]

        self.stock += self.trans[0]
        self.trans.pop(-1)

        self._check_stock_and_trans(action[0])
        self.week += 1
        state, cost, done = self._deal_after_each_week()

        return state, cost, done
