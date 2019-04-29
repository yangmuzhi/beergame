"""
Beer game 啤酒游戏
四个agent
action: 每周向上游订单数量

player: F:制造商；D：经销商; W：批发商；R：零售商
player与论文一致：按顺序分别为3，2，1，0
注意结算的顺序：
预定订单从0到3开始，有顺序。送货从3到0结算。

可以使用generator的send

demand: 是一个 generator
"""

import numpy as np

class BeerGameEnv:

    def __init__(self, demand, lag=1):
        self.demand_gen = demand
        self.trans_lag = lag
        self.feature_num = 5
        self.end_week = None
        self.lag = lag

    def _init_chain(self):
        """初始化库存"""
        self.stock = np.zeros(4)
        self.stock_minus = np.zeros(4)
        # OO, AO
        self.on_order = [np.zeros(4)]
        self.arr_order = [np.zeros(4)]

        # 正在运过来的
        self.trans = [np.zeros(4) for _ in range(self.trans_lag)] # several weeks delay
        self.cost = np.zeros(4)
        self.cost_his = []
        self.a_his = [np.zeros(4)]
        self.week = 0
        self.max_week = np.random.randint(30,60)

    @property
    def done(self):
        d = False
        if self.week >= self.max_week:
            d = True
        return d

    # def _deal_after_each_week(self):
    #     """已废弃   每周结算, 返回 state，r, d   """
    #     self.cost_his.append(self.cost.copy())
    #     for i in range(4):
    #         self.state[i].pop(-1)
    #     new_state = []
    #     for _ in range(4):
    #         new_state.append(np.zeros(self.feature_num))
    #     for i in range(4):
    #         new_state[i][0] = self.stock[i]
    #         # new_state[i][1] = self.back_order[i]
    #         new_state[i][2] = self.trans[0][i]
    #         new_state[i][3] = self.on_order[i]
    #     for i in range(4):
    #         self.state[i].append(new_state[i])
    #
    #     return self.state, self.cost, self.done

    def _return_state_for_every_agent(self, agent_idx):
        """返回state t+1时刻"""
        self.state[agent_idx][-1][0] = self.stock[agent_idx]
        self.state[agent_idx][-1][1] = self.on_order[-1][agent_idx]
        self.state[agent_idx][-1][2] = self.arr_order[-1][agent_idx]
        self.state[agent_idx][-1][0] = self.trans[-self.trans_lag][agent_idx]

        return self.state[agent_idx],self.cost[agent_idx],self.done

    def _init_state(self, m=1):
        """初始化 state """
        self.state = []
        demand = next(self.demand_gen)
        for i in range(4):
            state = []
            for j in range(m):
                state.append(np.zeros(self.feature_num))
            self.state.append(state)

    def _check_stock_and_trans(self, a):
        """货先到，再发货"""
        trans = np.zeros(4)
        # 计算下个月的到货
        trans[3] = a
        arr_trans = self.trans[-(self.trans_lag)]
        for i in reversed(range(4)):
            # 到货
            self.stock[i] += arr_trans[i]
            self.on_order[-1][i] -= arr_trans[i]

            # 库存足够，全部发出,接受库存惩罚
            if self.stock[i] >= self.arr_order[-2][i]:
                if i is not 3:
                    trans[i] = self.stock[i]
                self.stock[i] -= self.arr_order[-2][i]
                assert self.stock[i] >= 0
                self.cost[i] = - self.stock[i]

            else: # 库存不够,把库存全发出，然后接受惩罚
                if i is not 3:
                    trans[i] = self.stock[i]
                # self.back_order[i] = - (self.stock[i] - self.on_order[i])
                assert (self.stock[i] - self.arr_order[-2][i]) < 0
                self.cost[i] = 2 * (self.stock[i] - self.arr_order[-2][i])
                self.stock[i] = 0
        self.trans.append(trans)

    def _do_ob_end(self):
        self.week += 1
        self.arr_order.append(np.zeros(4))
        self.on_order.append(np.zeros(4))

    def reset(self):
        """reset game"""
        self._init_chain()
        self._init_state(m=self.lag)
        # 保证初始化 OO 不报错
        self.on_order.append(np.zeros(4))
        self.arr_order.append(np.zeros(4))

    def start_play(self):
        """按照0，1，2，3的顺序给actions。结算要按照paper是3，2，1，0的顺序"""
        self.reset()
        while True:
            # 产生需求

            demand = next(self.demand_gen)
            self.arr_order[-1][0] += demand
            # retailer
            a = yield self._return_state_for_every_agent(0) #
            self.on_order[-1][0] = self.on_order[-2][0] + a
            self.arr_order[-1][1] += a

            # W
            a = yield self._return_state_for_every_agent(1)
            self.on_order[-1][1] += self.on_order[-2][1] + a
            self.arr_order[-1][2] += a

            # D
            a = yield self._return_state_for_every_agent(2)
            self.on_order[-1][2] += self.on_order[-2][2] + a
            self.arr_order[-1][3] += a

            # F
            a = yield self._return_state_for_every_agent(1)
            self.on_order[-1][3] += self.on_order[-2][3] + a

            self._check_stock_and_trans(a)
            self._do_ob_end()
