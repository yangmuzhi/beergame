"""
封装一下 env 和 agents
"""
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression

class Agent:

    def __init__(self, policy="simple_policy"):
        self.policy = policy
        self.cum_r = []
        self.action_his = []
        self._action_his = []
        self.demand_his = []
        self._demand_his = []

    def _init_agent(self):
        if len(self._demand_his):
            self.demand_his.append(self._demand_his)
        if len(self._action_his):
            self.action_his.append(self._action_his)
        self.x = []
        self.y = []
        self._demand_his = []
        self.need_reg = True
        self.stock = 0

    def get_action(self, env):
        """
        根据policy来选择action
        """

        if (self.policy == "simple_policy") | (env.week < 60):
            action = self.simple_policy(env)
            if self.policy == "ar":
                self.x.append(env.demand_now)
        elif self.policy == "ar":
            action = self.AR_one(env)
            # action = self.simple_policy(env)
        self._demand_his.append(env.shangyou_order)
        self._action_his.append(action)

        return action

    def simple_policy(self, env, a=0.1):
        if env.lack <= 0:
            assert env.shangyou_order>= 0, "shangyouorder should  more than 0"
            action = int(env.shangyou_order * (1 + a))
        else:
            assert env.shangyou_order>= 0, "shangyouorder should  more than 0"
            action = env.shangyou_order
        # if action < 0:
            # action = 0
        return action

    def AR_one(self, env):
        # lm 预测的是lag=2之后的需求量
        # 所以现在需要目前的stock减掉lag=0的action，lag=1天的预测值
        if self.need_reg:
            self.y = self.x[2:]
            self.x = self.x[:-2]
            self.lm = LinearRegression().fit(np.array(self.x).reshape(-1,1),
                    np.array(self.y).reshape(-1,1))
            self.need_reg = False
        next2_demand = self.lm.predict(np.array(env.shangyou_order).reshape(-1,1))
        action = next2_demand + env.next_demand + env.shangyou_order - env.agent_stock - env.agent_action_last
        if action < 0:
            action = 0
        assert action >= 0, "action should more than 0"
        env._agent_action_last[env.agent_idx] = action
        env._next_demand[env.agent_idx] = next2_demand
        return action

class chain_wrapper:

    def __init__(self, agents, env):
        self.env = env
        self.agents = agents
        self.agents_num = len(self.agents)
        self.week_his = []

    def play_policy(self, episode=100):
        """
        agent:4 policy
        """
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
            for i in range(4):
                self.agents[i]._init_agent()
            cum_r = np.zeros(4)
            api = self.env.start_play()
            state, _, _ = next(api)
            d = False
            while not d:
                # ob : state, reward, done, action, next_state
                for i in range(4):
                    a = self.agents[i].get_action(self.env)
                    next_state,r,d = api.send(a)
                    cum_r[i] += r
            for i in range(4):
                self.agents[i].cum_r.append(cum_r[i])
            self.week_his.append(self.env.week)

    def play(self, episode, train_freq=10):
        """要注意的是，agent_i 做完决策之后，并不能立刻获得下一个s,r,d.
            需要等到本周完成且agent_{i-1}完成才行 这个需要思考思考"""
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
            # for i in range(4):
            #     self.agents[i]._init_agent()
            cum_r = np.zeros(4)
            api = self.env.start_play()
            state, _, _ = next(api)
            d = False
            while not d:
                obs = [[], [], [], []]
                # ob : state, reward, done, action, next_state
                for i in range(4):
                    state_ = np.array(state).flatten()[np.newaxis,:]
                    a = self.agents[i].agent.e_greedy_action(state=state_)
                    next_state,r,d = api.send(a)
                    cum_r[i] += r
                    next_state_ = np.array(next_state).flatten()[np.newaxis,:]
                    obs[i].append([state_, r, d, a, next_state_])
                for i in range(4):
                    self.obs = obs
                    self.agents[i].sampling_pool.add_to_buffer(obs[i][0])
            for i in range(4):
                self.agents[i].cum_r.append(cum_r[i])
            self.week_his.append(self.env.week)

            # train
            if epi % train_freq == 0:
                for i in range(4):
                    self.agents[i].train_agent()

    def play_mixed(self, episode, train_freq=10):
        """agent0 dqn; agent 1,2,3 rule-based policy"""
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
            for i in range(1,4):
                self.agents[i]._init_agent()
            cum_r = np.zeros(4)
            api = self.env.start_play()
            state, _, _ = next(api)
            d = False
            while not d:
                obs = [[], [], [], []]
                # ob : state, reward, done, action, next_state
                for i in range(1):
                    state_ = np.array(state).flatten()[np.newaxis,:]
                    a = self.agents[i].agent.e_greedy_action(state=state_)
                    next_state,r,d = api.send(a)
                    cum_r[i] += r
                    next_state_ = np.array(next_state).flatten()[np.newaxis,:]
                    obs[i].append([state_, r, d, a, next_state_])
                    self.obs = obs
                    self.agents[i].sampling_pool.add_to_buffer(obs[i][0])
                for i in range(1,4):
                    a = self.agents[i].get_action(self.env)
                    next_state,r,d = api.send(a)
                    cum_r[i] += r

            for i in range(4):
                self.agents[i].cum_r.append(cum_r[i])
            self.week_his.append(self.env.week)

            # train
            if epi % train_freq == 0:
                for i in range(1):
                    self.agents[i].train_agent()
