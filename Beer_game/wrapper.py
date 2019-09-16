"""
封装chain类 和 agent类；
解决reward 的问题

"""
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.models import load_model

class chain_wrapper:

    def __init__(self, agents=[None], env=None, agents_list=[]):
        self.env = env
        self.agents = agents
        self.agents_num = len(self.agents)
        self.week_his = []
        self.agents_list = agents_list

    def play_4dqn(self, episode, train_freq=10, save_freq=1000):
        """
        要注意的是，这是多人游戏，所以每次agent_i决策完了获得的s,r,d
        是agent_{i+1}应该的state
        """
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
            cum_r = np.zeros(4)
            api = self.env.start_play()
            # agent 0 's state
            state, r, d = next(api)
            state_ = np.array(state).flatten()[np.newaxis,:]
            # d = False
            # save next state
            now_state_queue = [[], [], [], []]
            r_queue = [[], [], [], []]
            d_queue = [[], [], [], []]
            a_queue = [[], [], [], []]
            while not d:
                # ob : state, reward, done, action, next_state
                for i in range(4):
                    cum_r[i] += r
                    r_queue[i].append(r)
                    d_queue[i].append(d)
                    now_state_queue[i].append(state_)
                    a = self.agents[i].agent.e_greedy_action(state=state_)
                    a_queue[i].append(a)
                    # next_state 是 agent i+1 的 state
                    next_state,r,d = api.send(a)
                    next_state_ = np.array(state).flatten()[np.newaxis,:]
                    state_ = next_state_
            # for i in range(4):
            #     for j in range(len(now_state_queue[0])-1):
            #         # self.obs = obs
            #         self.agents[i].sampling_pool.add_to_buffer(
            #         [now_state_queue[i][j], r_queue[i][j],
            #         d_queue[i][j], a_queue[i][j], now_state_queue[i][j+1]])
                for i in range(4):
                    if i == 0:
                        k = 3
                    else:
                        k = i-1
                    for j in range(len(now_state_queue[0])-1):
                        self.agents[i].sampling_pool.add_to_buffer(
                        [now_state_queue[i][j], r_queue[i][j],
                        d_queue[i][j], a_queue[i][j], now_state_queue[k][j+1]])
            tqdm_e.set_description("Score: " + str(np.sum(cum_r)))
            tqdm_e.refresh()

            for i in range(4):
                self.agents[i].cum_r.append(cum_r[i])
            self.week_his.append(self.env.week)
            # train
            if epi % train_freq == 0:
                for i in range(4):
                    self.agents[i].train_agent()
            if epi % save_freq == 0:
                print("saving models ...")
                for i in range(4):
                    self.agents[i].save_model(f"agent_{i}-epis_{epi}.h5")
        for i in range(4):
            self.agents[i].save_model(f"final-agent_{i}-epis_{epi}.h5")

    def play_4policy(self, episode):
        """
        4个硬策略, 支持4个simple policy 和 3个simple policy和1个ar one
        """
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
            # init
            for i in range(4):
                self.agents[i]._init_agent()
            cum_r = np.zeros(4)
            api = self.env.start_play()
            # agent 0 's state
            state, r, d = next(api)
            # d = False
            r_queue = [[], [], [], []]
            while not d:
                # ob : state, reward, done, action, next_state
                for i in range(4):
                    cum_r[i] += r
                    r_queue[i].append(r)
                    a = self.agents[i].get_action()
                    # next_state 是 agent i+1 的 state
                    next_state,r,d = api.send(a)
            tqdm_e.set_description("Score: " + str(np.sum(cum_r)))
            tqdm_e.refresh()

            for i in range(4):
                self.agents[i].cum_r.append(cum_r[i])
            self.week_his.append(self.env.week)
    
    def play_1dqn_3policy(self, episode, train_freq=10, save_freq=1000):
        """
        2个simple policy, 1个ar_one, 1个dqn(第三个agent),
        """
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
            # init agent
            for i in [0,1,3]:
                self.agents[i]._init_agent()
            cum_r = np.zeros(4)
            api = self.env.start_play()
            # agent 0 's state
            state, r, d = next(api)
            state_ = np.array(state).flatten()[np.newaxis,:]
            # d = False
            # save next state
            now_state_queue = [[], [], [], []]
            r_queue = [[], [], [], []]
            d_queue = [[], [], [], []]
            a_queue = [[], [], [], []]
            while not d:
                # ob : state, reward, done, action, next_state
                for i in range(4):
                    cum_r[i] += r
                    r_queue[i].append(r)
                    d_queue[i].append(d)
                    now_state_queue[i].append(state_)
                    if i == 2:
                        a = self.agents[i].agent.e_greedy_action(state=state_)
                    else:
                        a = self.agents[i].get_action()
                    a_queue[i].append(a)
                    # next_state 是 agent i+1 的 state
                    next_state,r,d = api.send(a)
                    next_state_ = np.array(state).flatten()[np.newaxis,:]
                    state_ = next_state_
            # for i in range(4):
            #     for j in range(len(now_state_queue[0])-1):
            #         # self.obs = obs
            #         self.agents[i].sampling_pool.add_to_buffer(
            #         [now_state_queue[i][j], r_queue[i][j],
            #         d_queue[i][j], a_queue[i][j], now_state_queue[i][j+1]])

                for j in range(len(now_state_queue[0])-1):
                    self.agents[2].sampling_pool.add_to_buffer(
                    [now_state_queue[2][j], r_queue[2][j],
                    d_queue[2][j], a_queue[2][j], now_state_queue[1][j+1]])
            tqdm_e.set_description("Score: " + str(np.sum(cum_r)))
            tqdm_e.refresh()
                self.agents[3].cum_r.append(cum_r[3])
            self.week_his.append(self.env.week)
            # train
            if epi % train_freq == 0:
                for i in range(4):
                    self.agents[i].train_agent()
            if epi % save_freq == 0:
                print("saving models ...")
                for i in range(4):
                    self.agents[i].save_model(f"agent_{i}-epis_{epi}.h5")
        for i in range(4):
            self.agents[i].save_model(f"final-agent_{i}-epis_{epi}.h5")
    
    def play_4step(self, episode, train_freq=10, save_freq=1000):
        """
        4个阶段，1天4个simple policy，25天agent0 换ar1，
                50天agent2换dqn，75天全换dqn。
        """
        
        self.cum_r = [[],[],[],[]]
        self.cum_r_0 = [ [] for i in range(episode)]
        self.cum_r_1 = [ [] for i in range(episode)]
        self.cum_r_2 = [ [] for i in range(episode)]
        self.cum_r_3 = [ [] for i in range(episode)]

        self.r_0 = [ [] for i in range(episode)]
        self.r_1 = [ [] for i in range(episode)]
        self.r_2 = [ [] for i in range(episode)]
        self.r_3 = [ [] for i in range(episode)]

        tqdm_e = tqdm(range(episode))
        for i in range(4):
            self.agents_list[0][i]._init_agent()
        for epi in tqdm_e:
            r_4 = np.zeros(4)
            week = 0
            cum_r = np.zeros(4)
            api = self.env.start_play()
            state, _, _ = next(api)
            d = False
            now_state_queue = [[], [], [], []]
            r_queue = [[], [], [], []]
            d_queue = [[], [], [], []]
            a_queue = [[], [], [], []]
            for i in range(4):
                self.agents[i]._init_agent()
            while not d:
                # 1 , 2 step
                if week < 50:
                    for i in range(4):
                        a = self.agents_list[0][i].get_action(self.env)
                        state,r,d = api.send(a)
                        cum_r[i] += r
                        r_4[i] = r
                # policy 3
                elif week < 75:
                    # ob : state, reward, done, action, next_state
                    for i in [0, 1]:
                        a = self.agents_list[0][i].get_action(self.env)
                        state,r,d = api.send(a)
                        cum_r[i] += r
                        r_4[i] = r
                    for i in [2]:
                        # dqn
                        state_ = np.array(state).flatten()[np.newaxis,:]
                        a = self.agents_list[1][0].agent.e_greedy_action(state=state_)
                        next_state,r,d = api.send(a)
                        cum_r[i] += r
                        r_4[i] = r
                        next_state_ = np.array(next_state).flatten()[np.newaxis,:]
                        obs[i].append([state_, r, d, a, next_state_])
                        self.obs = obs
                        self.agents[i].sampling_pool.add_to_buffer(obs[i][0])

                    for i in [3]:
                        a = self.agents_list[0][3].get_action(self.env)
                        state,r,d = api.send(a)
                        cum_r[i] += r
                        r_4[i] = r

                # policy 4
                else:
                    # ob : state, reward, done, action, next_state
                    for i in range(4):
                        state_ = np.array(state).flatten()[np.newaxis,:]
                        a = self.agents_list[2][i].agent.e_greedy_action(state=state_)
                        state,r,d = api.send(a)
                        cum_r[i] += r
                        r_4[i] = r
                        next_state_ = np.array(next_state).flatten()[np.newaxis,:]
                        obs[i].append([state_, r, d, a, next_state_])
                        self.obs = obs
                        self.agents[i].sampling_pool.add_to_buffer(obs[i][0])

                self.cum_r_0[epi].append(cum_r[0])
                self.cum_r_1[epi].append(cum_r[1])
                self.cum_r_2[epi].append(cum_r[2])
                self.cum_r_3[epi].append(cum_r[3])
                self.r_0[epi].append(r_4[0])
                self.r_1[epi].append(r_4[1])
                self.r_2[epi].append(r_4[2])
                self.r_3[epi].append(r_4[3])

            for i in range(4):
                    self.cum_r[i].append(cum_r[i])

            tqdm_e.set_description("Score: " + str(np.sum(cum_r)))
            tqdm_e.refresh()
            self.week_his.append(self.env.week)
            #  train
            if epi % 10:
                self.agents_list[1][0].train_agent()
                for i in range(4):
                    self.agents_list[2][i].train_agent()




class Agent:

    def __init__(self, policy="simple_policy", step4=False):
        self.policy = policy
        self.cum_r = []
        self.action_his = []
        self._action_his = []
        self.demand_his = []
        self._demand_his = []
        if step4:
            self.exp_weeks = 25
        else:
            self.exp_weeks = 60

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

        if (self.policy == "simple_policy") | (env.week < self.exp_weeks):
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
