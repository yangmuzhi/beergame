"""
封装一下 env 和 agents
"""

from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
from keras.models import load_model

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

class chain_wrapper:

    def __init__(self, agents=[None], env=None, agents_list=[]):
        self.env = env
        self.agents = agents
        self.agents_num = len(self.agents)
        self.week_his = []
        self.agents_list = agents_list

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
            # print("cum_r  : {}".format(cum_r))
            self.week_his.append(self.env.week)

    def play(self, episode, train_freq=10, save_freq=1000):
        """要注意的是，agent_i 做完决策之后，并不能立刻获得下一个s,r,d.
            需要等到本周完成且agent_{i-1}完成才行 这个需要思考思考"""
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
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
                    # next_state_ = np.array(next_state).flatten()[np.newaxis,:]
                    obs[i].append([state_, r, d, a, next_state_])
                    state = next_state
                for i in range(4):
                    self.obs = obs
                    self.agents[i].sampling_pool.add_to_buffer(obs[i][0])
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

    def play_mixed(self, episode, train_freq=10):
        """agent0 ar1; agent2 dqn; agent 1,3 rule-based policy"""
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
            for i in [0,1,3]:
                self.agents[i]._init_agent()
            cum_r = np.zeros(4)
            api = self.env.start_play()
            state, _, _ = next(api)
            d = False
            while not d:
                obs = [[], [], [], []]
                # ob : state, reward, done, action, next_state
                for i in [0, 1]:
                    a = self.agents[i].get_action(self.env)
                    next_state,r,d = api.send(a)
                    cum_r[i] += r
                for i in [2]:
                    state_ = np.array(state).flatten()[np.newaxis,:]
                    a = self.agents[i].agent.e_greedy_action(state=state_)
                    next_state,r,d = api.send(a)
                    cum_r[i] += r
                    next_state_ = np.array(next_state).flatten()[np.newaxis,:]
                    obs[i].append([state_, r, d, a, next_state_])
                    self.obs = obs
                    self.agents[i].sampling_pool.add_to_buffer(obs[i][0])
                for i in [3]:
                    a = self.agents[i].get_action(self.env)
                    next_state,r,d = api.send(a)
                    cum_r[i] += r

            for i in range(4):
                self.agents[i].cum_r.append(cum_r[i])
            self.week_his.append(self.env.week)

            # train
            if epi % train_freq == 0:
                for i in [2]:
                    self.agents[i].train_agent()


    def play_4_policy(self, episode=10):
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

    # def play_(self, epsiode=100):
    #     """前50期 polciy2， 50至75 policy3
    #     """
    #     self.cum_r = [[],[],[],[]]
    #     self.cum_r_0 = [ [] for i in range(episode)]
    #     self.cum_r_1 = [ [] for i in range(episode)]
    #     self.cum_r_2 = [ [] for i in range(episode)]
    #     self.cum_r_3 = [ [] for i in range(episode)]
    #
    #     self.r_0 = [ [] for i in range(episode)]
    #     self.r_1 = [ [] for i in range(episode)]
    #     self.r_2 = [ [] for i in range(episode)]
    #     self.r_3 = [ [] for i in range(episode)]
    #
    #     tqdm_e = tqdm(range(episode))
    #     for i in range(4):
    #         self.agents_list[0][i]._init_agent()
    #     for epi in tqdm_e:
    #         r_4 = np.zeros(4)
    #         week = 0
    #         cum_r = np.zeros(4)
    #         api = self.env.start_play()
    #         state, _, _ = next(api)
    #         d = False
    #         while not d:
    #             # 1 , 2 step
    #             if week < 50:
    #                 for i in range(4):
    #                     a = self.agents_list[0][i].get_action(self.env)
    #                     state,r,d = api.send(a)
    #                     cum_r[i] += r
    #                     r_4[i] = r
    #             # policy 3
    #             elif week < 75:
    #                 # ob : state, reward, done, action, next_state
    #                 for i in [0, 1]:
    #                     a = self.agents_list[0][i].get_action(self.env)
    #                     state,r,d = api.send(a)
    #                     cum_r[i] += r
    #                     r_4[i] = r
    #                 for i in [2]:
    #                     state_ = np.array(state).flatten()[np.newaxis,:]
    #                     a = self.agents_list[1][0].agent.e_greedy_action(state=state_)
    #                     state,r,d = api.send(a)
    #                     cum_r[i] += r
    #                     r_4[i] = r
    #                 for i in [3]:
    #                     a = self.agents_list[0][3].get_action(self.env)
    #                     state,r,d = api.send(a)
    #                     cum_r[i] += r
    #                     r_4[i] = r
    #             # policy 4
    #             else:
    #                 # ob : state, reward, done, action, next_state
    #                 for i in range(4):
    #                     state_ = np.array(state).flatten()[np.newaxis,:]
    #                     a = self.agents_list[2][i].agent.e_greedy_action(state=state_)
    #                     state,r,d = api.send(a)
    #                     cum_r[i] += r
    #                     r_4[i] = r
    #             self.cum_r_0[epi].append(cum_r[0])
    #             self.cum_r_1[epi].append(cum_r[1])
    #             self.cum_r_2[epi].append(cum_r[2])
    #             self.cum_r_3[epi].append(cum_r[3])
    #             self.r_0[epi].append(r_4[0])
    #             self.r_1[epi].append(r_4[1])
    #             self.r_2[epi].append(r_4[2])
    #             self.r_3[epi].append(r_4[3])
    #
    #         for i in range(4):
    #                 self.cum_r[i].append(cum_r[i])
    #         self.week_his.append(self.env.week)
    #         tqdm_e.set_description("Score: " + str(np.sum(cum_r)))
    #         tqdm_e.refresh()
