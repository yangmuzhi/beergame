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

    def play(self, episode, train_freq=10, save_freq=1000):
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
            for i in range(4):
                for j in range(len(now_state_queue[0])-1):
                    # self.obs = obs
                    self.agents[i].sampling_pool.add_to_buffer(
                    [now_state_queue[i][j], r_queue[i][j],
                    d_queue[i][j], a_queue[i][j], now_state_queue[i][j+1]])
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
