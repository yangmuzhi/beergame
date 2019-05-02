"""
封装一下 env 和 agents
"""
from tqdm import tqdm
import numpy as np


class chain_wrapper:

    def __init__(self, agents, env):
        self.env = env
        self.agents = agents
        self.agents_num = len(self.agents)

    def play(self, episode, train_freq=10):
        """要注意的是，agent_i 做完决策之后，并不能立刻获得下一个s,r,d.
            需要等到本周完成且agent_{i-1}完成才行 这个需要思考思考"""
        tqdm_e = tqdm(range(episode))
        for epi in tqdm_e:
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
                    obs[i].append([state_, r, d, a, next_state])
                for i in range(4):
                    self.obs = obs
                    self.agents[i].sampling_pool.add_to_buffer(obs[i])

            # train
            if epi % train_freq == 0:
                for i in range(4):
                    self.agents[i].train_agent()
