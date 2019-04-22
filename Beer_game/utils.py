"""
封装一下 env 和 agents
"""
from tqdm import tqdm


class chain_wrapper:

    def __init__(self, agents, env):
        self.env = env
        self.agents = agents
        self.agents_num = len(self.agents)

    def play(self, episode):
        tqdm_e = tqdm(range(epsiode))
        for epi in tqdm_e:
            state = self.env.reset()
            d = False
            while not d:
                a = []
                for i in range(self.agents_numself.agents_num):
                    a.append(self.agents[i].agent.e_greedy_action(state[i]))
                next_state, cost, d = self.env.step(a)
                for i in range(self.agents_num):
                    ob = (state[i], cost[i], d[i], a[i], next_state[i])
                    self.agents[i].sampling_pool.add_to_buffer(ob)

            for i in range(self.agents_num):
                self.agents[i].train_agent()
            state = next_state
