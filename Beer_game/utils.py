"""
封装一下 env 和 agents
"""
from tqdm import tqdm


class chain_wrapper:

    def __init__(self, agents, env):
        self.env = env
        self.agents = agents

    def play(self, episode):
        tqdm_e = tqdm(range(epsiode))
        for epi in tqdm_e:
