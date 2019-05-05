"""
目的：
sampling pool可以自己打乱顺序
遇到state shape的问题：如batch (10,210,160,3),一个则为（210,160,3)
从sample buffer中取出的对于batch 更新可以直接用。
"""
import numpy as np

class Sampling_Pool:
    def __init__(self,mem_size=20000):
        self.mem_size = mem_size
        self.buffer = {} # buffer list
        self.buffer["state"] = []
        self.buffer["reward"] = []
        self.buffer["done"] = []
        self.buffer["action"] = []
        self.buffer["next_state"] = []
        self.size = self.get_size()

    def add_to_buffer(self, ob):
        # ob : state, reward, done, action, next_state
        self.size = self.get_size()
        if self.size >= self.mem_size:
            self.buffer["state"].pop(0)
            self.buffer["reward"].pop(0)
            self.buffer["done"].pop(0)
            self.buffer["action"].pop(0)
            self.buffer["next_state"].pop(0)
        else:
            self.buffer["state"].append(ob[0])
            self.buffer["reward"].append(ob[1])
            self.buffer["done"].append(ob[2])
            self.buffer["action"].append(ob[3])
            self.buffer["next_state"].append(ob[4])

    def shuffle(self):
        self.size = self.get_size()
        idx = list(range(self.size))
        np.random.shuffle(idx)
        return idx

    def get_sample(self, batch_size=32, shuffle=True):
        if shuffle :
            idx = self.shuffle()[0:batch_size]
        else:
            idx = list(range(self.get_size()))
        state = np.array([self.buffer["state"][i] for i in idx ])
        reward = np.array([self.buffer["reward"][i] for i in idx ])
        done = np.array([self.buffer["done"][i] for i in idx ])
        action = np.array([self.buffer["action"][i] for i in idx ])
        next_state = np.array([self.buffer["next_state"][i] for i in idx ])
        return np.squeeze(state), np.squeeze(reward), np.squeeze(done), np.squeeze(action), np.squeeze(next_state)

    def get_size(self):
        return len(self.buffer["state"])

    def clear(self):
        self.buffer["state"] = []
        self.buffer["reward"] = []
        self.buffer["done"] = []
        self.buffer["action"] = []
        self.buffer["next_state"] = []
