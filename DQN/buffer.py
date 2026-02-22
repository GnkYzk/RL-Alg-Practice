from collections import deque
import random
class ReplayBuffer:
    def __init__(self,maxlen,batch_size):
        self.buffer = deque([],maxlen)
        self.batch_size=batch_size
    def __len__(self):
        return len(self.buffer)
    def sample(self):
        return random.sample(self.buffer,self.batch_size)
    def add(self,transition):
        self.buffer.append(transition)