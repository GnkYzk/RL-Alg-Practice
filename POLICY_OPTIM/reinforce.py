from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
class Policy(nn.Module):
    def __init__(self,state_n,action_n):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(state_n,128)
        self.fc2 = nn.Linear(128,action_n)
    def forward(self,x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
class Reinforce:
    def __init__(self,env,discrete_state,discrete_action,n_episodes,discount = 0.95,lr = 3e-4):
        self.env = env
        if discrete_state:
            self.state_n = env.observation_space.n
        else:
            state,info = env.reset()
            self.state_n= len(state)
        if discrete_action:
            self.action_n = env.action_space.n
        else:
            action = env.observation_space.sample(1)
            self.action_n= len(action)
        self.model = Policy(self.state_n,self.action_n)
        self.n_episodes = n_episodes
        self.discount = discount
        self.lr = lr
    def generate_episode(self,start_state):
        state = torch.tensor(start_state)
        done = False
        transitions = []
        while not done:
            probs = self.model(state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            next_state,reward,terminated,truncated,info = self.env.step(action.item())
            transitions.append((log_prob,reward))
            state=torch.tensor(next_state)
            if terminated or truncated:
                done = True
        return transitions
    def train(self):

        optimizer = torch.optim.Adam(self.model.parameters(),lr = self.lr)
        for i_episode in range(self.n_episodes):

            g = 0
            start_state,info = self.env.reset()
            episode = self.generate_episode(start_state)
            t = len(episode)
            loss = torch.tensor(0,dtype = torch.float32)
            for transition in reversed(episode):
                log_prob,reward = transition
                g = g * self.discount + reward
                loss -= g * log_prob

            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        
    
