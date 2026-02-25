from torch import nn
import torch
import matplotlib.pyplot as plt
import numpy as np
class ActorModel(nn.Module):
    def __init__(self,state_n,action_n):
        super(ActorModel,self).__init__()
        self.fc1 = nn.Linear(state_n,128)
        self.fc2 = nn.Linear(128,action_n)
    def forward(self,x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return torch.softmax(x, dim=-1)
class CriticModel(nn.Module):
    def __init__(self,state_n):
        super(CriticModel,self).__init__()
        self.fc1 = nn.Linear(state_n,128)
        self.fc2 = nn.Linear(128,1)
    def forward(self,x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
class ActorCritic:
    def __init__(self,env,discrete_state,discrete_action,n_episodes,discount = 0.95,lr =1e-4):
        self.env=env
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
        self.actormodel = ActorModel(self.state_n,self.action_n)
        self.criticmodel= CriticModel(self.state_n)
        self.n_episodes=n_episodes
        self.discount=discount
        self.actor_optimizer = torch.optim.Adam(self.actormodel.parameters(),lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.criticmodel.parameters(),lr=lr)
    def train(self):
        for i_episode in range(self.n_episodes):
            done = False
            state,info = self.env.reset()
            while not done:
                
                probs = self.actormodel(torch.tensor(state,dtype=torch.float32))
                dist = torch.distributions.Categorical(probs)
                action=dist.sample()
                log_prob = dist.log_prob(action)
                next_state,reward,terminated,truncated,info = self.env.step(action.item())
                if terminated or truncated:
                    done = True
                vs = self.criticmodel(torch.tensor(state,dtype=torch.float32))
                if done:
                    vns = torch.tensor(0.0)
                else:
                    vns = self.criticmodel(torch.tensor(next_state,dtype=torch.float32))
                advantage = reward + self.discount*vns - vs
                loss_actor = -log_prob * advantage.detach()
                loss_critic = advantage.pow(2)
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()
                state = next_state
