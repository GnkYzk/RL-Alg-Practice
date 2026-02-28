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
    
class PPO:
    def __init__(self,env,discrete_state,discrete_action,n_episodes,discount = 0.95,lr =1e-4,clip=0.1,rollout_n = 1024,epochs=10,lmbda = 0.95):
        self.env=env
        if discrete_state:
            self.state_n = env.observation_space.n
        else:
            state,info = env.reset()
            self.state_n= len(state)
        if discrete_action:
            self.action_n = env.action_space.n
        else:
            action = env.action_space.sample(1)
            self.action_n= len(action)
        self.actormodel = ActorModel(self.state_n,self.action_n)
        self.criticmodel= CriticModel(self.state_n)
        self.n_episodes=n_episodes
        self.discount=discount
        self.clip = clip
        self.rollout_n = rollout_n
        self.epochs = epochs
        self.lmbda = lmbda
        self.actor_optimizer = torch.optim.Adam(self.actormodel.parameters(),lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.criticmodel.parameters(),lr=lr)
    def train(self):
        for episode in range(self.n_episodes):
            
            transitions= self.rollout()
            states,actions,rewards,dones,values,next_values,log_probs = map(torch.stack,zip(*transitions))
            advantages,returns = self.gae(rewards,dones,values,next_values)
            for _ in range(self.epochs):
                dist = torch.distributions.Categorical(self.actormodel(states))
                new_log_probs = dist.log_prob(actions)
                ratio = torch.exp(new_log_probs-log_probs)
                surr1 = ratio * advantages.detach()
                surr2 = torch.clamp(ratio, 1-self.clip, 1+self.clip) * advantages.detach()

                loss_actor = -torch.min(surr1, surr2).mean()
                loss_critic = (returns - self.criticmodel(states).squeeze()).pow(2).mean()
                self.actor_optimizer.zero_grad()
                loss_actor.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                loss_critic.backward()
                self.critic_optimizer.step()
    def gae(self,rewards,dones,values,next_values):
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        r=0
        g=0
        for step in reversed(range(len(rewards))):
            reward,done,value,next_value = rewards[step],dones[step],values[step],next_values[step]
            change = reward + self.discount* next_value* (1-done.float()) - value
            g = change + self.discount * self.lmbda * (1 - done.float())* g
            advantages[step]=g

        returns = advantages + values
        return advantages.detach(),returns.detach()
    def rollout(self):
        state,info = self.env.reset()
        transitions=[]
        for i in range(self.rollout_n):
            with torch.no_grad():
                probs =  self.actormodel(torch.tensor(state,dtype = torch.float32))
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                value = self.criticmodel(torch.tensor(state,dtype = torch.float32))
            next_state,reward,terminated,truncated,info = self.env.step(action.item())
            return_ += reward
            done = True if terminated or truncated else False
            if done:
                next_value = torch.tensor(0.0)
            else:
                with torch.no_grad():
                    next_value = self.criticmodel(torch.tensor(next_state,dtype = torch.float32))
            
            
            transitions.append((torch.tensor(state,dtype=torch.float32),action.unsqueeze(0),torch.tensor(reward,dtype=torch.float32),torch.tensor(done,dtype = torch.bool),value.detach().view(1),next_value.detach().view(1),log_prob.detach().view(1)))
            if terminated or truncated:
                state,info = self.env.reset()
            else:
                state= next_state
        return transitions