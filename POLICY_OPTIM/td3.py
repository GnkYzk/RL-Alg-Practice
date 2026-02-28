from torch import nn
import torch
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
class Q_model(nn.Module):
    def __init__(self,state_n,action_n):
        super(Q_model,self).__init__()
        self.fc1 = nn.Linear(state_n+action_n,128)
        self.fc2 = nn.Linear(128,1)
    def forward(self,state,action):
        x = torch.cat([state, action], dim=1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x
class Policy(nn.Module):
    def __init__(self,state_n,action_n):
        super(Policy,self).__init__()
        self.fc1 = nn.Linear(state_n,128)
        self.fc2 = nn.Linear(128,action_n)
    def forward(self,x):
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x


class Td3:
    def __init__(self,env,capacity,batch_size,n_episodes,discount = 0.95,lr =1e-4,polyak = 0.95,noise_clip = 0):
        self.env=env
        state,info = env.reset()
        self.state_n= len(state)
        action = env.action_space.sample()
        self.action_n= len(action)
        self.batch_size = batch_size
        self.Q1 = Q_model(self.state_n,self.action_n)
        self.Q2 = Q_model(self.state_n,self.action_n)
        self.target_Q1 = Q_model(self.state_n,self.action_n)
        self.target_Q2 = Q_model(self.state_n,self.action_n)
        self.policy = Policy(self.state_n,self.action_n)
        self.target_policy = Policy(self.state_n,self.action_n)
        self.target_Q1.load_state_dict(self.Q1.state_dict())
        self.target_Q2.load_state_dict(self.Q2.state_dict())
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.buffer = ReplayBuffer(capacity,batch_size)
        self.n_episodes=n_episodes
        self.noise_clip = noise_clip
        self.discount=discount
        self.polyak = polyak
        self.Q1_optimizer = torch.optim.Adam(self.Q1.parameters(),lr=lr)
        self.Q2_optimizer = torch.optim.Adam(self.Q2.parameters(),lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr)
    def replay(self,policy_update):
        batch = self.buffer.sample()
        states,actions,rewards,next_states,dones = map(torch.stack,zip(*batch))
        with torch.no_grad():
            next_actions = self.target_policy(next_states)
            next_actions += torch.clip(0.1 *torch.randn(next_actions.shape),-self.noise_clip,self.noise_clip)
            next_values1 = self.target_Q1(next_states,next_actions)
            next_values2 = self.target_Q2(next_states,next_actions)
        values1 = self.Q1(states,actions)
        values2 = self.Q2(states,actions)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1).float()
        targets = rewards + self.discount * (1 - dones)*torch.min(next_values1, next_values2)
        loss_Q1 = (values1-targets).pow(2).mean()
        loss_Q2 = (values2-targets).pow(2).mean()
        actions_pred = self.policy(states)
        loss_policy = -self.Q1(states, actions_pred).mean()
        self.Q1_optimizer.zero_grad()
        loss_Q1.backward()
        self.Q1_optimizer.step()
        self.Q2_optimizer.zero_grad()
        loss_Q2.backward()
        self.Q2_optimizer.step()
        if policy_update:
            self.policy_optimizer.zero_grad()
            loss_policy.backward()
            self.policy_optimizer.step()


    def train(self):
        for i_episode in range(self.n_episodes):
            state,info = self.env.reset()
            done=False
            policy_update=True
            while not done:
                with torch.no_grad():
                    action = self.policy(torch.tensor(state,dtype=torch.float32).unsqueeze(0))#is clipped in gymnasium environment with ClipAction
                action +=torch.clip(0.1 *torch.randn(action.shape),-self.noise_clip,self.noise_clip)
                next_state,reward,terminated,truncated,info = self.env.step(action.squeeze(0).numpy())
                if terminated or truncated:
                    done =True
                self.buffer.add((torch.tensor(state,dtype=torch.float32),action.squeeze(0).detach(),torch.tensor(reward,dtype=torch.float32),torch.tensor(next_state,dtype=torch.float32),torch.tensor(done,dtype=torch.bool)))
                if len(self.buffer)> self.batch_size:
                    self.replay(policy_update)
                    policy_update= not policy_update
                if policy_update:
                    Q1_state_dict = self.Q1.state_dict()
                    target_Q1_state_dict = self.target_Q1.state_dict()
                    Q2_state_dict = self.Q2.state_dict()
                    target_Q2_state_dict = self.target_Q2.state_dict()
                    policy_state_dict = self.policy.state_dict()
                    target_policy_state_dict = self.target_policy.state_dict()
                    for key in Q1_state_dict:
                        target_Q1_state_dict[key] = target_Q1_state_dict[key]*self.polyak + Q1_state_dict[key]*(1-self.polyak)
                    self.target_Q1.load_state_dict(target_Q1_state_dict)
                    for key in Q2_state_dict:
                        target_Q2_state_dict[key] = target_Q2_state_dict[key]*self.polyak + Q2_state_dict[key]*(1-self.polyak)
                    self.target_Q2.load_state_dict(target_Q2_state_dict)
                    for key in policy_state_dict:
                        target_policy_state_dict[key] = target_policy_state_dict[key]*self.polyak + policy_state_dict[key]*(1-self.polyak)
                    self.target_policy.load_state_dict(target_policy_state_dict)


                state=next_state