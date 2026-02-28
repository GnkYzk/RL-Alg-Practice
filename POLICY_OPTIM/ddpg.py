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
        return nn.Tanh()(x)


class Ddpg:
    def __init__(self,env,capacity,batch_size,n_episodes,discount = 0.95,lr =1e-4,polyak = 0.95):
        self.env=env
        state,info = env.reset()
        self.state_n= len(state)
        action = env.action_space.sample()
        self.action_n= len(action)
        self.batch_size = batch_size
        self.Q = Q_model(self.state_n,self.action_n)
        self.target_Q = Q_model(self.state_n,self.action_n)
        self.policy = Policy(self.state_n,self.action_n)
        self.target_policy = Policy(self.state_n,self.action_n)
        self.target_Q.load_state_dict(self.Q.state_dict())
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.buffer = ReplayBuffer(capacity,batch_size)
        self.n_episodes=n_episodes
        self.discount=discount
        self.polyak = polyak
        self.Q_optimizer = torch.optim.Adam(self.Q.parameters(),lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(),lr=lr)
    def replay(self):
        batch = self.buffer.sample()
        states,actions,rewards,next_states,dones = map(torch.stack,zip(*batch))
        with torch.no_grad():
            next_actions = self.target_policy(next_states)
            next_values = self.target_Q(next_states,next_actions)
        values = self.Q(states,actions)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1).float()
        targets = rewards + self.discount * (1 - dones)*next_values
        loss_Q = (values-targets).pow(2).mean()
        actions_pred = self.policy(states)
        loss_policy = -self.Q(states, actions_pred).mean()
        self.Q_optimizer.zero_grad()
        loss_Q.backward()
        self.Q_optimizer.step()
        self.policy_optimizer.zero_grad()
        loss_policy.backward()
        self.policy_optimizer.step()


    def train(self):
        for i_episode in range(self.n_episodes):
            state,info = self.env.reset()
            done=False
            while not done:
                with torch.no_grad():
                    action = self.policy(torch.tensor(state,dtype=torch.float32).unsqueeze(0))#is clipped in gymnasium environment with ClipAction
                action +=0.1 *torch.randn(action.shape)
                next_state,reward,terminated,truncated,info = self.env.step(action.squeeze(0).numpy())
                if terminated or truncated:
                    done =True
                self.buffer.add((torch.tensor(state,dtype=torch.float32),action.squeeze(0).detach(),torch.tensor(reward,dtype=torch.float32),torch.tensor(next_state,dtype=torch.float32),torch.tensor(done,dtype=torch.bool)))
                if len(self.buffer)> self.batch_size:
                    self.replay()

                Q_state_dict = self.Q.state_dict()
                target_Q_state_dict = self.target_Q.state_dict()
                policy_state_dict = self.policy.state_dict()
                target_policy_state_dict = self.target_policy.state_dict()
                for key in Q_state_dict:
                    target_Q_state_dict[key] = target_Q_state_dict[key]*self.polyak + Q_state_dict[key]*(1-self.polyak)
                self.target_Q.load_state_dict(target_Q_state_dict)

                for key in policy_state_dict:
                    target_policy_state_dict[key] = target_policy_state_dict[key]*self.polyak + policy_state_dict[key]*(1-self.polyak)
                self.target_policy.load_state_dict(target_policy_state_dict)


                state=next_state