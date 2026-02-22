from torch import nn
from .buffer import ReplayBuffer
import random
import torch
import logging
class Model(nn.Module):
    def __init__(self,state_n,action_n):
        super(Model,self).__init__()
        self.fc1 = nn.Linear(state_n, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_n)
        self.relu = nn.ReLU()
    def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
class DDQN:
    def __init__(self,env,discrete,maxlen,batch_size,n_epsiodes,discount=0.99,lr = 3e-4,tau=0.005):
        self.env = env
        self.batch_size = batch_size
        self.action_n = env.action_space.n
        if discrete:
            self.state_n = env.observation_space.n
        else:
            state,info = env.reset()
            self.state_n = len(state)
        self.model = Model(self.state_n,self.action_n)
        self.target = Model(self.state_n,self.action_n)
        self.target.load_state_dict(self.model.state_dict())
        self.buffer = ReplayBuffer(maxlen,batch_size)
        self.n_episodes=n_epsiodes
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.997
        self.discount=discount
        self.lr = lr
        self.optimizer = torch.optim.Adam(params=self.model.parameters(),lr =lr )
        self.criterion = nn.SmoothL1Loss()
        self.tau=tau
    def choose_action(self,state):
        if random.random()<self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            s = self.normalize_state(state)
            return torch.argmax(self.model(s)).numpy()
    def normalize_state(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        state[0] /= 4.8      # cart position
        state[1] /= 5.0      # cart velocity
        state[2] /= 0.418    # pole angle (~24 degrees)
        state[3] /= 5.0      # pole angular velocity
        return state
    def experience_replay(self):
        if len(self.buffer)<self.batch_size:
            return
        batch = self.buffer.sample() #B,T where T is tuple (S,A,R,NS)
        states, actions, rewards, next_states,done = map(torch.stack, zip(*batch))
        states = states.float()
        next_states = next_states.float()
        rewards = rewards.float()
        q= self.model(states).gather(1,actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(1)
            next_q = self.target(next_states)\
                .gather(1, next_actions.unsqueeze(1))\
                .squeeze(1)
            t = rewards + self.discount * next_q *(1-done)
            t = t.unsqueeze(1)
        loss = self.criterion(q,t)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

    def train(self):
        print("start")
        for episode in range(self.n_episodes):
            done =False
            state,info = self.env.reset()
            while not done:
                action = self.choose_action(state)
                next_state,reward,terminated,truncated,info = self.env.step(action)
                if terminated or truncated:
                    done = True
                self.buffer.add((self.normalize_state(state),torch.tensor(action,dtype=torch.long),torch.tensor(reward),self.normalize_state(next_state),torch.tensor(done,dtype=torch.float32)))
                self.experience_replay()
                state = next_state
            target_net_state_dict = self.target.state_dict()
            policy_net_state_dict = self.model.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
            self.target.load_state_dict(target_net_state_dict)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            if "episode" in info:
                episode_data = info["episode"]
                logging.info(f"Episode {episode}: "
                            f"reward={episode_data['r']:.1f}, "
                            f"length={episode_data['l']}, "
                            f"time={episode_data['t']:.2f}s")

                # Additional analysis for milestone episodes
                if episode % 1000 == 0:
                    # Look at recent performance (last 100 episodes)
                    recent_rewards = list(self.env.return_queue)[-100:]
                    if recent_rewards:
                        avg_recent = sum(recent_rewards) / len(recent_rewards)
                        print(f"  -> Average reward over last 100 episodes: {avg_recent:.1f}")
                        print(f"Episode {episode}")