import random
import numpy as np
class MCSolver:
    def __init__(self,env,itrs,epsilon=0.1,discount=0.9,step_size=0.1):
        self.env=env
        self.itrs=itrs
        self.discount=discount
        self.epsilon=epsilon
        self.policy = [[0.25 for _ in env.actions] for _ in env.states]
        self.values = [[0 for _ in env.actions] for _ in env.states]
        self.step_size=step_size
    def solve(self):
        for i in range(self.itrs):
            print(i)
            episode = self.sample()
            self.update_values(episode)
            self.update_policy()
    def action_selection(self,state):
        return random.choices(self.env.actions,weights=self.policy[state])[0]
    def sample(self):
        state = self.env.startstate
        episode=[]
        while state not in self.env.terminal_states:
            action = self.action_selection(state)
            next_state,reward = self.env.transition(state,action)
            transition = (state,action,reward)
            episode.append(transition)
            state=next_state
        return episode
    def update_values(self,sample):
        G=0

        for transition in reversed(sample):
            state,action,reward = transition
            G = reward + self.discount * G
            q=self.values[state][action]
            self.values[state][action] +=  self.step_size*(G-q)
    def update_policy(self):
        for ind in range(len(self.policy)):
            state = self.env.states[ind]
            if state in self.env.terminal_states:
                continue
            new =[self.epsilon/len(self.env.actions) for _ in self.env.actions]
            
            max_q = max(self.values[state])
            best_actions = [action for action in self.env.actions if self.values[state][action] == max_q]
            best_action = random.choice(best_actions)
            new[best_action]=1- self.epsilon + self.epsilon/len(self.env.actions)
            self.policy[ind] = new
