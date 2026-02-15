import random
import numpy as np
class Q:
    def __init__(self,env,itrs,step_size,discount):
        self.env=env
        self.itrs = itrs
        self.step_size = step_size
        self.discount = discount
        self.policy = [[0.25 for _ in env.actions] for _ in env.states ] 
        self.behavior_policy = [[1/len(env.actions) for _ in env.actions]for _ in env.states]
        self.values = [[0 for _ in env.actions] for _ in env.states ]
    def solve(self):
        for itr in range(self.itrs):
            state = self.env.startstate
            action = self.choose_behavior_action(state)
            while state not in self.env.terminal_states:
                next_state,reward = self.env.transition(state,action)
                if next_state in self.env.terminal_states:
                    self.values[state][action] += self.step_size *(reward - self.values[state][action])
                    self.update_policy(state)
                    state = next_state
                else:
                    self.values[state][action] += self.step_size *(reward + self.discount*max(self.values[next_state]) - self.values[state][action])
                    self.update_policy(state)
                    state= next_state
                    action = self.choose_behavior_action(state)
    def choose_behavior_action(self,state):
        return random.choices(self.env.actions,weights=self.behavior_policy[state])[0]
    def update_policy(self,state):
        best_action = np.argmax(self.values[state])
        new = [0 for _ in self.env.actions]
        new[best_action] = 1
        self.policy[state] = new