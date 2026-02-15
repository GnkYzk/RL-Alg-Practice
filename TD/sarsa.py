import random
import numpy as np
class Sarsa:
    def __init__(self,env,itrs,step_size=0.1,discount =0.9,epsilon = 0.1):
        self.env = env
        self.itrs = itrs
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.startstate = env.startstate
        self.policy = [[0.25 for _ in env.actions] for _ in env.states]
        self.values = [[0 for _ in env.actions] for _ in env.states]
    def solve(self):
        for itr in range(self.itrs):
            state = self.startstate
            action = self.choose_action(state)
            while state not in self.env.terminal_states:
                
                next_state, reward = self.env.transition(state,action)
                if next_state in self.env.terminal_states:
                    self.values[state][action] += self.step_size * (reward-self.values[state][action])
                    self.update_policy(state)
                    state = next_state
                else:

                    next_action = self.choose_action(next_state)
                    self.values[state][action] += self.step_size * (reward + self.discount*self.values[next_state][next_action]-self.values[state][action])
                    self.update_policy(state)
                    state = next_state
                    action = next_action
    def choose_action(self,state):
        return random.choices(self.env.actions,weights=self.policy[state])[0]
    def update_policy(self,state):
        
        best_action = np.argmax(self.values[state])
        new_policy = [self.epsilon / len(self.env.actions) for _ in self.env.actions]
        new_policy[best_action] = 1 - self.epsilon + self.epsilon / len(self.env.actions)
        self.policy[state] = new_policy