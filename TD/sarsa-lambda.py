import random
import numpy as np
class Sarsa_trace:
    def __init__(self,env,itrs,step_size=0.1,discount =0.9,epsilon = 0.1,trace_decay =0.1,trace_type="dutch"):
        self.env = env
        self.itrs = itrs
        self.step_size = step_size
        self.discount = discount
        self.epsilon = epsilon
        self.trace_decay = trace_decay
        self.startstate = env.startstate
        self.trace_type=trace_type #accumulating,replacing,dutch
        self.policy = [[0.25 for _ in env.actions] for _ in env.states]
        self.values = [[0 for _ in env.actions] for _ in env.states]
        self.eligibility = [[0 for _ in env.actions] for _ in env.states]
    def solve(self):
        for itr in range(self.itrs):
            self.eligibility = [[0 for _ in self.env.actions] for _ in self.env.states]
            state = self.startstate
            action = self.choose_action(state)
            while state not in self.env.terminal_states:
                change=0
                next_state, reward = self.env.transition(state,action)
                if next_state in self.env.terminal_states:
                    change = reward -self.values[state][action]
                    if self.trace_type=="accumulating":

                        self.eligibility[state][action] +=1
                    elif self.trace_type=="replacing":
                        self.eligibility[state][action] =1
                    else:
                        self.eligibility[state][action] = (1-self.step_size)*self.eligibility[state][action] + 1
                    self.update_values(change)
                    self.update_policy(state)
                    state = next_state
                else:
                    next_action = self.choose_action(next_state)
                    change = reward + self.discount*self.values[next_state][next_action]-self.values[state][action]
                    if self.trace_type=="accumulating":

                        self.eligibility[state][action] +=1
                    elif self.trace_type=="replacing":
                        self.eligibility[state][action] =1
                    else:
                        self.eligibility[state][action] = (1-self.step_size)*self.eligibility[state][action] + 1
                
                    self.update_values(change)
                    self.update_policy(state)
                    state = next_state
                    action = next_action
    def update_values(self,change):
        for state in self.env.states:
            for action in self.env.actions:
                self.values[state][action] += self.step_size * change * self.eligibility[state][action]
                self.eligibility[state][action]*=self.discount*self.trace_decay
    def choose_action(self,state):
        return random.choices(self.env.actions,weights=self.policy[state])[0]
    def update_policy(self,state):
        
        best_action = np.argmax(self.values[state])
        new_policy = [self.epsilon / len(self.env.actions) for _ in self.env.actions]
        new_policy[best_action] = 1 - self.epsilon + self.epsilon / len(self.env.actions)
        self.policy[state] = new_policy