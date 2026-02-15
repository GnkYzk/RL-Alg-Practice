import random
import numpy as np
class DynaQ:
    def __init__(self,env,itrs,n,step_size=0.1,discount=0.9):
        self.env=env
        self.itrs=itrs
        self.n=n
        self.step_size =step_size
        self.discount=discount
        self.policy =[[0.25 for _ in env.actions] for _ in env.states]
        self.behavior_policy =[[0.25 for _ in env.actions] for _ in env.states]
        self.values =[[0 for _ in env.actions] for _ in env.states]
        self.model = [[() for _ in env.actions] for _ in env.states]
        self.seen =set()
    def solve(self):

        
        for itr in range(self.itrs):
            state = self.env.startstate
            action = self.choose_behavior_action(state)
            while state not in self.env.terminal_states:
                next_state,reward = self.env.transition(state,action)
                if next_state in self.env.terminal_states:
                    self.values[state][action]+= self.step_size * (reward -self.values[state][action])
                    self.model[state][action] = (next_state,reward)
                    self.seen.add((state,action))
                else:
                    self.values[state][action]+= self.step_size * (reward + self.discount *max(self.values[next_state])-self.values[state][action])
                    self.model[state][action] = (next_state,reward)
                    self.seen.add((state,action))
                for _ in range(self.n):
                    s,a = random.choice(list(self.seen))
                    ns,r = self.model[s][a]
                    if ns in self.env.terminal_states:
                        self.values[s][a]+= self.step_size * (r -self.values[s][a])
                    else:
                        self.values[s][a]+= self.step_size * (r + self.discount *max(self.values[ns])-self.values[s][a])
                self.update_policy(state)
                state=next_state
                action=self.choose_behavior_action(next_state)
    def choose_behavior_action(self,state):
        return random.choices(self.env.actions,self.behavior_policy[state])[0]
    def update_policy(self,state):
        
        best_actions = [action for action in self.env.actions if self.values[state][action]==max(self.values[state])]
        action = random.choice(best_actions)
        new = [0 for _ in self.env.actions]
        new[action]=1
        self.policy[state] = new