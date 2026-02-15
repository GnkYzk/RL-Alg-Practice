import random
import numpy as np
class DynaQ:
    def __init__(self,env,itrs,n,step_size=0.1,discount=0.9,k=0.1):
        self.env=env
        self.itrs=itrs
        self.n=n
        self.step_size =step_size
        self.discount=discount
        self.k=k
        self.policy =[[0.25 for _ in env.actions] for _ in env.states]
        self.behavior_policy =[[0.25 for _ in env.actions] for _ in env.states]
        self.values =[[0 for _ in env.actions] for _ in env.states]
        self.model = [[() for _ in env.actions] for _ in env.states]
        self.seen =[]
        self.elapsed={}
    def solve(self):

        
        for itr in range(self.itrs):
            state = self.env.startstate
            action = self.choose_behavior_action(state)
            while state not in self.env.terminal_states:
                next_state,reward = self.env.transition(state,action)
                for transition in self.seen:
                    self.elapsed[transition] += 1
                if (state,action) not in self.seen:
                        self.seen.append((state,action))
                        self.elapsed.update({(state,action):0})
                else:
                    self.elapsed[(state,action)]=0
                if next_state in self.env.terminal_states:
                    self.values[state][action]+= self.step_size * (reward -self.values[state][action])
                    self.model[state][action] = (next_state,reward)
                    
                else:

                    self.values[state][action]+= self.step_size * (reward + self.discount *max(self.values[next_state])-self.values[state][action])
                    self.model[state][action] = (next_state,reward)
                for n in range(self.n):
                    s,a = random.choice(self.seen)
                    elapsed = self.elapsed[(s,a)]
                    bonus = self.k * elapsed**(1/2)
                    ns,r = self.model[s][a]
                    if ns in self.env.terminal_states:
                        self.values[s][a]+= self.step_size * (r+bonus -self.values[s][a])
                    else:
                        self.values[s][a]+= self.step_size * (r +bonus+ self.discount *max(self.values[ns])-self.values[s][a])
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