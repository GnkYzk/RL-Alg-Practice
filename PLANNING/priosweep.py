import random
import numpy as np
class PrioritySweep:
    def __init__(self,env,itrs,n,step_size=0.1,discount=0.9,threshold = 0.1):
        self.env=env
        self.itrs=itrs
        self.step_size=step_size
        self.discount=discount
        self.n=n
        self.values = [[0 for _ in env.actions] for _ in env.states]
        self.policy = [[0.25 for _ in env.actions] for _ in env.states]
        self.queue = {}
        self.policy =[[0.25 for _ in env.actions] for _ in env.states]
        self.behavior_policy =[[0.25 for _ in env.actions] for _ in env.states]
        self.model = [[() for _ in env.actions] for _ in env.states]
        self.threshold = threshold
        self.predecessors = {}
    def solve(self):
        for itr in range(self.itrs):
            state = self.env.startstate
            action = self.choose_behavior_action(state)
            while state not in self.env.terminal_states:
                next_state,reward = self.env.transition(state,action)

                self.model[state][action] = (next_state,reward)
                self.predecessors.setdefault(next_state, set()).add((state, action))

                p = abs(reward + self.discount*max(self.values[next_state]) - self.values[state][action])
                if p>self.threshold:
                    if (state,action) not in self.queue or p > self.queue[(state,action)]:
                        self.queue[(state,action)] = p

                for _ in range(self.n):
                    if len(self.queue)==0:break

                    s,a = max(self.queue, key=self.queue.get)
                    next_s,r = self.model[s][a]
                    self.values[s][a] +=self.step_size*(r + self.discount*max(self.values[next_s])- self.values[s][a])
                    self.queue.pop((s,a))

                    for pre_s, pre_a in self.predecessors.get(s, []):
                        pre_next_s,pre_r = self.model[pre_s][pre_a]
                        p = abs(pre_r + self.discount*max(self.values[pre_next_s]) - self.values[pre_s][pre_a])
                        if p>self.threshold:
                            self.queue.update({(pre_s,pre_a):p})
                state= next_state
                action = self.choose_behavior_action(state)
    def choose_behavior_action(self,state):
        return random.choices(self.env.actions,self.behavior_policy[state])[0]
    def update_policy(self,state):
        
        best_actions = [action for action in self.env.actions if self.values[state][action]==max(self.values[state])]
        action = random.choice(best_actions)
        new = [0 for _ in self.env.actions]
        new[action]=1
        self.policy[state] = new