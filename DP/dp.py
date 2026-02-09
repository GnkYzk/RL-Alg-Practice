class DPMethodV:
    def __init__(self,env,itrs,discount=0.9,target =1e-3,mode=0):
        self.env=env
        n = len(env.states)
        self.v = [0]*n
        self.itrs = itrs
        self.discount=discount
        self.policy = [[0.25 for _ in range(4)] for _ in range(n)]
        self.target=target
        self.mode=mode #mode 0 = policy evaluation, mode 1 = value iteration
    def solve(self):
        if self.mode:
            self.value_iteration()
            self.policy_improvement()

        else:
            while True:
                self.policy_evaluation()
                if self.policy_improvement():
                    break
            
    def policy_evaluation(self):
        whileflag=True
        
        while whileflag:
            change=0
            for state in self.env.states:
                    if state in self.env.terminal_states:
                        continue
                    v = self.v[state]
                    new_v=0
                    for action in self.env.actions:
                        next_state,reward = self.env.transition(state,action)
                        
                        new_v+=self.policy[state][action] *(reward + self.discount*self.v[next_state])
                    self.v[state]=new_v
                    change = max(change,abs(new_v-v))
                    
            if change<self.target:
                whileflag=False
    def value_iteration(self):
        whileflag=True
        
        while whileflag:
            change=0
            for state in self.env.states:
                if state in self.env.terminal_states:
                        continue
                v = self.v[state]
                values = []
                for action in self.env.actions:
                    next_state,reward = self.env.transition(state,action)
                    values.append(reward + self.discount * self.v[next_state])
                self.v[state] = max(values)
                change = max(change,abs(self.v[state]-v))
            if change<self.target:
                whileflag=False
            
    def policy_improvement(self):
    
        conv=True
        for state in self.env.states:
            if state in self.env.terminal_states:
                        continue
            curr_action =self.policy[state].index(max(self.policy[state]))
            values=[]
            for action in self.env.actions:
                next_state,reward = self.env.transition(state,action)
                values.append(reward + self.discount*self.v[next_state])
            best_action=values.index(max(values))
            if best_action!=curr_action:
                conv=False
                np = [0,0,0,0]
                np[best_action]=1
                self.policy[state]=np
        return conv
    def fullmove(self):
        terminal_states =self.env.terminal_states
        accum_reward=0
        state=0
        
        while state not in terminal_states:
            reward=0
            next_action = self.policy[state].index(max(self.policy[state]))
            next_state,reward=self.env.transition(state,next_action)
            accum_reward += reward
            print(f"At state {state} I take action {next_action} and arrive at {next_state} with reward {reward}. ")
            state=next_state
        print(f"Terminal state {state}.Accumulated Reward {accum_reward}")

class DPMethodQ:
    def __init__(self,env,itrs,discount=0.9,target =1e-3,mode=0):
        self.env=env
        n = len(env.states)
        self.q = [[0 for _ in range(4)] for _ in range(n)]
        self.itrs = itrs
        self.discount=discount
        self.policy = [[0.25 for _ in range(4)] for _ in range(n)]
        self.target=target
        self.mode=mode #mode 0 = policy evaluation, mode 1 = value iteration
    def solve(self):
        if self.mode:
            self.value_iteration()
            self.policy_improvement()

        else:
            while True:
                self.policy_evaluation()
                if self.policy_improvement():
                    break
            
    def policy_evaluation(self):
        whileflag=True
        
        while whileflag:
            change=0
            for state in self.env.states:
                    if state in self.env.terminal_states:
                        
                        continue
                    for action in self.env.actions:
                        q = self.q[state][action]
                        new_q=0
                    
                        next_state,reward = self.env.transition(state,action)
                        next_qs = 0
                        for next_action in self.env.actions:
                             next_qs+=self.policy[next_state][next_action]*self.q[next_state][next_action]
                        new_q+= reward + self.discount*next_qs
                        self.q[state][action]=new_q
                        change = max(change,abs(new_q-q))
                    
            if change<self.target:
                whileflag=False
    def value_iteration(self):
        whileflag=True
        
        while whileflag:
            change=0
            for state in self.env.states:
                if state in self.env.terminal_states:
                        
                        continue
                for action in self.env.actions:
                    q = self.q[state][action]
                    new_q=0
                    values=[]
                    next_state,reward = self.env.transition(state,action)
                    next_qs = self.q[next_state]
                    new_q = reward + self.discount * max(next_qs)
                    self.q[state][action]=new_q
                    change = max(change,abs(new_q-q))
            if change<self.target:
                whileflag=False
            
    def policy_improvement(self):
    
        conv=True
        for state in self.env.states:
            if state in self.env.terminal_states:
                        continue
            curr_action =self.policy[state].index(max(self.policy[state]))
            best_action=self.q[state].index(max(self.q[state]))
            if best_action!=curr_action:
                conv=False
                np = [0,0,0,0]
                np[best_action]=1
                self.policy[state]=np
        return conv
    def fullmove(self):
        terminal_states =self.env.terminal_states
        accum_reward=0
        state=0
        
        while state not in terminal_states:
            reward=0
            next_action = self.policy[state].index(max(self.policy[state]))
            next_state,reward=self.env.transition(state,next_action)
            accum_reward += reward
            print(f"At state {state} I take action {next_action} and arrive at {next_state} with reward {reward}. ")
            state=next_state
        print(f"Terminal state {state}.Accumulated Reward {accum_reward}")