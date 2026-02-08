class gridworld():
    def __init__(self,states,rewards,probability,x_dim,mode="normal",debug=False):
        #If mode = normal,  all movement possibilities, shape is almost-rectangle,lowest row can be shorter
        #If mode = fullrect, then the shape is a correct rectangle
        #If mode = maze, then certain borders in the rect 
        self.states = states
        self.actions = [0,1,2,3]
        self.rewards = rewards 
        self.probability = probability
        self.debug=debug
        self.n = len(states)
        self.x_dim = x_dim
        assert mode == "normal" or mode == "fullrect" or mode == "maze"
        if mode == "fullrect":
            assert type(self.n/self.x_dim) == int, f"state size {self.n} and x_dim {self.x_dim} unfit for fullrect"
        self.maze_borders={}
    def transition(self,state,action):
        #0=L,1=R,2=U,3=D
        def check_border(state,action,next_state):
            #borderup
            if state < self.x_dim and action == 2:
                if self.debug:    
                    print("Up-most state can not go up")
                return True
            #borderleft
            if state %self.x_dim ==0 and action == 0:
                if self.debug:    
                    print("Left-most state can not go left")
                return True
            #borderright
            if (state+1) %self.x_dim ==0 and action == 1:
                if self.debug:    
                    print("Right-most state can not go right")
                return True
            #borderdown
            if state>=self.n-self.x_dim and action == 3:
                if self.debug:    
                    print("Down-most state can not go down")
                return True
            #extra check to ensure
            if next_state < 0 or next_state >= self.n:
                if self.debug:    
                    print("Next state not in state space")
                return True
            #check maze borders
            if state in self.maze_borders.keys() and next_state in self.maze_borders[state]:
                if self.debug:    
                    print("Border between two states")
                return True
        move = -1 * (action==0) + 1 * (action==1) - self.x_dim * (action==2) + self.x_dim * (action==3)
        next_state = state+ move
        if self.debug:
                action_letter = ["L","R","U","D"][action]
                print(f"State: {state},Action: {action_letter}, Next State: {next_state}")
        if check_border(state,action,next_state):
            return state,0
        return next_state,self.rewards[next_state]

    def add_maze_borders(self,state1,state2):
        def check_possible_next_state(state1,state2):
            if state1 + 1 == state2 or state1 - 1 == state2 or state1 + self.x_dim == state2 or state1 - self.x_dim == state2:
                return True
            return False
        def add_border(state1,state2):
            if state1 not in self.maze_borders.keys():
                self.maze_borders.update({state1:[state2]})
                if self.debug:
                    print(f"Added new key {state1}")
            else:
                self.maze_borders[state1].append(state2)
                if self.debug:
                    print(f"Updated key {state1}")
        assert check_possible_next_state(state1,state2), f"States {state1} and {state2} are not neighbours"
        add_border(state1,state2)
        add_border(state2,state1)
        if self.debug:
                    print(f"Added border between states {state1} and {state2}")
