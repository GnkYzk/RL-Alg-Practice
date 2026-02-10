from ENV.gridworld import Gridworld
from DP.dp import DPMethodV
from DP.dp import DPMethodQ
from MC.mc import MCSolver
ng=Gridworld([0,1,2,3,4,5,6,7,8,9],0,{0:0,1:0,2:0,3:0,4:-3,5:10,6:0,7:0,8:0,9:0},5,terminal_states=[4,5],debug=False)
ng.add_maze_borders(0,5)
ng.add_maze_borders(1,6)
ng.add_maze_borders(2,7)

solver =MCSolver(ng,10000)
solver.solve()
print("V values:")
for s in ng.states:
    for a in ng.actions:
        print(s,a,solver.values[s][a])
