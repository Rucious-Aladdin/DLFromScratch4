import sys
if ".." not in sys.path: sys.path.append("..")
from common import grid_world
import numpy as np

if __name__ == '__main__':
    env = grid_world.GridWorld()
    V = {}
    for state in env.states():
        V[state] = np.random.randn()
    env.render_v(V, savefig=True, filename='dummy_value_state_func.png')