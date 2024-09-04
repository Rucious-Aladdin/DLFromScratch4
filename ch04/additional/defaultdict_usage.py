import sys
if "C:\\Deep_Learning_Study\\DLFromScratch4" not in sys.path: sys.path.append("C:\\Deep_Learning_Study\\DLFromScratch4")
from common.grid_world import GridWorld
from collections import defaultdict

if __name__ == "__main__":
    # Dictionary
    env = GridWorld()
    V = {}

    for state in env.states():
        V[state] = 0

    state = (1, 2)
    print(V[state])

    # Default Dictionary
    env = GridWorld()
    V = defaultdict(lambda: 0)

    pi = defaultdict(lambda: {0:.25, 1:.25, 2:.25, 3:.25})
    state = (0, 1)

    print(pi[state])