import sys
if "../../" not in sys.path:
    sys.path.append("../../")
from common.grid_world import GridWorld

if __name__ == "__main__":
    env = GridWorld()
    action = 0
    next_state, reward, done = env.step(action)

    print("next_state:", next_state)
    print("reward:", reward)
    print("done:", done)