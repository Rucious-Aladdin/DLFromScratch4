import sys
if "../" not in sys.path:
    sys.path.append("../")
from collections import defaultdict
import numpy as np

class TdAgent:
    def __init__(self):
        self.gamma = .9
        self.alpha = .01
        self.action_size = 4

        random_actions = {0:.25, 1:.25, 2:.25, 3:.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys()) 
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)   

    def eval(self, state, reward, next_state, done):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V

        self.V[state] += self.alpha * (target - self.V[state])

if __name__ == "__main__":
    from common.grid_world import GridWorld
    env = GridWorld()
    agent = TdAgent()

    episodes = 10000
    for episode in range(episodes):
        state  = env.reset()
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.eval(state, reward, next_state, done)
            if done:
                break
            state = next_state
    
    import os
    os.makedirs("./viz_images/", exist_ok=True)
    env.render_v(agent.V, savefig=True, filename="td_eval.png")