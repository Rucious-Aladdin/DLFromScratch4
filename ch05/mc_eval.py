import numpy as np
import sys
if ".." not in sys.path:
    sys.path.append("..")
from common.grid_world import GridWorld
from collections import defaultdict

class RandomAgent:
    def __init__(self, env):
        self.gamma = 0.9
        self.action_size = 4

        random_actions = {
            i: .25 for i in range(self.action_size)
        }
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)
        self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def add(self, state, action, reward):
        data = (state, action, reward)
        self.memory.append(data)
    
    def reset(self):
        self.memory.clear()
    
    def eval(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward ## 뒤에서부터 계산
            self.cnts[state] += 1
            self.V[state] += (G - self.V[state]) / self.cnts[state]

if __name__ == "__main__":
    env = GridWorld()
    agent = RandomAgent(env)

    episodes = 100000
    for episode in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.eval()
                break
                
            state = next_state
    env.render_v(agent.V, savefig=True, filename='mc_eval.png')