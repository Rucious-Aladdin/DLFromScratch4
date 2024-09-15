import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")  
from common.grid_world import GridWorld
from collections import defaultdict

def greedy_probs(Q, state, epsilon=0.1, action_size=4):
    qs = [Q[(state, a)] for a in range(action_size)]
    max_action = np.argmax(qs)

    base_prob = epsilon / action_size
    action_probs = {action: base_prob for action in range(action_size)}
    action_probs[max_action] += (1 - epsilon)
    return action_probs

class McAgent:
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 0.1
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {i : .25 for i in range(self.action_size)}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        #self.cnts = defaultdict(lambda: 0)
        self.memory = []

    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def add(self, state, action, reward):
        self.memory.append((state, action, reward))
    
    def reset(self):
        self.memory.clear()

    def update(self):
        G = 0
        for data in reversed(self.memory):
            state, action, reward = data
            G = self.gamma * G + reward
            key = (state, action)
            #self.cnts[key] += 1

            #self.Q[key] += (G - self.Q[key]) / self.cnts[key]
            
            self.Q[key] += self.alpha * (G - self.Q[key]) # 과거에 얻은 Q함수의 값을 지수적으로 감소시키는 방법
            # self.Q[key] = (1 - self.alpha) * self.Q[key] + self.alpha * G
            # self.Q[key] 값이 한타임이 지날때마다 self.alpha * 100 % 만큼 지수적으로 감소한다.

            self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

if __name__ == '__main__':
    env = GridWorld()
    agent = McAgent()

    episodes = 10000
    for i in range(episodes):
        state = env.reset()
        agent.reset()

        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.add(state, action, reward)
            if done:
                agent.update()
                break

            state = next_state
    env.render_q(agent.Q, savefig=True, filename='mc_control.png')