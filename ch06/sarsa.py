from collections import defaultdict, deque
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from common.utils import greedy_probs
from common.grid_world import GridWorld

class SarsaAgent:
    def __init__(self):
        self.gamma = .9
        self.alpha = .8
        self.epsilon = .1
        self.action_size = 4

        random_actions = {0:.25, 1:.25, 2:.25, 3:.25}
        self.pi = defaultdict(lambda: random_actions)
        self.Q = defaultdict(lambda: 0)
        self.memory = deque(maxlen=2)
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys()) 
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    
    def reset(self):
        self.memory.clear()
    
    def update(self, state, action, reward, done):
        self.memory.append((state, action, reward, done))
        if len(self.memory) < 2:
            return

        state, action, reward, done = self.memory[0]
        next_state, next_action, _, _ = self.memory[1]

        next_q = 0 if done else self.Q[next_state, next_action]
        # print(next_state, next_action)
        # default dictionary를 이용한 예외처리
        # next_action이 None일 경우 0을 반환

        target = reward + self.gamma * next_q
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

        self.pi[state] = greedy_probs(self.Q, state, self.epsilon)

if __name__ == "__main__":
    env = GridWorld()
    agent = SarsaAgent()

    episodes = 10000
    for episode in range(episodes):
        state  = env.reset()
        agent.reset()
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            agent.update(state, action, reward, done)

            if done:
                agent.update(next_state, None, None, None)
                break
            state = next_state
    
    env.render_q(agent.Q, savefig=True, filename="sarsa.png")