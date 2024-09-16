from collections import defaultdict
import numpy as np
import sys
if "../" not in sys.path:
    sys.path.append("../")
from common.utils import greedy_probs
from common.grid_world import GridWorld

class RandomAgent: ## 분포모델과 샘플링을 통해 행동을 선택하는 에이전트
    def __init__(self):
        random_actions = {0:.25, 1:.25, 2:.25, 3:.25}
        self.pi = defaultdict(lambda: random_actions)
    
    def get_action(self, state):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)
    """
    def get_action(self, state): # 샘플모델을 이용한 행동 선택
        return np.random.choice(4)
    """

class QLearningAgent:
    def __init__(self) -> None:
        self.gamma = .9         # Discount factor
        self.alpha = .1         # Exponentially decaying learning rate
        self.epsilon = .5       # epsilon-greedy policy
        self.action_size = 4    # Number of actions

        random_actions = {0:.25, 1:.25, 2:.25, 3:.25}
        # self.pi = ...
        self.Q = defaultdict(lambda: 0)
    
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            qs = [self.Q[state, a] for a in range(self.action_size)]
            return np.argmax(qs)
    
    def update(self, state, action, reward, next_state, done):
        if done:
            next_q_max = 0
        else:
            next_qs = [self.Q[next_state, a] for a in range(self.action_size)]
            next_q_max = max(next_qs)
        
        target = reward + self.gamma * next_q_max
        self.Q[state, action] += (target - self.Q[state, action]) * self.alpha

if __name__ == "__main__":
    env = GridWorld()
    agent = QLearningAgent()

    episodes = 10000
    for episode in range(episodes):
        state = env.reset()
        
        while True:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            if done:
                break
            state = next_state
    env.render_q(agent.Q, savefig=True, filename="q_learning_with_sample.png")