from bandit import Agent
import numpy as np
import matplotlib.pyplot as plt

class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        self.rates += 0.1 * np.random.randn(self.arms)
        if rate > np.random.rand():
            return 1
        else:
            return 0

class AlphaAgent:
    def __init__(self, epsilon, alpha, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha
    
    def update(self, action, reward):
        self.Qs[action] = self.Qs[action] + self.alpha * (reward - self.Qs[action])
    
    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)

if __name__ == "__main__":
    steps = 10000
    runs = 200
    all_rates_const = np.zeros((runs, steps))
    all_rates_alpha = np.zeros((runs, steps))

    for run in range(runs):
        bandit = NonStatBandit()
        const_agent = Agent(0.1) # (epsilon=0.1)
        alpha_agent = AlphaAgent(0.1, 0.8) # (epsilon=0.1, alpha=0.8)
        total_rewards_const = 0
        total_rewards_alpha = 0
        
        for step in range(steps):
            # Const Agent 행동 및 업데이트
            action_const = const_agent.get_action()
            reward_const = bandit.play(action_const)
            const_agent.update(action_const, reward_const)
            total_rewards_const += reward_const
            
            # Alpha Agent 행동 및 업데이트
            action_alpha = alpha_agent.get_action()
            reward_alpha = bandit.play(action_alpha)
            alpha_agent.update(action_alpha, reward_alpha)
            total_rewards_alpha += reward_alpha
            
            # 평균 보상률 저장
            all_rates_const[run, step] = total_rewards_const / (step + 1)
            all_rates_alpha[run, step] = total_rewards_alpha / (step + 1)
        rates_const = all_rates_const.mean(axis=0)
        rates_alpha = all_rates_alpha.mean(axis=0)
    
    plt.title("Non-stationary bandit VS Stationary bandit")
    plt.xlabel("Steps")
    plt.ylabel("Rates")
    plt.plot(rates_const, label="Stationary bandit")
    plt.plot(rates_alpha, label="Non-stationary bandit")
    plt.legend(loc="best")
    plt.savefig("./result_images/non_stationary_vs_stationary.png")
    plt.show()
