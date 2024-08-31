import numpy as np

class Bandit:
    def __init__(self, arms=10):
        self.rates = np.random.rand(arms)

    def play(self, arm):
        rate = self.rates[arm]
        if rate > np.random.rand():
            return 1
        else:
            return 0
        
class Agent:
    def __init__(self, epsilon, actions=10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.ns = np.zeros(actions)
    
    def update(self, action, reward):
        self.ns[action] += 1
        self.Qs[action] = self.Qs[action] + (reward - self.Qs[action]) / self.ns[action]
    
    def get_action(self):
        # if epsilon = 0.1, 10% of the time to explore
        # 90% of the time to exploit
        if self.epsilon > np.random.rand():
            return np.random.randint(0, len(self.Qs))
        else:
            return np.argmax(self.Qs)
    
if __name__ == "__main__":
    bandit = Bandit()
    Q = 0

    # estimation of slot machine rates (machine 0)
    print("Estimation of slot machine rates (machine 0)")
    for i in range(1, 11):
        reward = bandit.play(0)
        Q = Q + (reward - Q) / i
        print(f"iterations:{i}, Rewards:{Q:.4f}")
    print()

    # estimation of slot machine rates (machine 0~9)
    print("Estimation of slot machine rates (machine 0~9)")
    Qs = np.zeros(10)
    ns = np.zeros(10)
    for n in range(10):
        action = np.random.randint(0, 10)
        reward = bandit.play(action)

        ns[action] += 1
        Qs[action] = Qs[action] + (reward - Qs[action]) / ns[action]
        print(f"iterations:{n+1}, Rewards:{Qs}")
    print("Ground truth:", bandit.rates)
    print()

    # epsilon-greedy
    import matplotlib.pyplot as plt
    steps = 1000
    epsilon = 0.1

    bandit = Bandit()
    agent = Agent(epsilon)
    total_reward = 0
    total_rewards = []
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action, reward)
        total_reward += reward

        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))
    print(total_reward)

    plt.title("Episilon-Greedy_Rewards (epsilon=0.1)")
    plt.ylabel("Total rewards")
    plt.xlabel("Steps")
    plt.plot(total_rewards)
    plt.savefig("./result_images/bandit_total_rewards(epsilon=0.1).png")
    plt.show()

    plt.title("Episilon-Greedy_Rates (epsilon=0.1)")
    plt.ylabel("Rates")
    plt.xlabel("Steps")
    plt.plot(rates)
    plt.savefig("./result_images/bandit_rates(epsilon=0.1).png")
    plt.show()

    # Avearge Results(iterations = 10)
    print("Average Results(iterations = 10)")
    exps = []
    import copy
    steps = 1000
    for exp_num in range(10):
        total_reward = 0
        total_rewards = []
        rates = []
        for step in range(steps):
            action = agent.get_action()
            reward = bandit.play(action)
            agent.update(action, reward)
            total_reward += reward

            total_rewards.append(total_reward)
            rates.append(total_reward / (step + 1))
        exps.append((total_rewards, rates))

    plt.title("Episilon-Greedy_Rewards_10exps (epsilon=0.1)")
    for i, exp in enumerate(exps):
        plt.plot(exp[1], label=f"exp{i+1}")
    plt.ylabel("Total rates")
    plt.xlabel("Steps")
    plt.legend()
    plt.savefig("./result_images/bandit_total_rates_10exps(epsilon=0.1).png")
    plt.show()

