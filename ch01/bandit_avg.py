from bandit import Bandit, Agent
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    runs = 200
    steps = 1000
    epsilons = np.logspace(-2, 0, 5)
    print(epsilons)

    # tqdm for runs
    avg_rates_per_epsilon = []
    for epsilon in epsilons:
        all_rates = np.zeros((runs, steps))
        for run in tqdm(range(runs), desc="Runs"):
            bandit = Bandit()
            agent = Agent(epsilon)
            total_reward = 0
            rates = []

            for step in range(steps):
                action = agent.get_action()
                reward = bandit.play(action)
                agent.update(action, reward)
                total_reward += reward
                rates.append(total_reward / (step + 1))
            
            all_rates[run] = rates
        avg_rates = all_rates.mean(axis=0)
        avg_rates_per_epsilon.append(avg_rates)

    plt.title("Average reward")
    plt.xlabel("Steps")
    plt.ylabel("Rates")
    for i, avg_rates in enumerate(avg_rates_per_epsilon):
        plt.plot(avg_rates, label=f"epsilon={epsilons[i]:.2f}")
    plt.legend(loc="best")
    plt.savefig("./result_images/bandit_avg_rates.png")
    plt.show()
