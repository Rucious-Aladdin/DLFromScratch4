import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    rewards = []

    for n in range(1, 1001):
        reward = np.random.rand()
        rewards.append(reward)
        Q = sum(rewards) / n
        print(f"iterations:{n}, Rewards:{Q:.4f}")

    """
    iterations:2, Rewards:0.6320
    iterations:3, Rewards:0.6223
    iterations:4, Rewards:0.6029
    
    ...
    
    iterations:998, Rewards:0.4960
    iterations:999, Rewards:0.4957
    iterations:1000, Rewards:0.4959
    """