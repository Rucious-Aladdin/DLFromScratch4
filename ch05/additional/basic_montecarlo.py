import numpy as np
import matplotlib.pyplot as plt

def sample(dices = 2):
    x = 0
    for _ in range(dices):
        x += np.random.randint(1, 7)
    return x

if __name__ == "__main__":
    exps = 10000

    avgs = []
    for _ in range(exps):
        trial = 1000
        samples = []
        for _ in range(trial):
            samples.append(sample())
        V = sum(samples) / trial
        avgs.append(V) 
    
    avgs = np.array(avgs)
    print(avgs.mean())
    print(avgs.std())
    plt.hist(avgs, bins=80)
    plt.savefig('../viz_images/dice_montecarlo.png')
    plt.show()
