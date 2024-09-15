import numpy as np
import matplotlib.pyplot as plt

def sample(dices = 2):
    x = 0
    for _ in range(dices):
        x += np.random.randint(1, 7)
    return x

if __name__ == "__main__":
    V, n = 0, 0
    trial = 1000
    samples = []
    for _ in range(trial):
        s = sample()
        n += 1
        V += (s - V) / n
        print(V)