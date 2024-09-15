import numpy as np

def sample(dices = 2):
    x = 0
    for _ in range(dices):
        x += np.random.randint(1, 7)
    return x

if __name__ == "__main__":
    print(sample())
    print(sample())
    print(sample())