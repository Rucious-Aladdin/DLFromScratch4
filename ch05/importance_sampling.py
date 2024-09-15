import numpy as np

if __name__ == "__main__":
    x = np.array([1, 2, 3])
    pi = np.array([.1, .1, .8])

    e = np.sum(x* pi)
    print(f"true expectation: {e}")

    n = 100
    samples = []
    for _ in range(n):
        s = np.random.choice(x, p=pi)
        samples.append(s)
    mean = np.mean(samples)
    var = np.var(samples)
    print(f"Monte Carlo estimate: {mean:.2f} (Var: {var:.2f})")

    """
    true expectation: 2.7
    Monte Carlo estimate: 2.73 (Var: 0.36)
    """