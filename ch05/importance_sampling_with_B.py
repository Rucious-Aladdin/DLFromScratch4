import numpy as np

if __name__ == "__main__":
    b = np.array([1/3, 1/3, 1/3])
    n = 100
    samples = []

    x = np.array([1, 2, 3])
    pi = np.array([.2, .2, .6])

    for _ in range(n):
        idx = np.arange(len(b))
        i = np.random.choice(idx, p=b)
        s = x[i]
        rho = pi[i]/ b[i]
        samples.append(s * rho)
    
    mean = np.mean(samples)
    var = np.var(samples)
    print(f"Importance sampling estimate: {mean:.2f} (Var: {var:.2f})")