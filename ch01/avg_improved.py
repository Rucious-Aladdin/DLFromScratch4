import numpy as np

# time complexity: O(n^2) to O(n)
# but has problem w.r.t floating point precision
if __name__ == '__main__':
    Q = 0

    for n in range(1, 1001):
        reward = np.random.rand()
        Q = Q + (reward - Q) / n
        print(f"iterations:{n}, Rewards:{Q:.4f}")