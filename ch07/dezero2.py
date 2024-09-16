import numpy as np
from dezero import Variable

def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (1 - x0) ** 2
    return y

if __name__ == "__main__":
    x0 = Variable(np.array(0.0))
    x1 = Variable(np.array(2.0))
    y = rosenbrock(x0, x1)

    iters = 10000
    lr = 0.001

    for i in range(iters):
        x0.cleargrad()
        x1.cleargrad()
        y = rosenbrock(x0, x1)
        y.backward()

        x0.data -= lr * x0.grad.data
        x1.data -= lr * x1.grad.data
        print(f"iters: {i}, x0: {x0.data:.3f}, x1: {x1.data:.3f}, y: {y.data:.3f}")


    print(x0, x1)
    """
    ...
    iters: 9997, x0: 0.994, x1: 0.989, y: 0.000
    iters: 9998, x0: 0.994, x1: 0.989, y: 0.000
    iters: 9999, x0: 0.994, x1: 0.989, y: 0.000
    variable(0.9944984367782456) variable(0.9890050527419593)
    """