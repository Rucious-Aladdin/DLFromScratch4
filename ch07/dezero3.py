import numpy as np
import matplotlib.pyplot as plt 
from dezero import Variable
import dezero.functions as F

def predict(x):
    y = F.matmul(x, W) + b
    return y

def mse(x0, x1):
    diff = x0 - x1
    return F.sum(diff ** 2) / len(diff)

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = 5 + 2 * x + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)
    
    W = Variable(np.zeros((1, 1)))
    b = Variable(np.zeros(1))

    lr = .1
    iters = 100

    for i in range(iters):
        y_pred = predict(x)
        loss = mse(y, y_pred)

        W.cleargrad()
        b.cleargrad()
        loss.backward()

        W.data -= lr * W.grad.data
        b.data -= lr * b.grad.data
        if i % 10 == 0:
            print(f"iters: {i}, W: {W.data}, b: {b.data}, loss: {loss.data:.3f}")

    print("====")
    print(f"W = {W.data}")
    print(f"b = {b.data}")

    plt.scatter(x.data, y.data, s=5)
    lx = np.arange(0, 1, .01).reshape(100, 1)
    ly = predict(lx)
    plt.text(0, 7.0, f"mse_loss:{loss.data:.3f}")
    plt.plot(lx, ly.data, color='red')
    plt.show()
