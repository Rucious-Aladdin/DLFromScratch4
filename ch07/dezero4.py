from dezero import Variable
import dezero.functions as F
import numpy as np
import matplotlib.pyplot as plt

def predict(x):
    y = F.linear(x, W1, b1)
    y = F.sigmoid(y)
    y = F.linear(y, W2, b2)
    return y

if __name__ == "__main__":
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    I, H, O = 1, 10, 1
    W1 = Variable(0.01 * np.random.randn(I, H))
    b1 = Variable(np.zeros(H))
    W2 = Variable(0.01 * np.random.randn(H, O))
    b2 = Variable(np.zeros(O))

    lr = 0.2
    iters = 10000

    for i in range(iters):
        y_pred = predict(x)
        loss = F.mean_squared_error(y, y_pred)

        W1.cleargrad()
        b1.cleargrad()
        W2.cleargrad()
        b2.cleargrad()

        loss.backward(create_graph=True)

        W1.data -= lr * W1.grad.data
        b1.data -= lr * b1.grad.data
        W2.data -= lr * W2.grad.data
        b2.data -= lr * b2.grad.data

        if i % 1000 == 0:
            print(f"iters: {i}, loss: {loss.data}")
    
    plt.scatter(x, y, s=5)
    lx = np.arange(0, 1, .01).reshape(100, 1)
    ly = predict(lx)
    plt.text(0, 7.0, f"mse_loss:{loss.data:.3f}")
    plt.plot(lx, ly.data, color='red')
    plt.show()