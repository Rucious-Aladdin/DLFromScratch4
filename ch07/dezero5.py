import numpy as np
import dezero.layers as L
import dezero.functions as F
from dezero import Model
import matplotlib.pyplot as plt

class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
    
    def forward(self, x):
        y = F.sigmoid(self.l1(x))
        y = self.l2(y)
        return y
    

if __name__ == "__main__":
    linear = L.Linear(10)
    batch_size, input_size = 100, 5
    
    x = np.random.randn(batch_size, input_size)
    y = linear(x)
    print(f"y.shape: {y.shape}")
    print(f"params_shape: {linear.W.shape}, {linear.b.shape}")

    for params in linear.params():
        print(params.name, params.shape)

    """
    y.shape: (100, 10)
    params_shape: (5, 10), (10,)
    W (5, 10)
    b (10,)
    """

    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)

    lr = .2
    iters = 10000
    model = TwoLayerNet(10, 1)
    for i in range(iters):
        y_pred = model.forward(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        for p in model.params():
            p.data -= lr * p.grad.data
        
        if i % 1000 == 0:
            print(f"iters: {i}, loss: {loss.data}")
    
    plt.scatter(x, y, s=5)
    lx = np.arange(0, 1, .01).reshape(100, 1)
    ly = model.forward(lx)
    plt.text(0, 7.0, f"mse_loss:{loss.data:.3f}")
    plt.plot(lx, ly.data, color='red')
    plt.show()