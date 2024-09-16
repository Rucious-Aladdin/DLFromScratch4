import numpy as np
from dezero import Variable
from dezero import optimizers
import dezero.functions as F    
import dezero.layers as L
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
    np.random.seed(0)
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    x, y = Variable(x), Variable(y)
    lr = .2
    iters = 10000
    model = TwoLayerNet(10, 1)
    optimizer = optimizers.SGD(lr)
    """
    adam_args = {
        "alpha": lr,
        "beta1": .9,
        "beta2": .999,
    }
    optimizer = optimizers.Adam(**adam_args)
    """
    optimizer.setup(model)

    model.to_gpu()
    x.to_gpu()
    y.to_gpu()

    for i in range(iters):
        y_pred = model(x)
        loss = F.mean_squared_error(y, y_pred)

        model.cleargrads()
        loss.backward()

        optimizer.update()
        
        if i % 1000 == 0:
            print(f"iters: {i}, loss: {loss.data}")
    
    model.to_cpu()
    x = np.random.rand(100, 1)
    y = np.sin(2 * np.pi * x) + np.random.rand(100, 1)
    plt.scatter(x, y, s=5)
    lx = np.arange(0, 1, .01).reshape(100, 1)
    ly = model.forward(lx)
    plt.text(0, 7.0, f"mse_loss:{loss.data:.3f}")
    plt.plot(lx, ly.data, color='red')
    plt.show()