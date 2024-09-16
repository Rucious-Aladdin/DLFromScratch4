import sys
if 'C:\\Deep_Learning_Study\\DLFromScratch3' not in sys.path:
    sys.path.append('C:\\Deep_Learning_Study\\DLFromScratch3')
from dezero import Variable
import numpy as np

if __name__ == "__main__":
    x_np = np.array(5.0)
    x = Variable(x_np)

    y = 3 * x ** 2
    print(y)
    y.backward()
    print(x.grad)