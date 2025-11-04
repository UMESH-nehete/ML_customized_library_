from my_ml_lib.nn.autograd import Value
import numpy as np

x = Value(np.array([1.0, 2.0]))
w = Value(np.array([[0.5, -0.2],
                    [0.1,  0.3]]))
b = Value(np.array([0.0, 0.0]))

out = (x @ w + b).relu().sum()
out.backward()

print("out:", out.data)
print("x.grad:", x.grad)
print("w.grad:", w.grad)
print("b.grad:", b.grad)
