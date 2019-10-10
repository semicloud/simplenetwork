import numpy as np
import matplotlib.pyplot as plt


def function(x):
    return 1 / 20 * x[0] ** 2 + x[1] ** 2


def _numerical_gradient_no_batch(f, x):
    h = 1e-4
    grad = np.zeros_like(x)  # x的梯度肯定与x的维度相同

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad


def gradient_descent(f, init_x, lr=0.01, step_num=20):
    x = init_x
    x_history = []

    for i in range(step_num):
        x_history.append(x.copy())
        grad = numerical_gradient(f, x)  # x的梯度（其实是个二维函数）
        x -= lr * grad  # 更新x的值

    return x, np.array(x_history)


x = np.arange(-11, 11, 0.1)
y = np.arange(-11, 11, 0.1)
xx, yy = np.meshgrid(x, y)
# xx = xx.flatten()
# yy = yy.flatten()

# print(numerical_gradient(function, np.array([-3.0, 4.0])))

zz = function(np.array([xx, yy]))  # input is a n*2 matrix
x, x_history = gradient_descent(function, np.array([-7.0, 2.0]), 0.9, 40)
plt.contour(xx, yy, zz, levels=50)
plt.plot(x_history[:, 0], x_history[:, 1], 'o--')
plt.show()
