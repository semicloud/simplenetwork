import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm

x = np.arange(-10, 10, 0.1)
y = np.arange(-10, 10, 0.1)

X, Y = np.meshgrid(x, y)

Z = 1/20*(X**2) + Y**2

# plt.subplot(121)
fig = plt.figure(figsize=(8,4 ))
fig.add_subplot(121, projection='3d')
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3)
ax.set_xlabel('X')
ax.set_xlim(-10, 10)
ax.set_ylabel('Y')
ax.set_ylim(-10, 10)
ax.set_zlabel('Z')
ax.set_zlim(0, 120)

plt.subplot(122)
plt.contour(X, Y, Z, levels=40)

plt.show()
