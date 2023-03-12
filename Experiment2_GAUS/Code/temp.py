import numpy as np
def getZ(X, Y):
    return np.sin(np.sqrt(X ** 2 + Y ** 2))

X = np.linspace(-5, 5, 100)
Y = np.linspace(-5, 5, 100)

X, Y = np.meshgrid(X, Y)
Z = getZ(X,Y)

print(X, Y)


print(Z)

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot_surface (X, Y, Z
                , rstride=1 # default value is one
                , cstride=1 # default value is one
                , cmap='winter'
                , edgecolor='none'
                )
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Fine-Mesh Surface Plot');
plt.show()
plt.savefig('refinedSurface3D.png')