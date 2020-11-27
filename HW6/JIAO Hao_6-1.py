import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

points=np.loadtxt('rand_points.txt')

plt.ion()
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(points[:,0],points[:,1],points[:,2],',')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
