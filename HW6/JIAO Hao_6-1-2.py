import numpy as np
import matplotlib.pyplot as plt

n=100000
x=1e8*np.random.rand(n)
y=1e8*np.random.rand(n)
z=1e8*np.random.rand(n)

plt.ion()
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(x,y,z,',')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()





