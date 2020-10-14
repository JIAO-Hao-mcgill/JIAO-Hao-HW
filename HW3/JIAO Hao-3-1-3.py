import numpy as np
from matplotlib import pyplot as plt

data=np.loadtxt('dish_zenith.txt')
x_data=data[:,0]
y_data=data[:,1]
z_data=data[:,2]

# A: c1+c2*x+c3*x*x+c4*x*y+c5*y+c6*y*y
A=np.zeros([len(x_data),6])
A[:,0]=1
A[:,1]=x_data
A[:,2]=x_data*x_data
A[:,3]=x_data*y_data
A[:,4]=y_data
A[:,5]=y_data*y_data

coeff=np.linalg.inv(A.T@A)@(A.T@z_data)
print('The new parameters are ',coeff)

# Sorry that I do not know how to use Python to solve these parameters,
# so I use Mathematica to solve them.


