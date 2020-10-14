import numpy as np
from matplotlib import pyplot as plt

data=np.loadtxt('dish_zenith.txt')
x_data=data[:,0]
y_data=data[:,1]
z_data=data[:,2]

# A: c1+c2*x+c3*x*x+c4*y+c5*y*y
A=np.zeros([len(x_data),4])
A[:,0]=1
A[:,1]=x_data
A[:,2]=y_data
A[:,3]=x_data*x_data+y_data*y_data

coeff=np.linalg.inv(A.T@A)@(A.T@z_data)
print('The new parameters are ',coeff)

a=coeff[3]
x0=-coeff[1]/(2*a)
y0=-coeff[2]/(2*a)
z0=coeff[0]-a*x0*x0-a*y0*y0

print('\nx0=',x0,'\ny0=',y0,'\nz0=',z0,'\na=',a,'\n')

plt.ion()
plt.plot(z_data-z0,a*((x_data-x0)**2+(y_data-y0)**2),'.')
plt.plot([0,1500],[0,1500],'-')
plt.xlabel('z_data-z0')
plt.ylabel('a*((x_data-x0)^2+(y_data-y0)^2)')
plt.show()



