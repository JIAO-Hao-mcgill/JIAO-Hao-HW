import numpy as np
from matplotlib import pyplot as plt

#x0=-1.3604886221977293
#y0=58.22147608157934
#z0=-1512.8772100367873
#a=0.00016670445477401342

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

a=coeff[3]
x0=-coeff[1]/(2*a)
y0=-coeff[2]/(2*a)
z0=coeff[0]-a*x0*x0-a*y0*y0

r1=z_data-z0-a*((x_data-x0)**2+(y_data-y0)**2)
r=z_data-A@coeff
rms=np.std(r)
print('The rms error is ',rms)
N=np.outer(r,r)

error=np.linalg.inv(A.T@(np.linalg.inv(N)@A))
err,v=np.linalg.eig(error)
print('coeff is ',coeff)
print('error is ',err)
print('error of a is ',err[3])

plt.ion()
plt.plot(a*((x_data-x0)**2+(y_data-y0)**2),r,'.')
plt.show()



