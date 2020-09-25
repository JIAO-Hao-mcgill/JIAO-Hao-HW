import numpy as np
import matplotlib.pyplot as plt

def simpson(x,y):
    return (x[1]-x[0])*(y[0]+4*np.sum(y[1::2])+2*np.sum(y[2:-1:2])+y[-1])/3

def recursion(fun,x1,x2,err):
    n=3 # number of points
    error=2*err
    while error>err:
        x=np.linspace(x1,x2,n)
        y=fun(x)
        area1=simpson(x,y)
        n=2*n-1
        x=np.linspace(x1,x2,n)
        y=fun(x)
        area2=simpson(x,y)
        print('n=',n,'\t delta x=',x[1]-x[0])
        error=np.abs(area1-area2)
    print('The integration of ',fun,'(x) from',x1,'to',x2,'is',area2,'\n')
    return area2

def rec_class(fun,x1,x2,err):
    x=np.linspace(x1,x2,5)
    y=fun(x)
    print('n=',int(4/(x2-x1)+1),'\t delta x=',(x2-x1)/4)
    area1=(x2-x1)*(y[0]+4*y[2]+y[4])/6
    area2=(x2-x1)*(y[0]+4*y[1]+2*y[2]+4*y[3]+y[4])/12
    error=np.abs(area1-area2)
    if error<err:
        return area2
    else:
        xm=0.5*(x1+x2)
        a1=rec_class(fun,x1,xm,err/2)
        a2=rec_class(fun,xm,x2,err/2)
        return a1+a2

classway=rec_class(np.exp,0,1,0.00001)
print('The integration in class of np.exp(x) from 0 to 1 is',classway,'\n')

myway=recursion(np.exp,0,1,0.00001)

print('For N times recursive,\n\
the way in class calls the function (2**(N+1)-1) times,\n\
while my way calls the function N times.\n\
So I save (2**(N-1)-(N-1)) times.\n\
(PS, calculate the area1 and area2 is once.)')

