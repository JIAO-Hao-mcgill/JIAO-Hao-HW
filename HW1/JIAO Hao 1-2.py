import numpy as np
import matplotlib.pyplot as plt

temp=[]
vol=[]
dVdT=[]
dTdV=[]

print("I use Hermite interpolation here, ")
print("so the estimate of error is approximate to")
print("f''''(x)*(x-x1)^2*(x-x2)^2/4!")

with open("lakeshore.txt") as fo:
    for line in fo.readlines():
        linelist=[float(i) for i in line.split()]
        temp.append(linelist[0])
        vol.append(linelist[1])
        dVdT.append(0.001*linelist[2])
        dTdV.append(1000./linelist[2])
#print(temp[0],vol[0],dVdT[0],dTdV[0])

def Temperature(V):
    n1=-1
    for i in range(len(temp)):
        if(vol[i]>V):
            n1=i # get the range where V is in
    x1=vol[n1]
    x2=vol[n1+1]
    y1=temp[n1]
    y2=temp[n1+1]
    yy1=dTdV[n1]
    yy2=dTdV[n1+1]
# Hermite interpolation
# H_3(x)=h1(x)y1+h2(x)y2+g1(x)yy1+g2(x)yy2
    h1=(1-2*(V-x1)/(x1-x2))*((V-x2)/(x1-x2))**2
    h2=(1-2*(V-x2)/(x2-x1))*((V-x1)/(x2-x1))**2
    g1=(V-x1)*((V-x2)/(x1-x2))**2
    g2=(V-x2)*((V-x1)/(x2-x1))**2
    Temp=h1*y1+h2*y2+g1*yy1+g2*yy2
    return Temp


#plot
x=np.linspace(vol[0],vol[-1],1001)
y=[]
for i in range(len(x)):
    y.append(Temperature(x[i]))
plt.plot(vol,temp,'.',x,y,'-')
plt.legend(['data','Hermite'],loc='best')
plt.show()

V=input("Please enter the voltage value (0.091<V<1.644): ")
print("The corresponding temperature is approximate to ",Temperature(float(V))," K")


