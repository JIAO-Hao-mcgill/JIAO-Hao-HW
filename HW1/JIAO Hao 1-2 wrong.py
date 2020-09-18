import numpy as np
import sympy

temp=[]
vol=[]
dVdT=[]
dTdV=[]

with open("lakeshore.txt") as fo:
    for line in fo.readlines():
        linelist=[float(i) for i in line.split()]
        temp.append(linelist[0])
        vol.append(linelist[1])
        dVdT.append(0.001*linelist[2])
        dTdV.append(1000./linelist[2])
#print(temp[10],vol[10],dVdT[10],dTdV[10])
#print(temp[11],vol[11],dVdT[11],dTdV[11])
#print(temp[-1],vol[-1],dVdT[-1],dTdV[-1])

def Temperature(V):
    n1=0
    for i in range(len(temp)):
        if(vol[i]>V):
            n1=i # get the range where V is in
            print(vol[i])
    n2=n1+1
#    print(n1)
    
# f(x)=a*x**3+b*x**2+c*x+d
# f'(x)=3*a*x**2+2*b*x+c
# f(x1)=y1, f'(x2)=yy1
# f(x2)=y2, f'(x2)=yy2
    x1=vol[n1]
    x2=vol[n2]
    y1=temp[n1]
    y2=temp[n2]
    yy1=dTdV[n1]
    yy2=dTdV[n2]

    a,b,c,d=sympy.symbols('a b c d')
#    b=sympy.symbol('b')
#    c=sympy.symbol('c')
#    d=sympy.symbol('d')
    para=sympy.solve(\
        [y1/(x1**3)-b/x1-c/(x1**2)-d/(x1**3),\
         yy1/(2*x1)-1.5*a*x1-c/(2*x1),\
         yy2-3*a*x2**2-2*b*x2,\
         y2-a*x2**3-b*x2**2-c*x2],\
        [a,b,c,d])
    print(para)
    aa=para[a]
    bb=para[b]
    cc=para[c]
    dd=para[d]
    Temp=aa*V**3+bb*V**2+cc*V+dd
    print(Temp)
    return[Temp]

T16=Temperature(1.6)

## too large error: may due to inaccurate solution of a,b,c,d
