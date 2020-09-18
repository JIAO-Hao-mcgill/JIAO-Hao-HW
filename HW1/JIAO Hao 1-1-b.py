# f'(x)={8[f(x+δ)-f(x-δ)]-[f(x+2δ)-f(x-2δ)]}/12δ

import numpy as np

# f(x)=exp(x)
# f'(x)=exp(x)
expvals=np.linspace(-12,-2,21)
x0=3
minexp=1
# set initial value of minimal exp index
truth=np.exp(x0)
f0=np.exp(x0)
for myexp in expvals:
    delta=10**myexp
    f1=np.exp(x0+delta)
    f2=np.exp(x0-delta)
    f3=np.exp(x0+2*delta)
    f4=np.exp(x0-2*delta)
    deriv=(8*f1-8*f2-f3+f4)/(12*delta)
    diff=np.abs(deriv-truth)
    if minexp>diff:
        minexp=diff
        minnum=myexp
    print(myexp,deriv,diff)
print(minnum,minexp)

# Result: delta for minimal error:
#    For x0=0, delta_min=10**-3, error=2.7e-14
#    For x0=1, delta_min=10**-4, error=1.2e-13
#    For x0=2, delta_min=10**-3, error=9.8e-13
#    For x0=3, delta_min=10**-3, error=2.4e-12


################################################################################


# g(y)=exp(0.01y)
# g'(y)=0.01exp(0.01y)
expvals1=np.linspace(-10,0,21)
y0=1000
print(y0)
minexp1=1
# set initial value of minimal exp index
truth1=0.01*np.exp(0.01*y0)
f0=np.exp(0.01*y0)
for myexp in expvals1:
    delta1=10**myexp
    g1=np.exp(0.01*y0+0.01*delta1)
    g2=np.exp(0.01*y0-0.01*delta1)
    g3=np.exp(0.01*y0+0.02*delta1)
    g4=np.exp(0.01*y0-0.02*delta1)
    deriv1=(8*g1-8*g2-g3+g4)/(12*delta1)
    diff1=np.abs(deriv1-truth1)
    if minexp1>diff1:
        minexp1=diff1
        minnum1=myexp
    print(myexp,deriv1,diff1)
print(minnum1,minexp1)

# Result: delta for minimal error:
#    For y0=0, delta1_min=10**-1, error1=2.7e-16
#    For y0=1, delta1_min=10**-1, error1=1.5e-15
#    For y0=10, delta1_min=10**-1, error1=2.5e-16
#    For y0=100, delta1_min=10**-2, error1=1.1e-15
#    For y0=1000, delta1_min=10**-1, error1=1.1e-10

