import numpy as np
import matplotlib.pyplot as plt
plt.ion()

# First, get the Green's function:
def make_Ax(x0,mask):
    x=x0.copy()
    x[mask]=0
    tot=np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)
    x=x-tot/4.  # Ax
    x[mask]=0
    return x
def cg_Ax(b,x0,mask,ninter=601):
    Ax=make_Ax(x0,mask)
    rk=b-Ax
    pk=rk.copy()
    x=x0.copy()
    rtr=np.sum(rk*rk)
    for i in range(ninter):
        Apk=make_Ax(pk,mask)
        pAp=np.sum(pk*Apk)
        ak=rtr/pAp
        x=x+ak*pk
        rk_new=rk-ak*Apk
        rtr_new=np.sum(rk_new*rk_new)
        bk=rtr_new/rtr
        pk=rk_new+bk*pk
        rk=rk_new
        rtr=rtr_new
    return x
def make_Green(n): 
    mask=np.zeros([n,n],dtype='bool')
    mask[0,:]=True
    mask[-1,:]=True
    mask[:,0]=True
    mask[:,-1]=True
    source=np.zeros([n,n])
    source[n//2,n//2]=1
    G=cg_Ax(source,0*source,mask)
    G=np.roll(G,n//2+1,axis=0)
    G=np.roll(G,n//2+1,axis=1)
    return G,mask

# Second, get the conjugate-gradient for G⊗ρ= V
def make_lhs(x0,G,mask):
    # x0 is the charge density
    # input G to avoid calculate it for many times
    x=x0.copy()
    #x[mask]=0
    lhs=np.fft.ifftn(np.fft.fftn(G)*np.fft.fftn(x))  # I find that rfftn cannot solve the matrix with odd lines
    lhs=np.real(lhs)
    #lhs[mask]=0
    return lhs
def cg(V,x0,G,mask,ninter=601):
    Ax=make_lhs(x0,G,mask)
    rk=V-Ax
    pk=rk.copy()
    x=x0.copy()
    rtr=np.sum(rk*rk)
    for i in range(ninter):
        Apk=make_lhs(pk,G,mask)
        pAp=np.sum(pk*Apk)
        ak=rtr/pAp
        x=x+ak*pk
        rk_new=rk-ak*Apk
        rtr_new=np.sum(rk_new*rk_new)
        bk=rtr_new/rtr
        pk=rk_new+bk*pk
        rk=rk_new
        rtr=rtr_new
        if i%10==0:
            plt.clf()
            plt.imshow(rk)
            plt.colorbar()
            plt.pause(0.0001)
        if i%100==0:
            print('residue of',i,'is',rtr)
    return x



n=201
G,mask=make_Green(n)
#V=np.ones([n,n])
#V[mask]=0
#V[1,:]=0
#V[-2,:]=0
#V[:,1]=0
#V[:,-2]=0
b=np.zeros([n,n])
b[50:-50,50:-50]=1
V1=cg_Ax(b,0*b,mask)
plt.imshow(V1)
plt.pause(1)
e=cg(V1,0*V1,G,mask)






