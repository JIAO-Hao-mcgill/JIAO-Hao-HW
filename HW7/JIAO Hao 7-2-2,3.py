import numpy as np
import matplotlib.pyplot as plt
plt.ion()

PartC=True

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
        if rtr<1e-30:
            break
    #x=x-x[n//2,n//2]+1
    return x
def make_Green(n,N):  # N is the number of additional grid
    mask1=np.zeros([n+2*N,n+2*N],dtype='bool')
    mask1[0,:]=True
    mask1[-1,:]=True
    mask1[:,0]=True
    mask1[:,-1]=True
    source=np.zeros([n+2*N,n+2*N])
    source[n//2+N,n//2+N]=1
    G=cg_Ax(source,0*source,mask1)
    G=np.roll(G,n//2+N+1,axis=0)
    G=np.roll(G,n//2+N+1,axis=1)
    return G

# Second, get the conjugate-gradient for G⊗ρ= V
def make_lhs(x0,G,N):
    # x0 is the charge density
    # input G to avoid calculate it for many times
    x=np.zeros([n+2*N,n+2*N])
    x[N:-N,N:-N]=1*x0
    lhs=np.fft.ifftn(np.fft.fftn(G)*np.fft.fftn(x))  # I find that rfftn cannot solve the matrix with odd lines
    lhs=np.real(lhs)
    return lhs[N:-N,N:-N]
def cg(V,x0,G,N,mask,ninter=601):
    Ax=make_lhs(x0,G,N)
    rk=V-Ax
    pk=rk.copy()
    x=x0.copy()
    rtr=np.sum(rk*rk)
    for i in range(ninter):
        Apk=make_lhs(pk,G,N)
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
        if rtr<1e-30:
            break
    plt.clf()
    plt.imshow(x)
    plt.colorbar()
    plt.savefig('7-2-2-1.png')
    plt.pause(1)
    return x


#-------------------------- Part B ---------------------------
n=41
N=20
mask=np.zeros([n,n],dtype='bool')
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True

G=make_Green(n,N)

V=np.ones([n,n])
V[mask]=0
e=cg(V,0*V,G,N,mask)
plt.clf()
plt.plot(e[0,:])
plt.plot(e[1,:])
plt.plot(e[10,:])
plt.legend(['n=0','n=1','n=10'],loc='best')
plt.savefig('7-2-2-2.png')
plt.pause(1)

#-------------------------- Part C -----------------------------
if PartC==False:
    assert(1==0)
# potential inside the box
VV=cg_Ax(e,0*e,mask)
plt.clf()
plt.imshow(VV)
plt.colorbar()
plt.savefig('7-2-3-1.png')
plt.pause(1)
plt.clf()
plt.plot(VV[0,:])
plt.plot(VV[1,:])
plt.plot(VV[10,:])
plt.legend(['n=0','n=1','n=10'],loc='best')
plt.savefig('7-2-3-2.png')
plt.pause(1)

# potential both inside and outside the box
NN=40
mask=np.zeros([n+2*NN,n+2*NN],dtype='bool')
mask[0,:]=True
mask[-1,:]=True
mask[:,0]=True
mask[:,-1]=True
ee=np.zeros([n+2*NN,n+2*NN])
ee[NN:-NN,NN:-NN]=1*e
VVV=cg_Ax(ee,0*ee,mask)
plt.clf()
plt.imshow(VVV)
plt.colorbar()
plt.savefig('7-2-3-3.png')
plt.pause(1)
plt.clf()
plt.plot(VVV[0,:])
plt.plot(VVV[1,:])
plt.plot(VVV[NN,:])
plt.plot(VVV[NN+n//2,:])
plt.legend(['n=0','n=1','n=40','n=60'],loc='best')
plt.savefig('7-2-3-4.png')






