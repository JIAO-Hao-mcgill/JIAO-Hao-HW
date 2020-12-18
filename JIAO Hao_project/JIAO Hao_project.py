import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import time
#from numba import njit

plt.ion()

#-------------------------------------------------------------------------------
# Set the # of problems
Part=3
#-------------------------------------------------------------------------------

# Parameters
ngrid=125
#ngrid=75  # one case for non-periodic BC in Part 3 and many many particels Part 4
G=1
" For simplification, we set the size of every grid cell is 1 "
" It is not very difficult to set dpos≠1: "
" We need to consider the dpos=(pos_max-pos_min)/ngrid, "
" Just take x for example: "
" The coordinate of each particle should like this form: "
"     pos_x=int((x-pos_min)/dpos) "
" And the initial position is:  "
"     self.x=pos_min+(pos_max-pos_min)*np.random.rand(self.opts['n']) "
" Besides, the function 'outside' should also change: "
" For periodic case: "
"     self.x=(self.x-pos_min)%(pos_max-pos_min)+pos_min"
" For non-periodic case: "
"     ind11=self.x>pos_min "
"     ind12=self.x<pos_max "

#-------------------------------------------------------------------------------


#def make_Ax(x0,mask):
#    x=x0.copy()
#    x[mask]=0
#    tot=np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)
#    tot=tot+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)
#    tot=tot+np.roll(x,1,axis=2)+np.roll(x,-1,axis=2)
#    x=(tot/6-x)  # Ax
#    x[mask]=0
#    return x
#def cg(b,x0,mask,ninter=800):
#    Ax=make_Ax(x0,mask)
#    rk=b-Ax
#    pk=rk.copy()
#    x=x0.copy()
#    rtr=np.sum(rk*rk)
#    for i in range(ninter):
#        Apk=make_Ax(pk,mask)
#        pAp=np.sum(pk*Apk)
#        ak=rtr/pAp
#        x=x+ak*pk
#        rk_new=rk-ak*Apk
#        rtr_new=np.sum(rk_new*rk_new)
#        bk=rtr_new/rtr
#        pk=rk_new+bk*pk
#        rk=rk_new
#        rtr=rtr_new
#        if rtr<1e-30:
#            return x
#    return x
" At first, I try to solve the non-periodic case by function cg, "
" but then found that this method is too slow and not very accurate."

def make_Green(n):  # N is the number of additional grid
    dx=np.arange(n)
    dx[n//2:]=dx[n//2:]-n
    xmat,ymat,zmat=np.meshgrid(dx,dx,dx)
    dr=np.sqrt(xmat*xmat+ymat*ymat+zmat*zmat)
    dr[0,0,0]=1
    pot=-1/dr
    pot=pot-pot[n//2,n//2,n//2]
    pot[0,0,0]=pot[1,0,0]-1/6  # From the gc method, the difference is 0.25
    return pot
Green=make_Green(ngrid)
Green_ft=np.fft.fftn(Green)
# The center of Green's function is not infinity, so it's softed
if Part==3:  # only in part 3 we need to consider the non-periodic case
    Green_nonperi=make_Green(2*(ngrid//2)+ngrid)  # # of ngrid must be odd
    Green_nonperi_ft=np.fft.fftn(Green_nonperi)
"The periodic case use the green function with the same size as the volume;"
"While the non-periodic case use the green function with each side twice the length of the space"

def get_potential(x,y,z,m,G,periodic=True):
    n=len(x)
    energy=0
    if periodic:
        density=np.zeros([ngrid,ngrid,ngrid])
        for i in range(n):
            pos_x=int(x[i])
            pos_y=int(y[i])
            pos_z=int(z[i])
            density[pos_x,pos_y,pos_z]+=m[i]
        pot=G*np.fft.ifftn(np.fft.fftn(density)*Green_ft)
        pot=np.real(pot)
        energy=np.sum(density*pot)
        return pot,energy
    else:
        N=ngrid//2
        density=np.zeros([ngrid+2*N,ngrid+2*N,ngrid+2*N])
        for i in range(n):
            pos_x=int(x[i])
            pos_y=int(y[i])
            pos_z=int(z[i])
            density[pos_x+N,pos_y+N,pos_z+N]+=m[i]
        pot=G*np.real(np.fft.ifftn(np.fft.fftn(density)*Green_nonperi_ft))
        energy=np.sum(pot*density)
        NN=N-1
        return pot[NN:-NN,NN:-NN,NN:-NN],energy

def get_force(x,y,z,fx,fy,fz,pot,periodic=True):
    for i in range(len(x)):
        pos_x=int(x[i])#; xx=x[i]-pos_x
        pos_y=int(y[i])#; yy=y[i]-pos_y
        pos_z=int(z[i])#; zz=z[i]-pos_z
        if periodic:  # potentail: [ngrid,ngrid,ngrid]
#            fx[i]=(1-xx)*(pot[(pos_x-1)%ngrid,pos_y,pos_z]-pot[pos_x,pos_y,pos_z])\
#                   +xx*(pot[pos_x,pos_y,pos_z]-pot[(pos_x+1)%ngrid,pos_y,pos_z])
#            fy[i]=(1-yy)*(pot[pos_x,(pos_y-1)%ngrid,pos_z]-pot[pos_x,pos_y,pos_z])\
#                   +yy*(pot[pos_x,pos_y,pos_z]-pot[pos_x,(pos_y+1)%ngrid,pos_z])
#            fz[i]=(1-zz)*(pot[pos_x,pos_y,(pos_z-1)%ngridpoten]-pot[pos_x,pos_y,pos_z])\
#                   +zz*(pot[pos_x,pos_y,pos_z]-pot[pos_x,pos_y,(pos_z+1)%ngrid])
#            This cannot get rid of the potential from it self
#            (and particle in the same grid)
            fx[i]=(pot[(pos_x-1)%ngrid,pos_y,pos_z]-pot[(pos_x+1)%ngrid,pos_y,pos_z])/2
            fy[i]=(pot[pos_x,(pos_y-1)%ngrid,pos_z]-pot[pos_x,(pos_y+1)%ngrid,pos_z])/2
            fz[i]=(pot[pos_x,pos_y,(pos_z-1)%ngrid]-pot[pos_x,pos_y,(pos_z+1)%ngrid])/2
        else:  # potentail: [ngrid+2,ngrid+2,ngrid+2]
            fx[i]=(pot[(pos_x-1)+1,pos_y+1,pos_z+1]-pot[(pos_x+1)+1,pos_y+1,pos_z+1])/2
            fy[i]=(pot[pos_x+1,(pos_y-1)+1,pos_z+1]-pot[pos_x+1,(pos_y+1)+1,pos_z+1])/2
            fz[i]=(pot[pos_x+1,pos_y+1,(pos_z-1)+1]-pot[pos_x+1,pos_y+1,(pos_z+1)+1])/2

def density_part4(ngrid):  # n=ngrid
    #R1=np.random.rand(ngrid,ngrid,ngrid)
    #R2=np.random.rand(ngrid,ngrid,ngrid)
    #Rft=R1+1j*R2
    R=np.random.rand(ngrid,ngrid,ngrid)
    Rft=np.fft.fftn(R)
    dk=np.arange(ngrid)
    dk[ngrid//2:]=dk[ngrid//2:]-ngrid
    kx,ky,kz=np.meshgrid(dk,dk,dk)
    k=np.sqrt(kx*kx+ky*ky+kz*kz)
    k[0,0,0]=1
    dens_ft=Rft/(k*np.sqrt(k))
    dens_ft[0,0,0]=0
    dens=np.fft.ifftn(dens_ft)
    dens=np.real(dens)
    dens=dens-np.min(dens)    # let the desnity alway positive
    dens=dens/np.max(dens)    # let the biggest density = 1
    return dens
def get_dens_part4(n):
    x=[]; y=[]; z=[]
    dens=density_part4(ngrid)
    #plt.clf()
    #plt.imshow(dens[:,:,ngrid//2])
    #plt.colorbar()
    #plt.title('density distribution at z=ngrid/2')
    #plt.savefig('JH_4_density.png')
    for i in range(n):
        j=0
        while j==0:
            pos=ngrid*np.random.rand(3)
            prob=np.random.rand()
            if prob<dens[int(pos[0]),int(pos[1]),int(pos[2])]:
                x.append(pos[0])
                y.append(pos[1])
                z.append(pos[2])
                j=1
    return np.array(x),np.array(y),np.array(z)

#-------------------------------------------------------------------------------
class particles:
    def __init__(self,m=1.0,npart=1000,G=1.0,dt=0.1,dist_type='random'):
        self.opts={}  #options
        self.opts['n']=npart
        self.opts['G']=G
        self.opts['dt']=dt
        self.m=np.ones(self.opts['n'])*m
        # initial position (with different distribution)
        if dist_type=='random':
            self.x=ngrid*np.random.rand(self.opts['n'])
            self.y=ngrid*np.random.rand(self.opts['n'])
            self.z=ngrid*np.random.rand(self.opts['n'])
        if dist_type=='gaussian':  # don't need use this condition
            self.x=ngrid/2+ngrid*np.random.randn(self.opts['n'])/10
            self.y=ngrid/2+ngrid*np.random.randn(self.opts['n'])/10
            self.z=ngrid/2+ngrid*np.random.randn(self.opts['n'])/10
            ind=self.outside(periodic=False)
        if dist_type=='circular':
            if self.opts['n']!=2:
                print('Not 2 particles! Cannot generate circular orbit!')
                assert(1==0)
            self.x=np.zeros(2)
            self.y=np.ones(2)*ngrid/2
            self.z=1*self.y
            self.x[0]=ngrid/2+ngrid/10; self.x[1]=ngrid/2-ngrid/10
        if dist_type=='scaleinv':
            n=self.opts['n']
            self.x,self.y,self.z=get_dens_part4(self.opts['n'])
        # initial velocity
        self.vx=np.zeros(self.opts['n'])
        self.vy=np.zeros(self.opts['n'])
        self.vz=np.zeros(self.opts['n'])
        if dist_type=='circular':
            self.vy[0]=np.sqrt(self.opts['G']*self.m[1]/(2*np.abs(self.x[0]-self.x[1])))
            self.vy[1]=-1*np.sqrt(self.opts['G']*self.m[0]/(2*np.abs(self.x[0]-self.x[1])))
            # mv**2/r=GmM/(2*r)**2
        self.fx=np.zeros(self.opts['n'])
        self.fy=np.zeros(self.opts['n'])
        self.fz=np.zeros(self.opts['n'])
        
    def get_forces(self,periodic=True):
        potential,energy=get_potential(self.x,self.y,self.z,self.m,\
                                       self.opts['G'],periodic=periodic)
        get_force(self.x,self.y,self.z,self.fx,self.fy,self.fz,potential)
        return energy
    
    def outside(self,periodic=True):  # if particles move outside the boundary
        if periodic:
            self.x=self.x%ngrid
            self.y=self.y%ngrid
            self.z=self.z%ngrid
        else:
            disappear=True
            if disappear:  # particles disappear when moving outside
                ind11=self.x>0
                ind12=self.x<ngrid
                ind21=self.y>0
                ind22=self.y<ngrid
                ind31=self.z>0
                ind32=self.z<ngrid
                ind=ind11&ind12&ind21&ind22&ind31&ind32
                self.x=self.x[ind]
                self.y=self.y[ind]
                self.z=self.z[ind]
                self.vx=self.vx[ind]
                self.vy=self.vy[ind]
                self.vz=self.vz[ind]
                self.fx=self.fx[ind]
                self.fy=self.fy[ind]
                self.fz=self.fz[ind]
                self.m=self.m[ind]
                self.opts['n']=len(part.x)
                if all(ind.flatten()):
                    return []
                else:
                    return ind
            else:   # bounce back when moving outside
                    # keep total energy conserved
                ind=self.x%ngrid==0
                self.x[ind]=self.x[ind]+0.01
                ind=self.x>ngrid
                self.x[ind]=ngrid-self.x[ind]%ngrid; self.vx[ind]=-self.vx[ind]
                ind=self.x<0
                self.x[ind]=(-self.x[ind])%ngrid; self.vx[ind]=-self.vx[ind]
                ind=self.y%ngrid==0
                self.y[ind]=self.y[ind]+0.01
                ind=self.y>ngrid
                self.y[ind]=ngrid-self.y[ind]%ngrid; self.vy[ind]=-self.vy[ind]
                ind=self.y<0
                self.y[ind]=(-self.y[ind])%ngrid; self.vy[ind]=-self.vy[ind]
                ind=self.z%ngrid==0
                self.z[ind]=self.z[ind]+0.01
                ind=self.z>ngrid
                self.z[ind]=ngrid-self.z[ind]%ngrid; self.vz[ind]=-self.vz[ind]
                ind=self.z<0
                self.z[ind]=(-self.z[ind])%ngrid; self.vz[ind]=-self.vz[ind]
        return []
    
    def evolve(self,periodic=True):
        dt=self.opts['dt']
        x=1.*self.x; y=1.*self.y; z=1.*self.z
        self.x=self.x+0.5*self.vx*dt
        self.y=self.y+0.5*self.vy*dt
        self.z=self.z+0.5*self.vz*dt
        ind=self.outside(periodic=periodic)
        energy_pot=self.get_forces(periodic=periodic) # get the force
        vvx=self.vx+0.5*dt*self.fx
        vvy=self.vy+0.5*dt*self.fy
        vvz=self.vz+0.5*dt*self.fz
        if len(ind)==0:
            self.x=x+dt*vvx
            self.y=y+dt*vvy
            self.z=z+dt*vvz
        else:  # remove the paeticles outside the box
            self.x=x[ind]+dt*vvx
            self.y=y[ind]+dt*vvy
            self.z=z[ind]+dt*vvz
        ind=self.outside(periodic=periodic)
        self.vx+=dt*self.fx
        self.vy+=dt*self.fy
        self.vz+=dt*self.fz
        if len(ind)==0:
            energy_kin=np.sum(self.m*(vvx**2+vvy**2+vvz**2))/2
        else:
            energy_kin=np.sum(self.m*(vvx[ind]**2+vvy[ind]**2+vvz[ind]**2))/2
        return energy_pot,energy_kin
        

################################################################################


# Plot 3D gif figure



# Part 1 : one rest particle
if Part==1:
    dt=0.1
    part=particles(m=1,npart=1,dt=dt)
    image_list=[]
    fig=plt.figure()
    for i in range(10):
        pot,kin=part.evolve()
        plt.clf()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(part.x,part.y,part.z,'.',c='blue')
        t=(i+1)*dt; time='%.2f' % t; 
        ax.text(ngrid,ngrid,ngrid*1.12,'t='+time+'s')
        ax.set_xlim(0,ngrid)
        ax.set_ylim(0,ngrid)
        ax.set_zlim(0,ngrid)
        plt.savefig('0.png')
        image_list.append(imageio.imread('0.png'))
        print(time,'s: the total energy is',pot+kin)
        print('position:',part.x[0],part.y[0],part.z[0])
    imageio.mimsave('JH_1.gif',image_list,duration=0.2)
    # The particle keep rest at any position


#-------------------------------------------------------------------------------

# Part 2 : a pair particle in a circular orbit
if Part==2:
    dt=0.005
    ni=1000
    nj=100
    part=particles(m=1,npart=2,dt=dt,dist_type='circular')
    image_list=[]
    x0_past=[]
    x1_past=[]
    y0_past=[]
    y1_past=[]
    fig=plt.figure(figsize=(6,6))
    ax=fig.add_subplot(111)
    ax.plot(part.x[0],part.y[0],'.',c='blue')
    ax.plot(part.x[1],part.y[1],'.',c='green')
    ax.plot(ngrid/2,ngrid/2,'.',c='lightgray')
    ax.text(ngrid*0.8,ngrid*0.95,'t=0.0s')
    ax.set_xlim(0,ngrid)
    ax.set_ylim(0,ngrid)
    plt.savefig('0.png')
    plt.show()
    image_list.append(imageio.imread('0.png'))
    for i in range(ni):
        for j in range(nj):
            pot,kin=part.evolve()
        x0_past.append(part.x[0])
        x1_past.append(part.x[1])
        y0_past.append(part.y[0])
        y1_past.append(part.y[1])
        plt.clf()
        ax=fig.add_subplot(111)
        ax.plot(x0_past,y0_past,'-',c='lightsteelblue')
        ax.plot(x1_past,y1_past,'--',c='lightgreen')
        ax.plot(part.x[0],part.y[0],'.',c='blue')
        ax.plot(part.x[1],part.y[1],'.',c='green')
        ax.plot(ngrid/2,ngrid/2,'.',c='lightgray')
        # since the two particle only move in xy-plane, we don't show z axis
        t=nj*(i+1)*dt; time='%.2f' % t 
        ax.text(ngrid*0.8,ngrid*0.95,'t='+time+'s')
        ax.set_xlim(0,ngrid)
        ax.set_ylim(0,ngrid)
        plt.savefig('0.png')
        image_list.append(imageio.imread('0.png'))
        if i%25==0:
            imageio.mimsave('JH_2.gif',image_list,duration=0.01)
            print('Saved')
        print(time,'s: the total energy is',pot+kin)
        print('\tr=',np.sqrt((part.x[0]-part.x[1])**2+(part.y[0]-part.y[1])**2))
        print('\tx=',part.x[0]-ngrid/2,', z=',part.z[0]-ngrid/2)
        # output the z coordinate so we can check if it is 0
    imageio.mimsave('JH_2.gif',image_list,duration=0.02)


#-------------------------------------------------------------------------------

# Part 3 :

peri=True  # change here to get periodic/non-periodic BC

if Part==3:
    # periodic: n=5000; dt=0.005; n1=1000; nj=50
    n=5000
    dt=0.005
    ni=200
    nj=20
    part=particles(m=1,npart=n,dt=dt,dist_type='random')
    image_list=[]    # plot gif
    fig=plt.figure()
    energy_past=[]   # total energy
    pot_past=[]      # total potentail energy
    if not peri:
        n_past=[]    # number of particle
    plt.clf()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot(part.x,part.y,part.z,',',c='blue',alpha=0.4)
    ax.text(ngrid,ngrid,ngrid*1.12,'t=0.00s')
    ax.set_xlim(0,ngrid)
    ax.set_ylim(0,ngrid)
    ax.set_zlim(0,ngrid)
    plt.savefig('0.png')
    plt.show()
    image_list.append(imageio.imread('0.png'))
    for i in range(ni):
        for j in range(nj):
            pot,kin=part.evolve(periodic=peri)
        plt.clf()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(part.x,part.y,part.z,',',c='blue',alpha=0.4)
        t=nj*(i+1)*dt; time='%.2f' % t; 
        ax.text(ngrid,ngrid,ngrid*1.12,'t='+time+'s')
        ax.set_xlim(0,ngrid)
        ax.set_ylim(0,ngrid)
        ax.set_zlim(0,ngrid)
        plt.savefig('0.png')
        image_list.append(imageio.imread('0.png'))
        if peri:
            print(time,'s: energy: ',pot,pot+kin)
        else:
            print(time,'s:  n=',part.opts['n'],', the total energy is',pot+kin)
            n_past.append(part.opts['n'])
        energy_past.append(pot+kin)
        pot_past.append(pot)
        if i%25==0:
            if peri:
                imageio.mimsave('JH_3_peri.gif',image_list,duration=0.05)
            else:
                imageio.mimsave('JH_3_nonperi.gif',image_list,duration=0.04)
            print('Saved')
        if part.opts['n']<1000:
            break
    if peri:
        imageio.mimsave('JH_3_peri.gif',image_list,duration=0.04)
        plt.clf()
        plt.plot(pot_past)
        plt.plot(energy_past)
        plt.legend(['total potential','total enegry'],loc='best')
        plt.savefig('JH_3_peri_energy.png')
    else:
        imageio.mimsave('JH_3_nonperi.gif',image_list,duration=0.04)
        plt.clf()
        plt.plot(pot_past)
        plt.plot(energy_past)
        plt.legend(['total potential','total enegry'],loc='best')
        plt.savefig('JH_3_nonperi_energy.png')
#        plt.clf()
#        plt.plot(n_past)
#        plt.savefig('JH_3_nonperi_particlenumber.png')

" It shows that: "
" 'MemoryError: Unable to allocate 29.8 MiB for an array "
" with shape(125, 125, 125) and data type complex128'!!! "
" so I cannot run more steps. "

#-------------------------------------------------------------------------------

# Part 4 :

if Part==4:
    n=420000
    dt=0.0005
    ni=500
    nj=20
    part=particles(m=0.4,npart=n,dt=dt,dist_type='scaleinv')
    energy_past=[]   # total energy
    pot_past=[]      # total potentail energy
#    fig=plt.figure(figsize=(6,6))
#    plt.clf()
#    plt.plot(part.x,part.y,',',markersize=4,alpha=0.5)
#    plt.title('initial distribution of (x,y)')
#    plt.savefig('JH_4_initial.png')
    image_list=[]    # plot gif
    fig=plt.figure()
    plt.clf()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot(part.x,part.y,part.z,',',c='blue',alpha=0.05)
    ax.text(ngrid,ngrid,ngrid*1.12,'t=0.00s')
    ax.set_xlim(0,ngrid)
    ax.set_ylim(0,ngrid)
    ax.set_zlim(0,ngrid)
    plt.savefig('0.png')
    plt.show()
    image_list.append(imageio.imread('0.png'))
    for i in range(ni):
        for j in range(nj):
            pot,kin=part.evolve(periodic=peri)
        plt.clf()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(part.x,part.y,part.z,',',c='blue',alpha=0.05)
        t=nj*(i+1)*dt; time='%.2f' % t; 
        ax.text(ngrid,ngrid,ngrid*1.12,'t='+time+'s')
        ax.set_xlim(0,ngrid)
        ax.set_ylim(0,ngrid)
        ax.set_zlim(0,ngrid)
        plt.savefig('0.png')
        plt.show()
        image_list.append(imageio.imread('0.png'))
        print(time,'s: the total energy is',pot+kin)
        energy_past.append(pot+kin)
        pot_past.append(pot)
        if i%25==0:
            imageio.mimsave('JH_4.gif',image_list,duration=0.04)
            print('Saved!')
    imageio.mimsave('JH_4.gif',image_list,duration=0.04)
    plt.clf()
    plt.plot(pot_past)
    plt.plot(energy_past)
    plt.legend(['total potential','total enegry'],loc='best')
    plt.savefig('JH_4_energy.png')






