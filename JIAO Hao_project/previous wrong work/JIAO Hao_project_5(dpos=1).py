import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import time
from numba import njit

plt.ion()

#-------------------------------------------------------------------------------
# Set the # of problems
Part=2
#-------------------------------------------------------------------------------

# Parameters
ngrid=125
G=1

#-------------------------------------------------------------------------------


def make_Ax(x0,mask):
    x=x0.copy()
    x[mask]=0
    tot=np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)
    tot=tot+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)
    tot=tot+np.roll(x,1,axis=2)+np.roll(x,-1,axis=2)
    x=(tot/6-x)  # Ax
    x[mask]=0
    return x
def cg(b,x0,mask,ninter=800):
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
            return x
    print('not very precise!')
    return x
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

def get_potential(x,y,z,m,G,periodic=True):
    n=len(x)
    energy=0
    if periodic:
        density=np.zeros([ngrid,ngrid,ngrid]) # N is difined at the beginning
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
        density=np.zeros([ngrid+2,ngrid+2,ngrid+2])
        for i in range(n):
            pos_x=int(x[i])
            pos_y=int(y[i])
            pos_z=int(z[i])
            density[pos_x+1,pos_y+1,pos_z+1]+=m[i]
        density=density
        mask=np.zeros([ngrid+2,ngrid+2,ngrid+2],dtype='bool')
        mask[0,:,:]=True
        mask[-1,:,:]=True
        mask[:,0,:]=True
        mask[:,-1,:]=True
        mask[:,:,0]=True
        mask[:,:,-1]=True
        pot=cg(4*np.pi*G*density,0*density,mask)
        energy=np.sum(density*pot)
        return pot,energy

def get_force(x,y,z,fx,fy,fz,pot,periodic=True):
    for i in range(len(x)):
        pos_x=int(x[i]); xx=x[i]-pos_x
        pos_y=int(y[i]); yy=y[i]-pos_y
        pos_z=int(z[i]); zz=z[i]-pos_z
        if periodic:
#            fx[i]=(1-xx)*(pot[(pos_x-1)%ngrid,pos_y,pos_z]-pot[pos_x,pos_y,pos_z])\
#                   +xx*(pot[pos_x,pos_y,pos_z]-pot[(pos_x+1)%ngrid,pos_y,pos_z])
#            fy[i]=(1-yy)*(pot[pos_x,(pos_y-1)%ngrid,pos_z]-pot[pos_x,pos_y,pos_z])\
#                   +yy*(pot[pos_x,pos_y,pos_z]-pot[pos_x,(pos_y+1)%ngrid,pos_z])
#            fz[i]=(1-zz)*(pot[pos_x,pos_y,(pos_z-1)%ngridpoten]-pot[pos_x,pos_y,pos_z])\
#                   +zz*(pot[pos_x,pos_y,pos_z]-pot[pos_x,pos_y,(pos_z+1)%ngrid])
            fx[i]=(pot[(pos_x-1)%ngrid,pos_y,pos_z]-pot[(pos_x+1)%ngrid,pos_y,pos_z])/2
            fy[i]=(pot[pos_x,(pos_y-1)%ngrid,pos_z]-pot[pos_x,(pos_y+1)%ngrid,pos_z])/2
            fz[i]=(pot[pos_x,pos_y,(pos_z-1)%ngrid]-pot[pos_x,pos_y,(pos_z+1)%ngrid])/2
        else:
            fx[i]=(pot[(pos_x-1)+1,pos_y+1,pos_z+1]-pot[(pos_x+1)+1,pos_y+1,pos_z+1])/2
            fy[i]=(pot[pos_x+1,(pos_y-1)+1,pos_z+1]-pot[pos_x+1,(pos_y+1)+1,pos_z+1])/2
            fz[i]=(pot[pos_x+1,pos_y+1,(pos_z-1)+1]-pot[pos_x+1,pos_y+1,(pos_z+1)+1])/2
            # Only this can get rid of the potential from it self
            # (and particle in the same grid)

def density_part4(n):
    dk=np.arange(n)
    dk[n//2:]=dk[n//2:]-n
    kx,ky,kz=np.meshgrid(dk,dk,dk)
    k=np.sqrt(kx*kx+ky*ky+kz*kz)
    k[0,0,0]=1
    dnes_ft=1/(k*k*k)
    dens=np.real(np.fft.ifftn(dens_ft))
    dens=dens
    dens=np.fft.fftshift(dens)
    return dens

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
        if dist_type=='gaussian':
            self.x=ngrid/2+ngrid*np.random.randn(self.opts['n'])/10
            self.y=ngrid/2+ngrid*np.random.randn(self.opts['n'])/10
            self.z=ngrid/2+ngrid*np.random.randn(self.opts['n'])/10
            ind1=np.abs(self.x)<(ngrid)/2  # constrain the particles in [-5,5]
            ind2=np.abs(self.y)<(ngrid)/2
            ind3=np.abs(self.z)<(ngrid)/2
            self.x=self.x[ind1&ind2&ind3]
            self.y=self.y[ind1&ind2&ind3]
            self.z=self.z[ind1&ind2&ind3]
            self.opts['n']=len(self.x)
        if dist_type=='circular':
            if self.opts['n']!=2:
                print('Not 2 particles! Cannot generate circular orbit!')
                assert(1==0)
            self.x=np.zeros(2)
            self.y=np.ones(2)*ngrid/2
            self.z=1*self.y
            self.x[0]=ngrid/2+ngrid/10; self.x[1]=ngrid/2-ngrid/10
        if dist_type=='scalarinv':
            self.opts['n']=ngrid**3
            dx=np.arange(n+1)
            dx=dx[1:]+dx[:-1]  # in the center of grid cells
            xmat,ymat,zmat=np.meshgrid(dx,dx,dx)
            part.x=xmat.flatten()
            part.y=ymat.flatten()
            part.z=zmat.flatten()
            part.m=density_part4(ngrid).flatten()
        # initial velocity
        self.vx=np.zeros(self.opts['n'])
        self.vy=np.zeros(self.opts['n'])
        self.vz=np.zeros(self.opts['n'])
        if dist_type=='circular':
            self.vy[0]=np.sqrt(self.opts['G']*self.m[1]/(2*np.abs(self.x[0]-self.x[1])))
            self.vy[1]=-np.sqrt(self.opts['G']*self.m[0]/(2*np.abs(self.x[0]-self.x[1])))
        self.fx=np.zeros(self.opts['n'])
        self.fy=np.zeros(self.opts['n'])
        self.fz=np.zeros(self.opts['n'])
        
    def get_forces(self,periodic=True):
        potential,energy=get_potential(self.x,self.y,self.z,self.m,\
                                       self.opts['G'],periodic=periodic)
        get_force(self.x,self.y,self.z,self.fx,self.fy,self.fz,potential)
        return energy
        
    def evolve(self,periodic=True):
        dt=self.opts['dt']
        x=self.x; y=self.y; z=self.z
        if periodic:
            self.x=(self.x+0.5*self.vx*dt)%ngrid
            self.y=(self.y+0.5*self.vy*dt)%ngrid
            self.z=(self.z+0.5*self.vz*dt)%ngrid
        else:
            self.x=self.x+0.5*self.vx*dt  # x
            ind=self.x%ngrid==0
            self.x[ind]=self.x[ind]+0.01
            ind=self.x>ngrid
            self.x[ind]=ngrid-self.x[ind]%ngrid; self.vx[ind]=-self.vx[ind]
            ind=self.x<0
            self.x[ind]=(-self.x[ind])%ngrid; self.vx[ind]=-self.vx[ind]
            self.y=self.y+0.5*self.vy*dt  # y
            ind=self.y%ngrid==0
            self.y[ind]=self.y[ind]+0.01
            ind=self.y>ngrid
            self.y[ind]=ngrid-self.y[ind]%ngrid; self.vy[ind]=-self.vy[ind]
            ind=self.y<0
            self.y[ind]=(-self.y[ind])%ngrid; self.vy[ind]=-self.vy[ind]
            self.z=self.z+0.5*self.vz*dt  #z
            ind=self.z%ngrid==0
            self.z[ind]=self.z[ind]+0.01
            ind=self.z>ngrid
            self.z[ind]=ngrid-self.z[ind]%ngrid; self.vz[ind]=-self.vz[ind]
            ind=self.z<0
            self.z[ind]=(-self.z[ind])%ngrid; self.vz[ind]=-self.vz[ind]
        energy_pot=self.get_forces(periodic=periodic) # get the force
        vvx=self.vx+0.5*dt*self.fx
        vvy=self.vy+0.5*dt*self.fy
        vvz=self.vz+0.5*dt*self.fz
        if periodic:
            self.x=(x+dt*vvx)%ngrid
            self.y=(y+dt*vvy)%ngrid
            self.z=(z+dt*vvz)%ngrid
        else:
            self.x=x+dt*vvx  # x
            ind=self.x%ngrid==0
            self.x[ind]=self.x[ind]+0.01
            ind=self.x>ngrid
            self.x[ind]=ngrid-self.x[ind]%ngrid; self.vx[ind]=-self.vx[ind]
            ind=self.x<0
            self.x[ind]=(-self.x[ind])%ngrid; self.vx[ind]=-self.vx[ind]
            self.y=y+dt*vvy  # y
            ind=self.y%ngrid==0
            self.y[ind]=self.y[ind]+0.01
            ind=self.y>ngrid
            self.y[ind]=ngrid-self.y[ind]%ngrid; self.vy[ind]=-self.vy[ind]
            ind=self.y<0
            self.y[ind]=(-self.y[ind])%ngrid; self.vy[ind]=-self.vy[ind]
            self.z=z+dt*vvz  #z
            ind=self.z%ngrid==0
            self.z[ind]=self.z[ind]+0.01
            ind=self.z>ngrid
            self.z[ind]=ngrid-self.z[ind]%ngrid; self.vz[ind]=-self.vz[ind]
            ind=self.z<0
            self.z[ind]=(-self.z[ind])%ngrid; self.vz[ind]=-self.vz[ind]
        self.vx+=dt*self.fx
        self.vy+=dt*self.fy
        self.vz+=dt*self.fz
        energy_kin=np.sum(self.m*(vvx*vvx+vvy*vvy+vvz*vvz))/2
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
        t=i*dt; time='%.2f' % t; 
        ax.text(ngrid,ngrid,ngrid*1.12,'t='+time+'s')
        ax.set_xlim(0,ngrid)
        ax.set_ylim(0,ngrid)
        ax.set_zlim(0,ngrid)
        plt.savefig('0.png')
        image_list.append(imageio.imread('0.png'))
        print(time,'s: the total energy is',pot+kin)
        print('position:',part.x[0],part.y[0],part.z[0])
    imageio.mimsave('JH_1.gif',image_list,duration=0.2)


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
    for i in range(ni):
        for j in range(nj):
            pot,kin=part.evolve(periodic=True)
            x0_past.append(part.x[0])
            x1_past.append(part.x[1])
            y0_past.append(part.y[0])
            y1_past.append(part.y[1])
        plt.clf()
        ax=fig.add_subplot(111)
        ax.plot(x0_past,y0_past,',',c='lightsteelblue')
        ax.plot(part.x[0],part.y[0],'.',c='blue')
        ax.plot(x1_past,y1_past,',',c='lightgreen')
        ax.plot(part.x[1],part.y[1],'.',c='green')
        ax.plot(ngrid/2,ngrid/2,'.',c='lightgray')
        # since the two particle only move in xy-plane, we don't show z axis
        t=nj*(i+1)*dt; time='%.2f' % t; 
        ax.text(ngrid*0.8,ngrid*0.95,'t='+time+'s')
        ax.set_xlim(0,ngrid)
        ax.set_ylim(0,ngrid)
        plt.savefig('0.png')
        image_list.append(imageio.imread('0.png'))
        print(time,'s: the total energy is',pot+kin)
        print('\tr=',np.sqrt((part.x[0]-part.x[1])**2+(part.y[0]-part.y[1])**2))
        print('\tx=',part.x[0]-ngrid/2,', z=',part.z[0]-ngrid/2)
        if i%25==0:
            imageio.mimsave('JH_2.gif',image_list,duration=0.001)
            print('Saved')
        # output the z coordinate so we can 
    imageio.mimsave('JH_2.gif',image_list,duration=0.015)


#-------------------------------------------------------------------------------

# Part 3 : 
if Part==3:
    n=5000
    dt=0.005
    ni=2500
    nj=20
    part=particles(m=1,npart=n,dt=dt,dist_type='random')
    image_list=[]
    fig=plt.figure()
    energy_past=[]
    pot_past=[]
    for i in range(ni):
        for j in range(nj):
            pot,kin=part.evolve(periodic=False)  # change here to get non-periodic BC
        plt.clf()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(part.x,part.y,part.z,',',c='blue',alpha=0.4)
        t=nj*i*dt; time='%.2f' % t; 
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
    #imageio.mimsave('JH_3_peri.gif',image_list,duration=0.025)
    imageio.mimsave('JH_3_nonperi.gif',image_list,duration=0.025)


#-------------------------------------------------------------------------------

# Part 4 : 
if Part==4:
    n=5000
    dt=0.005
    ni=1000
    nj=50
    part=particles(m=1,npart=n,dt=dt,dist_type='')
    image_list=[]
    fig=plt.figure()
    for i in range(ni):
        for j in range(nj):
            pot,kin=part.evolve(periodic=True)  # change here to get non-periodic BC
        plt.clf()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(part.x,part.y,part.z,',',c='blue',alpha=0.4)
        t=nj*i*dt; time='%.2f' % t; 
        ax.text(ngrid,ngrid,ngrid*1.12,'t='+time+'s')
        ax.set_xlim(0,ngrid)
        ax.set_ylim(0,ngrid)
        ax.set_zlim(0,ngrid)
        plt.savefig('0.png')
        plt.show()
        image_list.append(imageio.imread('0.png'))
        print(time,'s: the total energy is',pot+kin)
    imageio.mimsave('JH_3_peri.gif',image_list,duration=0.025)







