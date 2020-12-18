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
ngrid=45
pos_min=-4.5
pos_max=4.5
dpos=(pos_max-pos_min)/ngrid
N=ngrid//2
soft=0.0001
G=1

#-------------------------------------------------------------------------------


def make_Ax(x0,mask):
    x=x0.copy()
    x[mask]=0
    tot=np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)
    tot=tot+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)
    tot=tot+np.roll(x,1,axis=2)+np.roll(x,-1,axis=2)
    x=(tot/6-x)/(dpos*dpos)  # Ax
    x[mask]=0
    return x
def cg(b,x0,mask,ninter=500):
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
    return x
def make_Green(n,N):  # N is the number of additional grid
    mask1=np.zeros([n+2*N,n+2*N,n+2*N],dtype='bool')
    mask1[0,:,:]=True
    mask1[-1,:,:]=True
    mask1[:,0,:]=True
    mask1[:,-1,:]=True
    mask1[:,:,0]=True
    mask1[:,:,-1]=True
    source=np.zeros([n+2*N,n+2*N,n+2*N])
    source[n//2+N,n//2+N,n//2+N]=4*np.pi*G
    Green=cg(source,0*source,mask1)
    Green=np.roll(Green,n//2+N+1,axis=0)
    Green=np.roll(Green,n//2+N+1,axis=1)
    Green=np.roll(Green,n//2+N+1,axis=2)
    return Green
Green=make_Green(ngrid,N)
# The center of Green's function is not infinity, so it's softed

def get_potential(x,y,z,m,G,soft=0.0001,periodic=True):
    n=len(x)
    energy=0
    if periodic:
        density=np.zeros([ngrid+2*N,ngrid+2*N,ngrid+2*N]) # N is difined at the beginning
        for i in range(n):
            pos_x=int((x[i]-pos_min)/dpos)
            pos_y=int((y[i]-pos_min)/dpos)
            pos_z=int((z[i]-pos_min)/dpos)
            density[pos_x+N,pos_y+N,pos_z+N]+=m[i]
        density=density/(dpos*dpos*dpos)
        pot=G*np.fft.ifftn(np.fft.fftn(density)*np.fft.fftn(Green))
        pot=np.real(pot)
        energy=np.sum(density*pot)
        return pot[N-1:-N+1,N-1:-N+1,N-1:-N+1],energy
    else:
        density=np.zeros([ngrid+2,ngrid+2,ngrid+2])
        for i in range(n):
            pos_x=int((x[i]-pos_min)/dpos)
            pos_y=int((x[i]-pos_min)/dpos)
            pos_z=int((x[i]-pos_min)/dpos)
            density[pos_x+1,pos_y+1,pos_z+1]+=m[i]
        density=density/(dpos*dpos*dpos)
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
    n=len(x)
    for i in range(n):
        xx=(x[i]-pos_min)/dpos; pos_x=int(xx); xx=xx-pos_x
        yy=(y[i]-pos_min)/dpos; pos_y=int(yy); yy=yy-pos_y
        zz=(z[i]-pos_min)/dpos; pos_z=int(zz); zz=zz-pos_z
#        fx[i]=(1-xx)*(pot[(pos_x-1)%ngrid,pos_y,pos_z]-pot[pos_x,pos_y,pos_z])\
#               +xx*(pot[pos_x,pos_y,pos_z]-pot[(pos_x+1)%ngrid,pos_y,pos_z])
#        fy[i]=(1-yy)*(pot[pos_x,(pos_y-1)%ngrid,pos_z]-pot[pos_x,pos_y,pos_z])\
#               +yy*(pot[pos_x,pos_y,pos_z]-pot[pos_x,(pos_y+1)%ngrid,pos_z])
#        fz[i]=(1-zz)*(pot[pos_x,pos_y,(pos_z-1)%ngrid]-pot[pos_x,pos_y,pos_z])\
#               +zz*(pot[pos_x,pos_y,pos_z]-pot[pos_x,pos_y,(pos_z+1)%ngrid])
        fx[i]=(pot[(pos_x-1)+1,pos_y+1,pos_z+1]-pot[(pos_x+1)+1,pos_y+1,pos_z+1])/2
        fy[i]=(pot[pos_x+1,(pos_y-1)+1,pos_z+1]-pot[pos_x+1,(pos_y+1)+1,pos_z+1])/2
        fz[i]=(pot[pos_x+1,pos_y+1,(pos_z-1)+1]-pot[pos_x+1,pos_y+1,(pos_z+1)+1])/2
    fx=fx/dpos
    fy=fy/dpos
    fz=fz/dpos

class particles:
    def __init__(self,m=1.0,npart=1000,soft=0.0001,G=1.0,dt=0.1,dist_type='random'):
        self.opts={}  #options
        self.opts['n']=npart
        self.opts['G']=G
        self.opts['dt']=dt
        self.opts['soft']=soft
        self.m=np.ones(self.opts['n'])*m
        # initial position (with different distribution)
        if dist_type=='random':
            self.x=pos_min+(pos_max-pos_min)*np.random.rand(self.opts['n'])
            self.y=pos_min+(pos_max-pos_min)*np.random.rand(self.opts['n'])
            self.z=pos_min+(pos_max-pos_min)*np.random.rand(self.opts['n'])
        if dist_type=='gaussian':
            self.x=(pos_min+pos_max)/2+np.random.randn(self.opts['n'])
            self.y=(pos_min+pos_max)/2+np.random.randn(self.opts['n'])
            self.z=(pos_min+pos_max)/2+np.random.randn(self.opts['n'])
            ind1=np.abs(self.x)>(pos_max-pos_min)/2  # constrain the particles in [-5,5]
            ind2=np.abs(self.y)>(pos_max-pos_min)/2
            ind3=np.abs(self.z)>(pos_max-pos_min)/2
            self.x=self.x[ind1&ind2&ind3]
            self.y=self.y[ind1&ind2&ind3]
            self.z=self.z[ind1&ind2&ind3]
            self.opts['n']=len(self.x)
        if dist_type=='circular':
            if self.opts['n']!=2:
                print('Not 2 particles! Cannot generate circular orbit!')
                assert(1==0)
            self.x=np.zeros(2)
            self.y=np.zeros(2)
            self.z=np.zeros(2)
            #r=0.25*(pos_max-pos_min)*np.random.rand()
            #self.x[0]=(pos_min+pos_max)/2+r; self.x[1]=(pos_min+pos_max)/2-r
            self.x[0]=2; self.x[1]=-2
        # initial velocity
        self.vx=0*self.x
        self.vy=self.vx.copy()
        self.vz=self.vx.copy()
        if dist_type=='circular':
            self.vy[0]=np.sqrt(self.opts['G']*self.m[1]/(4*self.x[0]))
            self.vy[1]=-np.sqrt(self.opts['G']*self.m[0]/(4*self.x[0]))
        self.fx=np.zeros(self.opts['n'])
        self.fy=np.zeros(self.opts['n'])
        self.fz=np.zeros(self.opts['n'])
        
    def get_forces(self,periodic=True):
        potential,energy=get_potential(self.x,self.y,self.z,\
                                       self.m,self.opts['G'],periodic=periodic)
        get_force(self.x,self.y,self.z,self.fx,self.fy,self.fz,potential)
        return energy
        
    def evolve(self,periodic=True):
        dt=self.opts['dt']
        x=self.x; y=self.y; z=self.z
        self.x+=0.5*self.vx*dt
        self.y+=0.5*self.vy*dt
        self.z+=0.5*self.vz*dt
        energy_pot=self.get_forces(periodic=periodic) # get the force
        vvx=self.vx+0.5*dt*self.fx
        vvy=self.vy+0.5*dt*self.fy
        vvz=self.vz+0.5*dt*self.fz
        if periodic:
            self.x=(x+dt*vvx-pos_min)%(pos_max-pos_min)+pos_min
            self.y=(y+dt*vvy-pos_min)%(pos_max-pos_min)+pos_min
            self.z=(z+dt*vvz-pos_min)%(pos_max-pos_min)+pos_min
        else:
            self.x=x+dt*vvx
            ind=self.x>pos_max
            self.x[ind]=2*pos_max-self.x[ind]
            ind=self.x<pos_min
            self.x[ind]=2*pos_min-self.x[ind]
            self.y=y+dt*vvy
            ind=self.y>pos_max
            self.y[ind]=2*pos_max-self.y[ind]
            ind=self.y<pos_min
            self.y[ind]=2*pos_min-self.y[ind]
            self.z=z+dt*vvz
            ind=self.z>pos_max
            self.z[ind]=2*pos_max-self.z[ind]
            ind=self.z<pos_min
            self.z[ind]=2*pos_min-self.z[ind]
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
        ax.text(pos_max,pos_max,pos_max+1,'t='+time+'s')
        ax.set_xlim(pos_min,pos_max)
        ax.set_ylim(pos_min,pos_max)
        ax.set_zlim(pos_min,pos_max)
        plt.savefig('0.png')
        plt.show()
        image_list.append(imageio.imread('0.png'))
        print(pot+kin)
    imageio.mimsave('JH_1.gif',image_list,duration=0.2)


#-------------------------------------------------------------------------------

# Part 2 : a pair particle in a circular orbit
if Part==2:
    dt=0.002
    ni=250
    nj=50
    part=particles(m=1,npart=2,dt=dt,dist_type='circular')
    image_list=[]
    x0_past=[]
    x1_past=[]
    y0_past=[]
    y1_past=[]
    fig=plt.figure(figsize=(6,6))
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
        ax.plot(0,0,'.',c='lightgray')
        # since the two particle only move in xy-plane, we don't show z axis
        t=nj*i*dt; time='%.2f' % t; 
        ax.text(pos_max-1.5,pos_max-1,'t='+time+'s')
        ax.set_xlim(pos_min,pos_max)
        ax.set_ylim(pos_min,pos_max)
        plt.savefig('0.png')
        image_list.append(imageio.imread('0.png'))
        print(time,'s: the total energy is',pot+kin)
    imageio.mimsave('JH_2.gif',image_list,duration=0.2)


#-------------------------------------------------------------------------------

# Part 3 : 
if Part==2:
    n=1000
    dt=0.002
    ni=1000
    nj=50
    part=particles(m=1,npart=n,dt=dt,dist_type='random')
    image_list=[]
    fig=plt.figure()
    for i in range(ni):
        for j in range(nj):
            pot,kin=part.evolve(periodic=True)
        plt.clf()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(part.x,part.y,part.z,',',c='blue')
        t=nj*i*dt; time='%.2f' % t; 
        ax.text(pos_max,pos_max,pos_max+1,'t='+time+'s')
        ax.set_xlim(pos_min,pos_max)
        ax.set_ylim(pos_min,pos_max)
        ax.set_zlim(pos_min,pos_max)
        plt.savefig('0.png')
        plt.show()
        image_list.append(imageio.imread('0.png'))
        print(time,'s: the total energy is',pot+kin)
    imageio.mimsave('JH_3_peri.gif',image_list,duration=0.2)








