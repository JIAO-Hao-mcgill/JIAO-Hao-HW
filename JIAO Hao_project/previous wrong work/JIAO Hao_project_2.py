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
ngrid=41
pos_min=-5
pos_max=5
dpos=(pos_max-pos_min)/ngrid
soft=0.0001
G=1

# grid & r: don't need to calculate for many times
# For periodic case
grid=np.linspace(pos_min,pos_max,ngrid+1)
grid=grid[:-1]+grid[1:]
N=ngrid//2
grid1=np.linspace(1.5*pos_min-0.5*pos_max,-0.5*pos_min+1.5*pos_max,ngrid+1+2*N)
grid1=grid1[:-1]+grid1[1:]
r1=np.zeros([2*N+ngrid,2*N+ngrid,2*N+ngrid])
for ix in range(2*N+ngrid):
    for iy in range(2*N+ngrid):
        for iz in range(2*N+ngrid):
            r1[ix,iy,iz]=np.sqrt(grid1[ix]**2+grid1[iy]**2+grid1[iz]**2+soft)
#r=r1[N:-N,N:-N,N:-N]
r1_inv=1/r1
r1_inv_ft=np.fft.fftn(r1_inv)

# mask: don't need to calculate for many times
# For non-periodic case
mask=np.zeros([ngrid+2,ngrid+2,ngrid+2],dtype='bool')
mask[0,:,:]=True
mask[-1,:,:]=True
mask[:,0,:]=True
mask[:,-1,:]=True
mask[:,:,0]=True
mask[:,:,-1]=True

#-------------------------------------------------------------------------------


def make_Ax(x0):
    if mask.shape!=x0.shape:
        print('the shape of mask have problem!')
        assert(1==0)
    x=x0.copy()
    x[mask]=0
    tot=np.roll(x,1,axis=0)+np.roll(x,-1,axis=0)
    tot=tot+np.roll(x,1,axis=1)+np.roll(x,-1,axis=1)
    tot=tot+np.roll(x,1,axis=2)+np.roll(x,-1,axis=2)
    x=x-tot/6.  # Ax
    x[mask]=0
    return x
def cg(b,x0,ninter=500):
    Ax=make_Ax(x0)
    rk=b-Ax
    pk=rk.copy()
    x=x0.copy()
    rtr=np.sum(rk*rk)
    for i in range(ninter):
        Apk=make_Ax(pk)
        pAp=np.sum(pk*Apk)
        ak=rtr/pAp
        x=x+ak*pk
        rk_new=rk-ak*Apk
        rtr_new=np.sum(rk_new*rk_new)
        bk=rtr_new/rtr
        pk=rk_new+bk*pk
        rk=rk_new
        rtr=rtr_new
        if rtr<1e-6:
            return x
    return x

def get_potential(x,y,z,m,G,soft=0.0001,ngrid=100,periodic=True):
    n=len(x)
    energy=0
    if periodic:
        density=np.zeros([ngrid+2*N,ngrid+2*N,ngrid+2*N]) # N is difined at the beginning
        for i in range(n):
            pos_x=int((x[i]-pos_min)/dpos)
            pos_y=int((y[i]-pos_min)/dpos)
            pos_z=int((z[i]-pos_min)/dpos)
            density[pos_x+N,pos_y+N,pos_z+N]+=m[i]
        pot=G*np.fft.ifftn(np.fft.fftn(density)*r1_inv_ft)
        energy=np.sum(density*pot)
        return np.real(pot[N:-N,N:-N,N:-N]),energy
    else:
        density=np.zeros([ngrid+2,ngrid+2,ngrid+2])
        for i in range(n):
            pos_x=int((x[i]-pos_min)/dpos)
            pos_y=int((x[i]-pos_min)/dpos)
            pos_z=int((x[i]-pos_min)/dpos)
            density[pos_x+1,pos_y+1,pos_z+1]+=m[i]
        pot=cg(G*4*np.pi*density,0*density)
        energy=np.sum(density*pot)
        return pot[1:-1,1:-1,1:-1],energy

def get_force(x,y,z,fx,fy,fz,pot,ngrid=100):
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
        fx[i]=(pot[(pos_x-1)%ngrid,pos_y,pos_z]-pot[(pos_x+1)%ngrid,pos_y,pos_z])/2
        fy[i]=(pot[pos_x,(pos_y-1)%ngrid,pos_z]-pot[pos_x,(pos_y+1)%ngrid,pos_z])/2
        fz[i]=(pot[pos_x,pos_y,(pos_z-1)%ngrid]-pot[pos_x,pos_y,(pos_z+1)%ngrid])/2
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
            self.vy[0]=np.sqrt(self.opts['G']*self.m[1]*np.abs(self.x[0]))
            self.vy[1]=-np.sqrt(self.opts['G']*self.m[0]*np.abs(self.x[0]))
            
    def get_forces(self,periodic=True):
        self.fx=np.zeros(self.opts['n'])
        self.fy=np.zeros(self.opts['n'])
        self.fz=np.zeros(self.opts['n'])
        potential,energy=get_potential(self.x,self.y,self.z,self.m,self.opts['G'],\
                          ngrid=ngrid,periodic=periodic)
        get_force(self.x,self.y,self.z,self.fx,self.fy,self.fz,potential,ngrid=100)
        return energy
        
    def evolve(self,periodic=True):
        dt=self.opts['dt']
        x=self.x; y=self.y; z=self.z
        self.x+=0.5*self.vx*dt
        self.y+=0.5*self.vy*dt
        self.z+=0.5*self.vz*dt
        energy_pot=self.get_forces(periodic=periodic)
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
    dt=0.01
    part=particles(m=1,npart=2,dt=dt,dist_type='circular')
    image_list=[]
    fig=plt.figure()
    for i in range(50):
        for j in range(10):
            pot,kin=part.evolve(periodic=False)
        plt.clf()
        ax=fig.add_subplot(111,projection='3d')
        ax.plot(part.x,part.y,part.z,'.',c='blue')
        t=5*i*dt; time='%.2f' % t; 
        ax.text(pos_max,pos_max,pos_max+1,'t='+time+'s')
        ax.set_xlim(pos_min,pos_max)
        ax.set_ylim(pos_min,pos_max)
        ax.set_zlim(pos_min,pos_max)
        plt.savefig('0.png')
        plt.show()
        image_list.append(imageio.imread('0.png'))
        print(pot+kin)
    imageio.mimsave('JH_2.gif',image_list,duration=0.2)


#-------------------------------------------------------------------------------

# Part 3 : 




