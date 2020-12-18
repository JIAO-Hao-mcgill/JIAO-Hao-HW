import numpy as np
from matplotlib import pyplot as plt
import time
from numba import njit
@njit

Part=1

# Parameters
ngrid=100
soft=0.0001

# grid
grid=np.linspace(-5,5,ngrid+1)
grid=position[:-1]+position[1:]
r=np.zeros([ngrid,ngrid,ngrid])
for ix in range(ngrid):
    for iy in range(ngrid):
        for iz in range(ngrid):
            r[i,j,k]=np.sqrt(grid[i]**2+grid[j]**2+grid[k]**2+soft)


def get_force(x,y,z,fx,fy,fz,m,G,soft=0.0001):  # no G
    n=len(x)
    pot=0
    for i in np.arange(n):
        for j in np.arange(i,n):
            dx=x[i]-x[j]
            dy=y[i]-y[j]
            dz=z[i]-z[j]
            rsqr=dx*dx+dy*dy+dz*dz
            if rsqr<soft:
                rsqr=soft
            #rsqr=rsqr+soft
            rinv=1/np.sqrt(rsqr)
            r3=rinv/rsqr
            fx[i]=fx[i]-G*m[i]*m[j]*dx*r3  # change the force
            fx[j]=fx[j]+G*m[i]*m[j]*dx*r3
            fy[i]=fy[i]-G*m[i]*m[j]*dy*r3
            fy[j]=fy[j]+G*m[i]*m[j]*dy*r3
            fz[i]=fz[i]-G*m[i]*m[j]*dz*r3
            fz[j]=fz[j]+G*m[i]*m[j]*dz*r3
            pot=pot-G*m[i]*m[j]*rinv
    return -1*pot

def get_potential(x,y,z,m,soft=0.0001,ngrid=200,periodic=True):
    n=len(x)
    density=np.zeros([ngrid,ngrid,ngrid])
    dpos=10./ngrid
    pot=0
    for i in np.arange(n):
        pos_x=int(x/dpos)
        pos_y=int(y/dpos)
        pos_z=int(z/dpos)
        density[pos_x,pos_y,pos_z]+=m[i]
    if periodic:
        g=1/r
        potential=np.ifft(np.fft.fftn(density)*np.fft.fftn(g))
    else:
        potential=np.zeros()
        ind=np.nonzero(density)
        for ix in range(ngrid):
            for iy in range(ngrid):
                for iz in range(ngrid):
                    for i in range(len(ind[0])):
                        dx=ix-ind[0][i]
                        dy=iy-ind[1][i]
                        dz=iz-ind[2][i]
                        R=dpos*np.sqrt(dx**2+dy**2+dz**2)  # R=|r-r'|
                        potential[ix,iy,iz]+=density[ind[0][i],ind[1][i],ind[2][i]]/R
                    potential[ix,iy,iz]=potential[ix,iy,iz]*dpos**3

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
            self.x=-5+10*np.random.rand(self.opts['n'])
            self.y=-5+10*np.random.rand(self.opts['n'])
            self.z=-5+10*np.random.rand(self.opts['n'])
        if dist_type=='gaussian':
            self.x=np.random.randn(self.opts['n'])
            self.y=np.random.randn(self.opts['n'])
            self.z=np.random.randn(self.opts['n'])
            ind1=np.abs(self.x)>5  # constrain the particles in [-5,5]
            ind2=np.abs(self.y)>5
            ind3=np.abs(self.z)>5
            self.x=self.x[ind1&ind2&ind3]
            self.y=self.y[ind1&ind2&ind3]
            self.z=self.z[ind1&ind2&ind3]
            self.opts['n']=len(self.x)
        if dist_type=='circular':
            if self.opts['n']!=2:
                print('Not 2 particles! Cannot generate circular orbit!')
                assert(1==0)
            self.x=np.zeros[2]
            self.y=np.zeros[2]
            self.z=np.zeros[2]
            r=-5+10*np.random.rand()
            self.x[0]=r; self.x[1]=-r
        # initial velocity
        self.vx=0*self.x
        self.vy=self.vx.copy()
        self.vz=self.vx.copy()
        if dist_type=='circular':
            self.vy[0]=np.sqrt(self.opts['G']*self.m[1]/np.abs(2*self.x))
            self.vy[1]=np.sqrt(self.opts['G']*self.m[0]/np.abs(2*self.x))
            
    def get_forces(self):
        self.fx=np.zeros(self.opts['n'])
        self.fy=np.zeros(self.opts['n'])
        self.fz=np.zeros(self.opts['n'])
        pot=get_force(self.x,self.y,self.z,self.fx,self.fy,self.fz,\
                      self.m,self.opts['G'],self.opts['soft']**2)
        return pot
    def get_potential(self,ngrid=1000,periodic=True):  # also get the forces
        self.fx=np.zeros(self.opts['n'])
        self.fy=np.zeros(self.opts['n'])
        self.fz=np.zeros(self.opts['n'])
        
    def evolve(self,potent=True):
        self.x+=self.vx*self.opts['dt']
        self.y+=self.vy*self.opts['dt']
        if potent:
            pot=self.get_potential()
            
        else:
            pot=self.get_forces()
        
        self.vx+=self.fx*self.opts['dt']
        self.vy+=self.fy*self.opts['dt']
        kinetic=0.5*np.sum(self.m*(self.vx**2+self.vy**2))
        return pot,kinetic


################################################################################

# Part 1 : one rest particle
if Part==1:
    


#-------------------------------------------------------------------------------

# Part 2 : a pair particle in a circular orbit
if Part==2:





