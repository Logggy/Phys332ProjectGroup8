# This will be the file for implementing the Chebyshev FFT method

import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.special as sp

hbar = 1
m    = 1
L    = 2.5
T    = 1000
v    = 10   #number of Chebyshelv expansion terms
# hbar = 1.05457182e-34
# m    = 9.1093837e-31
def TDSE_solve(J,minmaxx,dt0,minmaxt,fBNC,fINC,potential):
    
    
   
    N  = int((minmaxt[1]-minmaxt[0])/dt0)+1
    dt = (minmaxt[1]-minmaxt[0])/float(N-1) # recalculate, to make exact
    dx = (minmaxx[1]-minmaxx[0])/float(J)
    x  = minmaxx[0]+(np.arange(J)+0.5)*dx
    t  = minmaxt[0]+np.arange(N)*dt
    
    potential_func=np.zeros(x.size)
    for i in range(x.size):
        potential_func[i]=potential(x[i])
    Emin=np.min(potential_func)
    Emax=((hbar**2*np.pi**2)/(2*m*dx**2))+np.max(potential_func)
    
    deltaE=(Emax-Emin)/2
    a=np.zeros(v,dtype=complex)
    a[0]=np.exp(-1j*(deltaE+Emin)*dt/hbar)*sp.jv(0,deltaE*dt/hbar)
    for i in range(1,v):
        a[i]=np.exp(-1j*(deltaE+Emin)*dt/hbar)*2*sp.jv(i,deltaE*dt/hbar)
    print(a)
    print('[TDSE_solve]: E_min = %13.5e' % (Emin)) 
    print('[TDSE_solve]: E_max = %e    ' % (Emax))
    print('[TDSE_solve]: N     = %7i' % (N))
    y        = CBsolver(x,t,dx,dt,a,Emin,Emax,deltaE,fBNC,fINC,potential)
    return x,t,y


def CBsolver(x,t,dx,dt,a,Emin,Emax,deltaE,fBNC,fINC,potential):
    J        = x.size
    N        = t.size
    y        = np.zeros((J+2,N),dtype=complex)
    
    
    xb=np.zeros(J+2,dtype=complex)
    xb[0]=x[0]-dx
    xb[-1]=x[-1]+dx
    for i in range(J):
        xb[i+1]=x[i]
    
    y[:,0]=fINC(xb)
    y[0,0]=fBNC(0,y[:,0])
    y[-1,0]=fBNC(1,y[:,0])
    
    
    

    for n in range(1,N):
        expansion=np.zeros([xb.size,v],dtype=complex)
        expansion[:,0]=y[:,n-1]
        expansion[:,1]=1j*(-1/deltaE)*Hamiltonian(xb,expansion[:,0],dx,potential)+1j*(1+(Emin/deltaE))*expansion[:,0]
        for m in range(2,v):
            expansion[:,m]=1j*(-2/deltaE)*Hamiltonian(xb,expansion[:,m-1],dx,potential)+1j*(2+(2*Emin/deltaE))*expansion[:,m-1]+expansion[:,m-2]

        sum=0
        for m in range(v):
            sum=sum+a[m]*dt*expansion[:,m]
        
        y[:,n]=sum
        y[0,n]=fBNC(0,y[:,n])
        y[-1,n]=fBNC(1,y[:,n])
            

    return y[1:J+1,:]


def init(problem,inc):
    if (problem == 'well'):
        fBNC    = Bnon
        potential = Vwell
        minmaxx = np.array([-L,L])
        minmaxt = np.array([0.0,T])
    if (problem == 'free'):
        fBNC    = Bnon
        potential = Vfree
        minmaxx = np.array([-L,L])
        minmaxt = np.array([0.0,T])
    if (problem == 'wall'):
        fBNC    = Bnon
        potential = Vwall
        minmaxx = np.array([-L,L])
        minmaxt = np.array([0.0,T])  
    else:
        print('[init]: invalid problem %s' % (problem))


    if (inc=='gaussian'):
        fINC    = gaussian_wavepacket


    return fBNC,fINC,potential,minmaxx,minmaxt 

#-------------------------------------------------------

def Vfree(x):
    return np.zeros(x.size)


def Vwell(x):

   if x<-L/2 or x>L/2:
       return 1e5
   else:
       return 0 
       
def Vwall(x):
    if x>L/2 and x<L/2+0.5:
        return 1e5
    else:
        return 0
def Bnon(iside,y):
    if(iside==0):
        return y[1]
    else:
        return y[-2]

    
def gaussian_wavepacket(x, a=0.3, x0=0, k0=0):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """ 
    return ((a * np.sqrt(np.pi)) ** (-0.5)
            * np.exp(-0.5 * ((x - x0) * 1. / a) ** 2 + 1j * x * k0))


#-----------------------------------------------------------


def dydx2(x,psi):
    dx=x[1]-x[0]
    secondderiv = np.zeros(x.size, dtype=complex)
    for i in range(1, x.size - 1):
        secondderiv[i] = (psi[i+1] + psi[i-1] - 2. * psi[i]) / (dx * dx)
    secondderiv[0]         = (psi[1] + psi[x.size-1] - 2. * psi[0]) / (dx * dx)  
    secondderiv[x.size-1] = (psi[0] + psi[x.size-2] - 2. * psi[x.size-1]) / (dx * dx) 
    secondderiv[0]=secondderiv[1]
    secondderiv[-1]=secondderiv[-2]
    return secondderiv

def Hamiltonian(x,psi,dx,potential):
    potential_func=np.zeros(x.size,dtype=complex)
    for i in range(x.size):
        potential_func[i]=potential(x[i])
    return (-hbar**2/(2*m))*dydx2(x,psi)+psi*potential_func
    








def update(i,x,y,y1,y2,y3,line1,line2,line3):
    y1 = np.abs(y[:,i+1])
    y2 = y[:,i+1].real
    y3 = y[:,i+1].imag
    line1.set_data(x,y1)
    line2.set_data(x,y2)
    line3.set_data(x,y3)


def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of spatial support points (including boundaries)")
    parser.add_argument("dt",type=float,
                        help="timestep")
    parser.add_argument("problem",type=str,
                        help="potential function:\n"
                             "    free    : constant 0 potential\n"
                             "    well    : potential well\n      "
                             '    wall    : tunneling  '
                             )
    parser.add_argument("inc", type=str,
                        help="initial condition:\n"
                             "    gaussian   : gaussian wavepacket")

    args         = parser.parse_args()
    J            = args.J
    dt           = args.dt
    problem      = args.problem
    inc          = args.inc

    fBNC,fINC,potential,minmaxx,minmaxt = init(problem,inc)
    x,t,y        = TDSE_solve(J,minmaxx,dt,minmaxt,fBNC,fINC,potential)
    
    
    potential_func=np.zeros(x.size)
    for i in range(x.size):
        potential_func[i]=potential(x[i])
   
    fig1, (ax1,ax2,ax3) = plt.subplots(nrows=3,figsize=(15,15))
    y1=np.abs(y[:,0])
    y2=y[:,0].real
    y3=y[:,0].imag
    line1,=ax1.plot(x,y1)
    line2,=ax2.plot(x,y2)
    line3,=ax3.plot(x,y3)
    ax1.set_xlim(minmaxx)
    ax1.set_ylim([-1,1])
    ax2.set_xlim(minmaxx)
    ax2.set_ylim([-1,1])
    ax3.set_xlim(minmaxx)
    ax3.set_ylim([-1,1])
    
    ax1.set_xlabel("x")
    ax1.set_ylabel("Ψ^2")
    ax2.set_xlabel("x")
    ax2.set_ylabel("Real part of Ψ")
    ax3.set_xlabel('x')
    ax3.set_ylabel('Imag part of Ψ')
    
    ax1.plot(x,potential_func)
    ax2.plot(x,potential_func)
    ax3.plot(x,potential_func)
    anim = FuncAnimation(fig1, update, frames=t.size-1, repeat=True,fargs=(x,y,y1,y2,y3,line1,line2,line3))  
    anim.save('simulation.mp4', fps=1, extra_args=['-vcodec', 'libx264'])
  
    
    # x=np.arange(0,3*np.pi,np.pi/10)
    # y=np.sin(x)
    # plt.plot(x,y)
    # plt.plot(x,dydx2(x,y))
    
    
    
    
  
    
        
#========================================

main()


    