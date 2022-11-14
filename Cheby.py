# This will be the file for implementing the Chebyshev FFT method

import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import cmath
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, ifft, fftfreq, fftshift

hbar = 1
m    = 1



def TDSE_solve(J,minmaxx,dt0,minmaxt,fBNC,fINC,potential,**kwargs):
    # hbar = 1.05457182e-34
    # m    = 9.1093837e-31
    
   
    N  = int((minmaxt[1]-minmaxt[0])/dt0)+1
    dt = (minmaxt[1]-minmaxt[0])/float(N-1) # recalculate, to make exact
    dx = (minmaxx[1]-minmaxx[0])/float(J)
    x  = minmaxx[0]+(np.arange(J)+0.5)*dx
    t  = minmaxt[0]+np.arange(N)*dt

    Emin=np.min(potential(x))
    Emax=(hbar**2*np.pi**2)/(2*m*dx**2)+np.max(potential(x))
    print('[TDSE_solve]: E_min = %13.5e' % (Emin)) 
    print('[TDSE_solve]: E_max = %e    ' % (Emax))
    y        = CBsolver(x,t,Emin,Emax,fBNC,fINC,potential)
    print('[TDSE_solve]: N     = %7i' % (N))
    return x,t,y


def CBsolver(x,t,Emin,Emax,fBNC,fINC,potential):
    J        = x.size
    N        = t.size
    y        = np.zeros((J+2,N),dtype=complex)

    deltaE=(Emax-Emin)/2
    dx=x[1]-x[0]
    
    # from here ??????
    xb=np.zeros(J+2,dtype=complex)
    xb[0]=x[0]-(x[1]-x[0])
    xb[-1]=x[-1]+(x[1]-x[0])
    for i in range(J):
        xb[i+1]=x[i]
    
    y[:,0]=fINC(xb)
    y[0,0]=fBNC(0,y[:,0])
    y[-1,0]=fBNC(1,y[:,0])

    y[:,1]=complex(0,-1/deltaE)*Hamiltonian(xb,y[:,0],dx,potential)+complex(0,1+Emin/deltaE)*y[:,0]
    y[0,1]=fBNC(0,y[:,1])
    y[-1,1]=fBNC(1,y[:,1])

    for n in range(2,N-1):
        y[:,n]=complex(0,-2/deltaE)*Hamiltonian(xb,y[:,n-1],dx,potential)+complex(0,2+2*Emin/deltaE)*y[:,n-1]+y[:,n-2]
        y[0,n]=fBNC(0,y[:,n])
        y[-1,n]=fBNC(1,y[:,n])
            

    # to here ??????
    return y[1:J+1,:]


def init(problem,inc):
    if (problem == 'well'):
        fBNC    = Bnon
        potential = Vwell
        minmaxx = np.array([-0.5,0.5])
        minmaxt = np.array([0.0,0.5])
    
    else:
        print('[init]: invalid problem %s' % (problem))


    if (inc=='gaussian'):
        fINC    = gaussian_wavepacket


    return fBNC,fINC,potential,minmaxx,minmaxt 

#-------------------------------------------------------

def Vfree(x):
    return np.zeros(x.size)

def Vwell(x):
    a=np.zeros(x.size)
    for i in range(x.size):
        if x[i]<-0.2 or x[i]>0.2:
            a[i]=1e20
        else:
            a[i]=0
    return a
       


def Bnon(iside,y):
    if(iside==0):
        return y[1]
    else:
        return y[-2]

    
def gaussian_wavepacket(x):
    """Gaussian wavepacket at x0 +/- sigma0, with average momentum, p0."""
    A = (2 * np.pi * 0.1**2)**(-0.25)
    return A * np.exp(1j*1*x - ((x - 0)/(2 * 0.1))**2)


#-----------------------------------------------------------

# Finding the operators of the hamiltonian
def Hamiltonian(x,psi,dx,potential):
    k=fftfreq(psi.size,dx)
    k=fftshift(k)
    dydx2=ifft(-4*np.pi**2*k**2*fft(psi)).real
    return (-hbar**2/(2*m))*dydx2+psi*potential(x)
    



def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of spatial support points (including boundaries)")
    parser.add_argument("dt",type=float,
                        help="timestep")
    parser.add_argument("problem",type=str,
                        help="potential function:\n"
                             "    free   : constant 0 potential\n"
                             "    well    : potential well\n      "
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
    
    
    
    
    
    for i in range(8):
      plt.plot(x,abs(y[:,10000*i]))
    plt.show()
    



   
   
      
    # Ploting graph
    
  
    
        
#========================================

main()


    