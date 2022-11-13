#============================================
# Partial differential equations: 
# Diffusion problem.
#============================================
import argparse                  # allows us to deal with arguments to main()
from argparse import RawTextHelpFormatter
import numpy as np
import cmath
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation





#============================================
# Solver for a tridiagonal matrix.
# a,b,c are the lower, center, and upper diagonals,
# r is the RHS vector.
def tridiag(a,b,c,r):
    n    = b.size
    gam  = np.zeros(n,dtype=complex)
    u    = np.zeros(n,dtype=complex)
    bet  = b[0]
    u[0] = r[0]/bet 
    for j in range(1,n):
        gam[j] = c[j-1]/bet
        bet    = b[j]-a[j]*gam[j]
        if (bet == 0.0):
            print('[tridiag]: matrix not invertible.')
            exit()
        u[j]   = (r[j]-a[j]*u[j-1])/bet
    for j in range(n-2,-1,-1):
        u[j] = u[j]-gam[j+1]*u[j+1]
    return u

#============================================
# Driver for the actual integrators. Sets the initial conditions
# and generates the support point arrays in space and time.
# input: J      : number of spatial support points
#        dt0    : timestep
#        minmaxx: 2-element array containing minimum and maximum of spatial domain
#        minmaxt: 2-element array, same for time domain
#        fINT   : integrator (one of ftcs, implicit, cranknicholson)
#        fBNC   : boundary condition function
#        fINC   : potential function
def TDSE_solve(J,minmaxx,dt0,minmaxt,fINT,fBNC,fINC,potential,**kwargs):
    # hbar = 1.05457182e-34
    # m    = 9.1093837e-31
    hbar = 1
    m    = 1
    # for key in kwargs:
    #     if (key=='kappa'):
    #         kappa = kwargs[key]
    # time and space discretization
    N  = int((minmaxt[1]-minmaxt[0])/dt0)+1
    dt = (minmaxt[1]-minmaxt[0])/float(N-1) # recalculate, to make exact
    dx = (minmaxx[1]-minmaxx[0])/float(J)
    x  = minmaxx[0]+(np.arange(J)+0.5)*dx
    t  = minmaxt[0]+np.arange(N)*dt

    q    = (hbar*dt)/(4*m*dx**2)
    r    = dt/(2*hbar)
    print('[TDSE_solve]: q = %13.5e' % (q)) 
    print('[TDSE_solve]: r = %e    ' % (r))
    print('[TDSE_solve]: N     = %7i' % (N))
    y        = fINT(x,t,q,r,fBNC,fINC,potential)
    return x,t,y

#--------------------------------------------
# Forward-time centered-space integrator.
# Returns the full solution array (including 
# initial conditions at t=0). Array should be
# of shape (J,N), with J the spatial and N
# the temporal support points.
def ftcs(x,t,q,r,fBNC,fINC,potential):
    J        = x.size
    N        = t.size
    y        = np.zeros((J+2,N),dtype=complex)
    # from here ??????
    xb=np.zeros(J+2,dtype=complex)
    xb[0]=x[0]-(x[1]-x[0])
    xb[-1]=x[-1]+(x[1]-x[0])
    for i in range(J):
        xb[i+1]=x[i]
    
    y[:,0]=fINC(xb)
    y[0,0]=fBNC(0,y[:,0])
    y[-1,0]=fBNC(1,y[:,0])
    for n in range(N-1):
        for j in range(1,J+1):
            y[j,n+1]=complex(0,q)*y[j-1,n]+(1-complex(0,2*q)-complex(0,r*potential(y[j,n]))*y[j,n]+complex(0,q)*y[j+1,n])
        y[0,n+1]=fBNC(0,y[:,n+1])
        y[-1,n+1]=fBNC(1,y[:,n+1])
            

    # to here ??????
    return y[1:J+1,:]

#--------------------------------------------
# Fully implicit integrator.
# Returns the full solution array (including 
# initial conditions at t=0). Array should be
# of shape (J,N), with J the spatial and N
# the temporal support points.
# Uses tridiag to solve the tridiagonal matrix.
def implicit(x,t,q,r,fBNC,fINC,potential):
    J        = x.size
    N        = t.size
    y        = np.zeros((J+2,N),dtype=complex)
    # from here ??????
    xb=np.zeros(J+2,dtype=complex)
    xb[0]=x[0]-(x[1]-x[0])
    xb[-1]=x[-1]+(x[1]-x[0])
    
    
    tri_mid=np.zeros(J+2,dtype=complex)
    tri_up=np.zeros(J+2,dtype=complex)
    tri_low=np.zeros(J+2,dtype=complex)
    tri_low[:]=-complex(0,q)
    for i in range(J+2):
        tri_mid[i]=1+complex(0,2*q)+complex(0,r*potential(xb[i]))
    tri_up[:]=-complex(0,q)
    
    for i in range(J):
        xb[i+1]=x[i]
    
    y[:,0]=fINC(xb)
    y[0,0]=fBNC(0,y[:,0])
    y[-1,0]=fBNC(1,y[:,0])
    
    for n in range(N-1):
        y[:,n+1]=tridiag(tri_low, tri_mid, tri_up, y[:,n])
        y[0,n+1]=fBNC(0,y[:,n+1])
        y[-1,n+1]=fBNC(1,y[:,n+1])
    # to here ??????
    return y[1:J+1,:]

#--------------------------------------------
# Crank-Nicholson integrator.
# Returns the full solution array (including 
# initial conditions at t=0). Array should be
# of shape (J,N), with J the spatial and N
# the temporal support points.
# Uses tridiag to solve the tridiagonal matrix.
def cranknicholson(x,t,q,r,fBNC,fINC,potential):
    J        = x.size
    N        = t.size
    y        = np.zeros((J+2,N),dtype=complex)
    # from here ??????
    xb=np.zeros(J+2,dtype=complex)
    for i in range(J):
        xb[i+1]=x[i]
    xb[0]=xb[0]-(x[1]-x[0])
    xb[-1]=xb[-1]+(x[1]-x[0])
   
    tri_mid=np.zeros(J+2,dtype=complex)
    tri_up=np.zeros(J+2,dtype=complex)
    tri_low=np.zeros(J+2,dtype=complex)
    
    tri_low[:]=-complex(0,q)
    for i in range(J+2):
        tri_mid[i]=1+complex(0,2*q)+complex(0,r*potential(xb[i]))
    tri_up[:]=-complex(0,q)
      
    
    
    y[:,0]=fINC(xb)
    y[0,0]=fBNC(0,y[:,0])
    y[-1,0]=fBNC(1,y[:,0])
    
    for n in range(N-1):
        RHS=y[:,n]
        for j in range(1,J):
            RHS[j]=complex(0,q)*y[j-1,n]+(complex(1,0)-complex(0,2*q)-complex(0,r*potential(x[j])))*y[j,n]+complex(0,q)*y[j+1,n]
        
        y[:,n+1]=tridiag(tri_low, tri_mid, tri_up, RHS)
        y[0,n+1]=fBNC(0,y[:,n+1])
        y[-1,n+1]=fBNC(1,y[:,n+1])
    # to here ??????
    return y[1:J+1,:]

#============================================

def init(solver,problem,inc):
    if (solver == 'ftcs'):
        fINT = ftcs
    elif (solver == 'implicit'):
        fINT = implicit
    elif (solver == 'CN'):
        fINT = cranknicholson
    else:
        print('[init]: invalid solver %s' % (solver))
 
    if (problem == 'free'):
        potential    = Vfree
        fBNC    = Bnon
        minmaxx = np.array([-0.5,0.5])
        minmaxt = np.array([0.0,0.005])
    if (problem == 'box'):
        potential    = Vbox
        fBNC    = Bnon
        minmaxx = np.array([-0.5,0.5])
        minmaxt = np.array([0.0,0.5])
    else:
        print('[init]: invalid problem %s' % (problem))
        
    if (inc =='gaussian'):
        fINC    = gaussian_wavepacket

    return fINT,fBNC,fINC,potential,minmaxx,minmaxt 

#============================================
# functions for setting the initial conditions (T....)
# and the boundary conditions (B.....)
def Vfree(x):
    return np.zeros(x.size)

def Vbox(x):

   if x<-0.2 or x>0.2:
       return 1e20
   else:
       return 0 


def Bnon(iside,y):
    if(iside==0):
        return y[1]
    else:
        return y[-2]

    
def gaussian_wavepacket(x):
    """Gaussian wavepacket at x0 +/- sigma0, with average momentum, p0."""
    A = (2 * np.pi * 0.1**2)**(-0.25)
    return A * np.exp(1j*1*x - ((x - 0)/(2 * 0.1))**2)
    



#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of spatial support points (including boundaries)")
    parser.add_argument("dt",type=float,
                        help="timestep")
    parser.add_argument("solver",type=str,
                        help="diffusion equation solver:\n"
                             "    ftcs    : forward-time centered-space\n"
                             "    implicit: fully implicit\n"
                             "    CN      : Crank-Nicholson")
    parser.add_argument("inc", type=str,
                        help="initial condition:\n"
                             "    gaussian   : gaussian wavepacket")
    parser.add_argument("problem",type=str,
                        help="potential function:\n"
                             "    free   : constant 0 potential\n"
                             "    box    : potential well\n      "
                             )

    args         = parser.parse_args()
    J            = args.J
    dt           = args.dt
    solver       = args.solver
    problem      = args.problem
    inc          = args.inc

    fINT,fBNC,fINC,potential,minmaxx,minmaxt = init(solver,problem,inc)
    x,t,y        = TDSE_solve(J,minmaxx,dt,minmaxt,fINT,fBNC,fINC,potential)
    
    
    
    
  
    


   
   
      
    # Ploting graph
    
  
    
        
#========================================

main()

