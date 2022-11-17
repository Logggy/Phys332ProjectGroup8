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
from scipy.sparse import diags


# hbar = 1.05457182e-34
# m    = 9.1093837e-31
hbar = 1
m    = 1
L    = 5
T    = 1
#============================================

def tridiag(diag_low,diag_mid,diag_up):
            #value     array     value
    N=diag_mid.size
    k = [diag_low*np.ones(N-1),diag_mid,diag_up*np.ones(N-1)]
    offset = [-1,0,1]
    return diags(k,offset).toarray()
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
def TDSE_solve(J,minmaxx,dt0,minmaxt,fINT,fBNC,fINC,potential):
    
    
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
    dx=x[1]-x[0]
    
    xb=np.zeros(J+2,dtype=complex)
    for i in range(J):
        xb[i+1]=x[i]
    xb[0]=xb[0]-dx
    xb[-1]=xb[-1]+dx
    
    
    potential_func=np.zeros(xb.size,dtype=complex)
    for i in range(xb.size):
        potential_func[i]=potential(xb[i])
    RHS_matrix=tridiag(1j*q,1-1j*2*q-1j*r*potential_func,1j*q)
    LHS_matrix=tridiag(-1j*q,1+1j*2*q+1j*r*potential_func,-1j*q)
    
    
    y[:,0]=fINC(xb)
    y[0,0]=fBNC(0,y[:,0])
    y[-1,0]=fBNC(1,y[:,0])
    
    
    
    for n in range(N-1):
        y[:,n+1]=np.dot(np.dot(np.linalg.inv(LHS_matrix),RHS_matrix),y[:,n])
        y[0,n+1]=fBNC(0,y[:,n+1])
        y[-1,n+1]=fBNC(1,y[:,n+1])
        
    return y[1:J+1,:]

#============================================

def init(solver,problem,inc):
    
    if (solver == 'CN'):
        fINT = cranknicholson
    else:
        print('[init]: invalid solver %s' % (solver))
 
    if (problem == 'free'):
        potential    = Vfree
        fBNC    = Bnon
        minmaxx = np.array([-L,L])
        minmaxt = np.array([0.0,T])
    if (problem == 'well'):
        potential    = Vwell
        fBNC    = Bnon
        minmaxx = np.array([-L,L])
        minmaxt = np.array([0.0,T])
    if (problem == 'wall'):
        potential =Vwall
        fBNC=Bnon
        minmaxx = np.array([-L,L])
        minmaxt = np.array([0.0,T])
    if (problem == 'osci'):
        potential =Voscillator
        fBNC=Bnon
        minmaxx = np.array([-L,L])
        minmaxt = np.array([0.0,T])
        
    if (inc =='gaussian'):
        fINC    = gaussian_wavepacket
    if (inc =='sin'):
        fINC=sin
    

    return fINT,fBNC,fINC,potential,minmaxx,minmaxt 

#============================================
# functions for setting the initial conditions (T....)
# and the boundary conditions (B.....)
def Vfree(x):
    return 0

def Vwell(x):

   if x<-L/2 or x>L/2:
       return 1e3
   else:
       return 0 

def Vwall(x):
    if x>2 and x<2.1:
        return 1e3
    else:
        return 0
    
def Voscillator(x,K=1):
    return 1/2*K*x**2
def Bnon(iside,y):
    if(iside==0):
        return y[1]
    else:
        return y[-2]



def Vspike(x):
    return np.exp(-10.0*x**2)

def Vrandom(x):
    return np.random.rand(x.size)+1.0

def Bdirichlet(iside,y):
    if (iside==0):
        return -y[1]
    else:
        return -y[y.size-2]

def Bperiodic(iside,y):
    if (iside==0):
        return y[y.size-2]
    else:
        return y[1]
def sin(x,a=L,n=4):
    #a length of the potential well
    #n mode
    y=np.zeros(x.size,dtype=complex) 
    for i in range(x.size):    
        if x[i]>-L/2 and x[i]<L/2:
            y[i]=np.sqrt(2/a)*np.sin(n*np.pi*x[i]/a)
        else:
            y[i]=0
    return y
def gaussian_wavepacket(x, sig=0.1, x0=0, p0=1):
    """
    a gaussian wave packet of width a, centered at x0, with momentum k0
    """
    return 1/(np.pi**(1/4)*sig**(1/2))*np.exp(-(x-x0)**2/(2*sig**2))*np.exp(1j*p0*x/hbar)





def update(i,x,y,y1,y2,y3,line1,line2,line3):
    y1 = np.abs(y[:,i+1])
    y2 = y[:,i+1].real
    y3 = y[:,i+1].imag
    line1.set_data(x,y1)
    line2.set_data(x,y2)
    line3.set_data(x,y3)
    



    

#============================================
def main():

    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument("J",type=int,
                        help="number of spatial support points (including boundaries)")
    parser.add_argument("dt",type=float,
                        help="timestep")
    parser.add_argument("solver",type=str,
                        help=" solver:\n"
                             "    CN      : Crank-Nicholson")
    parser.add_argument("problem",type=str,
                        help="potential function:\n"
                             "    free    : constant 0 potential\n"
                             "    well    : potential well\n      "
                             '    wall    : tunneling\n          '
                             '    osci    :   oscillator')
    parser.add_argument("inc", type=str,
                        help="initial condition:\n"
                             "    gaussian   : gaussian wavepacket\n"
                             '    sin        : sin wave\n         '
                             )
    

    args         = parser.parse_args()
    J            = args.J
    dt           = args.dt
    solver       = args.solver
    problem      = args.problem
    inc          = args.inc

    fINT,fBNC,fINC,potential,minmaxx,minmaxt = init(solver,problem,inc)
    x,t,y        = TDSE_solve(J,minmaxx,dt,minmaxt,fINT,fBNC,fINC,potential)
    
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
    ax1.set_ylim([-1,2])
    ax2.set_xlim(minmaxx)
    ax2.set_ylim([-1,2])
    ax3.set_xlim(minmaxx)
    ax3.set_ylim([-1,2])
    
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
    anim.save('simulation.gif', fps=30, extra_args=['-vcodec', 'libx264'])
   


  
    
        
#========================================


main()



