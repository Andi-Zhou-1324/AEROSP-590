import numpy as np
from calculateResidual import calculateResidual
def finiteVolume(x_start, x_end, N, CFL,t_end,u):
    #Calculating x domain coordinates
    dx = (x_end - x_start)/N
    x = np.arange(x_start+dx/2,x_end-dx/2+dx,dx)
    
    #Defining initial state array


    #Assembling E2N array. This series of array specifies the edge-element
    #connectivities
    E2N = np.vstack((np.arange(0,N-1), np.arange(1,N))).T
    E2N = np.vstack((E2N,[N-1,0]))

    #Looping through E2N to calculate finite volume residuals
    dt = 0.01

    t = 0

    while t < t_end:
        Residual_1 = calculateResidual(u,N,E2N)
        u_1 = u-dt*Residual_1/dx
        Residual_2 = calculateResidual(u_1,N,E2N)

        u = u - dt*(0.5*(Residual_1+Residual_2))

        t += dt
    
    return u