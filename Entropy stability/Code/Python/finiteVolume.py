import numpy as np
from calculateResidual import calculateResidual
def finiteVolume(x_start, x_end, N, CFL,t_end,u,EntropySwitch):
    #Calculating x domain coordinates
    dx = (x_end - x_start)/N
    
    #Defining initial state array


    #Assembling E2N array. This series of array specifies the edge-element
    #connectivities
    E2N = np.vstack((np.arange(0,N-1), np.arange(1,N))).T

    #Looping through E2N to calculate finite volume residuals
    dt = 0.0001

    t = 0
    iter = 0
    while t < t_end:
        iter += 1
        Residual_1 = calculateResidual(u,N,E2N,EntropySwitch)
        u = u - dt*Residual_1/dx
        #Residual_2 = calculateResidual(u_1,N,E2N,EntropySwitch)

        if iter % 100 == 0:
            print("Residual =", np.sum(np.abs(Residual_1)), ", t =", t)

       # u = u - dt*(0.5*(Residual_1+Residual_2))

        t += dt
    
    return u