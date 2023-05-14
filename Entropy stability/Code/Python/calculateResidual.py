import numpy as np
from roe_Flux import roe_1D

def calculateResidual(u,N,E2N,EntropySwitch):
    Residual = np.zeros((N,3))
    for i in range(E2N.shape[0]):
        L_indx = E2N[i,0]
        R_indx = E2N[i,1]

        uL = u[L_indx]
        uR = u[R_indx]
        
        #We are using professor Fidkowski's Roe flux function. Which does not support vectorization and is only 2D. We remove the 
        #2D velocity component v to make the function 1D only.
        F = roe_1D(uL,uR,EntropySwitch)
        #print(F)


        if L_indx != 0:
            Residual[L_indx, :] += F

        if R_indx != np.max(E2N):
            Residual[R_indx, :] -= F

    return Residual