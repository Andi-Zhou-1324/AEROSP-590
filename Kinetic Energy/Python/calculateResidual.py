import numpy as np
from flux import FluxFunction
def calculateResidual(u,N,E2N):
    gamma = 1.4
    EdgeFlux = np.zeros((E2N.shape[0],3))
    Residual = np.zeros((N,3))
    for i in range(E2N.shape[0]):
        L_indx = E2N[i,0]
        R_indx = E2N[i,1]

        uL = u[L_indx]
        uR = u[R_indx]

        uL_temp = np.array([[uL[0]],[uL[1]],[0],[uL[2]]])
        uR_temp = np.array([[uR[0]],[uR[1]],[0],[uR[2]]])
        
        #We are using professor Fidkowski's Roe flux function. Which does not support vectorization and is only 2D. We remove the 
        #2D velocity component v to make the function 1D only.
        F,smag = FluxFunction(uL_temp,uR_temp,gamma, np.array([[1],[0]]))

        F = np.array([F[0],F[1],F[3]])

        EdgeFlux[i] = F

        Residual[L_indx] -= F
        Residual[R_indx] += F

    return Residual