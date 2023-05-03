import numpy as np
#This file serves as the driver for a simple 1-D finite volume solver. 
from finiteVolume import finiteVolume


x_start = 0 #Starting position of the domain
x_end   = 1 #Ending position of the domain

N = 4 #Number of elements

CFL = 1 #The code is yet to taking into account the CFL condition. Left here for now

u = np.zeros((N,3))  + 1

t_end = 0.5 #Ending simulation time



u = finiteVolume(x_start, x_end, N, CFL,t_end,u)



print(u)
