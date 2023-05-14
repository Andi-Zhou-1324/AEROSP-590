import numpy as np
import matplotlib.pyplot as plt
#This file serves as the driver for a simple 1-D finite volume solver. 
from finiteVolume import finiteVolume
from PostProcessing import TadmorTest
from roe_Flux import roe_1D

#========================Solver Setting==========================================================
x_start = 0 #Starting position of the domain
x_end   = 1 #Ending position of the domain
gamma   = 1.4



N = 100 #Number of elements

dx = (x_end - x_start)/N
x = np.arange(x_start+dx/2,x_end-dx/2+dx,dx)
E2N = np.vstack((np.arange(0,N-1), np.arange(1,N))).T


CFL = 1 #The code is yet to taking into account the CFL condition. Left here for now

t_end = 0.2 #Ending simulation time

#=======================Setting initial condition=================================================
x_0 = 0.3
N = 100

u = np.zeros((N, 3))

rho_L = 1.0
P_L = 1.0

rho_R = 0.125
P_R = 0.1

u[:, 1] = 0

p = np.zeros(N)
p[:int(N * x_0)] = P_L
p[int(N * x_0):] = P_R

u[:int(N * x_0), 0] = rho_L
u[:int(N * x_0), 1] = 0.75 * rho_L
u[int(N * x_0):, 0] = rho_R

e = p/((gamma-1)*u[:,0])
u[:,2] = u[:,0]*(0.5*u[:,1]**2 + e)

#======================Begin Solving===========================================================
EntropySwitch = False
u = finiteVolume(x_start, x_end, N, CFL,t_end,u, EntropySwitch)

E_produced, m_diff = TadmorTest(u, E2N, roe_1D, EntropySwitch)


#=====================Plotting - Figure 1===============================
fig, axs = plt.subplots(2, 2)

# Subplot 1
axs[0, 0].scatter(x, u[:, 0], s=12, c='b', marker='o', alpha=1)
axs[0, 0].set_title('Density')

# Subplot 2
axs[0, 1].scatter(x, u[:, 1] / u[:, 0], s=12, c='b', marker='o', alpha=1)
axs[0, 1].set_title('Velocity')

# Subplot 3
p = (gamma - 1) * (u[:, 2] - 0.5 * (u[:, 1] ** 2 / u[:, 0]))
axs[1, 0].scatter(x, p, s=12, c='b', marker='o', alpha=1)
axs[1, 0].set_title('Pressure')

# Subplot 4
e = p / ((gamma - 1) * u[:, 0])
axs[1, 1].scatter(x, e, s=12, c='b', marker='o', alpha=1)
axs[1, 1].set_title('Internal Energy')

# Display the plot
plt.show()

# Adjust spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()



#==================================Figure 2===========================
plt.figure(2)
plt.plot(E_produced,label = "Entropy Produced")
plt.plot(m_diff,label = "Tadmor's Criterion")

# Labeling the axes
plt.xlabel('Edge Numbering')
plt.ylabel('Entropy Produced')

# Title of the plot
plt.legend()


plt.close('all')

#print(u)
