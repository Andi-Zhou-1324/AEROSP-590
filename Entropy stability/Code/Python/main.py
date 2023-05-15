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
x_E2N = np.linspace(0,1,E2N.shape[0])

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

#=======================Plotting Initial Condition=============================
plt.figure
plt.plot(x,u[:,0],label = 'Density',c = 'b')
plt.plot(x,u[:,1],label = 'Momentum', c = 'r')
plt.plot(x,u[:,2],label = 'Energy', c = 'g')
plt.xlabel('x')
plt.ylabel('States')
plt.legend()
plt.grid()
plt.show()
#======================Begin Solving===========================================================
EntropySwitch = False
u_nofix = finiteVolume(x_start, x_end, N, CFL,t_end,u, EntropySwitch)
E_produced_nofix, m_diff_nofix = TadmorTest(u_nofix, E2N, roe_1D, EntropySwitch)

EntropySwitch = True
u_fix = finiteVolume(x_start, x_end, N, CFL,t_end,u, EntropySwitch)
E_produced_fix, m_diff_fix = TadmorTest(u_fix, E2N, roe_1D, EntropySwitch)


#=====================Plotting - Figure 1===============================
fig, axs = plt.subplots(2, 2)

# Subplot 1
axs[0, 0].scatter(x, u_nofix[:, 0], s=12, c='b', marker='o', alpha=1, label = 'Without Entropy Fix')
axs[0, 0].scatter(x, u_fix[:,0], s=12, c='r', marker='o', alpha=1, label = 'With Entropy Fix')
axs[0, 0].set_title('Density')
# Subplot 2
axs[0, 1].scatter(x, u_nofix[:, 1] / u_nofix[:, 0], s=12, c='b', marker='o', alpha=1, label = 'Without Entropy Fix')
axs[0, 1].scatter(x, u_fix[:, 1] / u_fix[:, 0], s=12, c='r', marker='o', alpha=1, label = 'With Entropy Fix')
axs[0, 1].set_title('Velocity')

# Subplot 3
p_nofix = (gamma - 1) * (u_nofix[:, 2] - 0.5 * (u_nofix[:, 1] ** 2 / u_nofix[:, 0]))
p_fix = (gamma - 1) * (u_fix[:, 2] - 0.5 * (u_fix[:, 1] ** 2 / u_fix[:, 0]))

axs[1, 0].scatter(x, p_nofix, s=12, c='b', marker='o', alpha=1, label = 'Without Entropy Fix')
axs[1, 0].scatter(x, p_fix, s=12, c='r', marker='o', alpha=1, label = 'With Entropy Fix')

axs[1, 0].set_title('Pressure')

# Subplot 4
e_nofix = p_nofix / ((gamma - 1) * u_nofix[:, 0])
e_fix = p_fix / ((gamma - 1) * u_fix[:, 0])

axs[1, 1].scatter(x, e_nofix, s=12, c='b', marker='o', alpha=1, label = 'Without Entropy Fix')
axs[1, 1].scatter(x, e_fix, s=12, c='r', marker='o', alpha=1, label = 'With Entropy Fix')

axs[1, 1].set_title('Internal Energy')

# Display legend
axs[0, 0].legend()
axs[0, 1].legend()
axs[1, 0].legend()
axs[1, 1].legend()

# Turn on Grid
axs[0, 0].grid()
axs[0, 1].grid()
axs[1, 0].grid()
axs[1, 1].grid()

# Turn on Label
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('Density')

axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('Velocity')

axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('Pressure')

axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('Internal Energy')

# Display the plot
plt.legend()
# Adjust spacing between subplots

# Show the plot
plt.show()



#==================================Figure 2===========================
plt.figure(2)
plt.plot(x_E2N, E_produced_nofix,c = 'b', label = "Entropy Produced (Without Entropy Fix)")
plt.plot(x_E2N, E_produced_fix,c = 'r', label = "Entropy Produced (With Entropy Fix)")
plt.plot(x_E2N, m_diff_fix,c = 'g', label = "Criterion")

# Labeling the axes
plt.xlabel('x')
plt.ylabel('Entropy Produced')

# Title of the plot
plt.legend()
plt.grid()
plt.show()

#print(u)
