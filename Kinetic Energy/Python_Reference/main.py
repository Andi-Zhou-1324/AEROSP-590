"""

Let's learn python! 

"""

import numpy as np

from finite_volume import calculate_residual
from numerical_fluxes import roe_flux

###
# Inputs
#

nb_elements = 100
gas_constant = 287.
gamma = 1.4
rank = 3

state_array = np.zeros((nb_elements, rank))

###
# Grid setup
#

# domain boundaries
x0, x1 = 0., 1.

# grid coordinates
dx = (x1 - x0)/nb_elements

cell_centers = np.arange(x0 + 0.5*dx, x1, dx)

edge_to_elems = np.array(
    [[elem_id, elem_id+1] for elem_id in range(nb_elements-1)] + 
    [[nb_elements-1, 0]]
)
 
##
# Initial conditions: centered square wave
#

temperature = 300

# 
velocity = state_array[:,1]

# density
state_array[:,0] = 0.125

center_id = int(nb_elements/2)
offset = 10
state_array[center_id-offset:center_id+offset, 0] = 1

pressure = state_array[:,0] * gas_constant * temperature

state_array[:,2] = pressure / (gamma - 1) + 0.5 * state_array[:,0] * velocity**2 

##
# Time-stepping
#

cfl = 1
dt = 1e-6 
tfinal = 10

######################

import time
import matplotlib.pyplot as plt

plt.ion()
fig, ax = plt.subplots(1, figsize=(8,6))

t = 0
count = 0

density = state_array[:,0]
line, = ax.plot(cell_centers, density)

while t < tfinal:

    count += 1
     
    #
    start = time.perf_counter()

    residual_1 = calculate_residual(
        state_array, nb_elements, edge_to_elems, roe_flux
    )

    state_array_1 = state_array - dt/dx * residual_1

    #
    residual_2 = calculate_residual(
        state_array_1, nb_elements, edge_to_elems, roe_flux 
    )

    state_array -= 0.5 * dt * ( residual_1 + residual_2 )

    end = time.perf_counter()

    print(f'Elapsed time: {end-start:.6f} seconds')

    #
    if count % 100 == 0:
        res_norm = np.linalg.norm(residual_1)
        print(f"Residual = {res_norm:.3e}, t = {t:.3e}")
        density = state_array[:,0]
        line.set_ydata(density)
        plt.draw()
        plt.pause(0.01)

    #
    t += dt
