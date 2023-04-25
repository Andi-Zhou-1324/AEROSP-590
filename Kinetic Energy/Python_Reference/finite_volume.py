"""

Finite volume routines

"""

import numpy as np

from typing import Callable

def calculate_residual( 
    state_array: np.ndarray, 
    nb_elements: int,
    edge_to_elems: np.ndarray,
    flux_fun: Callable,
) -> np.ndarray:
    """
    This function computes a finite-volume residual

    Inputs:
      state_array: (nb_elements, 3) array of state values
      nb_elements: number of elements/cells
      edge_to_elems: (nb_edges, 2) edge-to-neighboring elements mapping 
      flux_fun: numerical flux function

    Outputs:
      residual: (nb_elements, 3) array of residual values

    """

    nb_edges = len(edge_to_elems)

    rank = 3
    edge_fluxes = np.zeros((nb_edges, rank))

    residual = np.zeros((nb_elements, rank))

    for edge_id in range(nb_edges):

        left_id, right_id = edge_to_elems[edge_id]

        left_state = state_array[left_id]
        right_state = state_array[right_id]

        num_flux = flux_fun(left_state, right_state)

        edge_fluxes[edge_id][:] = num_flux 

        residual[left_id][:]  += num_flux
        residual[right_id][:] -= num_flux

    return residual
