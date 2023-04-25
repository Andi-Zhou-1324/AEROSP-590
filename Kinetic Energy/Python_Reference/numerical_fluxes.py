"""

Numerical interface flux functions for the compressible
Euler equations in 1D 

Refs:
    [1] Roe, P.L. : Approximate Riemann solvers, parameter vectors, 
        and difference schemes, Journal of Computational Physics, 1981.

"""

import sys
import numpy as np

from numba import jit

sqrt = np.sqrt

@jit(nopython=True)
def roe_flux(left_state: np.ndarray, right_state: np.ndarray) -> np.ndarray:
    """
    Roe flux [1] in 1D

    Inputs:
      left_state: state vector in the left cell
      right_state: state vector in the right cell

    Outputs:
      flux: numerical interface flux function
    """

    gamma = 1.4
    gmi = gamma - 1.0

    # process left state
    rL = left_state[0]
    uL = left_state[1] / rL
    unL = uL
    pL = (gamma - 1) * (left_state[2] - 0.5 * rL * uL ** 2)

    if (pL <= 0) or (rL <= 0):
        print("roe_flux: non-physical left state.")

    rHL = left_state[2] + pL
    HL = rHL / rL
    cL = sqrt(gamma * pL / rL)

    # left flux
    left_flux = np.zeros(3)
    left_flux[0] = rL * unL
    left_flux[1] = rL * unL**2 + pL
    left_flux[2] = rHL * unL

    # process right state
    rR = right_state[0]
    uR = right_state[1] / rR
    unR = uR
    pR = (gamma - 1) * (right_state[2] - 0.5 * rR * uR ** 2)

    if (pR <= 0) or (rR <= 0):
        print("roe_flux: non-physical right state.")

    rHR = right_state[2] + pR
    HR = rHR / rR
    cR = sqrt(gamma * pR / rR)

    # right flux
    right_flux = np.zeros(3)
    right_flux[0] = rR * unR
    right_flux[1] = rR * unR**2 + pR
    right_flux[2] = rHR * unR

    # difference in states
    state_jump = right_state - left_state

    # Roe average
    di = sqrt(rR / rL)
    d1 = 1.0 / (1.0 + di)

    ui = (di * uR + uL) * d1
    Hi = (di * HR + HL) * d1

    af = 0.5 * ui * ui
    ucp = ui
    c2 = gmi * (Hi - af)

    if c2 <= 0:
        print("roe_flux: non-physical c2")

    ci = sqrt(c2)
    ci1 = 1.0 / ci

    # eigenvalues
    lamb = np.zeros(3)
    lamb[0] = ucp + ci
    lamb[1] = ucp - ci
    lamb[2] = ucp

    # entropy fix
    epsilon = ci * 0.1
    for i in range(3):
        if (-epsilon < lamb[i] < epsilon):
            lamb[i] = 0.5 * (epsilon + lamb[i] ** 2 / epsilon)

    lamb = np.abs(lamb)
    l3 = lamb[2]

    # average and half-difference of 1st and 2nd eigs
    s1 = 0.5 * (lamb[0] + lamb[1])
    s2 = 0.5 * (lamb[0] - lamb[1])

    # left eigenvector product generators (see Theory guide)
    G1 = gmi * (
        af * state_jump[0] - ui * state_jump[1] + state_jump[2]
    )
    G2 = -ucp * state_jump[0] + state_jump[1]

    # required functions of G1 and G2 (again, see Theory guide)
    C1 = G1 * (s1 - l3) * ci1 * ci1 + G2 * s2 * ci1
    C2 = G1 * s2 * ci1 + G2 * (s1 - l3)

    # flux assembly
    interface_flux = np.zeros(3)

    interface_flux[0] = 0.5 * (left_flux[0] + right_flux[0]) - 0.5 * (
        l3 * state_jump[0] + C1
    )
    interface_flux[1] = 0.5 * (left_flux[1] + right_flux[1]) - 0.5 * (
        l3 * state_jump[1] + C1 * ui + C2
    )
    interface_flux[2] = 0.5 * (left_flux[2] + right_flux[2]) - 0.5 * (
        l3 * state_jump[2] + C1 * Hi + C2 * ucp
    )

    return interface_flux
