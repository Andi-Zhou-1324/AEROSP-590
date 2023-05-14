import numpy as np

def entropyVariable(state_vec, idx):
    state = state_vec[idx, :]
    gamma = 1.4

    rho = state[0]
    u = state[1] / rho
    p = (gamma - 1) * (state[2] - 0.5 * state[1]**2 / state[0])
    E = state[2]

    m = rho * u

    S = np.log(p) - gamma * np.log(rho)

    v = np.zeros(3)
    v[0] = (gamma - S) / (gamma - 1) - rho * u**2 / (2 * p)
    v[1] = m / p
    v[2] = -rho / p

    return v

def TadmorTest(u, E2N, flux, EntropySwitch):
    E_produced = np.zeros(E2N.shape[0])
    m_diff = np.zeros(E2N.shape[0])
    for i in range(E2N.shape[0]):
        L_indx = E2N[i, 0]
        R_indx = E2N[i, 1]

        v_L = entropyVariable(u, L_indx)
        v_R = entropyVariable(u, R_indx)

        uL = u[L_indx, :]
        uR = u[R_indx, :]
        F = flux(uL, uR, EntropySwitch)

        E_produced[i] = np.dot(v_R - v_L, F)
        m_diff[i] = uR[1] - uL[1]

    return E_produced, m_diff

