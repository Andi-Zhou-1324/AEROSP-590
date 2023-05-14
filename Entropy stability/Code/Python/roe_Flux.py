import numpy as np

#This is the Roe flux function with an entropy fix switch, coded as per Toro textbook
def roe_1D (UL, UR, EntropySwitch):
    gamma = 1.4

    # Process Left States
    rho_L = UL[0]
    u_L = UL[1] / rho_L
    p_L = (gamma - 1) * (UL[2] - 0.5 * UL[1]**2 / UL[0])
    E_L = UL[2]
    a_L = np.sqrt(gamma * p_L / rho_L)

    H_L = E_L / rho_L + p_L / rho_L

    if p_L <= 0 or rho_L <= 0:
        raise ValueError('Non-physical state!')
    
    # Calculate Left Flux
    F_L = np.array([rho_L * u_L,
                    UL[1] * u_L + p_L,
                    u_L * (E_L + p_L)])

    # Process Right States
    rho_R = UR[0]
    u_R = UR[1] / rho_R
    p_R = (gamma - 1) * (UR[2] - 0.5 * UR[1]**2 / UR[0])
    E_R = UR[2]

    H_R = E_R / rho_R + p_R / rho_R
    a_R = np.sqrt(gamma * p_R / rho_R)

    if p_R <= 0 or rho_R <= 0:
        raise ValueError('Non-physical state!')

    # Calculate Right Flux
    F_R = np.array([rho_R * u_R,
                    UR[1] * u_R + p_R,
                    u_R * (E_R + p_R)])
    
    #Calculate the Roe averages

    u = (np.sqrt(rho_L) * u_L + np.sqrt(rho_R) * u_R) / (np.sqrt(rho_L) + np.sqrt(rho_R))
    H = (np.sqrt(rho_L) * H_L + np.sqrt(rho_R) * H_R) / (np.sqrt(rho_L) + np.sqrt(rho_R))
    V = u
    a = (gamma - 1) * (H - 0.5 * u**2)
    a = np.sqrt(a)

    # Calculating Eigenvalues
    lam = np.zeros(3)
    lam[0] = u - a
    lam[1] = u
    lam[2] = u + a

    # Calculating Eigenvectors
    K = np.zeros((3, 3))
    K[:, 0] = np.array([1, u - a, H - u * a])
    K[:, 1] = np.array([1, u, 0.5 * V**2])
    K[:, 2] = np.array([1, u + a, H + u * a])

    # Compute wave strength
    alpha = np.zeros(3)

    Delta_u1 = UR[0] - UL[0]
    Delta_u2 = UR[1] - UL[1]
    Delta_u3 = UR[2] - UL[2]

    alpha[1] = (gamma - 1) / (a**2) * (Delta_u1 * (H - u**2) + u * Delta_u2 - Delta_u3)
    alpha[0] = (1 / (2 * a)) * (Delta_u1 * (u + a) - Delta_u2 - a * alpha[1])
    alpha[2] = Delta_u1 - (alpha[0] + alpha[1])

    left_wave_active = False

    #If the entropy switch is on:
    if EntropySwitch:
        lam, left_wave_active = entropyFix(rho_L, u_L, E_L, a_L, rho_R, u_R, E_R, a_R, alpha, H, u, a, lam, left_wave_active)

    Roe_term = np.zeros(3)
    for i in range(3):
        if i == 0 and left_wave_active:
            temp = alpha[i] * lam[i] * K[:, i]
        else:
            temp = alpha[i] * abs(lam[i]) * K[:, i]
        Roe_term = Roe_term + temp

    F = 0.5 * (F_L + F_R) - 0.5 * Roe_term
    
    return F


#================================ HELPER FUNCTIONS ==========================
#Entropy fix: The Harten-Hyman fix

def entropyFix(rho_L, u_L, E_L, a_L, rho_R, u_R, E_R, a_R, alpha, H, u, a, lam, left_wave_active):
    
    gamma = 1.4

    # Left wave entropy fix
    rho_star_L = rho_L + alpha[0]
    u_star_L = (rho_L * u_L + alpha[0] * (u - a)) / (rho_L + alpha[0])
    p_star_L = (gamma - 1) * (E_L + alpha[0] * (H - u * a) - 0.5 * rho_star_L * u_star_L**2)
    a_star_L = np.sqrt((gamma * p_star_L) / rho_star_L)

    lambda_1_L = u_L - a_L
    lambda_1_R = u_star_L - a_star_L

    # Right wave entropy fix
    rho_star_R = rho_R - alpha[2]
    u_star_R = (rho_R * u_R - alpha[2] * (u + a)) / (rho_R - alpha[2])
    p_star_R = (gamma - 1) * (E_R - alpha[2] * (H + u * a) - 0.5 * rho_star_R * u_star_R**2)
    a_star_R = np.sqrt((gamma * p_star_R) / rho_star_R)

    lambda_3_L = u_star_R + a_star_R
    lambda_3_R = u_R + a_R

    if lambda_1_L <= 0 and lambda_1_R > 0:
        lam[0] = ((lambda_1_R + lambda_1_L) * lam[0] - 2 * (lambda_1_R * lambda_1_L)) / (lambda_1_R - lambda_1_L)
        left_wave_active = True

    if lambda_3_L < 0 and lambda_3_R > 0:
        raise ValueError("Unexpected Right Wave!")
    
    return lam,left_wave_active
    