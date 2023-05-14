function [F] = roe_1D(UL, UR, entropyFixSwitch)
    %Calculates Roe Flux WITH Entropy Fix following Toro's textbook
    gamma = 1.4;

    %Process Left States
    rho_L = UL(1);
    u_L = UL(2)/rho_L;
    p_L = (gamma-1).*(UL(3) - (0.5).*UL(2).^2/UL(1));
    E_L = UL(3);
    a_L = sqrt(gamma.*p_L./rho_L);

    H_L = UL(3)./rho_L + p_L./rho_L;

    if ((p_L<=0) || (rho_L<=0))
        error 'Non-physical state!', 
    end

    %Calculate Left Flux:
    F_L = [rho_L*u_L;
          UL(2)*u_L + p_L;
          u_L.*(E_L + p_L)];
    
    %Process Right States
    rho_R = UR(1);
    u_R = UR(2)/rho_R;
    p_R = (gamma-1).*(UR(3) - (0.5).*UR(2).^2/UR(1));
    E_R = UR(3);

    H_R = UR(3)./rho_R + p_R./rho_R;
    a_R = sqrt(gamma.*p_R./rho_R);

    if ((p_R<=0) || (rho_R<=0))
        error 'Non-physical state!', 
    end

    %Calculate Right Flux
    F_R =  [rho_R*u_R;
            UR(2)*u_R + p_R;
            u_R.*(E_R + p_R)];

    %Calculating Roe averages
    u   = (sqrt(rho_L).*u_L + sqrt(rho_R).*u_R)./(sqrt(rho_L) + sqrt(rho_R));
    H   = (sqrt(rho_L).*H_L + sqrt(rho_R).*H_R)./(sqrt(rho_L) + sqrt(rho_R));
    V   = u;
    a   = (gamma-1).*(H - (1/2).*u.^2);
    a   = sqrt(a);
   
    %Calculating Eigenvalues
    lambda = zeros(3,1);
    lambda(1) = u - a;
    lambda(2) = u;
    lambda(3) = u + a;

    %Calculating Eigenvectors
    K = zeros(3,3);
    K(:,1) = [1;u-a;H-u.*a];
    K(:,2) = [1;u;0.5.*V.^2];
    K(:,3) = [1;u+a;H+u.*a];

    %Compute wave strength
    alpha = zeros(3,1);
    Delta_u1 = UR(1) - UL(1);
    Delta_u2 = UR(2) - UL(2);
    Delta_u3 = UR(3) - UL(3);

    alpha(2) = (gamma-1)./(a.^2).*(Delta_u1.*(H-u.^2) + u.*Delta_u2 - Delta_u3);
    alpha(1) = (1./(2.*a)).*(Delta_u1.*(u+a)-Delta_u2 - a.*alpha(2));
    alpha(3) = Delta_u1 - (alpha(1) + alpha(2));
   
    left_wave_active = false;
    
    %% Entropy Fix

    if entropyFixSwitch
        [lambda,left_wave_active] = entropyFix(rho_L, u_L, E_L, a_L, rho_R, u_R, E_R, a_R, alpha, H, u, a, lambda, left_wave_active);
    end

    Roe_term = zeros(3,1);
    for i = 1:3
        if i == 1 && left_wave_active
            temp = alpha(i).*lambda(i).*K(:,i);
        else
            temp = alpha(i).*abs(lambda(i)).*K(:,i);
        end
        Roe_term = Roe_term + temp;
    end

    F = 0.5.*(F_L + F_R) - 0.5.*Roe_term;
    F = F';
end

function [lambda,left_wave_active] = entropyFix(rho_L, u_L, E_L, a_L, rho_R, u_R, E_R, a_R, alpha, H, u, a, lambda, left_wave_active) 
    gamma = 1.4;
    %Left wave entropy fix
    rho_star_L = rho_L + alpha(1);
    u_star_L   = (rho_L.*u_L + alpha(1).*(u - a))./(rho_L + alpha(1));
    p_star_L   = (gamma - 1).*(E_L + alpha(1).*(H - u.*a) - (1/2).*rho_star_L.*u_star_L.^2);
    a_star_L = sqrt((gamma.*p_star_L)./(rho_star_L));
    
    lambda_1_L = u_L - a_L;
    lambda_1_R = u_star_L - a_star_L;
    
    

    %Right wave entropy fix
    rho_star_R = rho_R + alpha(3);
    u_star_R   = (rho_R.*u_R + alpha(3).*(u - a))./(rho_R + alpha(3));
    p_star_R   = (gamma - 1).*(E_R + alpha(3).*(H - u.*a) - (1/2).*rho_star_R.*u_star_R.^2);
    a_star_R = sqrt((gamma.*p_star_R)./(rho_star_R));
    
    lambda_3_L = u_star_R + a_star_R;
    lambda_3_R = u_R + a_R;
    

    if (lambda_1_L <= 0) && (lambda_1_R > 0)
        lambda(1) = ((lambda_1_R + lambda_1_L).*lambda(1) - 2.*(lambda_1_R.*lambda_1_L))./(lambda_1_R - lambda_1_L);
        lambda(1);
        left_wave_active = true;
    end
    
    if (lambda_3_L < 0) && (lambda_3_R > 0)
       error("Unexpected Right Wave!") 
    end
end