function [F] = roe_1D(UL, UR)
    %Calculates Roe Flux following Toro textbook
    gamma = 1.4;

    %Process Left States
    rho_L = UL(1);
    u_L = UL(2)/rho_L;
    p_L = (gamma-1).*(UL(3) - (0.5).*UL(1).*(u_L).^2);
    H_L = UL(3)./rho_L + p_L./rho_L;
    rHL = UL(3) + p_L;
    if ((p_L<=0) || (rho_L<=0))
        error 'Non-physical state!', 
    end

    %Calculate Left Flux:
    F_L = [rho_L*u_L;
          UL(2)*u_L + p_L;
          rHL*u_L];
    %Process Right States
    rho_R = UR(1);
    u_R = UR(2)/rho_R;
    p_R = (gamma-1).*(UR(3) - (0.5).*UR(1).*(u_R).^2);
    H_R = UR(3)./rho_R + p_R./rho_R;
    rHR = UR(3) + p_R;

    if ((p_L<=0) || (rho_L<=0))
        error 'Non-physical state!', 
    end

    %Calculate Right Flux
    F_R = [rho_R*u_R;
          UR(2)*u_R + p_R;
          rHR*u_R];

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
    
    Roe_term = zeros(3,1);
    for i = 1:3
        temp = alpha(i).*abs(lambda(i)).*K(:,i);
        Roe_term = Roe_term + temp;
    end

    F = 0.5.*(F_L + F_R) - 0.5.*Roe_term;
    F = F';
end

