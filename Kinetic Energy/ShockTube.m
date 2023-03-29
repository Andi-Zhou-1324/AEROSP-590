%% Sod Shock Tube
clc
clear
close all
%%
N = 300;
gamma = 1.4;

u = zeros(N,3);

rho_L = 1.0;
P_L   = 1.0.*101325;

rho_R = 0.125;
P_R   = 0.1.*101325;

%Setting Initial Conditions
u(:,2) = 0;


u(1:round(N./2),1) = rho_L;
u(1:round(N./2),3) = (P_L)./(gamma - 1) + (1/2)*rho_L.*abs(u(1:round(N./2),2)).^2;
u(round(N./2)+1:end,1) = rho_R;
u(round(N./2)+1:end,3) = (P_R)./(gamma - 1) + (1/2)*rho_R.*abs(u(round(N./2)+1:end,2)).^2;


%Defining coordinates
u_coord = (0+(1/N/2):1/N:1-(1/N/2))';

E2N = [(1:N-1)',(2:N)'];

% E2N(1,1) = 0;
% E2N = [E2N;[100,0]];


CFL = 1; %CFL number of 1
d_x = 1./N;

T = 0.1;

%% Forward Euler Stepping
figure()
hold on


t = 0;
plot(u_coord,u(:,1));
count = 0;

while t < T
    count = count + 1;
    dt = 1E-6;
    %RK2
    [Residual_1] = calculateResidual(u,N,E2N,@roe);
    if mod(count,100) == 0
        fprintf("Residual = "+ sum(abs(Residual_1),'all')+", t = "+t+"\n");
    end
    u_1 = u - dt.*Residual_1./d_x;
    [Residual_2] = calculateResidual(u_1,N,E2N,@roe);
    u = u - dt.*(Residual_1 + Residual_2);

    %Advance in Time
    t = t + dt;
    
end
plot(u_coord,u(:,1))
legend('Initial','Final')
xlabel('x')
ylabel('rho')

%% Functions Declared

%Calculate residual for 1 timestep
function [Residual] = calculateResidual(u,N,E2N,flux)
    Residual = zeros(N,size(u,2)); 
    for i = 1:size(E2N,1)
        L_indx = E2N(i,1);
        R_indx = E2N(i,2);
        uL = u(L_indx,:);
        uR = u(R_indx,:);
        uL_temp = [uL(1),uL(2),0,uL(3)];
        uR_temp = [uR(1),uR(2),0,uR(3)];
        [F] = flux(uL_temp,uR_temp,[1;0]);
        F(3) = [];
        if L_indx ~= 1
            Residual(L_indx,:) = Residual(L_indx,:) + F;
        end
        if R_indx ~= max(max(E2N))
            Residual(R_indx,:) = Residual(R_indx,:) - F;
        end
    end
end
