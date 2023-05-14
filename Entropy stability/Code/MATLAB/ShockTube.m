%% Sod Shock Tube
clc
clear
close all
%%

%The States are AS FOLLOW:

% u = [rho; rho*u; E]

N = 100;
gamma = 1.4;
L = 1;
EntropySwitch = true;

x_0 = 0.3;

u = zeros(N,3);

rho_L = 1.0;
P_L   = 1.0;

rho_R = 0.125;
P_R   = 0.1;

%Setting Initial Conditions
u(:,2) = 0;

p = zeros(N,1);
p(1:round(N.*x_0)) = P_L;
p(round(N.*x_0)+1:end) = P_R;

u(1:round(N.*x_0),1) = rho_L;
u(1:round(N.*x_0),2) = 0.75.*rho_L;
u(round(N.*x_0)+1:end,1) = rho_R;

e  = p./((gamma-1).*u(:,1));
u(:,3) = u(:,1).*(0.5.*u(:,2).^2 + e);


%Defining coordinates
u_coord = (0+(1/N/2):1/N:1-(1/N/2))';

E2N = [(1:N-1)',(2:N)'];

% E2N(1,1) = 0;
% E2N = [E2N;[100,0]];


CFL = 1; %CFL number of 1
d_x = 1./N;

T = 0.2;

%% RK2 Time Stepping
figure()
hold on

[c1,c2,c3,c4] = plotStates(u_coord,u);

count = 0;
t = 0;

while t < T
    count = count + 1;
    dt = 1E-4;
    %RK2
    [Residual_1] = calculateResidual(u,N,E2N,@roe_1D_EntropyFix,EntropySwitch);
    if mod(count,100) == 0
        fprintf("Residual = "+ sum(abs(Residual_1),'all')+", t = "+t+"\n");
        delete([c1,c2,c3,c4]);
        [c1, c2, c3, c4] = plotStates(u_coord,u);
    end
    u = u - dt.*Residual_1./d_x;
%     [Residual_2] = calculateResidual(u_1,N,E2N,@roe_1D_EntropyFix,EntropySwitch);
%     u = u - dt.*(Residual_1 + Residual_2);

    %Advance in Time
    t = t + dt;
end

delete([c1,c2,c3,c4]);
[c1, c2, c3, c4] = plotStates(u_coord,u);

xlabel('x')
ylabel('rho')

%% Entropy Test
[E_produced,m_diff] = TadmorTest(u,E2N,@roe_1D_EntropyFix,EntropySwitch);
figure()
hold on
plot(E_produced,'LineWidth',1.5)
plot(m_diff,'LineWidth',1.5)
legend('Entropy Produced','Criterion')
grid on
xlabel('Edge Number')
ylabel('Entropy Produced')
%% Functions Declared
function [v] = entropyVariable(state_vec,idx)
    state = state_vec(idx,:);
    gamma = 1.4;

    rho = state(1);
    u   = state(2)/rho;
    p   = (gamma-1).*(state(3) - (0.5).*state(2).^2/state(1));
    E   = state(3);
    
    m   = rho.*u;

    S   = log(p) - gamma.*log(rho);

    v   = zeros(3,1);
    v(1) = (gamma - S)./(gamma-1) - rho.*u.^2./(2.*p);
    v(2) = m./p;
    v(3) = -rho./p;
end

function [E_produced,m_diff] = TadmorTest(u,E2N,flux,EntropySwitch)
    E_produced = zeros(size(E2N,1),1);
    m_diff     = zeros(size(E2N,1),1);
    for i = 1:size(E2N,1)
        L_indx = E2N(i,1);
        R_indx = E2N(i,2);
    
        v_L = entropyVariable(u,L_indx);
        v_R = entropyVariable(u,R_indx);
        
        uL = u(L_indx,:);
        uR = u(R_indx,:);
        [F] = flux(uL,uR,EntropySwitch);
        
        E_produced(i,:) = dot([v_R-v_L],F);
        m_diff(i,:) = uR(2) - uL(2);
    end
end

function [c1, c2, c3, c4] = plotStates(u_coord,u)
    gamma = 1.4;
    subplot(2,2,1)
    c1 = scatter(u_coord,u(:,1),12,'filled','MarkerFaceColor','b');
    drawnow
    
    subplot(2,2,2)
    c2 = scatter(u_coord,u(:,2)./u(:,1),12,'filled','MarkerFaceColor','b');
    drawnow
    
    subplot(2,2,3)
    p = (gamma-1).*(u(:,3) - 0.5.*(u(:,2).^2./u(:,1)));
    c3 = scatter(u_coord,p,12,'filled','MarkerFaceColor','b');
    drawnow

    subplot(2,2,4)
    e  = p./((gamma-1).*u(:,1));
    c4 = scatter(u_coord,e,12,'filled','MarkerFaceColor','b');
    drawnow

end


%Calculate residual for 1 timestep
function [Residual,flux_vector] = calculateResidual(u,N,E2N,flux,EntropySwitch)
    Residual = zeros(N,size(u,2)); 
    flux_vector = zeros(size(E2N,1),size(u,2));
    for i = 1:size(E2N,1)
        L_indx = E2N(i,1);
        R_indx = E2N(i,2);
        uL = u(L_indx,:);
        uR = u(R_indx,:);

        [F] = flux(uL,uR,EntropySwitch);
        F;
        flux_vector(i,:) = F;

        if L_indx ~= 1
            Residual(L_indx,:) = Residual(L_indx,:) + F;
        end
        if R_indx ~= max(max(E2N))
            Residual(R_indx,:) = Residual(R_indx,:) - F;
        end
    end
end