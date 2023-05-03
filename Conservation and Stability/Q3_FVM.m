clc
clear
close all
%%
N = 100;

tol = 1E-5;

u_coord = (0+(1/N/2):1/N:1-(1/N/2))';
E2N = [(1:N-1)',(2:N)'];
E2N = [N,1;E2N];

a = 1; % Wave speed
CFL = 0.8; %CFL number of 1
d_x = 1./N;

dt = CFL*d_x./abs(a); %Since cell size are uniform, timestep is fixed

T = 0.5;

%% Forward Euler Stepping
figure()
grid on
xlabel('t')
ylabel('total u')
hold on
u = gaussmf(u_coord,[0.1,0.5]);
t = 0;
% plot(u_coord,u,'LineWidth',1.5);
u_sum_history = [];
t_history = [];
while t < T
    %Calculating Residuals
    [Residual] = calculateResidual(u,N,a,E2N,@upwind);
    %Forward Euler
    u = u - dt.*Residual./d_x;
    %Advance in Time
    t = t + dt;
    t_history = [t_history,t];
    u_temp = sum(u,'all');
    u_sum_history = [u_sum_history, u_temp];
end
% plot(u_coord,u,'LineWidth',1.5)
plot(t_history,u_sum_history,'LineWidth',1.5)

hold on
u = gaussmf(u_coord,[0.1,0.5]);
t = 0;
u_sum_history = [];
t_history = [];

% plot(u_coord,u);
while t < T
    %Calculating Residuals
    [Residual] = calculateResidual(u,N,a,E2N,@central);
    %Forward Euler
    u = u - dt.*Residual./d_x;
    %Advance in Time
    t = t + dt;
    t_history = [t_history,t];
    u_temp = sum(u,'all');
    u_sum_history = [u_sum_history, u_temp];
end
% plot(u_coord,u)
plot(t_history,u_sum_history,'LineWidth',1.5,'LineStyle','--')

%% Backward Euler Stepping
%This becomes a root finding problem
% figure() 
hold on
u = gaussmf(u_coord,[0.1,0.5]);
t = 0;
% plot(u_coord,u);
u_sum_history = [];
t_history = [];

while t < T
    u_guess = u;
    fun = @(u) u + (dt./d_x)*calculateResidual(u,N,a,E2N,@central) - u_guess;

    u = fsolve(fun,u_guess);
    
    t = t + dt;
    t_history = [t_history,t];
    u_temp = sum(u,'all');
    u_sum_history = [u_sum_history, u_temp];
end
% plot(u_coord,u)
plot(t_history,u_sum_history,'LineWidth',1.5,'LineStyle',':')

% figure()
hold on
u = gaussmf(u_coord,[0.1,0.5]);
t = 0;
% plot(u_coord,u);
u_sum_history = [];
t_history = [];

while t < T
    u_guess = u;
    fun = @(u) u + (dt./d_x)*calculateResidual(u,N,a,E2N,@upwind) - u_guess;
    u = fsolve(fun,u_guess);
    t = t + dt;
    t_history = [t_history,t];
    u_temp = sum(u,'all');
    u_sum_history = [u_sum_history, u_temp];
end
% plot(u_coord,u)
plot(t_history,u_sum_history,'LineWidth',1.5)

%% Midpoint Rule
% figure()
hold on
u = gaussmf(u_coord,[0.1,0.5]);
t = 0;
% plot(u_coord,u);
u_sum_history = [];
t_history = [];

while t < T
    u_guess = u;
    fun = @(u) u + (dt./d_x)*(1/2.*(calculateResidual(u,N,a,E2N,@central) + u_guess)) - u_guess;
    u = fsolve(fun,u_guess);
    t = t + dt;
    t_history = [t_history,t];
    u_temp = sum(u,'all');
    u_sum_history = [u_sum_history, u_temp];
end
% plot(u_coord,u)
plot(t_history,u_sum_history,'LineWidth',1.5,'LineStyle','--')

% figure()
hold on
u = gaussmf(u_coord,[0.1,0.5]);
t = 0;
% plot(u_coord,u);
t_history = [];
u_sum_history = [];
while t < T
    u_guess = u;
    fun = @(u) u + (dt./d_x)*(1/2.*(calculateResidual(u,N,a,E2N,@upwind) + u_guess)) - u_guess;
    u = fsolve(fun,u_guess);
    t = t + dt;
    t_history = [t_history,t];
    u_temp = sum(u,'all');
    u_sum_history = [u_sum_history, u_temp];
end
% plot(u_coord,u)
plot(t_history,u_sum_history,'LineWidth',1.5,'LineStyle','-.')

legend('Forward Euler Upwind','Forward Euler Central','Backward Euler Central','Backward Euler Upwind','Midpoint Rule Central','Midpoint Rule Upwind')


%% Functions Declared

%Calculate residual for 1 timestep
function [Residual] = calculateResidual(u,N,a,E2N,flux)
    Residual = zeros(N,1); 

    for i = 1:size(E2N,1)
        L_indx = E2N(i,1);
        R_indx = E2N(i,2);
        F = flux(u(L_indx),u(R_indx),a);
        Residual(L_indx) = Residual(L_indx) + F;
        Residual(R_indx) = Residual(R_indx) - F;
    end

end

function [F] = upwind(L,R,a)
    F = 1/2*(a*L + a*R) - (1/2)*abs(a)*(R - L);
end

function [F] = central(L,R,a)
    F = 0.5*(a*L + a*R);
end