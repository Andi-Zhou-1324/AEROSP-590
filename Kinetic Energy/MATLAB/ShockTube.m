%% Sod Shock Tube
clc
clear
close all

%%
N = 100;
gamma = 1.4;

u = zeros(N,3);

T = 300;

%Setting Initial Conditions with a square wave at the center
u(:,1) = 0.125;
squareOffset = 10;
u(round(size(u,1)/2)-squareOffset:round(size(u,1)/2)+squareOffset,1) = 1;

P = u(:,1).*287.*T;
u(:,3) = (P)./(gamma - 1) + (1/2)*u(:,1).*abs(u(:,2)).^2;
%Defining coordinates
u_coord = (0+(1/N/2):1/N:1-(1/N/2))';

c1 = plot(u_coord,u(:,1),'Color',[0, 0.4470, 0.7410]);

E2N = [(1:N-1)',(2:N)'];

E2N = [E2N;N,1];


CFL = 1; %CFL number of 1
d_x = 1./N;

T = 10;

%% Forward Euler Stepping
figure()
hold on


t = 0;
c1 = plot(u_coord,u(:,1),'Color',[0, 0.4470, 0.7410]);
drawnow
count = 0;
u_history = zeros(size(u,1),size(u,2),1);
t_history = [];
save = 1;
while t < T
    count = count + 1;
    dt = 1E-6;
    %RK2
    [Residual_1] = calculateResidual_periodic(u,N,E2N,@roe);

    u_1 = u - dt.*Residual_1./d_x;
    [Residual_2] = calculateResidual_periodic(u_1,N,E2N,@roe);
    u = u - dt.*(0.5.*(Residual_1+Residual_2));
    
    if mod(count,100) == 0
        fprintf("Residual = "+ sum(abs(Residual_1),'all')+", t = "+t+"\n");
        delete(c1);
        c1 = plot(u_coord,u(:,1),'Color',[0, 0.4470, 0.7410]);
        drawnow
    end

    if mod(count,1E4) == 0
        u_history(:,:,save) = u;
        t_history = [t_history;t];
        save = save + 1;
        fprintf("Solution Saved!\n\n")
    end
    
    %Advance in Time
    t = t + dt;
   
end
plot(u_coord,u(:,1))
legend('Initial','Final')
xlabel('x')
ylabel('rho')



%% Functions Declared

%Calculate residual for 1 timestep
function [Residual] = calculateResidual_freestream(u,N,E2N,flux)
    EdgeFlux = zeros(size(E2N,1),3);
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
        EdgeFlux(i,:) = F;
    end

    for i = 1:size(EdgeFlux,1)
        L_indx = E2N(i,1);
        R_indx = E2N(i,2);
        if L_indx ~= 1
            Residual(L_indx,:) = Residual(L_indx,:) + EdgeFlux(i,:);
        end
        if R_indx ~= max(max(E2N))
            Residual(R_indx,:) = Residual(R_indx,:) - EdgeFlux(i,:);
        end
    end
end

function [Residual] = calculateResidual_periodic(u,N,E2N,flux)
    EdgeFlux = zeros(size(E2N,1),3);
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
        EdgeFlux(i,:) = F;
        Residual(L_indx,:) = Residual(L_indx,:) + F;
        Residual(R_indx,:) = Residual(R_indx,:) - F;
    end
end
