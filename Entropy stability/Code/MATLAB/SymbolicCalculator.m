clc
clear
close all
syms z1 z2 z3 gamma

assume (z1 > 0);
assume (z3 > 0);
assume (z2, 'real')
assume (gamma > 1)
u = [z1*z3;
     z2*z3;
     1/(gamma-1)*(z3/z1) + (1/2)*(z2*z3)^2/(z1*z3)];

v = [gamma/(gamma-1) + (gamma+1)/(gamma-1) * log(z1) + log(z3) - (1/2)*z2^2;
    z1*z2;
    -z1^2];

rhoU = z2.*z3;

dotProduct = simplify(dot(u,v));

f_star = simplify(rhoU/(dotProduct))*u;
pretty(f_star(1))

%% 
syms w1 w2 w3 w4 w5 gamma

% Matrix B
B = [2*w1, 0, 0, 0, 0;
     w2, w1, 0, 0, 0;
     w3, 0, w1, 0, 0;
     w4, 0, 0, w1, 0;
     w5/gamma, (gamma - 1)/gamma*w2, (gamma - 1)/gamma*w3, (gamma - 1)/gamma*w4, w1/gamma];

% Matrix C
C = [w2, w1, 0, 0, 0;
     (gamma - 1)/gamma*w5, (gamma - 1)/gamma*w2, -(gamma - 1)/gamma*w3, -(gamma - 1)/gamma*w4, (gamma - 1)/gamma*w1;
     0, w3, w2, 0, 0;
     0, w4, 0, w2, 0;
     0, w5, 0, 0, w2];

A = simplify(C*(B)^(-1))

