clc
clear
close all;

%%
syms u(x) x

int(u(x)*diff(u(x),x),x)