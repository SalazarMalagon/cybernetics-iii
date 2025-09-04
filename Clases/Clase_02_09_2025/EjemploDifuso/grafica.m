%
% preambulo
clear; close all; clc;
set(0, 'DefaultLineLineWidth',1.8) %linewidh on plots
set(0,'defaultfigurecolor', [1 1 1])
a = 4;
b = 6;
c = 6;
d = 10;

Delta_X = 0.1;
i = 1;
for x = 0:Delta_X:12
    MU(i) = MF_Trap(x, a, b, c, d);
    U(i) = x;
    i = i + 1;
end
plot(U, MU, 'g')
