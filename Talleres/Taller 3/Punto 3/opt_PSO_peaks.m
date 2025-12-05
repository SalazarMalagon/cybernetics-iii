% preambulo
clear;clc; close all;
set(0,'DefaultLineLineWidth',2)
set(0, 'defaultfigurecolor', [1 1 1])
%
x = -5:0.1:5;
y = x;
%Evaluación de todos los datos
for j=1:length(x)
for i=1:length(y)
z(i,j)=peaks([x(j);y(i)]);
end
end
%Figura de la función objetivo en 2D
figure
surf(x,y,z);
colormap(jet)
xlabel('x');ylabel('y');zlabel('J');
title('Funcion PEAKS 2D - PSO');
%Cantidad de variables
NV = 2;
%Límites de búsqueda
lb = [-5, -5];  % límites inferiores
ub = [5, 5];    % límites superiores
%Opciones para PSO
options_pso = optimoptions('particleswarm', ...
    'Display', 'iter', ...
    'SwarmSize', 25, ...
    'MaxIterations', 200, ...
    'PlotFcn', {@pswplotbestf});
%Función que implementa la optimización con PSO
y = particleswarm(@peaks, NV, lb, ub, options_pso)