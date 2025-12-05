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
z(i,j)=rastigin([x(j);y(i)]);
end
end
%Figura de la función objetivo en 2D
figure
surf(x,y,z);
colormap(jet)
xlabel('x');ylabel('y');zlabel('J');
title('Funcion de Rastrigin 2D');
%Cantidad de variables
NV = 2;
%
optionsga = optimoptions('ga', ...
'Display', 'iter', ...
'PopulationSize', 25, ...
'MaxGenerations', 200, ...
'PlotFcn', {@gaplotbestf, @gaplotbestindiv, @gaplotexpectation, @gaplotstopping});
%Función que implementa la optimización con AG
y = ga(@rastigin,NV,optionsga)