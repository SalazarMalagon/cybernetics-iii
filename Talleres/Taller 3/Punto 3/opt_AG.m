% preambulo
clear;clc;close all;
set(0, 'DefaultLineLinewidth', 2)
set(0,'defaultfigurecolor', [1 1 1]) 
%Superficie
x =-5:0.1:5;
y = x; 
%Evaluacion de todoslos datos
for j=1:length(x)
for i=1:length(y)
z(i,j)=PassinoE([x(j);y(i)]);
end
end 
%Figura de la función objetivo en 2D
figure
surf(x,y,z);
colormap(jet)
xlabel('x');ylabel('y');zlabel('J');
title('Funcion objetivo'); 
%Optimizacion empleando CN
%Punto inicial 
%X0 = [2 1];
X0 = [4 4];
%Opciones del algoritmo
options = optimset('Display','iter');
%Función que implementa la optimización con CN
X = fminunc(@PassinoE,X0,options)