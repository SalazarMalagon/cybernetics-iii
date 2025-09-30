%Aplicacion red neuronal FF nulticapa
clear;clc; close all;
%Rango de las entradas
R = [0 1;
    0 1];
%configuracion de capas y neuronas
S = [3 1];
%Creamos la red neuronal
net = newff(R, S, {'tansig', 'tansig'});
%Datos de entrada
P = [0 0 1 1;
    0 1 0 1];
%Datos de salida
T = [0 1 1 0];
%Entrenamiento
net = train(net, P, T);
Y = round(sim(net,P));