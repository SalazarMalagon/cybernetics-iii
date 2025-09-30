%Aplicacion red neuronal tipo perceptron
clear;clc; close all;
%Rango de las entradas
R = [0 1;
    0 1];
%configuracion de capas y neuronas
S = 1;
% Crea una red tipo percetron
net = newp(R, S, 'hardlim', 'learnp');
%Datos de entrada
P = [0 0 1 1;
    0 1 0 1 ];
% Datos de salida
T = [0 0 0 1];
%Simular sin entrenar
%Y = sim (net, p);
net.trainParam.epochs = 30;
net = train(net, P, T);
% Simular la red entrenada
Y = sim(net, P);