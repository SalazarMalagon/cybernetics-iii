% Aplicación red neuronal FF multicapa usando feedforward
clear; clc; close all;
% Datos de entrada (XOR)
P = [0 0 1 1; 
     0 1 0 1];   % Entradas
T = [0 1 1 0];   % Salidas esperadas (XOR)
% Crear red neuronal con 1 capa oculta de 3 neuronas
net = newff(minmax(P),[3 1],{'tansig','tansig'},'traingd');
% Configurar parámetros de entrenamiento (opcional)
net.trainParam.epochs = 1000;
net.trainParam.goal = 1e-9;
% Entrenar la red
net = train(net, P, T);
% Simular red entrenada
Y = net(P)
% Mostrar resultados
disp('Salidas esperadas:'); disp(T);
disp('Salidas de la red:'); disp(round(Y));
perf = perform(net,Y,T)