%Aplicacion red neuronal tipo perceptron
%preambulo
clear;clc; close all;
set(0,'DefaultLineLineWidth',2)
set(0, 'defaultfigurecolor', [1 1 1])
%preambulo
clear;clc; close all;
set(0,'DefaultLineLineWidth',2)
set(0, 'defaultfigurecolor', [1 1 1])
%Rango de las entradas
R=[0 1; 
   0 1];
%Configuracion capas y neuronas 
S=1;
%Crea una red neuronal tipo perceptron
net = newp(R,S,'hardlim','learnp');

%Datos entrada
P = [0 0 1 1; 
     0 1 0 1];
%Datos de salida
T = [0 1 1 0];
%Simulación sin entrenar
Y1 = sim(net,P)
%Entrenamiento
net.trainParam.epochs = 20;
net.trainParam.goal = 1e-6;
net = train(net,P,T);
pesos = net.iw(1,1)
bias = net.b(1,1)
%Simulacion con entrenamiento
Y = sim(net,P)
