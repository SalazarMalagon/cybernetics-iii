% Aplicación ANFIS para el modelado
clear,clc, close all
set(0,'DefaultLineLineWidth',2)
set(0, 'defaultfigurecolor', [1 1 1])
% Simular el sistema
datos = sim('modelado_ANFIS.slx');
X = datos.simout(:,1:3); %Entrada
Y = datos.simout(:,4); %Salida
datos = datos.simout;
% Sistema de inferencia TSK
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = [3 2 3];
opt.InputMembershipFunctionType = ["gbellmf" "pimf" "trimf"];
opt.OutputMembershipFunctionType = "linear";
%
in_fis = genfis(X, Y, opt);
%
% Épocas de entrenamiento
epoch_n = 100;
% Entrenamiento
out_fis = anfis(datos, in_fis, epoch_n);
% Resultado de la simulación
ys = evalfis(out_fis, X);
% Salida real
yr = datos(:,4);
% Gráficas
plot(ys, 'r')
hold on
plot(yr, '--b')
xlabel('Tiempo')
ylabel('y(t)')
title('Comparación')
