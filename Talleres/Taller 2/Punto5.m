%% MODELADO ANFIS - Sistema G(s) = 1 / (s^4 + 2s^3 + 2s^2 + 6s + 5)
clear; clc; close all;
set(0,'DefaultLineLineWidth',2)
set(0,'defaultfigurecolor',[1 1 1]);

%% Simular el sistema en Simulink
% Asegúrate de que el modelo se llama "anfis1.slx" y tiene el bloque modificado
sim_time = 20; % tiempo de simulación
datos = sim('anfis1.slx', 'StopTime', num2str(sim_time));

%% Preparar los datos de entrenamiento
data = datos.simout;   % columnas: [u(k) y(k-1) y(k-2) y(k)]
X = data(:,1:3);       % entradas al ANFIS
Y = data(:,4);         % salida real

%% Crear estructura inicial del FIS tipo TSK
opt = genfisOptions('GridPartition');
opt.NumMembershipFunctions = [3 2 3];
opt.InputMembershipFunctionType = ["gbellmf" "pimf" "trimf"];
opt.OutputMembershipFunctionType = "linear";

in_fis = genfis(X, Y, opt);

%% Entrenamiento ANFIS
epoch_n = 100;
[out_fis, trainError] = anfis([X Y], in_fis, epoch_n);

%% Evaluar desempeño
ys = evalfis(out_fis, X);   % salida estimada ANFIS
yr = Y;                     % salida real

%% Calcular errores
error_abs = abs(yr - ys);
error_max = max(error_abs) / max(yr) * 100;
rmse = sqrt(mean((yr - ys).^2)) / mean(yr) * 100;

fprintf('Error máximo = %.6f %%\n', error_max);
fprintf('RMSE = %.6f %%\n', rmse);


%% Graficar comparación
figure;
plot(ys,'r')
hold on
plot(yr,'--b');
xlabel('Tiempo (muestras)');
ylabel('Salida');
legend('Sistema Real', 'ANFIS');
title('Comparación de Salidas');

%% Graficar error de entrenamiento
figure;
plot(trainError, 'LineWidth', 1.5);
xlabel('Época');
ylabel('Error RMS');
title('Evolución del error de entrenamiento ANFIS');
grid on;

