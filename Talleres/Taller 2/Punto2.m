clear,clc, close all
set(0,'DefaultLineLineWidth',2)
set(0, 'defaultfigurecolor', [1 1 1])

% Generar señal triangular
t = linspace(0,2*pi, 1000);
x = 1*sawtooth(t+pi/2, 0.5);
% Señal objetivo
y = 1*sin(t);
% Visualizar las señales
figure;
plot(t, x, 'r')
hold on
plot(t, y, 'b')
title('Señal de entrada y objetivo')
xlabel('t'); ylabel('Amplitud')
grid on

% Crear red neuronal feedforward
hiddenlayers = [4 4 4]; 
net = feedforwardnet(hiddenlayers, 'trainscg');
net.trainParam.epochs = 2000;
net.trainParam.goal = 0.00005;
% Entrenar la red
[net, tr] = train(net, x, y);
% Salida de la red
y_pred = net(x);

% Calcular los errores
mse_error = mse(y-y_pred);
error_maximo = max(abs(y-y_pred));
error_relativo = error_maximo / max(abs(y)) * 100;

% Imprimir resultados en consola
fprintf('=== RESULTADOS DE LA RED NEURONAL ===\n');
fprintf('1. Error máximo: %.6f', error_maximo);
if error_relativo < 5
    fprintf(' (Inferior al 5%% ✓)\n');
else
    fprintf(' (Superior al 5%% ✗)\n');
end
fprintf('2. Error cuadrático medio: %.6f', mse_error);
if mse_error < 0.02
    fprintf(' (Inferior al 2%% ✓)\n');
else
    fprintf(' (Superior al 2%% ✗)\n');
end
fprintf('3. Error relativo: %.4f%%\n', error_relativo);
fprintf('=====================================\n');

% Visualizar resultados
figure;
plot(t, y, 'b')
hold on
plot(t, y_pred, 'g')
legend('Seno real','Salida de la red neuronal')
title('Comparación')
xlabel('t'); ylabel('Amplitud')
grid on