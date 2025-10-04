err_val = out.error;                  % valores de error
simTime = 10;                         % duración de la simulación (ajusta si usaste otro tiempo)
t = linspace(0, simTime, length(err_val));

% Calcular métricas
err_max = max(err_val);               % error absoluto máximo
err_mean = mean(err_val);             % error medio (promedio)
err_rmse = sqrt(mean(err_val.^2));    % error cuadrático medio (RMSE)

% Mostrar resultados
fprintf('Error máximo: %.4f\n', err_max);
fprintf('Error medio: %.4f\n', err_mean);
fprintf('Error RMSE : %.4f\n', err_rmse);

% Graficar
figure;
plot(t, err_val,'r','LineWidth',1.5);
grid on;
xlabel('Tiempo (s)');
ylabel('Error absoluto');
title('Error entre seno y FIS');




