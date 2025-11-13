%% MATLAB SCRIPT: Identificación ANFIS Multisalida (3 Modelos NARX)

clear; clc; close all;
set(0, 'DefaultLineLineWidth', 1.5);
set(0, 'defaultfigurecolor', [1 1 1]);

semilla = 42; 
rng(semilla, 'twister'); 
disp(['Semilla aleatoria fijada a: ', num2str(semilla)]);

% =================================
% 1. PARÁMETROS FÍSICOS Y DISCRETIZACIÓN (Igual)
% =================================
disp('1. Definiendo parámetros y discretizando el sistema.');
m_b = 250.0; m_w = 50.0; k_s = 16000.0; k_t = 190000.0; b_s = 1000.0; b_t = 0.0;
A = [0, 1, 0, 0; -k_s/m_b, -b_s/m_b, k_s/m_b, b_s/m_b; 0, 0, 0, 1; k_s/m_w, b_s/m_w, -(k_s + k_t)/m_w, -(b_s + b_t)/m_w];
B = [0, 0; 0, 1/m_b; 0, 0; k_t/m_w, 1/m_w];
C = [1, 0, 0, 0; -k_s/m_b, -b_s/m_b, k_s/m_b, b_s/m_b; 1, 0, -1, 0];
D = [0, 0; 0, 1/m_b; 0, 0];
Ts = 0.01;
sistema_continuo = ss(A, B, C, D);
sistema_discreto = c2d(sistema_continuo, Ts, 'zoh');
Ad = sistema_discreto.A; Bd = sistema_discreto.B; Cd = sistema_discreto.C; Dd = sistema_discreto.D;
num_states = size(Ad, 1); num_outputs = size(Cd, 1); 

% =================================
% 2. SIMULACIÓN MONTE CARLO (Generación de Datos)
% =================================
disp('2. Generando 20 simulaciones con baches aleatorios.');
num_simulaciones = 20; T_sim = 20.0; N = floor(T_sim / Ts); t = (0:N-1)' * Ts;
u_data_all = zeros(num_simulaciones, N); y_all_simulations = zeros(num_simulaciones, N, num_outputs); 
bump_height_min = 0.05; bump_height_max = 0.2; bump_duration_min = 1.0; bump_duration_max = 3.0; t_bump_start = 2.0;

for i = 1:num_simulaciones
    bump_height = bump_height_min + (bump_height_max - bump_height_min) * rand();
    bump_duration = bump_duration_min + (bump_duration_max - bump_duration_min) * rand();
    t_bump_end = t_bump_start + bump_duration;
    u_signal = zeros(N, 2);
    idx_start = floor(t_bump_start / Ts) + 1; idx_end = floor(t_bump_end / Ts) + 1;
    if idx_end > N; idx_end = N; end
    t_bump = t(idx_start:idx_end) - t_bump_start;
    bump = bump_height * 0.5 * (1 - cos(pi * (t_bump) / bump_duration)); 
    u_signal(idx_start:idx_end, 1) = bump; 
    
    x_k = zeros(num_states, 1); y_k_sim = zeros(N, num_outputs);
    for k = 1:N-1
        u_k = u_signal(k, :)';
        y_k = Cd * x_k + Dd * u_k;
        x_k_plus_1 = Ad * x_k + Bd * u_k;
        y_k_sim(k, :) = y_k';
        x_k = x_k_plus_1;
    end
    u_data_all(i, :) = u_signal(:, 1)'; 
    y_all_simulations(i, :, :) = y_k_sim;
end

% Parámetros NARX (fijos para las 3 salidas)
ny = 2; nu = 2; max_delay = max(ny, nu);
N_rows = num_simulaciones * (N - max_delay);
epoch_n = 100; mf_type = 'gaussmf'; numMF = 3;

% =================================
% 3. BUCLE DE ENTRENAMIENTO Y PREDICCIÓN PARA LAS 3 SALIDAS
% =================================
Results = cell(num_outputs, 5); % Para guardar las métricas de error
ys_all = zeros(N_rows, num_outputs); % Para guardar las predicciones de las 3 salidas

for output_idx = 1:num_outputs
    disp(['3. Preparando y entrenando ANFIS para Salida ', num2str(output_idx), '...']);
    
    Y_out = squeeze(y_all_simulations(:, :, output_idx)); % Salida objetivo actual (y1, y2 o y3)

    % 3.1. Estructura de Datos NARX para la Salida Actual
    X_train = zeros(N_rows, ny + nu);
    Y_train = zeros(N_rows, 1);
    row_index = 1;

    for i = 1:num_simulaciones
        u_sim = u_data_all(i, :)';
        y_sim = Y_out(i, :)';
        for k = max_delay+1 : N
            % Entradas ANFIS: [y_{i,k-1}, y_{i,k-2}, u_k, u_{k-1}]
            X_train(row_index, 1) = y_sim(k-1); X_train(row_index, 2) = y_sim(k-2);   
            X_train(row_index, 3) = u_sim(k); X_train(row_index, 4) = u_sim(k-1);   
            Y_train(row_index, 1) = y_sim(k);     
            row_index = row_index + 1;
        end
    end
    
    data_train_raw = [X_train, Y_train];

    % 3.2. Normalización (Normalización específica para cada salida)
    min_data = min(data_train_raw); max_data = max(data_train_raw); range_data = max_data - min_data;
    data_train_norm = (data_train_raw - min_data) ./ range_data;

    % 3.3. Entrenamiento ANFIS
    in_fis = genfis1(data_train_norm, numMF, mf_type, 'linear');
    [out_fis, trainError, ~] = anfis(data_train_norm, in_fis, epoch_n);
    
    % 3.4. Predicción y Desnormalización
    X_test_norm = data_train_norm(:, 1:end-1); 
    Y_test_norm = data_train_norm(:, end);
    ys_norm = evalfis(X_test_norm, out_fis);

    min_y = min_data(end); range_y = range_data(end);
    ys = (ys_norm .* range_y) + min_y;
    y_real = (Y_test_norm .* range_y) + min_y;
    
    ys_all(:, output_idx) = ys; % Guardar la predicción
    
    % 3.5. Cálculo y Almacenamiento de Métricas
    e = y_real - ys;
    MAE = mean(abs(e));
    MSE = mean(e.^2);
    RMSE = sqrt(MSE);
    ErrorMax = max(abs(e));

    Results(output_idx, :) = {['y', num2str(output_idx)], MAE, MSE, RMSE, ErrorMax};
end

% =================================
% 4. RESULTADOS Y VISUALIZACIÓN
% =================================

% 4.1. Mostrar Resultados en la Consola
disp(' ');
disp('===================================================================');
disp('=== Rendimiento de Identificación ANFIS para las TRES Salidas ===');
disp('===================================================================');
fprintf('Arquitectura: %d MFs (Gaussmf), %d Reglas, %d Épocas\n', numMF, numMF^4, epoch_n);
disp(' ');
ResultTable = cell2table(Results, ...
    'VariableNames', {'Salida','MAE','MSE','RMSE','ErrorMax'});
disp(ResultTable);
disp('-------------------------------------------------------------------');


% 4.2. Gráfica de Comparación de las 3 Salidas
disp('4. Generando gráfica de comparación de las 3 salidas.');

sim_plot = 1; % Simulación de ejemplo
start_idx = (sim_plot - 1) * (N - max_delay) + 1;
end_idx = sim_plot * (N - max_delay);
t_plot = t(max_delay+1 : N);

y1_real = squeeze(y_all_simulations(sim_plot, max_delay+1:N, 1));
y2_real = squeeze(y_all_simulations(sim_plot, max_delay+1:N, 2));
y3_real = squeeze(y_all_simulations(sim_plot, max_delay+1:N, 3));

y1_pred = ys_all(start_idx:end_idx, 1);
y2_pred = ys_all(start_idx:end_idx, 2);
y3_pred = ys_all(start_idx:end_idx, 3);

figure(1)
fig.Name = 'Comparación de las 3 Salidas';
fig.Position = [100 100 800 1000]; 
sgtitle(sprintf('Identificación ANFIS: Comparación de Predicción vs. Real (Simulación %d)', sim_plot));

% Plot 1: Salida 1 (Desplazamiento Carrocería)
subplot(3, 1, 1);
plot(t_plot, y1_real, 'b'); hold on;
plot(t_plot, y1_pred, 'r--');
title('Salida 1: Desplazamiento Carrocería ($y_1 = z_b$)');
legend('Real', 'ANFIS Predicción', 'Location', 'best');
ylabel('Posición (m)');
grid on;

% Plot 2: Salida 2 (Aceleración Carrocería)
subplot(3, 1, 2);
plot(t_plot, y2_real, 'b'); hold on;
plot(t_plot, y2_pred, 'r--');
title('Salida 2: Aceleración Carrocería ($y_2 = \ddot{z}_b$)');
legend('Real', 'ANFIS Predicción', 'Location', 'best');
ylabel('Aceleración ($m/s^2$)');
grid on;

% Plot 3: Salida 3 (Deflexión Suspensión)
subplot(3, 1, 3);
plot(t_plot, y3_real, 'b'); hold on;
plot(t_plot, y3_pred, 'r--');
title('Salida 3: Deflexión de la Suspensión ($y_3 = z_b - z_w$)');
legend('Real', 'ANFIS Predicción', 'Location', 'best');
ylabel('Deflexión (m)');
xlabel('Tiempo (s)');
grid on;

disp('Script Finalizado. Revise la tabla de métricas y la Figura 1.');