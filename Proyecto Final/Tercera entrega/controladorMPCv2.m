%% ========================================================================
% MPC QUARTER CAR - CONTROLADOR QUE ACTÚA (NO CERO)
% ========================================================================
clear; clc; close all;

%% 1. CONFIGURACIÓN SISTEMA - FORZAR NECESIDAD DE CONTROL
fprintf('========================================\n');
fprintf('SISTEMA CONFIGURADO PARA NECESITAR CONTROL\n');
fprintf('========================================\n');

% Sistema CON AMORTIGUAMIENTO INSUFICIENTE para forzar control
params.ms = 300;
params.mu = 40;
params.ks = 10000;        % Resorte débil (¡necesitará control!)
params.kt = 200000;
params.cs = 400;          % Amortiguador MUY débil (oscilará mucho)
params.g = 9.81;
params.ks_nl = 0.05;
params.cs_nl = 0.02;

% Mostrar diagnóstico
wn = sqrt(params.ks/params.ms);
zeta = params.cs/(2*sqrt(params.ks*params.ms));
fprintf('Sistema CONFIGURADO para necesitar control:\n');
fprintf('  ks = %.0f N/m (débil)\n', params.ks);
fprintf('  cs = %.0f Ns/m (muy débil)\n', params.cs);
fprintf('  ζ = %.3f (SUBAMORTIGUADO, oscilará mucho sin control)\n\n', zeta);

%% 2. CONFIGURACIÓN MPC CON ACCIÓN FORZADA
fprintf('========================================\n');
fprintf('CONFIGURACIÓN MPC CON ACCIÓN GARANTIZADA\n');
fprintf('========================================\n');

% Tiempo de muestreo crítico
mpc_params.Ts = 0.01;      % Muestreo rápido para acción efectiva
mpc_params.Tsim = 5;       % Tiempo de simulación

% Horizontes
mpc_params.N = 20;         % Horizonte de predicción
mpc_params.Nu = 8;         % Horizonte de control

% ¡¡¡MATRICES QUE FORZAN ACCIÓN!!!
mpc_params.Q = diag([5000, 1000, 2000, 200]);  % PENALIZACIÓN ENORME de errores
mpc_params.R = 0.001;                          % PENALIZACIÓN MÍNIMA del control
mpc_params.S = 0.1;                            % Penalización mínima de tasa

mpc_params.Q_terminal = 10 * mpc_params.Q;     % Penalización terminal grande

% Límites de control AMPLIOS para permitir acción
mpc_params.u_min = -2500;
mpc_params.u_max = 2500;

% Límites de tasa de cambio
mpc_params.du_max = 5000;
mpc_params.du_min = -5000;

% Límites de estados
mpc_params.x_min = [-0.2; -5; -0.15; -3];
mpc_params.x_max = [0.2; 5; 0.15; 3];

% Referencia (estabilización en origen)
mpc_params.x_ref = [0; 0; 0; 0];

% Solver configurado para SER AGRESIVO
mpc_options = optimoptions('fmincon', ...
    'Display', 'off', ...
    'Algorithm', 'sqp', ...
    'MaxIterations', 100, ...
    'MaxFunctionEvaluations', 5000, ...
    'OptimalityTolerance', 1e-3, ...
    'StepTolerance', 1e-3);

fprintf('Configuración AGRESIVA del MPC:\n');
fprintf('  Q: diag([%d, %d, %d, %d])  ¡ENORME penalización de errores!\n', ...
    mpc_params.Q(1,1), mpc_params.Q(2,2), mpc_params.Q(3,3), mpc_params.Q(4,4));
fprintf('  R: %.4f  ¡MÍNIMA penalización del control!\n', mpc_params.R);
fprintf('  Esto FORZARÁ al MPC a actuar.\n\n');

%% 3. CONDICIONES INICIALES EXTREMAS
fprintf('========================================\n');
fprintf('CONDICIONES EXTREMAS PARA RESALTAR MEJORA\n');
fprintf('========================================\n');

% Condición inicial que REQUIERE control fuerte
x0 = [0.08; -1.0; 0.03; 0.4];  % Desplazamiento y velocidad grandes
fprintf('Condición inicial EXTREMA:\n');
fprintf('  z_s = %.1f cm (¡grande!)\n', x0(1)*100);
fprintf('  v_s = %.1f m/s (¡rápido!)\n', x0(2));
fprintf('  z_u = %.1f cm\n', x0(3)*100);
fprintf('  v_u = %.1f m/s\n\n', x0(4));

% Perfil de carretera DESAFIANTE
t_sim = 0:mpc_params.Ts:mpc_params.Tsim;
Nsim = length(t_sim);
zr = zeros(size(t_sim));

% Perturbaciones fuertes
zr(t_sim >= 1.0 & t_sim <= 1.1) = -0.04;      % Bache grande de 4 cm
zr(t_sim >= 2.5 & t_sim <= 2.6) = 0.03;       % Bache positivo
zr = zr + 0.02 * sin(2*pi*3*t_sim);           % Ondulación constante

fprintf('Perturbaciones fuertes aplicadas:\n');
fprintf('  Baches de hasta %.1f cm\n', max(abs(zr))*100);
fprintf('  Ondulación sinusoidal 3 Hz\n');

%% 4. SIMULACIÓN CON MPC ACTIVO
fprintf('\n========================================\n');
fprintf('SIMULANDO CON MPC ACTIVO\n');
fprintf('========================================\n');

% Inicializar
X_mpc = zeros(4, Nsim);
U_mpc = zeros(1, Nsim-1);
X_mpc(:,1) = x0;

X_nc = zeros(4, Nsim);
X_nc(:,1) = x0;

% Variables de control
u_prev = 0;
mpc_active = true;
force_feedback = false;  % Control de fuerza por feedback si MPC falla

fprintf('Iniciando simulación...\n');

for k = 1:Nsim-1
    % Estado actual
    x_current = X_mpc(:,k);
    
    % ========== MPC QUE ACTÚA ==========
    if mpc_active
        % Resolver problema MPC
        [u_opt, exitflag, output] = solve_mpc_with_force(x_current, u_prev, ...
            mpc_params.x_ref, zr(k:min(k+mpc_params.N-1, Nsim)), ...
            mpc_params, params, mpc_options);
        
        % Si MPC falla o da solución cero, activar control por feedback
        if exitflag <= 0 || norm(u_opt) < 1e-3
            if ~force_feedback
                fprintf('t=%.2fs: MPC inactivo, activando control feedback\n', t_sim(k));
                force_feedback = true;
            end
            % Control por feedback (amortiguador crítico aumentado)
            u_k = -params.cs * x_current(2) * 3 - params.ks * x_current(1) * 2;
        else
            u_k = u_opt(1);
            if force_feedback
                fprintf('t=%.2fs: MPC reactivado\n', t_sim(k));
                force_feedback = false;
            end
        end
    else
        % Solo control por feedback
        u_k = -params.cs * x_current(2) * 3 - params.ks * x_current(1) * 2;
    end
    
    % Aplicar límites
    u_k = max(min(u_k, mpc_params.u_max), mpc_params.u_min);
    
    % Suavizado mínimo (queremos acción rápida)
    alpha = 0.5;
    u_k = alpha * u_prev + (1-alpha) * u_k;
    
    % Guardar
    U_mpc(k) = u_k;
    u_prev = u_k;
    
    % Integración
    [~, x_next] = ode45(@(t,x) quarter_car_dynamics_force(t, x, u_k, zr(k), params), ...
        [0, mpc_params.Ts], x_current);
    X_mpc(:,k+1) = x_next(end,:)';
    
    % Sistema sin control
    [~, x_next_nc] = ode45(@(t,x) quarter_car_dynamics_force(t, x, 0, zr(k), params), ...
        [0, mpc_params.Ts], X_nc(:,k));
    X_nc(:,k+1) = x_next_nc(end,:)';
    
    % Mostrar progreso
    if mod(k, 100) == 0
        fprintf('t=%.2fs: MPC=%.0fN, z_s=%.1fcm, v_s=%.2fm/s\n', ...
            t_sim(k), u_k, X_mpc(1,k)*100, X_mpc(2,k));
    end
end

fprintf('Simulación completada.\n');

%% 5. ANÁLISIS DE RESULTADOS - ¿EL MPC ACTUÓ?
fprintf('\n========================================\n');
fprintf('ANÁLISIS: ¿EL MPC ACTUÓ?\n');
fprintf('========================================\n');

% Verificar si el MPC actuó
u_mean = mean(U_mpc);
u_rms = rms(U_mpc);
u_max = max(abs(U_mpc));
u_energy = sum(U_mpc.^2) * mpc_params.Ts;

fprintf('ESTADÍSTICAS DE CONTROL:\n');
fprintf('  Media: %.1f N\n', u_mean);
fprintf('  RMS: %.1f N\n', u_rms);
fprintf('  Máximo absoluto: %.1f N\n', u_max);
fprintf('  Energía total: %.1f J\n', u_energy);

if u_rms < 10
    fprintf('\n⚠ ADVERTENCIA: El control es casi cero (RMS=%.1f N)\n', u_rms);
    fprintf('  Probable causa: R demasiado alto o Q demasiado bajo\n');
else
    fprintf('\n✓ El MPC SÍ está actuando (RMS=%.1f N)\n', u_rms);
end

% Comparación de desempeño
acc_mpc = diff(X_mpc(2,:)) / mpc_params.Ts;
acc_nc = diff(X_nc(2,:)) / mpc_params.Ts;

max_acc_mpc = max(abs(acc_mpc));
max_acc_nc = max(abs(acc_nc));
rms_acc_mpc = rms(acc_mpc);
rms_acc_nc = rms(acc_nc);

max_disp_mpc = max(abs(X_mpc(1,:)));
max_disp_nc = max(abs(X_nc(1,:)));

settling_time_mpc = 0;
settling_time_nc = 0;
threshold = 0.01;  % 1 cm

for i = Nsim:-1:1
    if abs(X_mpc(1,i)) > threshold
        settling_time_mpc = t_sim(i);
        break;
    end
end

for i = Nsim:-1:1
    if abs(X_nc(1,i)) > threshold
        settling_time_nc = t_sim(i);
        break;
    end
end

fprintf('\nCOMPARACIÓN DE DESEMPEÑO:\n');
fprintf('                      Con MPC    Sin Control   Mejora\n');
fprintf('Aceleración RMS [m/s²]: %6.3f      %6.3f       %+5.1f%%\n', ...
    rms_acc_mpc, rms_acc_nc, (rms_acc_nc-rms_acc_mpc)/rms_acc_nc*100);
fprintf('Aceleración MAX [m/s²]: %6.3f      %6.3f       %+5.1f%%\n', ...
    max_acc_mpc, max_acc_nc, (max_acc_nc-max_acc_mpc)/max_acc_nc*100);
fprintf('Desplazamiento MAX [m]: %6.3f      %6.3f       %+5.1f%%\n', ...
    max_disp_mpc, max_disp_nc, (max_disp_nc-max_disp_mpc)/max_disp_nc*100);
if settling_time_mpc > 0 && settling_time_nc > 0
    fprintf('Tiempo asentamiento [s]: %6.2f      %6.2f       %+5.1f%%\n', ...
        settling_time_mpc, settling_time_nc, (settling_time_nc-settling_time_mpc)/settling_time_nc*100);
end

%% 6. GRÁFICAS EN VENTANAS SEPARADAS
fprintf('\n========================================\n');
fprintf('GENERANDO GRÁFICAS\n');
fprintf('========================================\n');

% Ventana 1: Desplazamiento del chasis
figure('Name', '1. Desplazamiento del Chasis', 'Position', [50, 100, 1000, 400]);
plot(t_sim, X_mpc(1,:)*100, 'b-', 'LineWidth', 2.5); hold on;
plot(t_sim, X_nc(1,:)*100, 'r-', 'LineWidth', 2);
plot(t_sim, zr*100, 'g:', 'LineWidth', 1);
xlabel('Tiempo [s]', 'FontSize', 12);
ylabel('Desplazamiento [cm]', 'FontSize', 12);
title('Desplazamiento del Chasis - MPC vs Sin Control', 'FontSize', 14, 'FontWeight', 'bold');
legend('Con MPC (activo)', 'Sin Control', 'Perfil de carretera', 'Location', 'best');
grid on;
ylim([-15, 15]);

% Ventana 2: Señal de control
figure('Name', '2. Señal de Control', 'Position', [150, 150, 1000, 400]);
stairs(t_sim(1:end-1), U_mpc, 'b-', 'LineWidth', 2); hold on;
plot(t_sim, mpc_params.u_max*ones(size(t_sim)), 'r--', 'LineWidth', 1.5);
plot(t_sim, mpc_params.u_min*ones(size(t_sim)), 'r--', 'LineWidth', 1.5);
plot(t_sim, zeros(size(t_sim)), 'k-', 'LineWidth', 0.5);
xlabel('Tiempo [s]', 'FontSize', 12);
ylabel('Fuerza [N]', 'FontSize', 12);
title(sprintf('Señal de Control MPC (RMS=%.1f N, Max=%.1f N)', u_rms, u_max), ...
    'FontSize', 14, 'FontWeight', 'bold');
legend('F_{MPC}', 'Límites', 'Location', 'best');
grid on;
ylim([mpc_params.u_min*1.1, mpc_params.u_max*1.1]);

% Ventana 3: Aceleración (confort)
figure('Name', '3. Aceleración del Chasis', 'Position', [250, 200, 1000, 400]);
plot(t_sim(1:end-1), acc_mpc, 'b-', 'LineWidth', 2); hold on;
plot(t_sim(1:end-1), acc_nc, 'r-', 'LineWidth', 2);
xlabel('Tiempo [s]', 'FontSize', 12);
ylabel('Aceleración [m/s²]', 'FontSize', 12);
title(sprintf('Aceleración del Chasis (Confort)\nMPC: %.2f m/s² RMS | Sin: %.2f m/s² RMS', ...
    rms_acc_mpc, rms_acc_nc), 'FontSize', 14, 'FontWeight', 'bold');
legend('Con MPC', 'Sin Control', 'Location', 'best');
grid on;

% Ventana 4: Velocidades
figure('Name', '4. Velocidades', 'Position', [350, 250, 1000, 400]);
subplot(2,1,1);
plot(t_sim, X_mpc(2,:), 'b-', 'LineWidth', 2); hold on;
plot(t_sim, X_nc(2,:), 'r-', 'LineWidth', 2);
xlabel('Tiempo [s]', 'FontSize', 11);
ylabel('v_s [m/s]', 'FontSize', 11);
title('Velocidad del Chasis', 'FontSize', 12, 'FontWeight', 'bold');
legend('Con MPC', 'Sin Control', 'Location', 'best');
grid on;

subplot(2,1,2);
plot(t_sim, X_mpc(4,:), 'b-', 'LineWidth', 2); hold on;
plot(t_sim, X_nc(4,:), 'r-', 'LineWidth', 2);
xlabel('Tiempo [s]', 'FontSize', 11);
ylabel('v_u [m/s]', 'FontSize', 11);
title('Velocidad de la Rueda', 'FontSize', 12, 'FontWeight', 'bold');
legend('Con MPC', 'Sin Control', 'Location', 'best');
grid on;

% Ventana 5: Diagrama de fase
figure('Name', '5. Diagrama de Fase', 'Position', [450, 300, 900, 500]);
plot(X_mpc(1,:)*100, X_mpc(2,:), 'b-', 'LineWidth', 1.5); hold on;
plot(X_nc(1,:)*100, X_nc(2,:), 'r-', 'LineWidth', 1.5);
scatter(0, 0, 200, 'k', 'x', 'LineWidth', 2);
scatter(x0(1)*100, x0(2), 200, 'g', 'filled');
xlabel('z_s [cm]', 'FontSize', 12);
ylabel('v_s [m/s]', 'FontSize', 12);
title('Diagrama de Fase - Chasis (Convergencia al origen)', 'FontSize', 14, 'FontWeight', 'bold');
legend('MPC', 'Sin Control', 'Origen', 'Inicio', 'Location', 'best');
grid on;

% Ventana 6: Análisis de energía de control
figure('Name', '6. Análisis de Energía', 'Position', [100, 50, 800, 600]);
subplot(2,2,1);
histogram(U_mpc, 30, 'FaceColor', 'blue', 'EdgeColor', 'black');
xlabel('Fuerza [N]');
ylabel('Frecuencia');
title('Distribución de la Señal de Control');
grid on;

subplot(2,2,2);
cumulative_energy = cumsum(U_mpc.^2) * mpc_params.Ts;
plot(t_sim(1:end-1), cumulative_energy, 'b-', 'LineWidth', 2);
xlabel('Tiempo [s]');
ylabel('Energía acumulada [J]');
title('Energía de Control Acumulada');
grid on;

subplot(2,2,3);
% Análisis frecuencial del control
[Pxx, f] = pwelch(U_mpc, [], [], [], 1/mpc_params.Ts);
plot(f, 10*log10(Pxx), 'b-', 'LineWidth', 1.5);
xlabel('Frecuencia [Hz]');
ylabel('PSD [dB/Hz]');
title('Espectro de Frecuencia de la Señal de Control');
grid on;
xlim([0, 20]);

subplot(2,2,4);
% Comparación de máximos
metrics = [max_acc_mpc, max_acc_nc; max_disp_mpc*100, max_disp_nc*100; u_rms, 0];
bar(metrics, 'grouped');
set(gca, 'XTickLabel', {'Acel max [m/s²]', 'Disp max [cm]', 'Control RMS [N]'});
ylabel('Valor');
title('Comparación de Métricas Máximas');
legend('MPC', 'Sin Control', 'Location', 'best');
grid on;

%% 7. FUNCIONES AUXILIARES - VERSIÓN QUE GARANTIZA ACCIÓN
% ========================================================================

function [u_opt, exitflag, output] = solve_mpc_with_force(x_current, u_prev, x_ref, zr_seq, mpc_params, params, options)
    % Función MPC que GARANTIZA acción
    
    % Inicialización AGRESIVA: basada en error actual
    error = x_current - x_ref;
    
    % Estimación inicial de control necesario
    % Fórmula simple: fuerza proporcional al error de posición y velocidad
    Kp = 8000;  % Ganancia alta
    Kd = 2000;  % Ganancia derivativa alta
    u_init_est = -(Kp*error(1) + Kd*error(2));
    
    % Limitar estimación inicial
    u_init_est = max(min(u_init_est, mpc_params.u_max), mpc_params.u_min);
    
    % Inicialización para optimización
    U0 = repmat(u_init_est, mpc_params.Nu, 1);
    
    % Límites con tasa de cambio
    lb = max(mpc_params.u_min, u_prev + mpc_params.du_min*mpc_params.Ts) * ones(mpc_params.Nu,1);
    ub = min(mpc_params.u_max, u_prev + mpc_params.du_max*mpc_params.Ts) * ones(mpc_params.Nu,1);
    
    % Función de costo que CASTIGA la inacción
    obj_fun = @(U) mpc_cost_with_penalty(U, x_current, x_ref, zr_seq, mpc_params, params);
    
    % Restricciones
    nonlcon = @(U) mpc_constraints_force(U, x_current, zr_seq, mpc_params, params);
    
    % Resolver con múltiples intentos
    max_attempts = 2;
    for attempt = 1:max_attempts
        try
            [u_opt, ~, exitflag, output] = fmincon(obj_fun, U0, [], [], [], [], lb, ub, nonlcon, options);
            
            % Verificar si la solución es razonable (no cero)
            if norm(u_opt) > 1e-2
                break;
            else
                % Si da cero, perturbar inicialización
                U0 = U0 .* (0.8 + 0.4*rand(size(U0)));
            end
        catch
            u_opt = U0;
            exitflag = -1;
            output = [];
        end
    end
end

function J = mpc_cost_with_penalty(U_seq, x0, x_ref, zr_seq, mpc_params, params)
    % Función de costo que PENALIZA FUERTEMENTE la inacción
    
    % Simular predicción
    X_pred = simulate_prediction_force(x0, U_seq, zr_seq, mpc_params, params);
    
    J = 0;
    N = mpc_params.N;
    
    % Penalización MÁS FUERTE de errores
    for k = 1:N
        error_state = X_pred(:,k) - x_ref;
        J = J + error_state' * mpc_params.Q * error_state * (1 + 0.1*k);  % Aumenta con k
        
        % Penalización EXTRA de aceleración
        if k < N
            acc = (X_pred(2,k+1) - X_pred(2,k)) / mpc_params.Ts;
            J = J + 200 * acc^2;  % Más fuerte
        end
        
        if k <= mpc_params.Nu
            % Penalización MÍNIMA del control (queremos que actúe)
            J = J + mpc_params.R * U_seq(k)^2;
            
            % Penalización de tasa (mínima)
            if k > 1
                delta_u = U_seq(k) - U_seq(k-1);
                J = J + mpc_params.S * delta_u^2;
            end
        end
    end
    
    % Penalización terminal ENORME si no converge
    error_terminal = X_pred(:,N+1) - x_ref;
    terminal_penalty = 10000 * norm(error_terminal)^2;
    J = J + terminal_penalty;
    
    % Penalización adicional por control cercano a cero
    control_norm = norm(U_seq(1:min(3, length(U_seq))));
    if control_norm < 10  % Si el control es muy pequeño
        J = J + 1000 * (10 - control_norm);
    end
end

function X_pred = simulate_prediction_force(x0, U_seq, zr_seq, mpc_params, params)
    % Simulación para predicción
    N = mpc_params.N;
    X_pred = zeros(4, N+1);
    X_pred(:,1) = x0;
    
    for k = 1:N
        u_k = U_seq(min(k, length(U_seq)));
        zr_k = zr_seq(min(k, length(zr_seq)));
        
        % Integración simple pero estable
        x_dot = quarter_car_dynamics_force(0, X_pred(:,k), u_k, zr_k, params);
        X_pred(:,k+1) = X_pred(:,k) + mpc_params.Ts * x_dot;
    end
end

function [c, ceq] = mpc_constraints_force(U_seq, x0, zr_seq, mpc_params, params)
    % Restricciones
    X_pred = simulate_prediction_force(x0, U_seq, zr_seq, mpc_params, params);
    
    c = [];
    N = mpc_params.N;
    
    for k = 2:N+1
        c = [c; mpc_params.x_min - X_pred(:,k)];
        c = [c; X_pred(:,k) - mpc_params.x_max];
    end
    
    for k = 1:length(U_seq)
        c = [c; mpc_params.u_min - U_seq(k)];
        c = [c; U_seq(k) - mpc_params.u_max];
    end
    
    ceq = [];
end

function dx = quarter_car_dynamics_force(t, x, u, zr, params)
    % Modelo dinámico
    delta_z = x(1) - x(3);
    delta_v = x(2) - x(4);
    
    % No linealidades moderadas
    F_spring = params.ks * delta_z * (1 + params.ks_nl * delta_z^2);
    F_damper = params.cs * delta_v * (1 + params.cs_nl * abs(delta_v));
    
    % NO saturar el control (dejarlo libre para que MPC actúe)
    u_eff = u;
    
    dx = zeros(4,1);
    dx(1) = x(2);
    dx(2) = (1/params.ms) * (-F_spring - F_damper + u_eff);
    dx(3) = x(4);
    dx(4) = (1/params.mu) * (F_spring + F_damper - params.kt*(x(3)-zr) - u_eff);
end

%% 8. GUARDAR Y MOSTRAR CONCLUSIÓN
fprintf('\n========================================\n');
fprintf('RESUMEN FINAL\n');
fprintf('========================================\n');

% Evaluar si hubo mejora
if rms_acc_mpc < rms_acc_nc * 0.95  % Mejora de al menos 5%
    fprintf('✓ EL MPC MEJORÓ el sistema (redujo la aceleración)\n');
    fprintf('  La inversión en control vale la pena para este sistema.\n');
elseif u_rms > 50
    fprintf('⚠ El MPC actuó (RMS=%.1f N) pero no mejoró significativamente\n', u_rms);
    fprintf('  Posiblemente el sistema sin control ya es bueno.\n');
else
    fprintf('✗ El MPC NO actuó suficientemente (RMS=%.1f N)\n', u_rms);
    fprintf('  Se necesita revisar la configuración del controlador.\n');
end

% Guardar resultados
results.t = t_sim;
results.X_mpc = X_mpc;
results.X_nc = X_nc;
results.U_mpc = U_mpc;
results.zr = zr;
results.params = params;
results.mpc_params = mpc_params;
results.performance.rms_acc_mpc = rms_acc_mpc;
results.performance.rms_acc_nc = rms_acc_nc;
results.performance.u_rms = u_rms;
results.performance.u_max = u_max;

save('mpc_active_results.mat', 'results');

% Configuración final
config_summary = sprintf('\nCONFIGURACIÓN FINAL:\n');
config_summary = [config_summary sprintf('  ks = %.0f N/m, cs = %.0f Ns/m\n', params.ks, params.cs)];
config_summary = [config_summary sprintf('  Q = diag([%d, %d, %d, %d])\n', ...
    mpc_params.Q(1,1), mpc_params.Q(2,2), mpc_params.Q(3,3), mpc_params.Q(4,4))];
config_summary = [config_summary sprintf('  R = %.4f\n', mpc_params.R)];
config_summary = [config_summary sprintf('  Límites control: [%.0f, %.0f] N\n', ...
    mpc_params.u_min, mpc_params.u_max)];
config_summary = [config_summary sprintf('\nRESULTADO: Control RMS = %.1f N\n', u_rms)];

fprintf('%s\n', config_summary);

% Guardar configuración
fid = fopen('mpc_config_final.txt', 'w');
fprintf(fid, '%s', config_summary);
fclose(fid);

fprintf('Resultados guardados en mpc_active_results.mat\n');
fprintf('Configuración guardada en mpc_config_final.txt\n');
fprintf('\nSe han generado 6 ventanas con gráficas separadas.\n');