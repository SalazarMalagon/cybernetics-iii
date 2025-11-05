clear,clc, close all
set(0,'DefaultLineLineWidth',2)
set(0, 'defaultfigurecolor', [1 1 1])

% Sistema dinámico 2D (tipo Duffing forzado)
dt=0.01; T=8; t=0:dt:T;
b=0.3; a=0.35; w=1;

% Ecuación diferencial
sistema = @(t,x)([ x(2); ...
                   x(1)-(x(1)^3)-(a*x(2))+(b*cos(w*t))]);              
ode_options = odeset('RelTol',1e-10, 'AbsTol',1e-11);

% Generar datos de entrenamiento
input=[]; output=[];
figure(1)
for j=1:100
    x0 = 2*(rand(2,1)-0.5); % Rango inicial más amplio
    [~,y] = ode45(sistema,t,x0,ode_options);
    input = [input; y(1:end-1,:)];
    output = [output; y(2:end,:)];
    plot(y(:,1),y(:,2),'Color',[0.7 0.7 0.7]), hold on
end
title('Trayectorias de Entrenamiento')
xlabel('x_1'), ylabel('x_2')
grid on

% Diseño de la red neuronal
net = feedforwardnet([15 15 10]); % Más neuronas para mejor aproximación
net.trainFcn = 'trainlm'; % Algoritmo Levenberg-Marquardt
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'purelin';
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0.15;
net.divideParam.testRatio = 0.05;

% Entrenar la red
fprintf('Entrenando red neuronal...\n');
net = train(net,input.',output.');

% Validación con nuevas condiciones iniciales
figure(2)
colors = {'b-', 'r--', 'g-.', 'm:'};
errors = [];

for test_case = 1:4
    x0 = 0.5*(rand(2,1)-0.5);
    [~,y_real] = ode45(sistema,t,x0,ode_options);
    
    % Predicción con red neuronal
    y_pred = zeros(length(t),2);
    y_pred(1,:) = x0';
    
    for jj = 2:length(t)
        y_next = net(y_pred(jj-1,:)');
        y_pred(jj,:) = y_next';
    end
    
    % Calcular error
    error_traj = sqrt(sum((y_real - y_pred).^2, 2));
    max_error = max(error_traj);
    mean_error = mean(error_traj);
    errors = [errors; max_error, mean_error];
    
    % Graficar
    subplot(2,2,test_case)
    plot(y_real(:,1), y_real(:,2), 'b-', 'LineWidth', 2), hold on
    plot(y_pred(:,1), y_pred(:,2), 'r--', 'LineWidth', 2)
    plot(x0(1), x0(2), 'ko', 'MarkerSize', 8, 'MarkerFaceColor', 'k')
    title(sprintf('Test %d - Error máx: %.3f', test_case, max_error))
    xlabel('x_1'), ylabel('x_2')
    legend('Real', 'NN', 'CI', 'Location', 'best')
    grid on
end

% Análisis temporal
figure(3)
x0 = 0.5*(rand(2,1)-0.5);
[~,y_real] = ode45(sistema,t,x0,ode_options);

y_pred = zeros(length(t),2);
y_pred(1,:) = x0';
for jj = 2:length(t)
    y_next = net(y_pred(jj-1,:)');
    y_pred(jj,:) = y_next';
end

subplot(2,1,1)
plot(t, y_real(:,1), 'b-', t, y_pred(:,1), 'r--', 'LineWidth', 2)
xlabel('Tiempo'), ylabel('x_1')
legend('Real', 'NN')
title('Evolución temporal de x_1')
grid on

subplot(2,1,2)
plot(t, y_real(:,2), 'b-', t, y_pred(:,2), 'r--', 'LineWidth', 2)
xlabel('Tiempo'), ylabel('x_2')
legend('Real', 'NN')
title('Evolución temporal de x_2')
grid on

% Reporte de errores
fprintf('\n=== REPORTE DE RENDIMIENTO ===\n');
fprintf('Error máximo promedio: %.4f\n', mean(errors(:,1)));
fprintf('Error cuadrático medio promedio: %.4f\n', mean(errors(:,2)));
% fprintf('¿Error máximo < 5%%? %s\n', if mean(errors(:,1)) < 0.05; 'SÍ'; else 'NO'; end);
% fprintf('¿Error medio < 2%%? %s\n', if mean(errors(:,2)) < 0.02; 'SÍ'; else 'NO'; end);