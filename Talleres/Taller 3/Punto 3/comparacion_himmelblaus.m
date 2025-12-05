% Comparación entre GA y PSO para la función Himmelblau
% preambulo
clear;clc; close all;
set(0,'DefaultLineLineWidth',2)
set(0, 'defaultfigurecolor', [1 1 1])

% Óptimo teórico de la función Himmelblau
% Tiene 4 mínimos globales: (3,2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
% Todos con valor f = 0
optimo_teorico_x = 3;
optimo_teorico_y = 2;
optimo_teorico_valor = himmelblaus([optimo_teorico_x; optimo_teorico_y]);

fprintf('=== FUNCIÓN HIMMELBLAU - COMPARACIÓN GA vs PSO ===\n');
fprintf('Óptimo teórico (uno de 4): x = %.4f, y = %.4f, f = %.6f\n\n', ...
    optimo_teorico_x, optimo_teorico_y, optimo_teorico_valor);

% Parámetros comunes
NV = 2;
num_runs = 5; % Número de ejecuciones para promediar resultados

% Límites para PSO
lb = [-5, -5];
ub = [5, 5];

% Configuración GA
optionsga = optimoptions('ga', ...
    'Display', 'iter', ... % Se puede cambiar a 'off' para comparación limpia
    'PopulationSize', 25, ...
    'MaxGenerations', 200);
    %'PlotFcn', {@gaplotbestf, @gaplotbestindiv, @gaplotexpectation, @gaplotstopping}); 
    % Se puede añadir esta linea para ver todas las ejecuciones en porceso

% Configuración PSO
options_pso = optimoptions('particleswarm', ...
    'Display', 'iter', ... % Se puede cambiar a 'off' para comparación limpia
    'SwarmSize', 25, ...
    'MaxIterations', 200);
    %'PlotFcn', {@pswplotbestf}); 
    % Se puede añadir esta linea para ver todas las ejecuciones en porceso

% Variables para almacenar resultados
ga_results = zeros(num_runs, 3); % [x, y, f_val]
pso_results = zeros(num_runs, 3); % [x, y, f_val]
ga_iterations = zeros(num_runs, 1);
pso_iterations = zeros(num_runs, 1);

fprintf('Ejecutando %d ejecuciones de cada algoritmo...\n\n', num_runs);

for i = 1:num_runs
    fprintf('Ejecución %d/%d\n', i, num_runs);
    
    % Ejecutar GA
    tic;
    [x_ga, fval_ga, exitflag_ga, output_ga] = ga(@himmelblaus, NV, optionsga);
    time_ga = toc;
    ga_results(i, :) = [x_ga(1), x_ga(2), fval_ga];
    ga_iterations(i) = output_ga.generations;
    
    % Ejecutar PSO
    tic;
    [x_pso, fval_pso, exitflag_pso, output_pso] = particleswarm(@himmelblaus, NV, lb, ub, options_pso);
    time_pso = toc;
    pso_results(i, :) = [x_pso(1), x_pso(2), fval_pso];
    pso_iterations(i) = output_pso.iterations;
    
    fprintf('  GA:  x = [%.4f, %.4f], f = %.6f, gen = %d, tiempo = %.3fs\n', ...
        x_ga(1), x_ga(2), fval_ga, output_ga.generations, time_ga);
    fprintf('  PSO: x = [%.4f, %.4f], f = %.6f, iter = %d, tiempo = %.3fs\n\n', ...
        x_pso(1), x_pso(2), fval_pso, output_pso.iterations, time_pso);
end

% Calcular estadísticas
ga_mean = mean(ga_results);
ga_std = std(ga_results);
pso_mean = mean(pso_results);
pso_std = std(pso_results);

% Para Himmelblau, calcular distancia al mínimo más cercano de los 4 posibles
optimos = [3, 2; -2.805118, 3.131312; -3.779310, -3.283186; 3.584428, -1.848126];
ga_errors = zeros(num_runs, 1);
pso_errors = zeros(num_runs, 1);

for i = 1:num_runs
    % GA
    dist_ga = sqrt((ga_results(i,1) - optimos(:,1)).^2 + (ga_results(i,2) - optimos(:,2)).^2);
    ga_errors(i) = min(dist_ga);
    
    % PSO
    dist_pso = sqrt((pso_results(i,1) - optimos(:,1)).^2 + (pso_results(i,2) - optimos(:,2)).^2);
    pso_errors(i) = min(dist_pso);
end

fprintf('=== RESULTADOS ESTADÍSTICOS ===\n');
fprintf('ALGORITMOS GENÉTICOS (GA):\n');
fprintf('  Media: x = [%.4f, %.4f], f = %.6f\n', ga_mean(1), ga_mean(2), ga_mean(3));
fprintf('  Desv. Estándar: x = [%.4f, %.4f], f = %.6f\n', ga_std(1), ga_std(2), ga_std(3));
fprintf('  Error promedio desde óptimo más cercano: %.6f\n', mean(ga_errors));
fprintf('  Iteraciones promedio: %.1f\n\n', mean(ga_iterations));

fprintf('PARTICLE SWARM OPTIMIZATION (PSO):\n');
fprintf('  Media: x = [%.4f, %.4f], f = %.6f\n', pso_mean(1), pso_mean(2), pso_mean(3));
fprintf('  Desv. Estándar: x = [%.4f, %.4f], f = %.6f\n', pso_std(1), pso_std(2), pso_std(3));
fprintf('  Error promedio desde óptimo más cercano: %.6f\n', mean(pso_errors));
fprintf('  Iteraciones promedio: %.1f\n\n', mean(pso_iterations));

% Determinar el mejor algoritmo
if mean(ga_results(:,3)) < mean(pso_results(:,3))
    mejor_alg = 'GA';
else
    mejor_alg = 'PSO';
end

fprintf('=== CONCLUSIÓN ===\n');
fprintf('Mejor algoritmo (menor valor de función): %s\n', mejor_alg);
fprintf('Diferencia en valor de función: %.6f\n', abs(mean(ga_results(:,3)) - mean(pso_results(:,3))));

% Crear gráficas de comparación
figure;
subplot(2,2,1);
bar([mean(ga_results(:,3)), mean(pso_results(:,3))]);
set(gca, 'XTickLabel', {'GA', 'PSO'});
ylabel('Valor de función promedio');
title('Comparación de valores de función');
grid on;

subplot(2,2,2);
bar([mean(ga_errors), mean(pso_errors)]);
set(gca, 'XTickLabel', {'GA', 'PSO'});
ylabel('Error promedio desde óptimo');
title('Proximidad al óptimo teórico');
grid on;

subplot(2,2,3);
bar([mean(ga_iterations), mean(pso_iterations)]);
set(gca, 'XTickLabel', {'GA', 'PSO'});
ylabel('Número de iteraciones promedio');
title('Eficiencia computacional');
grid on;

subplot(2,2,4);
boxplot([ga_results(:,3), pso_results(:,3)], 'Labels', {'GA', 'PSO'});
ylabel('Valor de función');
title('Distribución de resultados');
grid on;

sgtitle('Comparación GA vs PSO - Función Himmelblau');

% Subplot 1: ¿Cuál algoritmo encuentra mejor valor de función?
% Subplot 2: ¿Cuál se acerca más al óptimo teórico?
% Subplot 3: ¿Cuál es más eficiente (menos iteraciones)?
% Subplot 4: ¿Cuál es más consistente (menos variabilidad)?