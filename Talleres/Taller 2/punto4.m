clear; clc; close all;
set(0, 'DefaultLineLineWidth', 2)
set(0, 'defaultfigurecolor', [1 1 1])

% --- Simulación del sistema ---
out = sim('puntosm');  % Asegúrate de tener el modelo puntosm.slx

% Señales
x = out.simout(:,1); % señal triangular (entrada)
y = out.simout(:,2); % señal seno (salida deseada)

figure(1)
plot(x,'r'); hold on;
plot(y,'b');
legend('Triangular','Seno');
title('Señales de entrada y salida deseada');
xlabel('Muestras'); ylabel('Amplitud');

% --- Parámetros de entrenamiento ---
epoch_n = 25;
mf_types = {'gaussmf','trimf'};
numMF_list = [5 7]; % dos configuraciones

% --- Inicialización de tabla de resultados ---
Resultados = [];

% --- Bucle principal ---
for t = 1:length(mf_types)
    for n = 1:length(numMF_list)
        tipo = mf_types{t};
        numMF = numMF_list(n);
        
        fprintf('\n=== Entrenando ANFIS con %d MF tipo %s ===\n', numMF, tipo);
        
        % Generar sistema difuso inicial
        in_fis = genfis1([x y], numMF, tipo, 'linear');
        
        % Entrenamiento del sistema difuso
        out_fis = anfis([x y], in_fis, epoch_n);
        
        % Evaluación del sistema entrenado
        ys = evalfis(x, out_fis);
        
        % Calcular errores
        e = y - ys;
        ECM = mean(e.^2);
        ErrorMax = max(abs(e)) / max(abs(y)) * 100;
        
        % Guardar resultados
        Resultados = [Resultados; {tipo, numMF, ECM, ErrorMax}];
        
        % --- Gráficos ---
        figure
        subplot(2,1,1)
        plot(y,'b'); hold on;
        plot(ys,'r');
        plot(x,'g')
        legend('Seno real', sprintf('ANFIS %s (%d MF)', tipo, numMF),'Triangular');
        title(sprintf('Aproximación con %s (%d MF)', tipo, numMF));
        xlabel('Muestras'); ylabel('Amplitud');
        
        % Gráfica del error
        subplot(2,1,2)
        plot(e,'k');
        title(sprintf('Error entre seno y ANFIS (%s - %d MF)', tipo, numMF));
        xlabel('Muestras'); ylabel('Error');
        grid on;
    end
end

% --- Mostrar tabla de resultados ---
ResultadosTable = cell2table(Resultados, ...
    'VariableNames', {'Tipo_MF','Num_MF','ECM','Error_Max_%'});

disp('=== Resultados globales ===');
disp(ResultadosTable);

% --- Exportar tabla a archivo Excel ---
filename = 'Resultados_ANFIS.xlsx';
writetable(ResultadosTable, filename, 'Sheet', 1, 'WriteMode', 'overwrite');
fprintf('\nTabla de resultados guardada como: %s\n', filename);




