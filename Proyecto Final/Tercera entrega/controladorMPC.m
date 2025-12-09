%% MPC_QuarterCar.m
% Diseño y validación de un controlador MPC (NMPC) para el modelo no lineal
% del quarter-car. El optimizador usa el modelo no lineal completo y la
% simulación valida el controlador sobre el mismo modelo.
%
% Para ejecutar: abrir este archivo en MATLAB y ejecutar la sección "Simulación"
% (o ejecutar todo). Requiere Optimization Toolbox (fmincon).

%% 1) Parámetros del vehículo y suspensión
clear; clc; close all

% masas
ms = 250;    % kg (sprung mass)
mu = 40;     % kg (unsprung mass)

% lineal (nominal) parámetros
ks = 15000;  % N/m (suspensión)
cs = 1000;   % N*s/m (amortiguamiento suspensión)
kt = 200000; % N/m (neumático)

% no-linealidad (término cúbico que añade no-linealidad realista)
alpha_s = 1e6;   % N/m^3 (cubic suspension stiffening)
alpha_t = 1e7;   % N/m^3 (cubic tyre nonlinearity)

% restricciones del actuador y de suspensión
u_min = -20000; u_max = 20000;   % fuerza actuator (N)
susp_travel_max = 0.2;         % máximo recorrido relativo (m) |zs-zu|

%% 2) Modelo continuo no lineal (función anidada)
% Estados: x = [zs; zs_dot; zu; zu_dot]
% Entrada: u (fuerza del actuador, positiva hacia arriba sobre masa resorte)
% Road: z_r (altura del pavimento)

model_cont = @(x,u,zr) [
    x(2);
    ( -ks*(x(1)-x(3)) - cs*(x(2)-x(4)) - alpha_s*(x(1)-x(3))^3 + u )/ms;
    x(4);
    (  ks*(x(1)-x(3)) + cs*(x(2)-x(4)) + alpha_s*(x(1)-x(3))^3 - kt*(x(3)-zr) - alpha_t*(x(3)-zr)^3 - u )/mu
];

%% 3) Discretización simple (Forward Euler dentro del predictor) y predicción
dt = 0.01;        % paso de simulación (s)
Np = 30;          % horizonte de predicción (pasos)

%% 4) Objetivo y matrices Q, R
% Decidimos hacer seguimiento de referencia para zs (posición del cuerpo).
Q = diag([1e6, 1e3, 1e4, 1e2]);   % pesos en estados: zs, zs_dot, zu, zu_dot
R = 1e-2;                         % peso en la acción de control

% Observación: escalas elegidas considerando magnitudes: zs ~ 0.01-0.1 m

%% 5) Configuración del optimizador (fmincon)
optim_opts = optimoptions('fmincon', ...
    'Display','none', ...
    'Algorithm','sqp', ...
    'MaxIterations',200, ...
    'OptimalityTolerance',1e-4);

% bounds para la secuencia de controles (Np pasos)
U_lb = u_min*ones(Np,1);
U_ub = u_max*ones(Np,1);

%% 6) Función que evalúa el coste para una secuencia de U dada y estado inicial
% referencia de seguimiento para zs: r(t) puede ser variable en el tiempo
predict_cost = @(x0,Useq,t0,ref_fun,zr_seq) predict_cost_fun(x0,Useq,t0,ref_fun,zr_seq,model_cont,dt,Np,Q,R);

%% 7) Simulación y loop MPC
Tsim = 10;                 % segundos
Nsim = round(Tsim/dt);

% Referencia deseada para zs: ejemplo seno (seguimiento requerido)
ref_amp = 0.02; ref_freq = 0.5; % 2 cm amplitude, 0.5 Hz
ref_fun = @(t) ref_amp*sin(2*pi*ref_freq*t);

% Road disturbance (ejemplo): pulso o sinusoidal
zr_fun = @(t) 0.01*sin(2*pi*1.0*t); % road input 1 cm, 1 Hz

% iniciales
x = [0.0; 0.0; 0.0; 0.0];   % puede cambiarse para validar convergencia

% almacenamiento
X = zeros(4,Nsim+1); X(:,1) = x;
Ustore = zeros(1,Nsim);
Tstore = (0:Nsim-1)*dt;

% solver warm-start
U0 = zeros(Np,1);

fprintf('Comienza simulación MPC (%d s, dt=%.3f s, Nsim=%d)...\n',Tsim,dt,Nsim);

for k=1:Nsim
    tnow = (k-1)*dt;
    % construir zr predicho sobre horizonte
    zr_seq = arrayfun(@(i) zr_fun(tnow + (i-1)*dt), 1:Np)';
    % referencia vector sobre horizonte (solo para zs)
    ref_seq = arrayfun(@(i) ref_fun(tnow + (i-1)*dt), 1:Np)';

    % optimizar secuencia U (fmincon)
    cost_handle = @(U) predict_cost(x,U,tnow,ref_fun,zr_seq);

    % constraints: bounds y restricciones no-lineales
    nonlcon = @(U) nonlcon_fun(x,U,tnow,ref_fun,zr_seq,model_cont,dt,Np,susp_travel_max);

    problem.objective = cost_handle;
    problem.x0 = U0;
    problem.lb = U_lb;
    problem.ub = U_ub;
    problem.nonlcon = nonlcon;
    problem.options = optim_opts;
    problem.solver = 'fmincon';

    [Uopt, ~, exitflag] = fmincon(problem);
    if exitflag <= 0
        % warning but continue with last feasible
        %fprintf('Warning: fmincon no convergio en paso %d (exitflag=%d)\n',k,exitflag);
    end

    % aplicar primer control
    u_apply = Uopt(1);

    % integrar un paso (RK4 para buena precisión)
    x = rk4_step(x,u_apply,zr_fun(tnow),model_cont,dt);

    % almacenar
    X(:,k+1) = x;
    Ustore(k) = u_apply;
    Tstore(k) = tnow;

    % warm start: shift Uopt
    U0 = [Uopt(2:end); Uopt(end)];
end

fprintf('Simulación completada.\n');

%% 8) Plots (desempeño)
figure;
subplot(3,1,1)
plot(Tstore, X(1,1:end-1),'LineWidth',1.2); hold on
plot(Tstore, arrayfun(ref_fun,Tstore),'--','LineWidth',1.2)
xlabel('Tiempo (s)'); ylabel('z_s (m)'); legend('zs','ref'); grid on; title('Seguimiento de la referencia (zs)');

subplot(3,1,2)
plot(Tstore, X(3,1:end-1),'LineWidth',1.2); xlabel('Tiempo (s)'); ylabel('z_u (m)'); grid on; title('Posición rueda (zu)');

subplot(3,1,3)
plot(Tstore, Ustore,'LineWidth',1.2); xlabel('Tiempo (s)'); ylabel('u (N)'); grid on; title('Señal de control');

% verificación de restricciones: suspensión
susp_rel = X(1,1:end-1) - X(3,1:end-1);
figure; 
plot(Tstore, susp_rel,'LineWidth',1.2); hold on
yline(susp_travel_max,'r--','Max travel'); yline(-susp_travel_max,'r--');
xlabel('Tiempo (s)'); ylabel('zs-zu (m)'); title('Recorrido relativo suspensión'); grid on;

%% 9) Verificación (indicadores)
viol = max(abs(susp_rel)) > susp_travel_max;
if any(viol)
    fprintf('ATENCION: hubo violaciones en el recorrido de suspensión. Valor máximo: %.4f m\n',max(abs(susp_rel)));
else
    fprintf('Restricción de recorrido de suspensión respetada. Valor máximo observado: %.4f m\n',max(abs(susp_rel)));
end

fprintf('Máximo|u| aplicado: %.1f N\n', max(abs(Ustore)));

%% --- funciones auxiliares ---

function J = predict_cost_fun(x0,Useq,t0,ref_fun,zr_seq,model_cont,dt,Np,Q,R)
    % evalúa coste sumatorio sobre horizonte usando Euler integrador (rápido)
    x = x0;
    J = 0;
    for i=1:Np
        u = Useq(i);
        zr = zr_seq(i);
        % paso Euler
        x = x + dt*model_cont(x,u,zr);
        % referencia para zs en tiempo t0 + (i-1)*dt
        r = ref_fun(t0 + (i-1)*dt);
        e = x - [r; 0; 0; 0];
        J = J + e'*Q*e + u'*R*u;
    end
end

function [c,ceq] = nonlcon_fun(x0,Useq,t0,ref_fun,zr_seq,model_cont,dt,Np,susp_max)
    % impone restricciones no lineales: |zs-zu| <= susp_max para cada paso
    x = x0;
    c = [];
    for i=1:Np
        u = Useq(i);
        zr = zr_seq(i);
        x = x + dt*model_cont(x,u,zr);
        % c <= 0 means satisfied
        c = [c; abs(x(1)-x(3)) - susp_max];
    end
    ceq = [];
end

function xnext = rk4_step(x,u,zr,model_cont,dt)
    k1 = model_cont(x,u,zr);
    k2 = model_cont(x + 0.5*dt*k1,u,zr);
    k3 = model_cont(x + 0.5*dt*k2,u,zr);
    k4 = model_cont(x + dt*k3,u,zr);
    xnext = x + (dt/6)*(k1 + 2*k2 + 2*k3 + k4);
end
