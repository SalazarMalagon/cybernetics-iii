
clear; clc; close all;

%% Datos del problema
Q = [4 1 0;
     1 2 0;
     0 0 2];

c = [-8; -3; -3];

a = [1; 1; 1];
b = 1;


%  SOLUCIÓN ANALÍTICA USANDO LAS ECUACIONES KKT

disp("=== SOLUCIÓN POR KKT ===");

KKT = [Q  a;
       a' 0];

rhs = [-c; b];

sol = KKT \ rhs;

x_kkt = sol(1:3);
lambda_kkt = sol(4);

disp("x* (KKT) = ");
disp(x_kkt);
disp("λ* (KKT) = ");
disp(lambda_kkt);

f_kkt = 0.5 * x_kkt' * Q * x_kkt + c' * x_kkt;
disp("Valor óptimo f(x*) = ");
disp(f_kkt);



%  ALGORITMO PRIMAL–DUAL (DINÁMICA CONTINUA)

disp("=== INTEGRACIÓN PRIMAL-DUAL ===");

dt = 0.01;          % paso pequeño
T  = 20;            % tiempo total
N  = T/dt;

x = zeros(3, N);
lambda = zeros(1, N);

% condiciones iniciales
x(:,1) = [0; 0; 0];
lambda(1) = 0;

for k = 1:N-1
    dx = -(Q*x(:,k) + c + a*lambda(k));
    dl = (a' * x(:,k) - b);

    % Euler hacia adelante
    x(:,k+1) = x(:,k) + dt*dx;
    lambda(k+1) = lambda(k) + dt*dl;
end

x_pd = x(:,end);
lambda_pd = lambda(end);

disp("x* (primal-dual) = ");
disp(x_pd);
disp("λ* (primal-dual) = ");
disp(lambda_pd);

%% Gráficas
figure;
plot(0:dt:T-dt, x);
title("Estados x(t) - Método primal–dual");
legend("x_1","x_2","x_3");

figure;
plot(0:dt:T-dt, lambda);
title("Multiplicador λ(t)");



% VALIDACIÓN CON fmincon

disp("=== VALIDACIÓN CON fmincon ===");

fun = @(x) 0.5*x'*Q*x + c'*x;
Aeq = a'; beq = b;

x0 = [0;0;0];

options = optimoptions('fmincon','Display','iter','Algorithm','interior-point');

x_fmincon = fmincon(fun, x0, [],[], Aeq, beq, [], [], [], options);

disp("x* (fmincon) = ");
disp(x_fmincon);

f_fmincon = fun(x_fmincon);
disp("Valor óptimo f(x*) con fmincon = ");
disp(f_fmincon);


