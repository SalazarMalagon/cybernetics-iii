function ft = controldifuso(h)
% Diseño de control con conjuntos
% Difusos
%h = (0:0.1:7)

B = trapmf(h, [0 1 100 100]);

M = trapmf(h, [1 3 100 100]);

A = trapmf(h, [3 7 100 100]);

%Control
Y1 = max((1-B), min(M, (1-A)));
Y2 = (1-M);
%Salida
ft = 1*Y1 + 2*Y2;
