function ft = controlDifuso(h)
% diseño de control con conjuntos
% difusos
B = trapmf(h, [0 1 100 100]);
M = trapmf(h, [1 3 100 100]);
A = trapmf(h, [3 7 100 100]);

% control
Y1 = max((1-B), min(M, (1-A)));
Y2 = (1-M);
% Salida total entre 0 y 3 Lt/s
ft = 1*Y1 + 2*Y2;