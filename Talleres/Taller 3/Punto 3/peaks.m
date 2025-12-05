function J = peaks(theta)
% Función PEAKS 2D
% theta es un vector de 2 elementos [x, y]

x = theta(1);
y = theta(2);

% Implementación de la función peaks estándar de MATLAB
J = 3*(1-x)^2*exp(-(x^2) - (y+1)^2) ...
    - 10*(x/5 - x^3 - y^5)*exp(-x^2-y^2) ...
    - (1/3)*exp(-(x+1)^2 - y^2);

% Como es un problema de minimización, podemos usar -J si queremos maximizar
% o mantener J si queremos minimizar
end