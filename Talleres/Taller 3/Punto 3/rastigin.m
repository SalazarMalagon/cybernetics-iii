function J = rastigin(theta)
% Función de Rastrigin 2D
% theta es un vector de 2 elementos [x, y]

x = theta(1);
y = theta(2);

% Implementación de la función de Rastrigin
% f(x,y) = A*n + sum(x_i^2 - A*cos(2*pi*x_i))
% donde A = 10 y n = 2 (dimensiones)
A = 10;
n = 2;

J = A*n + (x^2 - A*cos(2*pi*x)) + (y^2 - A*cos(2*pi*y));

end