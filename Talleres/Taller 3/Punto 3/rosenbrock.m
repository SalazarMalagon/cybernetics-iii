function J = rosenbrock(theta)
% Función de Rosenbrock 2D
% theta es un vector de 2 elementos [x, y]

x = theta(1);
y = theta(2);

% Implementación de la función de Rosenbrock
% f(x,y) = (a-x)^2 + b*(y-x^2)^2
% donde a = 1 y b = 100 (valores estándar)
a = 1;
b = 100;

J = (a - x)^2 + b*(y - x^2)^2;

end