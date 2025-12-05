function J = circles(theta)
% Función Circles 2D
% theta es un vector de 2 elementos [x, y]

x = theta(1);
y = theta(2);

% Implementación de la función Circles
% f(x,y) = sin(x)^2 + sin(y)^2 + 0.1*(x^2 + y^2)
J = sin(x)^2 + sin(y)^2 + 0.1*(x^2 + y^2);

end