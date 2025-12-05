function J = himmelblaus(theta)
% Función de Himmelblau 2D
% theta es un vector de 2 elementos [x, y]

x = theta(1);
y = theta(2);

% Implementación de la función de Himmelblau
% f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
J = (x^2 + y - 11)^2 + (x + y^2 - 7)^2;

end