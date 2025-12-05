function J = shaffer(theta)
% Función de Shaffer 2D
% theta es un vector de 2 elementos [x, y]

x = theta(1);
y = theta(2);

% Implementación de la función de Shaffer N. 2
% f(x,y) = 0.5 + (sin^2(x^2-y^2) - 0.5) / (1 + 0.001*(x^2+y^2))^2
numerador = sin(x^2 - y^2)^2 - 0.5;
denominador = (1 + 0.001*(x^2 + y^2))^2;

J = 0.5 + numerador / denominador;

end