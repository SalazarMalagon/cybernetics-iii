function u = Punto2(in)

r = in(1);
    y = in(2);

    % calcular el error
    e = r - y;

    % lógica booleana
    if e < -2
        u = -1.6;
    elseif e >= -2 && e < 0
        u = -0.8;
    elseif e == 0
        u = 0;
    elseif e > 0 && e < 2
        u = 0.8;
    else % e >= 2
        u = 1.6;
    end
end
