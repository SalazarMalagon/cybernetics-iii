function u = Punto2Difuso(in)

r = in(1);
y = in(2);
e = r - y; % error

% Conjuntos difusos
muNB = trapmf(e, [-2 -2 -1 0]);
muNS = trimf(e, [-2 -2  0]);
muZE = trimf(e, [-1  1  1]); 
muPS = trimf(e, [ 0  1  2]);
muPB = trapmf(e, [ 0  0.0001 2 2]);

% Reglas Sugeno 0: If e is {NB,NS,ZE,PS,PB} entonces u = {-1.6,-0.8,0,0.8,1.6}
w = [muNB, muNS, muZE, muPS, muPB];
z = [-1.6, -0.8,  0.0, 0.8,  1.6];

den = sum(w);
if den > 0
    u = sum(w .* z) / den;
else
    % respaldo
    if e < -2
        u = -1.6;
    elseif e < 0
        u = -0.8;
    elseif e == 0
        u = 0;
    elseif e < 2
        u = 0.8;
    else
        u = 1.6;
    end
end
end