function ft = TanqueAguaBooleano(h)
%sensor B
if h < 0.05
    B = 0;
else
    B = 1;
end
%sensor M
if h < 0.6
    M = 0;
else
    M = 1;
end
%sensor C
if h < 0.95
    C = 0;
else
    C = 1;
end  
%sensor A
if h < 1
    A = 0;
else
    A = 1;
end
%Y1 = max(min((1-M),B), min((1-A), C));
%Y2 = min((1-C),B);
%Y3 = (1-B);
Y1 = 1-A;
Y2 = max((1-M), min((1-A),C));
Y3 = 1-C;
%Salida
ft = 0.02*Y1 + 0.04*Y2 + 0.06*Y3;