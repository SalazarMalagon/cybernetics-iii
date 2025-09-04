function ft = controlbool(h)
%sensor B
if h < 1
    B=0
else
    B=1
end
%sensor M
if h < 3
    M=0
else
    M=1
end
%sensor A
if h < 5
    A=0
else
    A=1
end
Y1 = max((1-B), min(M, (1-A)));
Y2 = (1-M);
%Salida
ft = 1*Y1 + 2*Y2;