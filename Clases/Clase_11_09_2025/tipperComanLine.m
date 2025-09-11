clear; clc
% line comand
% abrir archivo fis
% fuzzyLogicDesigner('tirangularA2seno.fis')
%cargar un fis
%fis = readfis('tirangularA2seno.fis')
%fis.Inputs(1).MembershipFunctions
%plotfis(fis)
%plotmf(fis, 'input', 1)

fis = mamfis(Name='test_tip');
% crear entradas
fis = addInput(fis, [0 10], Name='servicio');
fis = addInput(fis, [0 10], Name='comida');
% funciones de membresia
fis = addMF(fis, 'servicio', 'gaussmf', [1.5 0], Name='pobre');
fis = addMF(fis, 'servicio', 'gaussmf', [1.5 5], Name='bueno');
fis = addMF(fis, 'servicio', 'gaussmf', [1.5 10], Name='excelente');
%

fis = addMF(fis, 'comida', 'trapmf', [-2 0 1 3], Name='maluca');
fis = addMF(fis, 'comida', 'trapmf', [7 9 10 12], Name='deliciosa');
% crear salida
fis = addOutput(fis, [0 30], Name='propina');
fis = addMF(fis, 'propina', 'trimf', [0 5 10], Name='mezquino');
fis = addMF(fis, 'propina', 'trimf', [10 15 20], Name='promedio');
fis = addMF(fis, 'propina', 'trimf', [20 25 30], Name='generoso');
%plotfis(fis)

%primera columna indice de la funcion de membresia de la primera entrada
%segunda columna indice de la funcion de membresia de la segunda entrada
%tercera columna indice de la funcion de membresia de la salida
%cuarta columna son los pesos siempre son 1
%quinta columan son la operacion, "and" o "or"
rulelist = [ 1 1 1 1 2;
             2 0 2 1 1;
             3 2 3 1 2;];
fis = addRule(fis, rulelist);
evalfis(fis,[3 8])
gensurf(fis)




