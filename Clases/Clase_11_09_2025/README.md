# Modelo Mamdani - Fuzzy Logic Designer

Este repositorio contiene instrucciones para trabajar con el modelo Mamdani utilizando MATLAB.

## Cómo abrir el modelo

Ejecuta el siguiente comando en MATLAB para abrir el diseñador de lógica difusa con el archivo `tipper.fis`:

```matlab
fuzzyLogicDesigner("tipper.fis")
```

## Contenido

- tirangularA2seno.fis
  - Tipo: Mamdani
  - Entradas/Salidas: 1 entrada (X), 1 salida (Y)
  - Rango: [-1.5, 1.5] para X e Y
  - MFs: 3 gaussianas (negativo, cero, positivo) en entrada y salida
  - Reglas: mapeo directo X→Y por categorías (negativo→negativo, cero→cero, positivo→positivo)

- tipper_1.fis
  - Tipo: Sugeno
  - Entradas: service (3 gauss), food (2 trapmf)
  - Salida: tip (3 constantes: ~5, 15, ~25)
  - Reglas: 3 reglas clásicas de propina

- mamdanitype1.fis
  - Tipo: Mamdani
  - Entradas: servicio (zmf/gauss/smf), comida (linzmf/linsmf)
  - Salida: output1 (3 trimf: mezquino, promedio, generoso)
  - Nota: si MATLAB no abre este FIS, ve “Solución de problemas” abajo.

## Requisitos

- MATLAB con Fuzzy Logic Toolbox
- Ubicarte en la carpeta de los .fis antes de abrirlos

## Abrir los modelos en el diseñador (GUI)

En MATLAB:

```matlab
% Ir a la carpeta del proyecto

% Abrir cada FIS en Fuzzy Logic Designer
fuzzyLogicDesigner("tirangularA2seno.fis")
fuzzyLogicDesigner("tipper_1.fis")
fuzzyLogicDesigner("mamdanitype1.fis")
```

Desde la GUI puedes:
- Editar funciones de membresía (Membership Function Editor)
- Ver y ajustar reglas (Rule Editor / Rule Viewer)
- Ver superficies de salida (Surface Viewer)
- Guardar cambios nuevamente al .fis

## Evaluación por código (programática)

### Utilidades comunes

```matlab
% Cargar un FIS
fis = readfis("tipper_1.fis");   % o "tirangularA2seno.fis", "mamdanitype1.fis"

% Ver MFs de una entrada
plotmf(fis, "input", 1); grid on; title("MFs de la entrada 1");

% Generar superficie (para 2 entradas)
if numel(fis.Inputs) == 2
    figure; gensurf(fis); title("Superficie de salida");
end
```

### Ejemplo 1: Mamdani 1D (tirangularA2seno.fis)

```matlab
fis = readfis("tirangularA2seno.fis");

% Barrido de la entrada X y evaluación
x = linspace(-1.5, 1.5, 200)';
y = evalfis(fis, x);

figure;
plot(x, y, "LineWidth", 1.5); grid on;
xlabel("X"); ylabel("Y"); title("Respuesta Mamdani 1D: X → Y");

% Ver MFs de entrada y salida
figure; plotmf(fis, "input", 1); grid on; title("MFs Entrada X");
figure; plotmf(fis, "output", 1); grid on; title("MFs Salida Y");
```

### Ejemplo 2: Sugeno 2D (tipper_1.fis)

```matlab
fis = readfis("tipper_1.fis");

% Evaluación puntual
inputSample = [7 8];     % service=7, food=8
tip = evalfis(fis, inputSample);

fprintf("Tip (Sugeno) para service=7, food=8: %.2f\n", tip);

% Superficie de salida
figure; gensurf(fis); title("Tip (Sugeno) - Superficie");
```

### Ejemplo 3: Mamdani 2D (mamdanitype1.fis)

```matlab
fis = readfis("mamdanitype1.fis");

% Evaluación puntual
inputSample = [7 8];     % servicio=7, comida=8
out = evalfis(fis, inputSample);

fprintf("Salida (Mamdani) para servicio=7, comida=8: %.2f\n", out);

% Superficie de salida
figure; gensurf(fis); title("Mamdani - Superficie");
```

## Diferencias clave (Mamdani vs. Sugeno)

- Mamdani:
  - Salidas con MFs (p.ej., trimf/gaussmf), agregación y defuzzificación (centroid).
  - Intuitivo para diseño con etiquetas lingüísticas.
- Sugeno:
  - Salidas como funciones (o constantes) y defuzzificación por promedio ponderado.
  - Conveniente para optimización e integración con control/ANFIS.

## Solución de problemas

- mamdanitype1.fis no abre:
  - Asegura que la primera línea del archivo comience exactamente con [System].
  - Si ves algo como fu[System], edítalo y deja solo [System], guarda y vuelve a abrir.

- Ruta incorrecta:
  - Usa cd a la carpeta exacta antes de llamar fuzzyLogicDesigner o readfis.

- Fuzzy Logic Toolbox:
  - Verifica que esté instalado: ver Add-Ons > Manage Add-Ons.


## Recursos 

- [Documentación oficial de Fuzzy Logic Designer](https://www.mathworks.com/help/fuzzy/fuzzy-logic-designer.html)
- [Introducción a sistemas difusos en MATLAB](https://www.mathworks.com/help/fuzzy/overview-of-fuzzy-logic.html)

---
