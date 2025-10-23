import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy import interpolate

# Limpiar variables y cerrar figuras
plt.close('all')

# Configuración inicial de la grilla
h = 0.1  # Tamaño del paso para la grilla
x_range = np.arange(-6, 6 + h, h)  # Rango en x de -6 a 6 con paso h
y_range = np.arange(-6, 6 + h, h)  # Rango en y de -6 a 6 con paso h
n = len(x_range)  # Número de puntos en cada dimensión

# Crear malla de coordenadas (meshgrid)
X, Y = np.meshgrid(x_range, y_range)

# Definir la función objetivo F (superficie con dos mínimos)
# Primera componente: pico gaussiano invertido centrado en (-3, -3)
F1 = 1.5 - 1.6 * np.exp(-0.05 * (3 * (X + 3)**2 + (Y + 3)**2))

# Segunda componente: pico gaussiano invertido centrado en (3, 3)
F2 = 0.5 - np.exp(-0.1 * (3 * (X - 3)**2 + (Y - 3)**2))

# Función objetivo total (suma de ambas componentes)
F = F1 + F2

# Calcular gradientes usando diferencias finitas
dFx, dFy = np.gradient(F, h, h)  # Gradiente en x y y respectivamente

# Configurar puntos iniciales para 3 experimentos diferentes
x0 = [4, 0, -5]   # Coordenadas x iniciales
y0 = [0, -5, 2]   # Coordenadas y iniciales
colors = ['red', 'blue', 'magenta']  # Colores para visualización
markers = ['o', 's', '^']  # Marcadores diferentes para cada experimento

# Listas para almacenar resultados de cada experimento
all_x_paths = []
all_y_paths = []
all_f_paths = []

# Ejecutar SGD para cada punto inicial
for jj in range(3):
    print(f"Ejecutando SGD desde punto inicial ({x0[jj]}, {y0[jj]})")
    
    # Selección aleatoria de subset de puntos para SGD (simulación de mini-batch)
    # Generar índices aleatorios para simular sampling estocástico
    q1 = np.random.permutation(n)  # Permutación aleatoria de índices
    i1 = np.sort(q1[:10])  # Tomar los primeros 10 índices y ordenarlos
    
    q2 = np.random.permutation(n)  # Segunda permutación para la otra dimensión
    i2 = np.sort(q2[:10])  # Tomar los primeros 10 índices y ordenarlos
    
    # Inicializar trayectoria
    x_path = [x0[jj]]  # Lista para almacenar posiciones x
    y_path = [y0[jj]]  # Lista para almacenar posiciones y
    
    # Evaluar función y gradiente en punto inicial usando interpolación
    # Nota: Se usa subset aleatorio de puntos para simular SGD
    y_sub = y_range[i1]
    x_sub = x_range[i2]
    F_sub = F[np.ix_(i1, i2)]
    dFx_sub = dFx[np.ix_(i1, i2)]
    dFy_sub = dFy[np.ix_(i1, i2)]
    f_interp = interpolate.RegularGridInterpolator((y_sub, x_sub), F_sub, bounds_error=False, fill_value=None)
    dfx_interp = interpolate.RegularGridInterpolator((y_sub, x_sub), dFx_sub, bounds_error=False, fill_value=None)
    dfy_interp = interpolate.RegularGridInterpolator((y_sub, x_sub), dFy_sub, bounds_error=False, fill_value=None)
    
    # Evaluar en punto inicial
    f_path = [float(f_interp((y_path[0], x_path[0])))]
    dfx = float(dfx_interp((y_path[0], x_path[0])))
    dfy = float(dfy_interp((y_path[0], x_path[0])))
    
    # Parámetros del algoritmo SGD
    tau = 2  # Tasa de aprendizaje (learning rate)
    max_iterations = 50  # Número máximo de iteraciones
    tolerance = 1e-6  # Tolerancia para convergencia
    
    # Bucle principal del SGD
    for j in range(max_iterations):
        # Actualizar posición usando gradiente descendente
        x_new = x_path[-1] - tau * dfx  # Paso en dirección opuesta al gradiente x
        y_new = y_path[-1] - tau * dfy  # Paso en dirección opuesta al gradiente y
        
        # Agregar nuevas posiciones a la trayectoria
        x_path.append(x_new)
        y_path.append(y_new)
        
        # Generar nuevo subset aleatorio de puntos (característica del SGD)
        q1 = np.random.permutation(n)
        i1 = np.sort(q1[:10])
        q2 = np.random.permutation(n)
        # Crear nuevos interpoladores con el subset actualizado
        y_sub = y_range[i1]
        x_sub = x_range[i2]
        F_sub = F[np.ix_(i1, i2)]
        dFx_sub = dFx[np.ix_(i1, i2)]
        dFy_sub = dFy[np.ix_(i1, i2)]
        f_interp = interpolate.RegularGridInterpolator((y_sub, x_sub), F_sub, bounds_error=False, fill_value=None)
        dfx_interp = interpolate.RegularGridInterpolator((y_sub, x_sub), dFx_sub, bounds_error=False, fill_value=None)
        dfy_interp = interpolate.RegularGridInterpolator((y_sub, x_sub), dFy_sub, bounds_error=False, fill_value=None)
        
        # Evaluar función y gradiente en la nueva posición
        f_new = float(f_interp((y_new, x_new)))
        f_path.append(f_new)
        dfx = float(dfx_interp((y_new, x_new)))
        dfy = float(dfy_interp((y_new, x_new)))
        dfx = float(dfx_interp(x_new, y_new))
        dfy = float(dfy_interp(x_new, y_new))
        
        # Verificar convergencia (cambio en función objetivo)
        if abs(f_path[-1] - f_path[-2]) < tolerance:
            print(f"  Convergencia alcanzada en iteración {j+1}")
            break
    
    # Almacenar resultados del experimento actual
    all_x_paths.append(x_path)
    all_y_paths.append(y_path)
    all_f_paths.append(f_path)
    
    print(f"  Punto final: ({x_path[-1]:.3f}, {y_path[-1]:.3f})")
    print(f"  Valor función final: {f_path[-1]:.6f}")

# Visualización 1: Contornos con trayectorias SGD
plt.figure(1, figsize=(10, 8))
# Crear contornos de la función (offset de -1 para mejor visualización)
contours = plt.contour(X, Y, F - 1, levels=10, colors='black', linewidths=1)
plt.hold = True  # Mantener gráfico para agregar más elementos

# Plotear trayectorias de cada experimento
for i in range(3):
    # Plotear puntos de la trayectoria
    plt.plot(all_x_paths[i], all_y_paths[i], 'o', color=colors[i], 
             markersize=6, label=f'Experimento {i+1}')
    # Plotear líneas conectando los puntos
    plt.plot(all_x_paths[i], all_y_paths[i], ':', color='black', linewidth=2)

plt.xlabel('X', fontsize=18)
plt.ylabel('Y', fontsize=18)
plt.title('SGD - Vista de Contornos', fontsize=18)
plt.tick_params(labelsize=16)
plt.legend()
plt.grid(True, alpha=0.3)

# Visualización 2: Superficie 3D con trayectorias
fig = plt.figure(2, figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Crear superficie 3D
surf = ax.plot_surface(X, Y, F, cmap='gray', alpha=0.7, 
                      linewidth=0, antialiased=True)

# Plotear trayectorias en 3D
for i in range(3):
    # Elevar ligeramente los puntos para mejor visualización
    z_offset = np.array(all_f_paths[i]) + 0.1
    
    # Plotear puntos de la trayectoria
    ax.plot(all_x_paths[i], all_y_paths[i], z_offset, 'o', 
            color=colors[i], markersize=6, label=f'Experimento {i+1}')
    # Plotear líneas de conexión
    ax.plot(all_x_paths[i], all_y_paths[i], all_f_paths[i], ':', 
            color='black', linewidth=2)

# Configurar vista y límites
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.view_init(elev=60, azim=-25)  # Ángulo de vista similar al código MATLAB

ax.set_xlabel('X', fontsize=18)
ax.set_ylabel('Y', fontsize=18)
ax.set_zlabel('F(X,Y)', fontsize=18)
ax.set_title('SGD - Vista 3D', fontsize=18)
ax.tick_params(labelsize=14)
ax.legend()

# Mostrar gráficos
plt.tight_layout()
plt.show()

print("\nResumen de resultados:")
print("=" * 50)
for i in range(3):
    print(f"Experimento {i+1}:")
    print(f"  Punto inicial: ({x0[i]}, {y0[i]})")
    print(f"  Punto final: ({all_x_paths[i][-1]:.3f}, {all_y_paths[i][-1]:.3f})")
    print(f"  Iteraciones: {len(all_x_paths[i]) - 1}")
    print(f"  Valor función final: {all_f_paths[i][-1]:.6f}")
    print()