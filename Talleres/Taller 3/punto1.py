import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are


def sistema_no_lineal(x, t):
    """
    Define el sistema de EDOs no lineal del robot manipulador con u=0.

    Argumentos:
        x: Vector de estado [x1, x2]
        t: Tiempo (variable independiente)

    Retorna:
        dx/dt: Vector de derivadas [x1_dot, x2_dot]
    """
    x1, x2 = x
    u = 0  # Condición sin control

    # Ecuación (1):
    # x1_dot = x2
    # x2_dot = 9.8 * sin(x1) - x2 + u

    x1_dot = x2
    x2_dot = 9.8 * np.sin(x1) - x2 + u

    return [x1_dot, x2_dot]

# --- 1. Comprobar trayectorias de estado del sistema no lineal ---
# Condiciones iniciales: x1(0) = 1, x2(0) = 0
x0 = [1.0, 0.0]

# Vector de tiempo: De 0 a 10 segundos
t = np.linspace(0, 10, 1001)

# Usamos odeint para resolver el sistema de EDOs
solucion = odeint(sistema_no_lineal, x0, t)

# Separar las variables de estado
x1_trayectoria = solucion[:, 0]
x2_trayectoria = solucion[:, 1]

plt.figure(figsize=(10, 6))

# Gráfica de x1(t)
plt.subplot(2, 1, 1)
plt.plot(t, x1_trayectoria, label='$x_1(t)$ (Posición)')
plt.title('Trayectorias de Estado del Sistema No Lineal (u=0)')
plt.ylabel('$x_1(t)$')
plt.grid(True)
plt.legend()

# Gráfica de x2(t)
plt.subplot(2, 1, 2)
plt.plot(t, x2_trayectoria, label='$x_2(t)$ (Velocidad)')
plt.xlabel('Tiempo (s)')
plt.ylabel('$x_2(t)$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()


# --- 2. Diseño del Controlador LQR (Basado en el sistema linealizado) ---

# Matrices del sistema linealizado (Ec. 3)
A = np.array([[0, 1],
              [9.8, -1]])

B = np.array([[0],
              [1]])

C = np.array([[1, 0],
              [0, 1]])

# Matrices de pesos (Ec. 6)
Q = np.array([[1, 0],
              [0, 0]])

R = np.array([[1]])

# Resolver la Ecuación de Riccati Algebraica (ARE)
P = solve_continuous_are(A, B, Q, R)
print(f"Matriz P (Solución de Riccati):\n{P}\n")

# Calcular la ganancia de realimentación K
# K = R^-1 @ B.T @ P
R_inv = np.linalg.inv(R)
K = R_inv @ B.T @ P

print(f"Matriz de Ganancia K:\n{K}\n")
k1, k2 = K[0]

# --- 3. Simulación del Sistema NO LINEAL en Lazo Cerrado ---

def sistema_lazo_cerrado(x, t, K):
    """
    Define el sistema NO LINEAL con la ley de control LQR u = -Kx.

    Argumentos:
        x: Vector de estado [x1, x2]
        t: Tiempo
        K: Matriz de ganancia LQR [k1, k2]

    Retorna:
        dx/dt: Vector de derivadas [x1_dot, x2_dot]
    """
    x1, x2 = x
    k1, k2 = K[0]

    # Ley de control LQR: u = -k1*x1 - k2*x2
    # El LQR está diseñado para estabilizar alrededor del origen (x=0).
    u = -(k1 * x1 + k2 * x2)

    # Dinámica NO LINEAL (Ec. 1):
    x1_dot = x2
    x2_dot = 9.8 * np.sin(x1) - x2 + u

    return [x1_dot, x2_dot]

# --- Parámetros de Simulación ---
# Condiciones iniciales: x1(0) = 1, x2(0) = 0
x0 = [1.0, 0.0]

# Vector de tiempo: 0 a 10 segundos
t = np.linspace(0, 10, 1001)

# --- Simulación ---
solucion_cerrado = odeint(sistema_lazo_cerrado, x0, t, args=(K,))

# Separar las variables de estado
x1_cerrado = solucion_cerrado[:, 0]
x2_cerrado = solucion_cerrado[:, 1]

# Calcular la señal de control u(t)
u_cerrado = -(k1 * x1_cerrado + k2 * x2_cerrado)

# --- Visualización de Resultados ---

plt.figure(figsize=(12, 8))

# Gráfica de x1(t) y x2(t)
plt.subplot(3, 1, 1)
plt.plot(t, x1_cerrado, label='$x_1(t)$ (Posición)')
plt.title('Sistema NO Lineal en Lazo Cerrado con Controlador LQR')
plt.ylabel('$x_1(t)$')
plt.grid(True)
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, x2_cerrado, label='$x_2(t)$ (Velocidad)', color='orange')
plt.ylabel('$x_2(t)$')
plt.grid(True)
plt.legend()

# Gráfica de la Señal de Control u(t)
plt.subplot(3, 1, 3)
plt.plot(t, u_cerrado, label='$u(t)$ (Control LQR)', color='red')
plt.xlabel('Tiempo (s)')
plt.ylabel('$u(t)$')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# --- 4. Definición de Sistemas Linealizados ---

def sistema_lineal_la(x, t):
    """
    Define el sistema linealizado en lazo abierto (u=0).
    dx/dt = A * x
    """
    x1, x2 = x
    # A = [[0, 1], [9.8, -1]]
    x1_dot = x2
    x2_dot = 9.8 * x1 - x2
    return [x1_dot, x2_dot]

def sistema_lineal_lc(x, t, K):
    """
    Define el sistema linealizado en lazo cerrado (u=-Kx).
    dx/dt = (A - B@K) * x
    """
    x1, x2 = x
    k1, k2 = K[0]
    
    # Matriz A_cl = A - B@K
    # A_cl = [[0, 1], [9.8 - k1, -1 - k2]]
    
    x1_dot = x2
    x2_dot = (9.8 - k1) * x1 - (1 + k2) * x2
    return [x1_dot, x2_dot]


# --- Generación y Gráfica de Planos de Fase ---

# Definición del rango del plano de fase
limite = 10
puntos_iniciales_por_eje = 10  # Número de puntos iniciales a muestrear en cada eje
t_simulacion = np.linspace(0, 5, 200) # Tiempo de simulación más corto para las trayectorias de fase

# Crear una malla de puntos iniciales
x1_iniciales = np.linspace(-limite, limite, puntos_iniciales_por_eje)
x2_iniciales = np.linspace(-limite, limite, puntos_iniciales_por_eje)
X1_iniciales, X2_iniciales = np.meshgrid(x1_iniciales, x2_iniciales)

# Aplanar los puntos iniciales para iterar
condiciones_iniciales = np.vstack([X1_iniciales.ravel(), X2_iniciales.ravel()]).T

# Configuración de las 4 subgráficas
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Planos de Fase ($x_1$ vs $x_2$)', fontsize=16)

# Función auxiliar para graficar una trayectoria en un eje dado
def plot_trayectoria(sistema, ax, titulo, args=()):
    for x0_i in condiciones_iniciales:
        # Resolver las EDOs para cada condición inicial
        sol = odeint(sistema, x0_i, t_simulacion, args=args)
        ax.plot(sol[:, 0], sol[:, 1], 'b-', linewidth=0.5, alpha=0.7)
    
    # Dibujar el punto de equilibrio (Origen)
    ax.plot(0, 0, 'ro', label='Punto de Equilibrio (0,0)')
    
    ax.set_title(titulo)
    ax.set_xlabel('$x_1$ (Posición)')
    ax.set_ylabel('$x_2$ (Velocidad)')
    ax.set_xlim(-limite, limite)
    ax.set_ylim(-limite, limite)
    ax.grid(True)
    ax.legend()


# 1. Plano de Fase: Linealizado sin Control (LA)
plot_trayectoria(sistema_lineal_la, axs[0, 0], 'Sistema Linealizado sin Control (Lazo Abierto)')

# 2. Plano de Fase: No Lineal sin Control (LA)
plot_trayectoria(sistema_no_lineal, axs[0, 1], 'Sistema NO Lineal sin Control (Lazo Abierto)')

# 3. Plano de Fase: Linealizado en Lazo Cerrado (LC)
plot_trayectoria(sistema_lineal_lc, axs[1, 0], 'Sistema Linealizado en Lazo Cerrado (LQR)', args=(K,))

# 4. Plano de Fase: No Lineal en Lazo Cerrado (LC)
plot_trayectoria(sistema_lazo_cerrado, axs[1, 1], 'Sistema NO Lineal en Lazo Cerrado (LQR)', args=(K,))


plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
