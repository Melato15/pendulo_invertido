import numpy as np
import cv2
from InvertedPendulum import InvertedPendulum
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import control

class MyLinearizedSystem:
    def __init__(self, d1=1.0, d2=0.5):
        g = 8.8
        L = 1.5
        m = 0.5
        M = 1.4
        self.d1 = d1
        self.d2 = d2

        _q = (m + M) * g / (M * L)
        self.A = np.array([[0, 1, 0, 0], 
                           [0, -self.d1, -g * m / M, 0],
                           [0, 0, 0, 1.],
                           [0, self.d1 / L, _q, -self.d2]])

        self.B = np.expand_dims(np.array([0, 1.0 / M, 0., -1 / (M * L)]), 1)

    def compute_K(self, Q, R):
        self.K, _, _ = control.lqr(self.A, self.B, Q, R)

    def get_K(self):
        return self.K

def fitness(params):
    d1, d2, q11, q22, q33, q44, r11 = params
    system = MyLinearizedSystem(d1, d2)
    Q = np.diag([q11, q22, q33, q44])
    R = np.diag([r11])

    try:
        system.compute_K(Q, R)
    except Exception as e:
        return float('inf')

    def u(t, y):
        return -np.matmul(system.get_K(), y - np.array([0, 0, np.pi / 2, 0]))[0]

    def y_dot(t, y):
        g = 9.8
        L = 1.5
        m = 1.0
        M = 5.0

        x_ddot = (u(t, y) - m * L * y[3] ** 2 * np.cos(y[2]) + m * g * np.cos(y[2]) * np.sin(y[2])) / \
                 (M + m - m * np.sin(y[2]) ** 2)

        theta_ddot = -g / L * np.cos(y[2]) - np.sin(y[2]) / L * x_ddot

        damping_x = -d1 * y[1]
        damping_theta = -d2 * y[3]

        return [y[1], x_ddot + damping_x, y[3], theta_ddot + damping_theta]

    sol = solve_ivp(y_dot, [0, 20], [0.0, 0., np.pi / 2 + 0.01, 0.], t_eval=np.linspace(0, 20, 100))
    cost = 0
    for i in range(len(sol.t)):
        theta_error = (sol.y[2, i] - np.pi / 2) ** 2
        cart_position_error = sol.y[0, i] ** 2
        cost += theta_error + cart_position_error

    return cost

# Define os limites para d1, d2 e matrizes Q e R
bounds = [(0.1, 2.0), (0.1, 2.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0), (0.1, 10.0)]
result = differential_evolution(fitness, bounds)

best_params = result.x
print("Melhores parâmetros:", best_params)

# Aplique os parâmetros otimizados
d1, d2, q11, q22, q33, q44, r11 = best_params
optimized_system = MyLinearizedSystem(d1, d2)
Q_opt = np.diag([q11, q22, q33, q44])
R_opt = np.diag([r11])
optimized_system.compute_K(Q_opt, R_opt)

# Controlador usando os parâmetros otimizados
def u(t, y):
    return -np.matmul(optimized_system.get_K(), y - np.array([0, 0, np.pi / 2, 0]))[0]

def y_dot(t, y):
    g = 9.8
    L = 1.5
    m = 1.0
    M = 5.0

    x_ddot = (u(t, y) - m * L * y[3] ** 2 * np.cos(y[2]) + m * g * np.cos(y[2]) * np.sin(y[2])) / \
             (M + m - m * np.sin(y[2]) ** 2)

    theta_ddot = -g / L * np.cos(y[2]) - np.sin(y[2]) / L * x_ddot

    damping_x = -d1 * y[1]
    damping_theta = -d2 * y[3]

    return [y[1], x_ddot + damping_x, y[3], theta_ddot + damping_theta]

# Simulação com interação
if __name__ == "__main__":
    sol = solve_ivp(y_dot, [0, 20], [0.0, 0., np.pi / 2 + 0.01, 0.], t_eval=np.linspace(0, 20, 100))
    syst = InvertedPendulum()

    for i, t in enumerate(sol.t):
        rendered = syst.step([sol.y[0, i], sol.y[1, i], sol.y[2, i], sol.y[3, i]], t)
        cv2.imshow('Imitação do Pêndulo Invertido', rendered)
        cv2.moveWindow('Imitação do Pêndulo Invertido', 100, 100)

        # Espera por uma tecla
        key = cv2.waitKey(100)  # Espera 100ms entre os frames
        if key == ord('q'):
            break
        elif key == ord('p'):  # Pressione 'p' para pausar
            while True:
                key = cv2.waitKey(100)
                if key == ord('r'):  # Pressione 'r' para retomar
                    break

    cv2.destroyAllWindows()
