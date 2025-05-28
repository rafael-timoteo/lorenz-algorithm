import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Definindo as equações de Lorenz
def lorenz(t, state, sigma, beta, rho):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]

# Parâmetros do sistema
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0

# Condição inicial
state0 = [1.0, 1.0, 1.0]

# Intervalo de tempo para a simulação
t_span = (0, 50)
t_eval = np.linspace(*t_span, 10000)

# Resolvendo o sistema de equações diferenciais
sol = solve_ivp(lorenz, t_span, state0, args=(sigma, beta, rho), t_eval=t_eval)

# Plotando o atrator de Lorenz
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(sol.y[0], sol.y[1], sol.y[2], color='c', linewidth=3)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()