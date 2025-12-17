# lorenz_solver.py
import numpy as np

def lorenz_system_equations(t, state, sigma, rho, beta):
    """
    Define as equações do sistema de Lorenz.
    Baseado na equação 5.1 do documento[cite: 171].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def runge_kutta_4th_order_solver(func, t0, y0, tf, dt, *args):
    """
    Implementa o método de Runge-Kutta de quarta ordem.
    Baseado nas equações 4.3 e 4.4 do documento[cite: 133, 136].
    """
    t_values = np.arange(t0, tf + dt, dt)
    n_steps = len(t_values)
    
    if y0 is None or not isinstance(y0, np.ndarray) or y0.ndim == 0:
        raise ValueError("y0 deve ser um array NumPy não vazio.")
        
    y_values = np.zeros((n_steps, len(y0)))
    y_values[0] = y0
    y = y0.copy() # Use .copy() para evitar modificar o y0 original se for passado como referência de outro lugar

    for i in range(n_steps - 1):
        t = t_values[i]
        k1 = dt * func(t, y, *args)
        k2 = dt * func(t + 0.5 * dt, y + 0.5 * k1, *args)
        k3 = dt * func(t + 0.5 * dt, y + 0.5 * k2, *args)
        k4 = dt * func(t + dt, y + k3, *args)
        y += (k1 + 2*k2 + 2*k3 + k4) / 6.0
        y_values[i+1] = y
        
    return t_values, y_values