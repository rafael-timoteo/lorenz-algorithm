# chaos_control.py

import numpy as np
from lorenz_solver import lorenz_system_equations

def lorenz_controlled_system_x(t, state, sigma, rho, beta, equilibrium_point, eta, epsilon, t_control):
    """
    Define o sistema de Lorenz com controle aplicado à variável x,
    baseado na lógica do arquivo ControleCaosX.py.
    """
    # Calcula as derivadas do sistema de Lorenz original
    dx_dt, dy_dt, dz_dt = lorenz_system_equations(t, state, sigma, rho, beta)
    
    x, y, z = state
    xe = equilibrium_point[0]
    
    u_control = 0
    if t >= t_control:
        # Superfície de deslizamento
        s = x - xe
        # Lei de controle suave usando tanh e uma camada limite (epsilon)
        u_control = -eta * np.tanh(s / epsilon)

    # A nova dinâmica para x é a original mais o controle
    dx_dt_controlled = dx_dt + u_control
    
    return np.array([dx_dt_controlled, dy_dt, dz_dt])