"""
Módulo core - Implementação base do sistema de Lorenz
"""

from .lorenz_solver import lorenz_system_equations, runge_kutta_4th_order_solver

__all__ = ['lorenz_system_equations', 'runge_kutta_4th_order_solver']
