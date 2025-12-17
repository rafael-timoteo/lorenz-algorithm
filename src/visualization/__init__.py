"""
Módulo visualization - Funções de plotagem e visualização
"""

from .attractor_generator import plot_and_save_attractor
from .timeseries_generator import plot_and_save_timeseries
from .poincare_sections import plot_and_save_poincare_section, calculate_poincare_points
from .poincare_sections_grad import get_stability_points, plot_poincare_data_on_ax, plot_stability_projection_on_ax

__all__ = [
    'plot_and_save_attractor',
    'plot_and_save_timeseries',
    'plot_and_save_poincare_section',
    'calculate_poincare_points',
    'get_stability_points',
    'plot_poincare_data_on_ax',
    'plot_stability_projection_on_ax'
]
