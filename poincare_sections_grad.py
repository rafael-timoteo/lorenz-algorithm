# poincare_sections.py
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_poincare_points(states, plane_coord_idx, plane_value, crossing_direction='positive'):
    """
    Calcula os pontos de interseção da trajetória com um plano de Poincaré.
    Baseado na abordagem geométrica descrita na Seção 4.2 do documento[cite: 485].
    """
    poincare_points = []
    for i in range(len(states) - 1):
        p1 = states[i]
        p2 = states[i+1]
        
        # Verifica se p1 e p2 estão em lados opostos do plano [cite: 486, 487]
        side1 = (p1[plane_coord_idx] - plane_value) >= 0
        side2 = (p2[plane_coord_idx] - plane_value) >= 0
        crossed = (side1 != side2) # Mudança de subespaço [cite: 488]

        if crossed:
            is_correct_direction = False
            # Registra passagens em um sentido específico [cite: 488]
            if crossing_direction == 'positive' and not side1 and side2: # Cruzou de negativo para positivo
                is_correct_direction = True
            # Adicionar a direção oposta se necessário, ou manter unilateral como no doc.
            # elif crossing_direction == 'negative' and side1 and not side2: # Cruzou de positivo para negativo
            #     is_correct_direction = True

            if is_correct_direction:
                # Interpolação linear para encontrar o ponto de interseção [cite: 489]
                # Soluciona (p2[idx] - p1[idx]) * r + p1[idx] = plane_value para r
                # r = (plane_value - p1[idx]) / (p2[idx] - p1[idx])
                # Esta é a parametrização da linha p = (p2-p1)r + p1 (Eq. 4.6)[cite: 490], 
                # e então resolve para r de forma que p[plane_coord_idx] = plane_value (Eq. 4.7)[cite: 490].
                # O ponto de interseção é p = (xi - xi-1)r + xi-1 (Eq. 4.8)[cite: 490].

                # Evitar divisão por zero se a trajetória for paralela ao plano nessa coordenada
                if np.abs(p2[plane_coord_idx] - p1[plane_coord_idx]) > 1e-9:
                    r = (plane_value - p1[plane_coord_idx]) / (p2[plane_coord_idx] - p1[plane_coord_idx])
                    if 0 <= r <= 1: # Garante que a interseção está entre p1 e p2
                        intersection_point = p1 + r * (p2 - p1)
                        poincare_points.append(intersection_point)
    return np.array(poincare_points)

def get_stability_points(rho, beta):
    """
    Calcula os pontos de estabilidade C+/- para um dado rho e beta do sistema de Lorenz.
    Para rho > 1, os pontos são C+/- = ( +/-sqrt(beta*(rho-1)), +/-sqrt(beta*(rho-1)), rho-1 ).
    Estes são estáveis para 1 < rho < rho_H (aprox 24.74 para sigma=10, beta=8/3).
    """
    if rho <= 1: # Origem é estável
        return [np.array([0.0, 0.0, 0.0])]
    
    # Para rho = 10, 15, 20, os pontos C+/- são os atratores estáveis. [cite: 539, 545]
    # O documento menciona que para rho=20 o comportamento é de caos transiente[cite: 183, 539],
    # mas ainda converge para um ponto estável[cite: 546].
    if rho > 1: #Consideramos apenas rho > 1 para C+/-
        val = np.sqrt(beta * (rho - 1))
        return [
            np.array([val, val, rho - 1]),
            np.array([-val, -val, rho - 1])
        ]
    return []


def plot_poincare_data_on_ax(ax, states, plane_coord_idx_sectioning, plane_value,
                             color_poincare, marker_poincare, label_poincare,
                             plot_axis_indices, point_size=10):
    """
    Calcula e plota pontos de Poincaré para um conjunto de estados em um Axes fornecido.
    """
    poincare_points = calculate_poincare_points(states, plane_coord_idx_sectioning, plane_value)
    
    coord_names = ['x', 'y', 'z']
    # As labels dos eixos são definidas uma vez pelo main.
    # xlabel_ax = f"{coord_names[plot_axis_indices[0]]}"
    # ylabel_ax = f"{coord_names[plot_axis_indices[1]]}"
    # ax.set_xlabel(xlabel_ax)
    # ax.set_ylabel(ylabel_ax)

    if len(poincare_points) > 0:
        ax.scatter(poincare_points[:, plot_axis_indices[0]], 
                   poincare_points[:, plot_axis_indices[1]], 
                   s=point_size, 
                   color=color_poincare,
                   marker=marker_poincare,
                   label=label_poincare if not ax.get_legend_handles_labels()[1] else None, # Evitar labels duplicadas para legenda geral
                   alpha=0.9,
                   edgecolors='none' if marker_poincare == 'o' else 'black', 
                   linewidths=0.5 if marker_poincare != 'o' else 0
                   )
    ax.grid(True)


def plot_stability_projection_on_ax(ax, stab_points_3d,
                                    color_stab, marker_stab, label_stab,
                                    plot_axis_indices, point_size=50):
    """
    Plota a projeção 2D dos pontos de estabilidade 3D em um Axes fornecido.
    """
    if stab_points_3d is not None:
        for pt_3d in stab_points_3d:
            ax.scatter(pt_3d[plot_axis_indices[0]], pt_3d[plot_axis_indices[1]],
                       s=point_size,
                       color=color_stab,
                       marker=marker_stab,
                       label=label_stab if not ax.get_legend_handles_labels()[1] else None, # Evitar labels duplicadas
                       edgecolors='black',
                       linewidths=0.5,
                       alpha=0.9)