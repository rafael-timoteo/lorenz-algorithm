# poincare_sections.py
import numpy as np
import matplotlib.pyplot as plt
import os

# A função calculate_poincare_points permanece a mesma de antes.
def calculate_poincare_points(states, plane_coord_idx, plane_value, crossing_direction='positive'):
    """
    Calcula os pontos de interseção da trajetória com um plano de Poincaré.
    Baseado na abordagem geométrica descrita na Seção 4.2 do documento. [cite: 140]

    Args:
        states (np.array): Array 2D com os estados (n_points, n_variables).
        plane_coord_idx (int): Índice da coordenada que define o plano (0 para x, 1 para y, 2 para z).
        plane_value (float): Valor da coordenada onde o plano está localizado.
        crossing_direction (str): 'positive' para cruzamentos no sentido positivo,
                                  'negative' para o sentido negativo.

    Returns:
        np.array: Array 2D com os pontos de Poincaré (n_poincare_points, n_variables).
    """
    poincare_points = []
    for i in range(len(states) - 1):
        p1 = states[i]
        p2 = states[i+1]
        
        side1 = (p1[plane_coord_idx] - plane_value) >= 0
        side2 = (p2[plane_coord_idx] - plane_value) >= 0
        crossed = (side1 != side2)

        if crossed:
            is_correct_direction = False
            if crossing_direction == 'positive' and not side1 and side2:
                is_correct_direction = True
            elif crossing_direction == 'negative' and side1 and not side2:
                 is_correct_direction = True # Opcional, o documento foca em um sentido

            if is_correct_direction: # O documento menciona registrar passagens em um sentido específico [cite: 143]
                if np.abs(p2[plane_coord_idx] - p1[plane_coord_idx]) > 1e-9:
                    r = (plane_value - p1[plane_coord_idx]) / (p2[plane_coord_idx] - p1[plane_coord_idx])
                    if 0 <= r <= 1: 
                        intersection_point = p1 + r * (p2 - p1) # Equação 4.8 do documento [cite: 145]
                        poincare_points.append(intersection_point)
    return np.array(poincare_points)

def plot_and_save_composite_poincare_section(
    states, 
    plane_coord_idx_sectioning, # Índice da coordenada cujos valores definem os planos de corte (e.g., 0 se x=k1, x=k2...)
    plane_values_list,         # Lista de valores para a coordenada de corte (e.g., [-10, -5, 0, 5, 10])
    sigma, rho, beta, 
    output_folder
):
    """
    Calcula e plota uma seção de Poincaré composta, mostrando pontos de múltiplos planos de corte em um único gráfico.
    """
    all_points_for_plot = {}
    coord_names = ['x', 'y', 'z']
    sectioning_coord_name = coord_names[plane_coord_idx_sectioning]

    for p_value in plane_values_list:
        current_points = calculate_poincare_points(states, plane_coord_idx_sectioning, p_value)
        if len(current_points) > 0:
            all_points_for_plot[p_value] = current_points
    
    if not all_points_for_plot:
        print(f"Nenhum ponto encontrado para a seção de Poincaré composta (cortes em {sectioning_coord_name} = {plane_values_list}) para rho={rho}.")
        return

    # Determinar eixos de plotagem: são as duas coordenadas que NÃO são a plane_coord_idx_sectioning
    plot_axis_indices = [i for i in range(3) if i != plane_coord_idx_sectioning]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Usar um mapa de cores para distinguir os diferentes planos de corte
    colors = plt.cm.viridis(np.linspace(0, 1, len(plane_values_list)))

    for i, p_value in enumerate(plane_values_list):
        if p_value in all_points_for_plot:
            points_to_plot = all_points_for_plot[p_value]
            ax.scatter(points_to_plot[:, plot_axis_indices[0]], 
                       points_to_plot[:, plot_axis_indices[1]], 
                       s=5,  # Tamanho do ponto
                       label=f'{sectioning_coord_name}={p_value}', 
                       color=colors[i],
                       alpha=0.7) # Transparência para melhor visualização de sobreposições

    xlabel = f"{coord_names[plot_axis_indices[0]]}"
    ylabel = f"{coord_names[plot_axis_indices[1]]}"
    
    # Determinar o nome do plano de visualização
    # Ex: se sectioning_coord_name é 'x', o plano de visualização é 'yz'
    view_plane_name = "".join([coord_names[idx] for idx in plot_axis_indices])


    title = f"Seção de Poincaré Composta - Plano de Visualização: {view_plane_name.upper()}"
    subtitle = f"Cortes em {sectioning_coord_name}; $\\sigma={sigma}, \\rho={rho}, \\beta={beta:.2f}$"
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{subtitle}")
    ax.legend(title=f"Valor de {sectioning_coord_name}", loc='best', markerscale=2)
    ax.grid(True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Nome do arquivo reflete o plano de visualização e a variável de corte
    filename = f"poincare_composite_view_{view_plane_name}_cuts_{sectioning_coord_name}_rho{rho}.png"
    filepath = os.path.join(output_folder, filename)

    plt.savefig(filepath, dpi=200) # DPI um pouco maior para clareza
    print(f"Seção de Poincaré composta salva em: {filepath}")
    plt.close(fig)