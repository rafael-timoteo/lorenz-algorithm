# poincare_sections.py
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_poincare_points(states, plane_coord_idx, plane_value, crossing_direction='positive'):
    """
    Calcula os pontos de interseção da trajetória com um plano de Poincaré.
    Baseado na abordagem geométrica descrita na Seção 4.2 do documento[cite: 140].

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
    
    # Para o sistema de Lorenz (x,y,z), n_variables = 3
    # plane_coord_idx: 0 para plano x=const, 1 para y=const, 2 para z=const
    # plane_normal_vector é [1,0,0] para x=const, [0,1,0] para y=const, [0,0,1] para z=const
    # plane_reference_point p0 pode ser [plane_value, 0, 0] para x=const, etc.
    # A condição <v_plano|x_i> - <v_plano|p_0> >= 0 [cite: 141] simplifica para:
    # x_i[plane_coord_idx] - plane_value >= 0

    for i in range(len(states) - 1):
        p1 = states[i]
        p2 = states[i+1]

        # Verifica se p1 e p2 estão em lados opostos do plano
        # val1 = p1[plane_coord_idx] - plane_value
        # val2 = p2[plane_coord_idx] - plane_value
        
        # Equivalente à checagem da inequação <v_plano|x_i> - <v_plano|p_0> >=0 [cite: 141]
        # e identificação da mudança de subespaço [cite: 143]
        side1 = (p1[plane_coord_idx] - plane_value) >= 0
        side2 = (p2[plane_coord_idx] - plane_value) >= 0

        crossed = (side1 != side2)

        if crossed:
            # Verifica a direção do cruzamento
            is_correct_direction = False
            if crossing_direction == 'positive' and not side1 and side2: # Cruzou de negativo para positivo
                is_correct_direction = True
            elif crossing_direction == 'negative' and side1 and not side2: # Cruzou de positivo para negativo
                is_correct_direction = True
            # O documento sugere registrar passagens em um sentido específico [cite: 143]
            # (e.g. "saindo do subespaço abaixo do plano para o subespaço acima")

            if is_correct_direction:
                # Interpolação linear para encontrar o ponto de interseção
                # Soluciona (p2[idx] - p1[idx]) * r + p1[idx] = plane_value para r
                # r = (plane_value - p1[idx]) / (p2[idx] - p1[idx])
                # Esta é a parametrização da linha p = (p2-p1)r + p1, e então resolve para r
                # de forma que p[plane_coord_idx] = plane_value, como em eq. 4.7 e 4.8 [cite: 145]

                # Evitar divisão por zero se a trajetória for paralela ao plano nessa coordenada
                if np.abs(p2[plane_coord_idx] - p1[plane_coord_idx]) > 1e-9:
                    r = (plane_value - p1[plane_coord_idx]) / (p2[plane_coord_idx] - p1[plane_coord_idx])
                    if 0 <= r <= 1: # Garante que a interseção está entre p1 e p2
                        intersection_point = p1 + r * (p2 - p1)
                        poincare_points.append(intersection_point)
    
    return np.array(poincare_points)

def plot_and_save_poincare_section(states, plane_coord_idx, plane_value, sigma, rho, beta, output_folder):
    """
    Calcula, plota e salva uma seção de Poincaré.
    """
    poincare_points = calculate_poincare_points(states, plane_coord_idx, plane_value)

    if len(poincare_points) == 0:
        print(f"Nenhum ponto encontrado para a seção de Poincaré (plane_coord_idx={plane_coord_idx}, plane_value={plane_value}).")
        return

    # Determinar quais eixos plotar. Se o plano é x=const, plotamos y vs z.
    # Se o plano é y=const, plotamos x vs z.
    # Se o plano é z=const, plotamos x vs y.
    plot_indices = [i for i in range(3) if i != plane_coord_idx]
    coord_names = ['x', 'y', 'z']
    
    xlabel = f"{coord_names[plot_indices[0]]}"
    ylabel = f"{coord_names[plot_indices[1]]}"
    title = f"Seção de Poincaré ({coord_names[plane_coord_idx]} = {plane_value})"
    subtitle = f"($\\sigma={sigma}, \\rho={rho}, \\beta={beta:.2f}$)"

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(poincare_points[:, plot_indices[0]], poincare_points[:, plot_indices[1]], s=2, c='blue')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{subtitle}")
    ax.grid(True)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = f"poincare_sec_plane_{coord_names[plane_coord_idx]}{plane_value}_rho{rho}.png"
    filepath = os.path.join(output_folder, filename)

    plt.savefig(filepath, dpi=150)
    print(f"Seção de Poincaré salva em: {filepath}")
    # plt.show()
    plt.close(fig)