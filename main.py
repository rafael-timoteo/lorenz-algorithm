# main.py
import numpy as np
import matplotlib.pyplot as plt 
import os

# Importar módulos locais
from lorenz_solver import lorenz_system_equations, runge_kutta_4th_order_solver
from attractor_generator import plot_and_save_attractor
from timeseries_generator import plot_and_save_timeseries
from fractal_dimension import estimate_fractal_dimension

# Para Abordagem 1: Seções de Poincaré Individuais
from poincare_sections import plot_and_save_poincare_section 

# Para Abordagem 2: Novas Imagens de Seções de Poincaré (1x3 subplots)
from poincare_sections_grad import (
    get_stability_points,
    plot_poincare_data_on_ax,
    plot_stability_projection_on_ax
)

def main():
    # --- Parâmetros da Simulação ---
    sigma = 10.0
    beta = 8.0 / 3.0
    rhos_to_simulate = [10.0, 15.0, 20.0, 28.0] 

    # Condições iniciais
    x0 = 0.0; y0 = 1.0; z0 = 20.0
    initial_state = np.array([x0, y0, z0])

    # Configurações de tempo
    t_initial = 0.0
    t_final = 150.0 
    dt_step_analysis = 0.01 

    # --- Configuração de Pastas de Saída ---
    output_plot_folder = "output_plots"
    poincare_individual_folder = os.path.join(output_plot_folder, "poincare_sections_individual")
    # Pasta para as novas 5 imagens de Poincaré (1x3)
    poincare_row_images_folder = os.path.join(output_plot_folder, "poincare_sections_row_images") 
    timeseries_folder = os.path.join(output_plot_folder, "timeseries")
    attractor_folder = os.path.join(output_plot_folder, "lorenz_attractor")

    for folder in [output_plot_folder, poincare_individual_folder, poincare_row_images_folder, timeseries_folder, attractor_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Pasta '{folder}' criada.")

    # --- Dicionário para armazenar os estados de todas as simulações ---
    all_sim_states = {}

    print("Iniciando simulações e geração de gráficos individuais...")
    for rho_current in rhos_to_simulate:
        print(f"\n--- Processando para rho = {rho_current} ---")
        time_points, states = runge_kutta_4th_order_solver(
            lorenz_system_equations, t_initial, initial_state, t_final, dt_step_analysis, sigma, rho_current, beta
        )
        all_sim_states[rho_current] = states

        print(f"Gerando atrator para rho = {rho_current}...")
        plot_and_save_attractor(
            time_points, states, sigma, rho_current, beta, attractor_folder
        )

        print(f"Gerando séries temporais para rho = {rho_current}...")
        plot_and_save_timeseries(time_points, states, sigma, rho_current, beta, timeseries_folder)

        # --- ABORDAGEM 1: Gerar e Salvar Seções de Poincaré Individuais ---
        print(f"Gerando seções de Poincaré INDIVIDUAIS para rho = {rho_current}:")
        plane_variables_config_individual = [{'idx': 0, 'name': 'x'}, {'idx': 1, 'name': 'y'}, {'idx': 2, 'name': 'z'}]
        plane_values_to_test_individual = [-10.0, -5.0, 0.0, 5.0, 10.0]
        for plane_var_config in plane_variables_config_individual:
            for p_value in plane_values_to_test_individual:
                print(f"  Plano individual: {plane_var_config['name']} = {p_value}")
                plot_and_save_poincare_section(
                    states, plane_var_config['idx'], p_value, sigma, rho_current, beta, poincare_individual_folder
                )
        
        if rho_current == 28.0: 
            print(f"\nTentando estimar a dimensão fractal para rho = {rho_current}...")
            estimate_fractal_dimension(states)
            
    print("\nSimulações individuais e geração de gráficos associados concluídas.")
    
    # --- ABORDAGEM 2: GERAÇÃO DE 5 IMAGENS DE SEÇÕES DE POINCARÉ (1x3 subplots cada) ---
    print("\nIniciando a geração das 5 imagens de Seções de Poincaré (1x3 subplots)...")

    rho_colors = {10.0: 'green', 15.0: 'red', 20.0: 'blue', 28.0: 'black'}
    rho_markers = {10.0: 'o', 15.0: 'o', 20.0: 'o', 28.0: 'o'}
    stab_rho_colors = {10.0: 'green', 15.0: 'red', 20.0: 'blue'}
    stab_rho_markers = {10.0: '^', 15.0: '^', 20.0: '^'}

    # Valores de corte que definirão cada uma das 5 imagens
    shared_cut_values_for_images = [-10.0, -5.0, 0.0, 5.0, 10.0] 
    coord_names = ['x', 'y', 'z']
    num_cols_per_image = 3

    for img_idx, shared_cut_value in enumerate(shared_cut_values_for_images):
        print(f"  Gerando Imagem {img_idx + 1} com valor de corte = {shared_cut_value}")
        fig, axes = plt.subplots(1, num_cols_per_image, figsize=(15, 5.5), squeeze=False) 
        # squeeze=False para garantir que axes[0,j] funcione
        fig.suptitle(f"Seções de Poincaré para o sistema de Lorenz (Planos em valor = {shared_cut_value})", fontsize=14, y=0.99)

        for j in range(num_cols_per_image): # Colunas dentro da imagem atual (0: yz, 1: xz, 2:xy)
            ax = axes[0, j] # Temos 1 linha, j é a coluna
            
            plane_coord_idx_sectioning = j # 0 para cortes em x, 1 para y, 2 para z
            current_plane_value = shared_cut_value # O valor de corte é o mesmo para os 3 subplots desta imagem
            
            current_plot_axis_indices = []
            ax_title_plot = ""

            if j == 0: # Subplot 1: Plano yz, cortes em x = shared_cut_value
                current_plot_axis_indices = [1, 2] # Plotar y vs z
                ax_title_plot = f"Plano yz, em x = {current_plane_value}"
                ax.set_xlabel("y")
                ax.set_ylabel("z")
            elif j == 1: # Subplot 2: Plano xz, cortes em y = shared_cut_value
                current_plot_axis_indices = [0, 2] # Plotar x vs z
                ax_title_plot = f"Plano xz, em y = {current_plane_value}"
                ax.set_xlabel("x")
                ax.set_ylabel("z")
            else: # j == 2, Subplot 3: Plano xy, cortes em z = shared_cut_value
                current_plot_axis_indices = [0, 1] # Plotar x vs y
                ax_title_plot = f"Plano xy, em z = {current_plane_value}"
                ax.set_xlabel("x")
                ax.set_ylabel("y")
            
            ax.set_title(ax_title_plot, fontsize=10)

            for rho_val in rhos_to_simulate:
                states_for_rho = all_sim_states[rho_val]
                plot_poincare_data_on_ax(ax, states_for_rho, plane_coord_idx_sectioning, current_plane_value,
                                         rho_colors[rho_val], rho_markers[rho_val], 
                                         label_poincare=f'ρ = {int(rho_val)}',
                                         plot_axis_indices=current_plot_axis_indices,
                                         point_size=5) 

            for rho_val_stab in [10.0, 15.0, 20.0]: 
                stab_points_3d_list = get_stability_points(rho_val_stab, beta)
                plot_stability_projection_on_ax(ax, stab_points_3d_list,
                                                stab_rho_colors[rho_val_stab], 
                                                stab_rho_markers[rho_val_stab],
                                                label_stab=f'Estab. ρ = {int(rho_val_stab)}',
                                                plot_axis_indices=current_plot_axis_indices,
                                                point_size=30) 
            
            # Ajustes de limites dos eixos (podem precisar de ajuste fino)
            if j == 0: ax.set_xlim([-18, 18]); ax.set_ylim([0, 50]) # Ajuste para yz (y lim, z lim)
            elif j == 1: ax.set_xlim([-18, 18]); ax.set_ylim([0, 50]) # Ajuste para xz (x lim, z lim)
            else: ax.set_xlim([-20, 20]); ax.set_ylim([-25, 25]) # Ajuste para xy (x lim, y lim)
            ax.tick_params(axis='both', which='major', labelsize=8)

        # Legenda para a figura atual
        legend_handles = []
        # Coleta handles e labels uma vez para a legenda da figura
        # (pegando do último eixo, assumindo que todos os itens foram plotados lá se necessário)
        # Uma forma mais robusta é criar os handles manualmente como antes
        temp_ax_for_legend = axes[0,0] # Usa o primeiro eixo para pegar os labels
        handles, labels = temp_ax_for_legend.get_legend_handles_labels()
        
        # Para evitar duplicatas na legenda da figura, criamos handles únicos
        unique_labels_handles = {}
        for handle, label in zip(handles, labels):
            if label not in unique_labels_handles:
                unique_labels_handles[label] = handle
        
        fig.legend(unique_labels_handles.values(), unique_labels_handles.keys(), 
                   loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02 if num_cols_per_image > 1 else 0.01), fontsize=9)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
        
        image_filename = os.path.join(poincare_row_images_folder, f"poincare_sections_cut_at_{str(shared_cut_value).replace('.', 'p')}.png")
        plt.savefig(image_filename, dpi=200)
        print(f"    Imagem de seções de Poincaré salva em: {image_filename}")
        plt.close(fig) # Fechar a figura atual para liberar memória antes de criar a próxima

    print("\n--- Processo de simulação e geração de gráficos (incluindo 5 imagens de Poincaré) concluído. ---")

if __name__ == "__main__":
    main()