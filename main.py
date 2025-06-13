# main.py
import numpy as np
import matplotlib.pyplot as plt 
import os

# Importar módulos locais
from lorenz_solver import lorenz_system_equations, runge_kutta_4th_order_solver
from attractor_generator import plot_and_save_attractor
from timeseries_generator import plot_and_save_timeseries
from lyapunov import LyapunovExponents
from fractal_dimension_kaplan_yorke import FractalDimension

# Para Abordagem 1: Seções de Poincaré Individuais
from poincare_sections import plot_and_save_poincare_section 

# Para Abordagem 2: Novas Imagens de Seções de Poincaré (1x3 subplots)
# Renomeei para poincare_sections_overlap conforme sua descrição anterior
from poincare_sections_grad import ( 
    get_stability_points,
    plot_poincare_data_on_ax,
    plot_stability_projection_on_ax
)

# Importar a função de box-counting
from fractal_dimension import box_counting_dimension # Nome da função como implementada antes

def main():
    # --- Parâmetros da Simulação ---
    sigma = 10.0
    beta = 8.0 / 3.0
    rhos_to_simulate = [10.0, 15.0, 20.0, 28.0] 

    # Condições iniciais
    x0 = 0.0; y0 = 1.0; z0 = 20.0
    initial_state = np.array([x0, y0, z0])

    # Configurações de tempo padrão
    t_initial = 0.0
    t_final_default = 150.0 
    dt_step_default = 0.01 

    # Configurações de tempo otimizadas para dimensão fractal (apenas para rho=28)
    t_final_fractal = 150.0  # Pode precisar de mais tempo para melhor convergência do atrator
    dt_step_fractal = 0.005 # dt menor para mais pontos no atrator

    # --- Configuração de Pastas de Saída ---
    output_plot_folder = "output_plots"
    poincare_individual_folder = os.path.join(output_plot_folder, "poincare_sections_individual")
    poincare_row_images_folder = os.path.join(output_plot_folder, "poincare_sections_row_images") 
    timeseries_folder = os.path.join(output_plot_folder, "timeseries")
    attractor_folder = os.path.join(output_plot_folder, "lorenz_attractor")
    # Pasta para o gráfico log-log da dimensão fractal, se gerado
    fractal_plot_output_folder = os.path.join(output_plot_folder, "fractal_dimension_analysis")
    
    # Novas pastas específicas para Lyapunov e Dimensão Fractal de Kaplan-Yorke
    lyapunov_folder = os.path.join(output_plot_folder, "lyapunov_exponents")
    kaplan_yorke_folder = os.path.join(output_plot_folder, "kaplan_yorke_dimension")

    for folder in [output_plot_folder, poincare_individual_folder, poincare_row_images_folder, 
                   timeseries_folder, attractor_folder, fractal_plot_output_folder,
                   lyapunov_folder, kaplan_yorke_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Pasta '{folder}' criada.")

    # --- Dicionário para armazenar os estados de todas as simulações ---
    all_sim_states = {}

    print("Iniciando simulações e geração de gráficos...")
    for rho_current in rhos_to_simulate:
        print(f"\n--- Processando para rho = {rho_current} ---")

        # Selecionar parâmetros de simulação
        current_t_final = t_final_default
        current_dt_step = dt_step_default
        if rho_current == 28.0: # Usar parâmetros otimizados para o caso caótico
            current_t_final = t_final_fractal
            current_dt_step = dt_step_fractal
        
        print(f"Simulando sistema de Lorenz (t_final={current_t_final}, dt={current_dt_step})...")
        time_points, states = runge_kutta_4th_order_solver(
            lorenz_system_equations, t_initial, initial_state, current_t_final, current_dt_step, 
            sigma, rho_current, beta
        )
        all_sim_states[rho_current] = {'states': states, 'dt': current_dt_step} # Armazenar dt também

        print(f"Gerando atrator para rho = {rho_current}...")
        plot_and_save_attractor( # Assumindo que esta função não precisa de high_res explicitamente ou tem default
            time_points, states, sigma, rho_current, beta, attractor_folder
        )

        print(f"Gerando séries temporais para rho = {rho_current}...")
        plot_and_save_timeseries(time_points, states, sigma, rho_current, beta, timeseries_folder)

        # --- ABORDAGEM 1: Gerar e Salvar Seções de Poincaré Individuais ---
        print(f"Gerando seções de Poincaré INDIVIDUAIS para rho = {rho_current}:")
        plane_variables_config_individual = [{'idx': 0, 'name': 'x'}, {'idx': 1, 'name': 'y'}, {'idx': 2, 'name': 'z'}]
        plane_values_to_test_individual = [-10.0, -5.0, 0.0, 5.0, 10.0]
        # Remover transientes para os plots individuais também pode ser uma boa ideia
        transient_time_individual = 30.0 
        transient_steps_individual = int(transient_time_individual / current_dt_step)
        states_for_poincare_individual = states[transient_steps_individual:] if len(states) > transient_steps_individual else states

        if len(states_for_poincare_individual) > 0:
            for plane_var_config in plane_variables_config_individual:
                coord_idx = plane_var_config['idx']
                # coord_name_str = plane_var_config['name'] # Não usado diretamente aqui
                for p_value in plane_values_to_test_individual:
                    # print(f"  Plano individual: {coord_name_str} = {p_value}") # Opcional
                    plot_and_save_poincare_section(
                        states_for_poincare_individual, coord_idx, p_value, sigma, rho_current, beta, poincare_individual_folder
                    )
        else:
            print(f"  Sem pontos suficientes para Poincaré individual para rho={rho_current} após remover transientes.")

        # --- Estimar Dimensão Fractal para o caso caótico ---
        if rho_current == 28.0: 
            print(f"\nEstimando a dimensão fractal (box-counting) para rho = {rho_current}...")
        
        # Obter os estados e o dt para rho=28
        sim_data_rho28 = all_sim_states[rho_current] # Assumindo que all_sim_states armazena {'states': ..., 'dt': ...}
        states_rho28_full = sim_data_rho28['states']
        dt_rho28 = sim_data_rho28['dt']

        transient_time_remove_fractal = 30.0 
        transient_steps_fractal = int(transient_time_remove_fractal / dt_rho28)
        
        if len(states_rho28_full) > transient_steps_fractal:
            states_for_fractal_calc = states_rho28_full[transient_steps_fractal:]
            
            dimension = box_counting_dimension(
                states_for_fractal_calc, 
                num_epsilons=15,  # Número de tamanhos de caixa a testar
                plot_loglog=True,
                output_folder_base=output_plot_folder # Passa a pasta base
            ) 
            if dimension is not None:
                print(f"  Dimensão Fractal Estimada (Box-Counting) para rho={rho_current}: {dimension:.3f}")
            else:
                print(f"  Não foi possível estimar a dimensão fractal para rho={rho_current}.")
        else:
            print(f"  Simulação para rho={rho_current} muito curta para remover transientes para cálculo da dimensão fractal.")
            
    print("\nSimulações individuais e geração de gráficos associados concluídas.")
    
    # --- ABORDAGEM 2: GERAÇÃO DE 5 IMAGENS DE SEÇÕES DE POINCARÉ (1x3 subplots cada) ---
    # Esta parte é executada APÓS todas as simulações terem sido coletadas em all_sim_states
    print("\nIniciando a geração das 5 imagens de Seções de Poincaré (1x3 subplots)...")

    rho_colors = {10.0: 'green', 15.0: 'red', 20.0: 'blue', 28.0: 'black'}
    rho_markers = {10.0: 'o', 15.0: 'o', 20.0: 'o', 28.0: 'o'}
    stab_rho_colors = {10.0: 'green', 15.0: 'red', 20.0: 'blue'}
    stab_rho_markers = {10.0: '^', 15.0: '^', 20.0: '^'}

    shared_cut_values_for_images = [-10.0, -5.0, 0.0, 5.0, 10.0] 
    coord_names = ['x', 'y', 'z']
    num_cols_per_image = 3
    
    transient_time_poincare_grid = 30.0 # Tempo de transiente para os plots da grade

    for img_idx, shared_cut_value in enumerate(shared_cut_values_for_images):
        # print(f"  Gerando Imagem {img_idx + 1} com valor de corte = {shared_cut_value}") # Opcional
        fig, axes = plt.subplots(1, num_cols_per_image, figsize=(15, 5.5), squeeze=False) 
        fig.suptitle(f"Seções de Poincaré para o sistema de Lorenz (Planos em valor = {shared_cut_value})", fontsize=14, y=0.99)

        for j in range(num_cols_per_image): 
            ax = axes[0, j]
            plane_coord_idx_sectioning = j 
            current_plane_value = shared_cut_value 
            current_plot_axis_indices = []; ax_title_plot = ""

            if j == 0: 
                current_plot_axis_indices = [1, 2]; ax_title_plot = f"Plano yz, em x = {current_plane_value}"; ax.set_xlabel("y"); ax.set_ylabel("z")
            elif j == 1: 
                current_plot_axis_indices = [0, 2]; ax_title_plot = f"Plano xz, em y = {current_plane_value}"; ax.set_xlabel("x"); ax.set_ylabel("z")
            else: 
                current_plot_axis_indices = [0, 1]; ax_title_plot = f"Plano xy, em z = {current_plane_value}"; ax.set_xlabel("x"); ax.set_ylabel("y")
            ax.set_title(ax_title_plot, fontsize=10)

            for rho_val in rhos_to_simulate:
                sim_data = all_sim_states[rho_val]
                states_for_rho_full = sim_data['states']
                dt_this_rho = sim_data['dt']
                
                transient_steps_grid = int(transient_time_poincare_grid / dt_this_rho)
                states_on_attractor = states_for_rho_full[transient_steps_grid:] if len(states_for_rho_full) > transient_steps_grid else states_for_rho_full

                if len(states_on_attractor) > 0:
                    plot_poincare_data_on_ax(ax, states_on_attractor, plane_coord_idx_sectioning, current_plane_value,
                                             rho_colors[rho_val], rho_markers[rho_val], 
                                             label_poincare=f'ρ = {int(rho_val)}',
                                             plot_axis_indices=current_plot_axis_indices,
                                             point_size=5) 

            for rho_val_stab in [10.0, 15.0, 20.0]: 
                stab_points_3d_list = get_stability_points(rho_val_stab, beta)
                plot_stability_projection_on_ax(ax, stab_points_3d_list,
                                                stab_rho_colors[rho_val_stab], stab_rho_markers[rho_val_stab],
                                                label_stab=f'Estab. ρ = {int(rho_val_stab)}',
                                                plot_axis_indices=current_plot_axis_indices,
                                                point_size=30) 
            
            if j == 0: ax.set_xlim([-18, 18]); ax.set_ylim([0, 50]) 
            elif j == 1: ax.set_xlim([-18, 18]); ax.set_ylim([0, 50]) 
            else: ax.set_xlim([-20, 20]); ax.set_ylim([-25, 25]) 
            ax.tick_params(axis='both', which='major', labelsize=8)

        legend_handles = []; unique_labels_handles = {}
        # Construir legenda para a figura atual
        for rho_val in rhos_to_simulate:
            label = f'ρ = {int(rho_val)}'
            if label not in unique_labels_handles:
                unique_labels_handles[label] = plt.Line2D([0], [0], linestyle='None', marker=rho_markers[rho_val], color='w', markerfacecolor=rho_colors[rho_val], markersize=6)
        for rho_val_stab in [10.0, 15.0, 20.0]:
            label = f'Estab. ρ = {int(rho_val_stab)}'
            if label not in unique_labels_handles:
                unique_labels_handles[label] = plt.Line2D([0], [0], linestyle='None', marker=stab_rho_markers[rho_val_stab], color='w', markerfacecolor=stab_rho_colors[rho_val_stab], markeredgecolor='black', markeredgewidth=0.5, markersize=7)
        
        fig.legend(unique_labels_handles.values(), unique_labels_handles.keys(), 
                   loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02 if num_cols_per_image > 1 else 0.01), fontsize=9)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
        
        image_filename = os.path.join(poincare_row_images_folder, f"poincare_sections_cut_at_{str(shared_cut_value).replace('.', 'p')}.png")
        plt.savefig(image_filename, dpi=200)
        # print(f"    Imagem de seções de Poincaré salva em: {image_filename}") # Opcional
        plt.close(fig) 
    print("Geração das 5 imagens de Poincaré (1x3) concluída.")
    print("\n--- Processo de simulação e geração de todos os gráficos concluído. ---")

    # Criar instância da classe
    lyap = LyapunovExponents()
        
    # Calcular o espectro de Lyapunov para cada valor de rho
    print("Iniciando cálculos para o espectro de Lyapunov...")
    results = lyap.compute_lorenz_spectrum(rhos_to_simulate)
    
    # Criar tabela com os resultados
    df_table = lyap.create_spectrum_table(results, rhos_to_simulate)
    print("\n Espectro de Lyapunov")
    print(df_table.to_string(index=False, float_format="%.2f"))
    
    # Gerar gráficos na pasta específica de Lyapunov
    lyap.plot_spectrum(results, rhos_to_simulate, 
                      filename=os.path.join(lyapunov_folder, 'Espectro_Lyapunov.png'),
                      csv_filename=os.path.join(lyapunov_folder, 'espectro_lyapunov_data.csv'))
    print(f"\nGráfico '{os.path.join(lyapunov_folder, 'Espectro_Lyapunov.png')}' foi salvo.")
    
    # Calcular o maior expoente de Lyapunov com um tempo de simulação maior
    print("\nIniciando cálculos para o maior expoente de Lyapunov")
    results_long = lyap.compute_lorenz_spectrum(rhos_to_simulate, t_span=(0, 100))
    
    # Exportar a tabela para CSV na pasta específica
    lyap.export_spectrum_table_to_csv(results, rhos_to_simulate, 
                                     filename=os.path.join(lyapunov_folder, 'espectro_lyapunov_tabela.csv'))
    print(f"Tabela de expoentes de Lyapunov exportada para '{os.path.join(lyapunov_folder, 'espectro_lyapunov_tabela.csv')}'")

    # Gerar gráfico do maior expoente na pasta específica
    lyap.plot_largest_exponent(results_long, rhos_to_simulate,
                              filename=os.path.join(lyapunov_folder, 'Maior_Expoente_Lyapunov.png'),
                              csv_filename=os.path.join(lyapunov_folder, 'maior_expoente_lyapunov_data.csv'))
    print(f"Gráfico '{os.path.join(lyapunov_folder, 'Maior_Expoente_Lyapunov.png')}' foi salvo.")

    print("--- Calculando a Dimensão Fractal (Kaplan-Yorke) para o Sistema de Lorenz ---\n")
    
    # Criar instância da classe
    fractal_calculator = FractalDimension()
    
    # Definir valores de rho para análise
    rho_values = [10.0, 15.0, 20.0, 28.0, 35.0, 40.0]
    
    print(f"Calculando dimensões fractais para valores de ρ: {rho_values}")
    
    # Calcular as dimensões fractais
    dimensions, results = fractal_calculator.compute_dimensions_for_lorenz(
        rho_values, 
        t_span=(0, 100),  # Tempo mais longo para melhor convergência
        dt=0.1,
        transient_time=20  # Descartar o transiente
    )
    
    # Exportar para CSV na pasta específica
    df = fractal_calculator.export_dimensions_to_csv(dimensions, rho_values,
                                                   filename=os.path.join(kaplan_yorke_folder, 'dimensoes_fractais.csv'))
    print(f"\nDados exportados para '{os.path.join(kaplan_yorke_folder, 'dimensoes_fractais.csv')}'")
    
    # Mostrar tabela de resultados
    print("\nTabela de Dimensões Fractais:")
    print(df.to_string(index=False, float_format="%.4f"))
    
    # Exibir detalhes para rho = 28.0 (regime caótico)
    fractal_calculator.print_kaplan_yorke_details(dimensions, 28)
    
    # Exibir detalhes no formato desejado para rho = 28.0
    fractal_calculator.print_lyapunov_dimension_details(dimensions, 28.0)

    # Exportar detalhes para arquivo texto no formato desejado
    fractal_calculator.export_lyapunov_dimension_to_txt(
        dimensions, 28.0, 
        os.path.join(kaplan_yorke_folder, 'dimensao_lyapunov_rho28.txt')
    )

    # Gerar gráficos na pasta específica
    fractal_calculator.plot_dimensions(dimensions, rho_values,
                                     filename=os.path.join(kaplan_yorke_folder, 'Dimensao_Fractal.png'))
    print(f"\nGráfico '{os.path.join(kaplan_yorke_folder, 'Dimensao_Fractal.png')}' foi salvo.")
    
    # Exportar resumo da análise para CSV na pasta específica
    fractal_calculator.export_analysis_summary(dimensions, rho_values, 
                                             filename=os.path.join(kaplan_yorke_folder, 'resumo_analise.csv'))
    print(f"Resumo da análise exportado para '{os.path.join(kaplan_yorke_folder, 'resumo_analise.csv')}'")

    # Exportar resumo da análise para TXT na pasta específica
    fractal_calculator.export_analysis_summary(dimensions, rho_values, 
                                             filename=os.path.join(kaplan_yorke_folder, 'resumo_analise.txt'), 
                                             format='txt')
    print(f"Resumo da análise exportado para '{os.path.join(kaplan_yorke_folder, 'resumo_analise.txt')}'")

    # Exportar análise detalhada para TXT na pasta específica
    fractal_calculator.export_analysis_summary_with_details(dimensions, rho_values, 
                                                          filename=os.path.join(kaplan_yorke_folder, 'analise_detalhada.txt'))
    print(f"Análise detalhada exportada para '{os.path.join(kaplan_yorke_folder, 'analise_detalhada.txt')}'")

if __name__ == "__main__":
    main()