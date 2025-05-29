# main.py
import numpy as np
import os

# Importar módulos locais
from lorenz_solver import lorenz_system_equations, runge_kutta_4th_order_solver
from attractor_generator import plot_and_save_attractor
from timeseries_generator import plot_and_save_timeseries
from poincare_sections import plot_and_save_poincare_section
from fractal_dimension import estimate_fractal_dimension

def main():
    # --- Parâmetros da Simulação ---
    sigma = 10.0
    beta = 8.0 / 3.0
    
    rhos_to_simulate = [10.0, 15.0, 20.0, 28.0] # Casos de rho a serem simulados

    # Condições iniciais (com e=0)
    x0 = 0.0
    y0 = 1.0 
    z0 = 20.0
    initial_state = np.array([x0, y0, z0])

    # Configurações de tempo
    t_initial = 0.0
    t_final = 100.0 
    dt_step_analysis = 0.01 

    # Pasta de saída para os gráficos
    output_plot_folder = "output_plots"
    poincare_folder = os.path.join(output_plot_folder, "poincare_sections")
    timeseries_folder = os.path.join(output_plot_folder, "timeseries")
    attractor_folder = os.path.join(output_plot_folder, "lorenz_attractor")

    for folder in [output_plot_folder, poincare_folder, timeseries_folder, attractor_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"Pasta '{folder}' criada.")

    # Definições dos planos para as seções de Poincaré
    plane_variables_config = [
        {'idx': 0, 'name': 'x'}, # Planos yz (x = constante)
        {'idx': 1, 'name': 'y'}, # Planos xz (y = constante)
        {'idx': 2, 'name': 'z'}  # Planos xy (z = constante)
    ]
    plane_values_to_test = [-10.0, -5.0, 0.0, 5.0, 10.0]

    for rho_current in rhos_to_simulate:
        print(f"\n--- Processando para rho = {rho_current} ---")

        # --- Executar a Simulação ---
        print(f"Simulando sistema de Lorenz (dt={dt_step_analysis})...")
        time_points, states = runge_kutta_4th_order_solver(
            lorenz_system_equations, t_initial, initial_state, t_final, dt_step_analysis, sigma, rho_current, beta
        )

        # --- Gerar e Salvar Atrator ---
        # Gerar atrator para cada rho para referência
        plot_and_save_attractor(
            time_points, states, sigma, rho_current, beta, attractor_folder, 
            high_res=(rho_current == 28.0) # Alta resolução apenas para o caso caótico principal
        )

        # --- Gerar e Salvar Séries Temporais ---
        plot_and_save_timeseries(time_points, states, sigma, rho_current, beta, timeseries_folder)

        # --- Gerar e Salvar Seções de Poincaré para todas as combinações ---
        print(f"\nGerando seções de Poincaré para rho = {rho_current}:")
        for plane_var_config in plane_variables_config:
            coord_idx = plane_var_config['idx']
            coord_name_str = plane_var_config['name']
            
            for p_value in plane_values_to_test:
                print(f"  Plano: {coord_name_str} = {p_value}")
                
                plot_and_save_poincare_section(
                    states, 
                    coord_idx, 
                    p_value, 
                    sigma, 
                    rho_current,
                    beta, 
                    poincare_folder
                )
        
        # --- Estimar Dimensão Fractal (Placeholder) ---
        if rho_current == 28.0: # Apenas para o caso caótico como exemplo
            print(f"\nTentando estimar a dimensão fractal para rho = {rho_current}...")
            estimate_fractal_dimension(states)

    print("\n--- Processo de simulação e geração de gráficos concluído para todos os valores de rho e planos. ---")

if __name__ == "__main__":
    main()