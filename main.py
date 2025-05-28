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
    rho = 28.0  # Valor clássico para caos

    # Condições iniciais [cite: 177] (com e=0)
    x0 = 0.0
    y0 = 1.0 
    z0 = 20.0
    initial_state = np.array([x0, y0, z0])

    # Configurações de tempo
    t_initial = 0.0
    t_final = 60.0  # Aumentar um pouco para seções de Poincaré mais povoadas
    dt_step_attractor = 0.005 # Para atrator de alta resolução
    dt_step_analysis = 0.01  # Para outras análises, pode ser um pouco maior

    # Pasta de saída para os gráficos
    output_plot_folder = "output_plots"
    if not os.path.exists(output_plot_folder):
        os.makedirs(output_plot_folder)

    # --- Executar a Simulação para o Atrator de Alta Resolução ---
    print(f"Simulando para atrator de alta resolução (dt={dt_step_attractor})...")
    time_points_attr, states_attr = runge_kutta_4th_order_solver(
        lorenz_system_equations, t_initial, initial_state, t_final, dt_step_attractor, sigma, rho, beta
    )
    plot_and_save_attractor(time_points_attr, states_attr, sigma, rho, beta, output_plot_folder, high_res=True)

    # --- Executar a Simulação para Outras Análises (se necessário com dt diferente) ---
    # Se dt_step_analysis for o mesmo que dt_step_attractor, podemos reutilizar states_attr
    if dt_step_analysis == dt_step_attractor:
        time_points_analysis, states_analysis = time_points_attr, states_attr
    else:
        print(f"Simulando para análises gerais (dt={dt_step_analysis})...")
        time_points_analysis, states_analysis = runge_kutta_4th_order_solver(
            lorenz_system_equations, t_initial, initial_state, t_final, dt_step_analysis, sigma, rho, beta
        )

    # --- Gerar e Salvar Séries Temporais ---
    plot_and_save_timeseries(time_points_analysis, states_analysis, sigma, rho, beta, output_plot_folder)

    # --- Gerar e Salvar Seções de Poincaré ---
    # Exemplo: Seção de Poincaré no plano z = rho - 1 (um valor comum para o Lorenz)
    # O documento usa vários planos, ex: yz (x=const), xz (y=const), xy (z=const) [cite: 198, 211, 218, 220]
    # Para o sistema de Lorenz com rho=28, os pontos de equilíbrio instáveis estão em z = rho - 1 = 27.
    # Vamos testar um plano y=0, como na Figura 24 [cite: 206] (para rho=28)
    
    plane_coord_idx_poincare = 1  # 0 para x, 1 para y, 2 para z
    plane_value_poincare = 0.0
    print(f"\nGerando seção de Poincaré para o plano {['x', 'y', 'z'][plane_coord_idx_poincare]} = {plane_value_poincare}")
    plot_and_save_poincare_section(states_analysis, plane_coord_idx_poincare, plane_value_poincare, sigma, rho, beta, output_plot_folder)

    # Outro exemplo de seção, z = 25, como na Figura 24/27 [cite: 208, 220, 222]
    plane_coord_idx_poincare_2 = 2 
    plane_value_poincare_2 = 25.0
    print(f"\nGerando seção de Poincaré para o plano {['x', 'y', 'z'][plane_coord_idx_poincare_2]} = {plane_value_poincare_2}")
    plot_and_save_poincare_section(states_analysis, plane_coord_idx_poincare_2, plane_value_poincare_2, sigma, rho, beta, output_plot_folder)


    # --- Estimar Dimensão Fractal (Placeholder) ---
    print("\nTentando estimar a dimensão fractal...")
    estimate_fractal_dimension(states_analysis)

    print("\nProcesso concluído.")

if __name__ == "__main__":
    main()