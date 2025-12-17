# fractal_dimension.py
import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt 
import os

def box_counting_dimension(states, num_epsilons=15, plot_loglog=False, output_folder_base="output_plots"):
    """
    Estima a dimensão fractal de um conjunto de pontos 3D usando o método de contagem de caixas.
    Baseado nos princípios descritos em artigos como MATH540_Budai_Fractal_Dimension.pdf.

    Args:
        states (np.array): Array Nx3 contendo os pontos (x, y, z) do atrator.
                           Espera-se que os transientes já tenham sido removidos.
        num_epsilons (int): Número de diferentes tamanhos de caixa (epsilon/l) a serem testados.
        plot_loglog (bool): Se True, plota o gráfico log-log usado para a regressão.
        output_folder_base (str): Pasta base onde a subpasta para o gráfico log-log será criada.

    Returns:
        float: Dimensão fractal estimada (box-counting dimension), ou None se não puder ser calculada.
    """
    if states is None or len(states) < 500: 
        print("Pontos insuficientes para estimar a dimensão fractal de forma confiável.")
        print(f"  Número de pontos fornecidos: {len(states) if states is not None else 0}")
        return None

    # 1. Determinar os limites dos dados para definir a grade
    min_coords = np.min(states, axis=0)
    max_coords = np.max(states, axis=0)
    data_range = max_coords - min_coords

    # Lidar com casos onde o atrator pode ser plano em alguma dimensão
    if np.any(data_range < 1e-9): 
        active_dims = data_range > 1e-9 # Identifica dimensões com range não desprezível
        print(f"Aviso: Atrator parece ter extensão próxima de zero em algumas dimensões. Usando {np.sum(active_dims)}D para cálculo se possível.")
        # Se todas as dimensões tiverem range zero (ex: um único ponto), data_range será ajustado.
        # Se algumas dimensões tiverem range zero, elas serão ajustadas para evitar problemas com log(0) ou divisão por zero.
        data_range[~active_dims] = np.mean(data_range[active_dims]) if np.sum(active_dims) > 0 else 1.0

    max_extent = np.max(data_range)
    if max_extent < 1e-9: # Se, mesmo após o ajuste, a extensão for minúscula
        print("Extensão total do atrator é próxima de zero. Não é possível calcular a dimensão.")
        return None

    # 2. Definir uma faixa de tamanhos de caixa (epsilon ou 'l' no artigo)
    min_epsilon = max_extent / (2**8) 
    max_epsilon = max_extent / (2**2)  
    
    if min_epsilon >= max_epsilon or min_epsilon < 1e-9 : 
        print(f"Faixa de epsilon inválida calculada (min: {min_epsilon:.2e}, max: {max_epsilon:.2e}). Verifique a extensão dos dados.")
        min_epsilon = 0.01 
        max_epsilon = 10.0 # Valor arbitrário maior
        if len(states) > 10000 and max_extent > 1: # Para datasets grandes e com boa extensão
             min_epsilon = 0.005 * max_extent # Escalar com max_extent
             max_epsilon = 0.25 * max_extent  # Escalar com max_extent
        elif max_extent <=1 : # Se o atrator for muito pequeno
            min_epsilon = max_extent / 100.0
            max_epsilon = max_extent / 2.0

        # Re-verificar após fallback
        if min_epsilon >= max_epsilon or min_epsilon < 1e-9:
            print(f"Faixa de epsilon de fallback ainda inválida (min: {min_epsilon:.2e}, max: {max_epsilon:.2e}). Não é possível prosseguir.")
            return None
        print(f"  Usando faixa de epsilon de fallback ajustada: [{min_epsilon:.2e}, {max_epsilon:.2e}]")
    
    # Gerar epsilons iniciais
    epsilons_generated = np.logspace(np.log10(min_epsilon), np.log10(max_epsilon), num=num_epsilons, base=10.0)
    
    if len(epsilons_generated) == 0:
        print("Erro: np.logspace não gerou valores de epsilon. Verifique min_epsilon e max_epsilon.")
        return None

    # Aplicar a heurística de filtro
    # A condição epsilons * np.sqrt(len(states)) > 1.0 é para evitar epsilons tão pequenos
    # que cada ponto caia em sua própria caixa ou que N(eps) seja muito próximo do número total de pontos.
    # Isso é equivalente a epsilons > 1.0 / np.sqrt(len(states))
    filter_threshold = 1.0 / np.sqrt(len(states))
    epsilons = epsilons_generated[epsilons_generated > filter_threshold]

    if len(epsilons) == 0:
        print("Erro: Nenhum valor de epsilon válido após a filtragem heurística (eps > 1/sqrt(N)).")
        print(f"  Epsilons gerados por logspace ({len(epsilons_generated)}): min={epsilons_generated.min():.3e}, max={epsilons_generated.max():.3e}" if len(epsilons_generated) > 0 else "np.logspace não gerou epsilons.")
        print(f"  Limite do filtro (1/sqrt(N)): {filter_threshold:.3e} para N={len(states)}")
        print(f"  Isso sugere que todos os epsilons gerados ({epsilons_generated.min():.3e} a {epsilons_generated.max():.3e}) eram <= {filter_threshold:.3e}")
        return None
        
    print(f"Estimando dimensão fractal com {len(epsilons)} tamanhos de caixa (l) variando de {epsilons[0]:.3e} a {epsilons[-1]:.3e}")

    counts_N_l = [] 
    valid_ls = []   

    for l_val in epsilons:
        if l_val <= 1e-9: continue # Epsilon deve ser positivo e não trivialmente pequeno

        occupied_boxes = set()
        for point in states:
            box_indices = tuple(np.floor((point - min_coords) / l_val).astype(int))
            occupied_boxes.add(box_indices)
        
        num_occupied = len(occupied_boxes)
        
        # Condições para uma boa região de scaling: N(l) não deve ser 1 nem N_total.
        if 1 < num_occupied < len(states):
            counts_N_l.append(num_occupied)
            valid_ls.append(l_val)
        # Se N(l) se torna 1, epsilons maiores não são mais úteis para a regressão na região de scaling.
        elif num_occupied == 1 and len(valid_ls) > 0 : 
            break 
        # Se N(l) == len(states), epsilons menores podem ainda ser úteis, mas este ponto pode estar fora da região linear.
        # A condição `1 < num_occupied < len(states)` já lida com isso.

    if len(counts_N_l) < 3: 
        print("Não foi possível obter contagens suficientes (1 < N(l) < N_total_pontos) para diferentes tamanhos de caixa na região de scaling.")
        print(f"  Contagens N(l) obtidas: {counts_N_l}")
        print(f"  Tamanhos de caixa 'l' correspondentes: {valid_ls}")
        return None

    log_N_l = np.log(np.array(counts_N_l))
    log_1_over_l = np.log(1.0 / np.array(valid_ls))

    try:
        slope, intercept, r_value, p_value, std_err = linregress(log_1_over_l, log_N_l)
        dimension = slope 
    except ValueError as e:
        print(f"Erro na regressão linear: {e}")
        return None

    print(f"  Dimensão fractal estimada (box-counting): {dimension:.3f} (R^2 da regressão = {r_value**2:.3f})")

    if plot_loglog:
        plt.figure(figsize=(8, 6))
        plt.scatter(log_1_over_l, log_N_l, label=r'Dados log-log (N($l$) vs 1/$l$)')
        plt.plot(log_1_over_l, intercept + slope * log_1_over_l, 'r', 
                 label=rf'Ajuste linear (D_f = {dimension:.3f}, $R^2={r_value**2:.3f}$)')
        plt.xlabel(r'log(1/$l$)')
        plt.ylabel(r'log(N($l$))')
        plt.title('Gráfico Log-Log para Estimativa da Dimensão Box-Counting')
        plt.legend()
        plt.grid(True)
        
        fractal_plot_output_folder = os.path.join(output_folder_base, "fractal_dimension_analysis")
        if not os.path.exists(fractal_plot_output_folder):
            os.makedirs(fractal_plot_output_folder)
            
        loglog_filename = os.path.join(fractal_plot_output_folder, "fractal_dimension_loglog_plot.png")
        try:
            plt.savefig(loglog_filename)
            print(f"  Gráfico log-log salvo em: {loglog_filename}")
        except Exception as e:
            print(f"  Erro ao salvar o gráfico log-log: {e}")
        plt.close()

    return dimension
