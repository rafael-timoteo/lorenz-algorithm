import numpy as np
import matplotlib.pyplot as plt

def calcular_dimensao_fractal(points):
    """
    Calcula a dimensão fractal de um conjunto de pontos 3D usando o método de contagem de caixas.
    """
    # 1. Aplicar a metodologia do artigo: filtrar para o octante x<0 e y<0
    filtered_points = points[(points[:, 0] < 0) & (points[:, 1] < 0)]
    
    print(f"   Pontos totais no atrator: {len(points)}. Pontos no octante analisado: {len(filtered_points)}")

    if len(filtered_points) < 500:
        print("   Pontos insuficientes para uma análise robusta.")
        return np.nan, [], [], []

    # 2. Definir a faixa de tamanhos de caixa para a contagem
    l_values = np.logspace(0.1, 1.5, 15)
    N_values = []

    # 3. Contar as caixas para cada tamanho 'l'
    for l in l_values:
        if l == 0: continue
        min_coords = filtered_points.min(axis=0)
        box_indices = np.floor((filtered_points - min_coords) / l)
        num_boxes = len(np.unique(box_indices, axis=0))
        N_values.append(num_boxes)

    # 4. Calcular a dimensão usando regressão linear
    valid_indices = np.array(N_values) > 0
    if not np.any(valid_indices):
        return np.nan, [], [], []

    log_l = np.log(np.array(l_values)[valid_indices])
    log_N = np.log(np.array(N_values)[valid_indices])

    coeffs = np.polyfit(log_l, log_N, 1)
    D = -coeffs[0]
    
    return D, l_values, N_values, coeffs

# --- Início do Código Principal ---

# --- PARÂMETROS GERAIS ---
h = 0.01
M = 20000
sigma = 10.0
b = 8.0 / 3.0

# --- LISTA DE VALORES DE R ATUALIZADA ---
r_values = [5, 10, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40]

results = {}

# --- Loop principal para simular e analisar cada valor de r ---
for r in r_values:
    print(f"\nProcessando para r = {r}...")
    x0, y0, z0 = 0.1, 0.1, 0.1
    xs, ys, zs = np.empty(M + 1), np.empty(M + 1), np.empty(M + 1)
    xs[0], ys[0], zs[0] = (x0, y0, z0)

    for i in range(M):
        x_old, y_old, z_old = xs[i], ys[i], zs[i]
        x_new = x_old + h * sigma * (y_old - x_old)
        y_new = y_old + h * (r * x_old - y_old - x_old * z_old)
        z_new = z_old + h * (x_old * y_old - b * z_old)
        xs[i + 1], ys[i + 1], zs[i + 1] = x_new, y_new, z_new

    points = np.column_stack([xs, ys, zs])
    D, l_vals, N_vals, coeffs = calcular_dimensao_fractal(points)
    
    results[r] = {'D': D, 'l_values': l_vals, 'N_values': N_vals, 'coeffs': coeffs}

# --- GERAÇÃO DAS TABELAS E GRÁFICOS DE RESULTADOS ---

print("\n\n--- Tabela de Resultados Finais ---")
print("="*35)
print(f"{'Parâmetro r':<15} | {'Dimensão Fractal (D)':<20}")
print("-"*35)
for r, data in results.items():
    if not np.isnan(data['D']):
        print(f"{r:<15.2f} | {data['D']:<20.4f}")
    else:
        print(f"{r:<15.2f} | {'Inconclusivo':<20}")
print("="*35)


# --- Gráfico 1: N(l) vs l ---
plt.figure(figsize=(12, 8))

# --- MUDANÇA PRINCIPAL AQUI: Gerando cores programaticamente com um mapa de cores adequado ---
colors = plt.cm.cool(np.linspace(0, 1, len(r_values)))

for i, (r, data) in enumerate(results.items()):
    if not np.isnan(data['D']):
        plt.plot(data['l_values'], data['N_values'], 'o-', label=f'r = {r}', color=colors[i])
plt.gca().invert_xaxis()
plt.xlabel("Tamanho da Caixa (l)")
plt.ylabel("Número de Caixas Ocupadas N(l)")
plt.title("Análise de Contagem de Caixas")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') # Ajusta a posição da legenda
plt.tight_layout() # Ajusta o layout para a legenda não cortar
plt.grid(True, which="both", ls="--")
plt.show()

# --- Gráfico 2: Gráfico Log-Log ---
plt.figure(figsize=(12, 8))
for i, (r, data) in enumerate(results.items()):
    if not np.isnan(data['D']):
        log_l = np.log(data['l_values'])
        valid_indices = np.array(data['N_values']) > 0
        log_N = np.log(np.array(data['N_values'])[valid_indices])
        
        plt.plot(log_l, log_N, 'o', label=f'Dados r = {r}', color=colors[i])
        fit_line = np.polyval(data['coeffs'], log_l)
        plt.plot(log_l, fit_line, '--', label=f'Ajuste D={data["D"]:.3f}', color=colors[i])
plt.xlabel("log(l)")
plt.ylabel("log(N(l))")
plt.title("Verificação da Dimensão Fractal (Gráfico Log-Log)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, which="both", ls="--")
plt.show()

# --- GRÁFICO FINAL: Dimensão Fractal vs. Rho (r) ---
plt.figure(figsize=(12, 8))
r_plot = [r for r, data in results.items() if not np.isnan(data['D'])]
D_plot = [data['D'] for r, data in results.items() if not np.isnan(data['D'])]

plt.plot(r_plot, D_plot, 'o-', color='crimson', markersize=10, linewidth=2)
plt.xlabel("Parâmetro rho (r)")
plt.ylabel("Dimensão Fractal (D)")
plt.title("Variação da Dimensão Fractal com o Parâmetro r")
plt.grid(True, which="both", ls="--")
plt.show()