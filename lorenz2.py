import numpy as np
import matplotlib.pyplot as plt
import os

def lorenz_system(t, state, sigma, rho, beta):
    """
    Define as equações do sistema de Lorenz.

    Args:
        t: tempo (não usado diretamente nas equações, mas necessário para o solver).
        state: array [x, y, z] com os valores atuais das variáveis de estado.
        sigma: parâmetro sigma do sistema de Lorenz.
        rho: parâmetro rho do sistema de Lorenz.
        beta: parâmetro beta do sistema de Lorenz.

    Returns:
        array com as derivadas [dx/dt, dy/dt, dz/dt].
    """
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return np.array([dxdt, dydt, dzdt])

def runge_kutta_4th_order(func, t0, y0, tf, dt, *args):
    """
    Implementa o método de Runge-Kutta de quarta ordem para resolver EDOs.
    Baseado nas equações 4.3 e 4.4 do documento.

    Args:
        func: a função que define as EDOs (ex: lorenz_system).
        t0: tempo inicial.
        y0: array com as condições iniciais.
        tf: tempo final.
        dt: passo de tempo (h no documento).
        *args: argumentos adicionais para a função 'func' (ex: sigma, rho, beta).

    Returns:
        t_values: array com os valores de tempo.
        y_values: array com os valores das variáveis de estado em cada passo de tempo.
    """
    t_values = np.arange(t0, tf + dt, dt)
    n_steps = len(t_values)
    y_values = np.zeros((n_steps, len(y0)))
    y_values[0] = y0
    y = y0

    for i in range(n_steps - 1):
        t = t_values[i]
        k1 = dt * func(t, y, *args)
        k2 = dt * func(t + 0.5 * dt, y + 0.5 * k1, *args)
        k3 = dt * func(t + 0.5 * dt, y + 0.5 * k2, *args)
        k4 = dt * func(t + dt, y + k3, *args)
        y = y + (k1 + 2*k2 + 2*k3 + k4) / 6.0
        y_values[i+1] = y
        
    return t_values, y_values

# --- Parâmetros da Simulação ---
# Parâmetros clássicos de Lorenz mencionados no documento
sigma = 10.0
beta = 8.0 / 3.0

# Escolha um valor para rho (ex: 28 para comportamento caótico)
rho = 28.0
# Outros valores de rho do documento:
# rho = 10.0 # Estável
# rho = 20.0 # Caos transiente

# Condições iniciais
e_perturbation = 0.0 # Para o sistema de referência, e=0
x0 = 0.0
y0 = 1.0 + e_perturbation
z0 = 20.0
initial_state = np.array([x0, y0, z0])

# Configurações de tempo
t_initial = 0.0
t_final = 50.0 # Tempo similar ao usado nas figuras do documento
# Diminuir o passo de tempo para maior resolução da trajetória
dt_step = 0.005  # Anteriormente 0.01, agora menor para mais pontos

# --- Criar pasta para salvar as imagens ---
output_folder = "imagens geradas"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
    print(f"Pasta '{output_folder}' criada.")
else:
    print(f"Pasta '{output_folder}' já existe.")

# Nomes dos arquivos de imagem
attractor_filename_high_res = os.path.join(output_folder, f"atrator_lorenz_rho{rho}.png")
timeseries_filename = os.path.join(output_folder, f"series_temporais_lorenz_rho{rho}.png") # Mantendo o nome do arquivo de séries temporais


# --- Executar a Simulação ---
print(f"Simulando o sistema de Lorenz com sigma={sigma}, rho={rho}, beta={beta:.2f}")
print(f"Condições iniciais: x0={x0}, y0={y0}, z0={z0}")
print(f"Intervalo de tempo: [{t_initial}, {t_final}] com dt={dt_step} (maior resolução de trajetória)")

time_points, states = runge_kutta_4th_order(lorenz_system, t_initial, initial_state, t_final, dt_step, sigma, rho, beta)

# Extrair as variáveis de estado para plotagem
x_t = states[:, 0]
y_t = states[:, 1]
z_t = states[:, 2]

# --- Plotar e Salvar os Resultados ---

# Plotar o atrator de Lorenz no espaço de fase com maior resolução de imagem
fig_attractor = plt.figure(figsize=(12, 10)) # Pode aumentar o tamanho da figura também
ax_attractor = fig_attractor.add_subplot(111, projection='3d')
ax_attractor.plot(x_t, y_t, z_t, lw=0.3) # Linha mais fina para melhor visualização com muitos pontos
ax_attractor.set_xlabel("X(t)")
ax_attractor.set_ylabel("Y(t)")
ax_attractor.set_zlabel("Z(t)")
ax_attractor.set_title(f"Atrator de Lorenz ($\\sigma={sigma}, \\rho={rho}, \\beta={beta:.2f}$) - Alta Resolução")
# Salvar a figura com maior DPI
plt.savefig(attractor_filename_high_res, dpi=300) 
print(f"Atrator de alta resolução salvo em: {attractor_filename_high_res}")
plt.show()


# Plotar as séries temporais (sem alterações na resolução aqui, mas pode ser feito se desejado)
fig_timeseries, axs_timeseries = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
fig_timeseries.suptitle(f"Séries Temporais do Sistema de Lorenz ($\\sigma={sigma}, \\rho={rho}, \\beta={beta:.2f}$)", fontsize=16)

axs_timeseries[0].plot(time_points, x_t, label='x(t)')
axs_timeseries[0].set_ylabel('x(t)')
axs_timeseries[0].grid(True)

axs_timeseries[1].plot(time_points, y_t, label='y(t)')
axs_timeseries[1].set_ylabel('y(t)')
axs_timeseries[1].grid(True)

axs_timeseries[2].plot(time_points, z_t, label='z(t)')
axs_timeseries[2].set_ylabel('z(t)')
axs_timeseries[2].set_xlabel('Tempo (t)')
axs_timeseries[2].grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(timeseries_filename) # Mantendo o DPI padrão ou pode adicionar dpi=300 aqui também
print(f"Séries temporais salvas em: {timeseries_filename}")
plt.show()


print("Simulação concluída e gráficos salvos com maior resolução para o atrator.")