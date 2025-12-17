# run_chaos_control.py

import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from mpl_toolkits.mplot3d import Axes3D

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.lorenz_solver import runge_kutta_4th_order_solver, lorenz_system_equations
from control.chaos_control import lorenz_controlled_system_x

# --- Parâmetros do Sistema de Lorenz (do artigo) ---
sigma = 10.0
rho = 28.0
beta = 8.0 / 3.0

# --- Ponto de Equilíbrio ---
xe = np.sqrt(beta * (rho - 1))
ye = np.sqrt(beta * (rho - 1))
ze = rho - 1
equilibrium_point = np.array([xe, ye, ze])

# --- Parâmetros da Simulação e do Novo Controle ---
initial_state = np.array([1.0, 1.0, 1.0])
t_start = 0.0
t_end = 400.0
dt = 0.01
t_control = 200.0

# Parâmetros da lei de controle
eta = 2.0
epsilon = 0.1

# Criar diretório para salvar os plots
output_dir = "output_plots/chaos_control"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Simulação com o Novo Controle ---
print("Iniciando simulação com a lógica de ControleCaosX...")
controlled_func = lambda t, state, *args: lorenz_controlled_system_x(t, state, *args)
t_controlled, sol_controlled = runge_kutta_4th_order_solver(
    controlled_func,
    t_start,
    initial_state,
    t_end,
    dt,
    sigma,
    rho,
    beta,
    equilibrium_point,
    eta,
    epsilon,
    t_control,
)
x_con, y_con, z_con = sol_controlled.T
print("Simulação concluída.")

# --- Sinal de controle u1(t) (aplicado em x) ---
u1_signal = []
for i, t in enumerate(t_controlled):
    if t < t_control:
        u1_signal.append(0.0)
    else:
        x, y, z = sol_controlled[i]
        s = x - xe
        u_control = -eta * np.tanh(s / epsilon)
        u1_signal.append(u_control)

# ====================================================
#  GRÁFICO 1: CONTROLE DE CAOS EM x(t), y(t), z(t)
#  (três figuras separadas no mesmo estilo do artigo)
# ====================================================

print("Gerando gráficos de controle de caos para x(t), y(t) e z(t)...")

# --- x(t) ---
fig_x, ax_x = plt.subplots(figsize=(12, 6))
ax_x.plot(t_controlled, x_con, "b-", linewidth=1.5, label="Variável $x(t)$")
ax_x.axhline(
    y=xe,
    color="g",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Ponto de Equilíbrio ($x_e = {xe:.4f}$)",
)
ax_x.axvline(
    x=t_control,
    color="orange",
    linestyle=":",
    linewidth=2,
    label=f"Controle ativado (t = {t_control})",
)
ax_x.set_xlabel("Tempo (t)", fontsize=13)
ax_x.set_ylabel("$x(t)$", fontsize=13)
ax_x.set_title("Controle de Caos na Variável x(t)", fontsize=16)
ax_x.legend(fontsize=12)
ax_x.grid(True, linestyle="--", alpha=0.6)
ax_x.set_xlim(0, t_end)
plt.savefig(os.path.join(output_dir, "grafico_controle_caos_x.png"))
plt.show()

# --- y(t) ---
fig_y, ax_y = plt.subplots(figsize=(12, 6))
ax_y.plot(t_controlled, y_con, "b-", linewidth=1.5, label="Variável $y(t)$")
ax_y.axhline(
    y=ye,
    color="g",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Ponto de Equilíbrio ($y_e = {ye:.4f}$)",
)
ax_y.axvline(
    x=t_control,
    color="orange",
    linestyle=":",
    linewidth=2,
    label=f"Controle ativado (t = {t_control})",
)
ax_y.set_xlabel("Tempo (t)", fontsize=13)
ax_y.set_ylabel("$y(t)$", fontsize=13)
ax_y.set_title("Controle de Caos na Variável y(t)", fontsize=16)
ax_y.legend(fontsize=12)
ax_y.grid(True, linestyle="--", alpha=0.6)
ax_y.set_xlim(0, t_end)
plt.savefig(os.path.join(output_dir, "grafico_controle_caos_y.png"))
plt.show()

# --- z(t) ---
fig_z, ax_z = plt.subplots(figsize=(12, 6))
ax_z.plot(t_controlled, z_con, "b-", linewidth=1.5, label="Variável $z(t)$")
ax_z.axhline(
    y=ze,
    color="g",
    linestyle="--",
    linewidth=2,
    alpha=0.7,
    label=f"Ponto de Equilíbrio ($z_e = {ze:.4f}$)",
)
ax_z.axvline(
    x=t_control,
    color="orange",
    linestyle=":",
    linewidth=2,
    label=f"Controle ativado (t = {t_control})",
)
ax_z.set_xlabel("Tempo (t)", fontsize=13)
ax_z.set_ylabel("$z(t)$", fontsize=13)
ax_z.set_title("Controle de Caos na Variável z(t)", fontsize=16)
ax_z.legend(fontsize=12)
ax_z.grid(True, linestyle="--", alpha=0.6)
ax_z.set_xlim(0, t_end)
plt.savefig(os.path.join(output_dir, "grafico_controle_caos_z.png"))
plt.show()

# ====================================================
#  GRÁFICOS 2 e 3: ATRATOR E SÉRIES SEM CONTROLE
# ====================================================

print("Gerando gráficos do artigo (sem controle)...")
t_uncontrolled, sol_uncontrolled = runge_kutta_4th_order_solver(
    lorenz_system_equations,
    0.0,
    initial_state,
    100.0,
    dt,
    sigma,
    rho,
    beta,
)
x_unc, y_unc, z_unc = sol_uncontrolled.T

# (se você tiver também o plot do atrator 3D sem controle,
# pode adicioná-lo aqui, como já faz em outras partes do projeto)

# ====================================================
#  GRÁFICO 4: VARIÁVEIS DE ESTADO COM CONTROLE
#  (Figuras 3, 4, 5 do artigo)
# ====================================================

print("Gerando gráficos do artigo (com controle)...")
fig_states, (ax_s1, ax_s2, ax_s3) = plt.subplots(
    3, 1, figsize=(12, 9), sharex=True
)
fig_states.suptitle(
    f"Variáveis de Estado (Controle em t={t_control}s)",
    fontsize=16,
)
ax_s1.plot(t_controlled, x_con, "b", lw=1), ax_s1.set_ylabel("x(t)"), ax_s1.grid(True)
ax_s2.plot(t_controlled, y_con, "b", lw=1), ax_s2.set_ylabel("y(t)"), ax_s2.grid(True)
ax_s3.plot(t_controlled, z_con, "b", lw=1), ax_s3.set_ylabel("z(t)"), ax_s3.grid(True)
ax_s3.set_xlabel("Tempo (t)")
ax_s3.set_xlim(0, t_end)
plt.savefig(
    os.path.join(output_dir, "artigo_figuras_3_4_5_series_com_controle.png")
)
plt.show()

# ====================================================
#  GRÁFICO 5: SINAL DE CONTROLE u1(t)
#  (Figura 6 do Artigo)
# ====================================================

plt.figure(figsize=(12, 5))
plt.plot(t_controlled, u1_signal, "b", lw=1.5)
plt.title("Sinal de Controle u(t)", fontsize=14)
plt.ylabel("u(t)")
plt.xlabel("Tempo (t)")
plt.grid(True)
plt.xlim(t_control - 5, t_control + 20)  # foco na ativação do controle
plt.savefig(os.path.join(output_dir, "sinal_controle.png"))
plt.show()

print(f"Todos os gráficos foram salvos em: {os.path.abspath(output_dir)}")