import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessário para projection='3d'


# =============================================================================
#  CONFIGURAÇÃO GERAL DA PARTE 1 (SINCRONIZAÇÃO)
# =============================================================================

# Modo de sincronização:
#   "xdrive"  → Pecora-Carroll clássico: subsistema (y,z) dirigido por x(t)
#   "ydrive"  → subsistema (x,z) dirigido por y(t)
#   "zdrive"  → subsistema (x,y) dirigido por z(t)
#   "master"  → master–slave 3D, acoplamento difusivo em x
mode = "xdrive"  # mude aqui: "ydrive", "zdrive" ou "master"

# Atraso no acoplamento (em unidades de tempo)
tau = 0.0

# Parâmetros do drive
sigma_d = 10.0
rho_d   = 28.0
beta_d  = 8.0 / 3.0

# Parâmetros do response antes do sync (diferentes para atratores distintos)
sigma_r_free = 9.0
rho_r_free   = 45.0
beta_r_free  = 3.0

# Parâmetros do response após o sync (para facilitar a sincronização
# igualamos aos do drive, mas você pode mudar se quiser experimentar)
sigma_r_cpl = sigma_d
rho_r_cpl   = rho_d
beta_r_cpl  = beta_d

# Ganho de acoplamento para o modo master–slave
k_master = 5.0

# Tempo de simulação
t0 = 0.0
tf = 60.0
dt = 0.01
t_grid = np.arange(t0, tf, dt)
N = len(t_grid)

# Instante em que o acoplamento é ligado
t_sync = 20.0
idx_sync = np.searchsorted(t_grid, t_sync)

# Tempo usado na fase 1 (inclui t_sync) e fase 2 (depois de t_sync)
t_before = t_grid[:idx_sync + 1]   # [0, ..., t_sync]
t_after  = t_grid[idx_sync + 1:]   # (t_sync, ..., tf]

# Condições iniciais do drive
x0_d, y0_d, z0_d = 1.0, 1.0, 1.0

# Condições iniciais do response (bem diferentes)
x0_r, y0_r, z0_r = -8.0, 12.0, 5.0


# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================

def lorenz3d(t, state, sigma, rho, beta):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]


def delayed_interp(var, t_grid, t, tau):
    """
    Interpola var(t - tau). Se t - tau < t_grid[0], usa o primeiro valor.
    """
    t_tau = t - tau
    if t_tau <= t_grid[0]:
        return var[0]
    return np.interp(t_tau, t_grid, var)


# --- subsistemas dirigidos ---

def response_xdrive_yz(t, state, x_drive, t_drive, rho, beta, tau):
    """Subsis. (y', z') com x-driving."""
    y_r, z_r = state
    x_del = delayed_interp(x_drive, t_drive, t, tau)
    dy = x_del * (rho - z_r) - y_r
    dz = x_del * y_r - beta * z_r
    return [dy, dz]


def response_ydrive_xz(t, state, y_drive, t_drive, sigma, rho, beta, tau):
    """Subsis. (x', z') com y-driving."""
    x_r, z_r = state
    y_del = delayed_interp(y_drive, t_drive, t, tau)
    dx = sigma * (y_del - x_r)
    dz = x_r * y_del - beta * z_r
    return [dx, dz]


def response_zdrive_xy(t, state, z_drive, t_drive, sigma, rho, beta, tau):
    """Subsis. (x', y') com z-driving."""
    x_r, y_r = state
    z_del = delayed_interp(z_drive, t_drive, t, tau)
    dx = sigma * (y_r - x_r)
    dy = x_r * (rho - z_del) - y_r
    return [dx, dy]


def response_master_slave(t, state, x_drive, t_drive,
                          sigma_r, rho_r, beta_r, k, tau):
    """
    Master–slave 3D com acoplamento difusivo em x:
      dx_r/dt = sigma_r (y_r - x_r) + k [ x_d(t - tau) - x_r ]
      dy_r/dt = x_r (rho_r - z_r) - y_r
      dz_r/dt = x_r y_r - beta_r z_r
    """
    x_r, y_r, z_r = state
    x_del = delayed_interp(x_drive, t_drive, t, tau)
    dx = sigma_r * (y_r - x_r) + k * (x_del - x_r)
    dy = x_r * (rho_r - z_r) - y_r
    dz = x_r * y_r - beta_r * z_r
    return [dx, dy, dz]


# =============================================================================
#  PARTE 1 – SIMULAÇÃO DO DRIVE
# =============================================================================

print("=" * 80)
print("PARTE 1 – SINCRONIZAÇÃO DE LORENZ COM ATRATORESS DISTINTOS E ATRASO")
print("=" * 80)
print(f"Modo de sincronização: {mode}")
print(f"Atraso tau = {tau}")
print(f"Parâmetros drive:  sigma={sigma_d}, rho={rho_d}, beta={beta_d:.4f}")
print(f"Parâmetros resp (livre): sigma={sigma_r_free}, rho={rho_r_free}, beta={beta_r_free:.4f}")
print(f"Parâmetros resp (acoplado): sigma={sigma_r_cpl}, rho={rho_r_cpl}, beta={beta_r_cpl:.4f}")
print(f"t ∈ [{t0}, {tf}], dt = {dt}, t_sync = {t_sync}")

# Drive
print("\n[1/3] Integrando drive 3D...")
sol_d = solve_ivp(
    lambda tt, s: lorenz3d(tt, s, sigma_d, rho_d, beta_d),
    [t0, tf],
    [x0_d, y0_d, z0_d],
    t_eval=t_grid,
    rtol=1e-9,
    atol=1e-12
)
x_d = sol_d.y[0]
y_d = sol_d.y[1]
z_d = sol_d.y[2]


# =============================================================================
#  PARTE 1 – SIMULAÇÃO DO RESPONSE (FASE 1: LIVRE)
# =============================================================================

print("[2/3] Integrando response livre (parâmetros diferentes, até t_sync)...")
sol_r1 = solve_ivp(
    lambda tt, s: lorenz3d(tt, s, sigma_r_free, rho_r_free, beta_r_free),
    [t0, t_sync],
    [x0_r, y0_r, z0_r],
    t_eval=t_before,
    rtol=1e-9,
    atol=1e-12
)
x_r_before = sol_r1.y[0]
y_r_before = sol_r1.y[1]
z_r_before = sol_r1.y[2]

# Estado do response em t_sync (ponto de partida para a fase 2)
x_r_sync = x_r_before[-1]
y_r_sync = y_r_before[-1]
z_r_sync = z_r_before[-1]


# =============================================================================
#  PARTE 1 – SIMULAÇÃO DO RESPONSE (FASE 2: ACOPLADO)
# =============================================================================

print("[3/3] Integrando response acoplado (modo =", mode, ") com atraso...")

if mode == "xdrive":
    # subsis (y', z'), acoplado por x_d(t - tau)
    y0_2 = y_r_sync
    z0_2 = z_r_sync

    sol_r2 = solve_ivp(
        lambda tt, s: response_xdrive_yz(
            tt, s, x_d, t_grid, rho_r_cpl, beta_r_cpl, tau
        ),
        [t_sync, tf],
        [y0_2, z0_2],
        t_eval=t_after,
        rtol=1e-9,
        atol=1e-12
    )
    y_r_after = sol_r2.y[0]
    z_r_after = sol_r2.y[1]
    x_r_after = x_d[idx_sync + 1:]  # reconstrução 3D

    # monta séries completas
    x_r_full = np.concatenate([x_r_before, x_r_after])
    y_r_full = np.concatenate([y_r_before, y_r_after])
    z_r_full = np.concatenate([z_r_before, z_r_after])

    err_x = x_d - x_r_full
    err_y = y_d - y_r_full
    err_z = z_d - z_r_full
    err_norm = np.sqrt(err_y**2 + err_z**2)  # erro no subespaço (y,z)

elif mode == "ydrive":
    # subsis (x', z'), acoplado por y_d(t - tau)
    x0_2 = x_r_sync
    z0_2 = z_r_sync

    sol_r2 = solve_ivp(
        lambda tt, s: response_ydrive_xz(
            tt, s, y_d, t_grid, sigma_r_cpl, rho_r_cpl, beta_r_cpl, tau
        ),
        [t_sync, tf],
        [x0_2, z0_2],
        t_eval=t_after,
        rtol=1e-9,
        atol=1e-12
    )
    x_r_after = sol_r2.y[0]
    z_r_after = sol_r2.y[1]
    y_r_after = y_d[idx_sync + 1:]  # reconstrução 3D

    x_r_full = np.concatenate([x_r_before, x_r_after])
    y_r_full = np.concatenate([y_r_before, y_r_after])
    z_r_full = np.concatenate([z_r_before, z_r_after])

    err_x = x_d - x_r_full
    err_y = y_d - y_r_full
    err_z = z_d - z_r_full
    err_norm = np.sqrt(err_x**2 + err_z**2)  # erro em (x,z)

elif mode == "zdrive":
    # subsis (x', y'), acoplado por z_d(t - tau)
    x0_2 = x_r_sync
    y0_2 = y_r_sync

    sol_r2 = solve_ivp(
        lambda tt, s: response_zdrive_xy(
            tt, s, z_d, t_grid, sigma_r_cpl, rho_r_cpl, beta_r_cpl, tau
        ),
        [t_sync, tf],
        [x0_2, y0_2],
        t_eval=t_after,
        rtol=1e-9,
        atol=1e-12
    )
    x_r_after = sol_r2.y[0]
    y_r_after = sol_r2.y[1]
    z_r_after = z_d[idx_sync + 1:]  # reconstrução 3D

    x_r_full = np.concatenate([x_r_before, x_r_after])
    y_r_full = np.concatenate([y_r_before, y_r_after])
    z_r_full = np.concatenate([z_r_before, z_r_after])

    err_x = x_d - x_r_full
    err_y = y_d - y_r_full
    err_z = z_d - z_r_full
    err_norm = np.sqrt(err_x**2 + err_y**2)  # erro em (x,y)

elif mode == "master":
    # master–slave 3D com acoplamento difusivo em x
    sol_r2 = solve_ivp(
        lambda tt, s: response_master_slave(
            tt, s, x_d, t_grid,
            sigma_r_cpl, rho_r_cpl, beta_r_cpl, k_master, tau
        ),
        [t_sync, tf],
        [x_r_sync, y_r_sync, z_r_sync],
        t_eval=t_after,
        rtol=1e-9,
        atol=1e-12
    )
    x_r_after = sol_r2.y[0]
    y_r_after = sol_r2.y[1]
    z_r_after = sol_r2.y[2]

    x_r_full = np.concatenate([x_r_before, x_r_after])
    y_r_full = np.concatenate([y_r_before, y_r_after])
    z_r_full = np.concatenate([z_r_before, z_r_after])

    err_x = x_d - x_r_full
    err_y = y_d - y_r_full
    err_z = z_d - z_r_full
    err_norm = np.sqrt(err_x**2 + err_y**2 + err_z**2)  # erro em 3D

else:
    raise ValueError(f"Modo desconhecido: {mode}")

print("\nErro de sincronização (norma):")
print(f"  ||erro||(t=0)  = {err_norm[0]:.6e}")
print(f"  ||erro||(t=tf) = {err_norm[-1]:.6e}")


# =============================================================================
#  FIGURAS – PARTE 1
# =============================================================================

plt.style.use("default")
t_full = t_grid

# 1) Séries de x(t)
figx, axx = plt.subplots(figsize=(12, 6))
axx.plot(t_full, x_d,       label=r"$x_{drive}(t)$",    lw=1.5)
axx.plot(t_full, x_r_full, label=r"$x_{response}(t)$", lw=1.2, ls="--")
axx.axvline(t_sync, color="k", lw=2, ls="--",
            label=rf"$t_{{sync}} = {t_sync}$")
axx.set_xlabel("t")
axx.set_ylabel("x(t)")
axx.set_title(f"Séries temporais de x(t) – modo {mode}, atraso τ={tau}")
axx.legend()
axx.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"lorenz_sync_{mode}_xtime.png", dpi=300)

# 2) Séries de y(t)
figy, axy = plt.subplots(figsize=(12, 6))
axy.plot(t_full, y_d,       label=r"$y_{drive}(t)$",    lw=1.5)
axy.plot(t_full, y_r_full, label=r"$y_{response}(t)$", lw=1.2, ls="--")
axy.axvline(t_sync, color="k", lw=2, ls="--",
            label=rf"$t_{{sync}} = {t_sync}$")
axy.set_xlabel("t")
axy.set_ylabel("y(t)")
axy.set_title(f"Séries temporais de y(t) – modo {mode}, atraso τ={tau}")
axy.legend()
axy.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"lorenz_sync_{mode}_ytime.png", dpi=300)

# 3) Séries de z(t)
figz, axz = plt.subplots(figsize=(12, 6))
axz.plot(t_full, z_d,       label=r"$z_{drive}(t)$",    lw=1.5)
axz.plot(t_full, z_r_full, label=r"$z_{response}(t)$", lw=1.2, ls="--")
axz.axvline(t_sync, color="k", lw=2, ls="--",
            label=rf"$t_{{sync}} = {t_sync}$")
axz.set_xlabel("t")
axz.set_ylabel("z(t)")
axz.set_title(f"Séries temporais de z(t) – modo {mode}, atraso τ={tau}")
axz.legend()
axz.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"lorenz_sync_{mode}_ztime.png", dpi=300)

# 4) ln do erro
fige, axe = plt.subplots(figsize=(12, 6))
axe.plot(t_full, np.log(err_norm + 1e-20), lw=1.5)
axe.axvline(t_sync, color="k", lw=2, ls="--",
            label=rf"$t_{{sync}} = {t_sync}$")
axe.set_xlabel("t")
axe.set_ylabel(r"$\ln \|erro\|$")
axe.set_title(f"Decaimento do erro de sincronização – modo {mode}, atraso τ={tau}")
axe.grid(True, alpha=0.3)
axe.legend()
plt.tight_layout()
plt.savefig(f"lorenz_sync_{mode}_log_error.png", dpi=300)

# 5) Atratores 3D antes/depois
fig3 = plt.figure(figsize=(14, 6))

# para limites comuns
x_all = np.concatenate([x_d, x_r_full])
y_all = np.concatenate([y_d, y_r_full])
z_all = np.concatenate([z_d, z_r_full])
x_min, x_max = x_all.min(), x_all.max()
y_min, y_max = y_all.min(), y_all.max()
z_min, z_max = z_all.min(), z_all.max()

ax3a = fig3.add_subplot(121, projection="3d")
ax3a.plot(x_d[:idx_sync + 1], y_d[:idx_sync + 1], z_d[:idx_sync + 1],
          lw=0.7, label="Drive (antes)")
ax3a.plot(x_r_full[:idx_sync + 1], y_r_full[:idx_sync + 1], z_r_full[:idx_sync + 1],
          lw=0.7, ls="--", label="Response (antes)")
ax3a.set_title(r"$t < t_{sync}$ – atratores distintos (sem acoplamento)")
ax3a.set_xlabel("x")
ax3a.set_ylabel("y")
ax3a.set_zlabel("z")
ax3a.legend()
ax3a.set_xlim(x_min, x_max)
ax3a.set_ylim(y_min, y_max)
ax3a.set_zlim(z_min, z_max)

ax3b = fig3.add_subplot(122, projection="3d")
ax3b.plot(x_d[idx_sync + 1:], y_d[idx_sync + 1:], z_d[idx_sync + 1:],
          lw=0.7, label="Drive (depois)")
ax3b.plot(x_r_full[idx_sync + 1:], y_r_full[idx_sync + 1:], z_r_full[idx_sync + 1:],
          lw=0.7, ls="--", label="Response (depois)")
ax3b.set_title(r"$t \geq t_{sync}$ – regime acoplado (com atraso)")
ax3b.set_xlabel("x")
ax3b.set_ylabel("y")
ax3b.set_zlabel("z")
ax3b.legend()
ax3b.set_xlim(x_min, x_max)
ax3b.set_ylim(y_min, y_max)
ax3b.set_zlim(z_min, z_max)

plt.suptitle(f"Sincronização de Lorenz – modo {mode}, atraso τ={tau}", fontsize=14)
plt.tight_layout()
plt.savefig(f"lorenz_sync_{mode}_attractors.png", dpi=300)

print("\nFiguras da PARTE 1 salvas (modo", mode, "):")
print("  -", f"lorenz_sync_{mode}_xtime.png")
print("  -", f"lorenz_sync_{mode}_ytime.png")
print("  -", f"lorenz_sync_{mode}_log_error.png")
print("  -", f"lorenz_sync_{mode}_attractors.png")


# =============================================================================
#  PARTE 2 – CLEs (MANTIDA COMO NO SEU SCRIPT ORIGINAL, PARA X-DRIVING)
# =============================================================================
#  (se quiser, depois a gente também generaliza CLEs para outros modos)
# =============================================================================

sigma_cle = 10.0
rho_cle   = 60.0
beta_cle  = 8.0 / 3.0

t0_cle          = 0.0
tf_cle          = 1000.0
renorm_interval = 0.1
t_transient_cle = 100.0

dt_drive_cle = 0.01
t_eval_drive_cle = np.arange(t0_cle, tf_cle, dt_drive_cle)

x0_drive_cle, y0_drive_cle, z0_drive_cle = (1.0, 1.0, 1.0)


def lorenz_drive_cle(t, state):
    x, y, z = state
    dx = sigma_cle * (y - x)
    dy = x * (rho_cle - z) - y
    dz = x * y - beta_cle * z
    return [dx, dy, dz]


def jacobian_response_xdrive_cle(x_value):
    return np.array([
        [-1.0,           -x_value],
        [ x_value,       -beta_cle]
    ])


def response_variational_xdrive_cle(t, state, x_drive, t_drive):
    y_r, z_r, dy1, dz1, dy2, dz2 = state
    x = np.interp(t, t_drive, x_drive)

    dy_r_dt = x * (rho_cle - z_r) - y_r
    dz_r_dt = x * y_r - beta_cle * z_r

    J = jacobian_response_xdrive_cle(x)

    delta1 = np.array([dy1, dz1])
    delta2 = np.array([dy2, dz2])

    d_delta1_dt = J @ delta1
    d_delta2_dt = J @ delta2

    return np.concatenate([[dy_r_dt, dz_r_dt], d_delta1_dt, d_delta2_dt])


def gram_schmidt_2d(v1, v2):
    norm1 = np.linalg.norm(v1)
    if norm1 < 1e-15:
        norm1 = 1e-15
    u1 = v1 / norm1
    proj = np.dot(v2, u1) * u1
    v2_orth = v2 - proj
    norm2 = np.linalg.norm(v2_orth)
    if norm2 < 1e-15:
        norm2 = 1e-15
    u2 = v2_orth / norm2
    return u1, u2, norm1, norm2


print("\n" + "=" * 80)
print("PARTE 2 – EXPOENTES DE LYAPUNOV CONDICIONAIS (CLEs) – X-DRIVING (rho = 60)")
print("=" * 80)

print("[CLE 1/3] Integrando drive (para CLEs)...")
sol_drive_cle = solve_ivp(
    lorenz_drive_cle,
    [t0_cle, tf_cle],
    [x0_drive_cle, y0_drive_cle, z0_drive_cle],
    method="DOP853",
    t_eval=t_eval_drive_cle,
    rtol=1e-10,
    atol=1e-13
)
x_drive_cle = sol_drive_cle.y[0]
t_drive_cle = sol_drive_cle.t

print("[CLE 2/3] Integrando subsistema (y', z') + variacionais...")
y0_resp_cle = 1.0
z0_resp_cle = 1.0
delta1_0 = np.array([1.0, 0.0])
delta2_0 = np.array([0.0, 1.0])

state_current = np.array([
    y0_resp_cle,
    z0_resp_cle,
    delta1_0[0], delta1_0[1],
    delta2_0[0], delta2_0[1]
])

t_current = t0_cle
sum_log_norms = np.zeros(2)
cle_evol = [[], []]
cle_times = []
transient_passed = False

while t_current < tf_cle - 1e-9:
    t_next = min(t_current + renorm_interval, tf_cle)

    sol_seg = solve_ivp(
        response_variational_xdrive_cle,
        [t_current, t_next],
        state_current,
        args=(x_drive_cle, t_drive_cle),
        method="DOP853",
        t_eval=[t_next],
        rtol=1e-10,
        atol=1e-13
    )

    state_current = sol_seg.y[:, -1]
    _, _, dy1_c, dz1_c, dy2_c, dz2_c = state_current

    delta1 = np.array([dy1_c, dz1_c])
    delta2 = np.array([dy2_c, dz2_c])

    u1, u2, norm1, norm2 = gram_schmidt_2d(delta1, delta2)
    t_current = t_next

    if not transient_passed and t_current >= t_transient_cle:
        transient_passed = True
        sum_log_norms[:] = 0.0
        cle_evol = [[], []]
        cle_times = []

    if transient_passed:
        sum_log_norms[0] += np.log(norm1)
        sum_log_norms[1] += np.log(norm2)

        t_eff = t_current - t_transient_cle
        cle1_inst = sum_log_norms[0] / (t_eff + 1e-15)
        cle2_inst = sum_log_norms[1] / (t_eff + 1e-15)

        cle_evol[0].append(cle1_inst)
        cle_evol[1].append(cle2_inst)
        cle_times.append(t_eff)

    state_current[2:4] = u1
    state_current[4:6] = u2

total_time_eff = tf_cle - t_transient_cle
cle1 = sum_log_norms[0] / total_time_eff
cle2 = sum_log_norms[1] / total_time_eff

print("[CLE 3/3] Resultados finais:")
print(f"  CLE1 (maior): {cle1:+.6f}")
print(f"  CLE2 (menor): {cle2:+.6f}")
print(f"  Soma CLE1 + CLE2: {cle1 + cle2:+.6f}")

fig_cle, ax_cle = plt.subplots(figsize=(10, 5))
ax_cle.plot(cle_times, cle_evol[0], label="CLE1 (maior)", lw=1.5)
ax_cle.plot(cle_times, cle_evol[1], label="CLE2 (menor)", lw=1.5, ls="--")
ax_cle.set_xlabel("Tempo efetivo (t - t_transiente)")
ax_cle.set_ylabel("CLE(t)")
ax_cle.set_title("Convergência dos CLEs – Lorenz x-driving (rho = 60)")
ax_cle.grid(True, alpha=0.3)
ax_cle.legend()
plt.tight_layout()
plt.savefig("lorenz_cle_evolution.png", dpi=300)
plt.show()

print("\nScript finalizado.")