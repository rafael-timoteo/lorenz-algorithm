#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparação de atratores do sistema de Lorenz com mesma escala em X, Y e Z.

- Atrator caótico (sem controle):   t < 215 s
- Atrator pós-controle:             t >= 215 s
  (controle ativado em t_control = 200 s, canal x)

Saídas em ./out:
  - attractor_uncontrolled_t_lt_215_same_scale.png
  - attractor_controlled_t_ge_215_same_scale.png
  - compare_summary.json
"""

import os
import json
import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ----------------------------
# Dinâmica de Lorenz
# ----------------------------

def lorenz_rhs(state, sigma, rho, beta, u=0.0):
    x, y, z = state
    dx = sigma * (y - x) + u
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)


# ----------------------------------
# Lei de controle saturada em x(t)
# ----------------------------------

def control_u_x(t, x, x_eq, eta, eps, t_control):
    """Controle só atua para t >= t_control; antes disso, u=0."""
    if t < t_control:
        return 0.0
    return -eta * np.tanh((x - x_eq) / float(eps))


# ----------------------------------
# Integração RK4 com f(t, s)
# ----------------------------------

def rk4_step(f, t, s, dt, *fargs):
    k1 = f(t, s, *fargs)
    k2 = f(t + 0.5*dt, s + 0.5*dt*k1, *fargs)
    k3 = f(t + 0.5*dt, s + 0.5*dt*k2, *fargs)
    k4 = f(t + dt,     s + dt*k3,     *fargs)
    return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)


def integrate(f, t0, s0, t_end, dt, *fargs):
    n = int(math.ceil((t_end - t0)/dt)) + 1
    t = np.empty(n, dtype=float)
    x = np.empty((n, 3), dtype=float)
    t[0], x[0] = t0, np.array(s0, dtype=float)
    for i in range(1, n):
        t[i] = t[i-1] + dt
        x[i] = rk4_step(f, t[i-1], x[i-1], dt, *fargs)
    return t, x


# -----------------------------------------
# Contagem aproximada de “órbitas”
# -----------------------------------------

def count_peaks_z(z):
    """Conta picos locais em z(t) (transição de derivada + para -)."""
    z = np.asarray(z)
    dz = np.diff(z)
    s = np.sign(dz)
    ds = np.diff(s)
    return int(np.sum(ds == -2))


# -----------------------------------------
# Cálculo de limites e mesma escala 3D
# -----------------------------------------

def limits_from_uncontrolled(traj_unc):
    """Limites de x,y,z obtidos do atrator sem controle."""
    x, y, z = traj_unc[:, 0], traj_unc[:, 1], traj_unc[:, 2]
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    zmin, zmax = np.min(z), np.max(z)

    def pad(a, b, frac=0.05):
        r = b - a
        return a - frac*r, b + frac*r

    return pad(xmin, xmax), pad(ymin, ymax), pad(zmin, zmax)


def apply_limits_equal_scale(ax, xlim, ylim, zlim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    # mesmíssima escala visual
    xr = xlim[1] - xlim[0]
    yr = ylim[1] - ylim[0]
    zr = zlim[1] - zlim[0]
    ax.set_box_aspect((xr, yr, zr))


def plot_attractor_3d(traj, title, path, xlim, ylim, zlim):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1.0)
    apply_limits_equal_scale(ax, xlim, ylim, zlim)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -----------------------------
# Programa principal / CLI
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    # parâmetros do Lorenz
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--rho",   type=float, default=28.0)
    ap.add_argument("--beta",  type=float, default=8.0/3.0)
    # integração
    ap.add_argument("--dt",        type=float, default=0.01)
    ap.add_argument("--t_end_unc", type=float, default=215.0,
                    help="Tempo final da simulação sem controle (t < 215).")
    ap.add_argument("--t_end_ctl", type=float, default=400.0,
                    help="Tempo final da simulação com controle.")
    # controle
    ap.add_argument("--t_control", type=float, default=200.0,
                    help="Instante em que o controle é ativado na simulação controlada.")
    ap.add_argument("--eta",       type=float, default=2.0)
    ap.add_argument("--epsilon",   type=float, default=0.1)
    # comparação
    ap.add_argument("--T_un",       type=float, default=215.0,
                    help="Janela de caos: t < T_un.")
    ap.add_argument("--T_ctrl_min", type=float, default=215.0,
                    help="Janela pós-controle: t >= T_ctrl_min.")
    # condição inicial
    ap.add_argument("--x0", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    # saída
    ap.add_argument("--out", type=str, default="out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ponto de equilíbrio (ρ=28)
    x_e = math.sqrt(args.beta * (args.rho - 1.0))
    y_e = x_e
    z_e = args.rho - 1.0

    # ----------------------------
    # 1) Caso sem controle
    # ----------------------------
    def f_unc(t, s, sigma, rho, beta):
        return lorenz_rhs(s, sigma, rho, beta, u=0.0)

    t_unc, traj_unc_full = integrate(
        f_unc, 0.0, args.x0, args.t_end_unc, args.dt,
        args.sigma, args.rho, args.beta
    )
    mask_unc = t_unc < args.T_un
    traj_unc = traj_unc_full[mask_unc]

    # ----------------------------
    # 2) Caso com controle
    # ----------------------------
    def f_ctl(t, s, sigma, rho, beta, x_eq, eta, eps, t_c):
        u = control_u_x(t, s[0], x_eq, eta, eps, t_c)
        return lorenz_rhs(s, sigma, rho, beta, u=u)

    t_ctl, traj_ctl_full = integrate(
        f_ctl, 0.0, args.x0, args.t_end_ctl, args.dt,
        args.sigma, args.rho, args.beta,
        x_e, args.eta, args.epsilon, args.t_control
    )
    mask_ctl = t_ctl >= args.T_ctrl_min
    traj_ctl = traj_ctl_full[mask_ctl]

    # ----------------------------
    # 3) Mesma escala em X, Y, Z
    # ----------------------------
    xlim, ylim, zlim = limits_from_uncontrolled(traj_unc)

    # atrator caótico
    path_unc = os.path.join(
        args.out, "attractor_uncontrolled_t_lt_215_same_scale.png"
    )
    plot_attractor_3d(
        traj_unc,
        f"Atrator — caos não controlado (t < {args.T_un:.0f}s)",
        path_unc, xlim, ylim, zlim
    )

    # atrator pós-controle
    path_ctl = os.path.join(
        args.out, "attractor_controlled_t_ge_215_same_scale.png"
    )
    plot_attractor_3d(
        traj_ctl,
        f"Atrator — pós-controle (t ≥ {args.T_ctrl_min:.0f}s, ctrl em t={args.t_control:.0f}s)",
        path_ctl, xlim, ylim, zlim
    )

    # ----------------------------
    # 4) Resumo / nº de órbitas
    # ----------------------------
    orbits_unc = count_peaks_z(traj_unc[:, 2])
    orbits_ctl = count_peaks_z(traj_ctl[:, 2])

    summary = {
        "windows": {
            "uncontrolled": {
                "t_window": f"t < {args.T_un}",
                "points": int(traj_unc.shape[0]),
            },
            "controlled": {
                "t_window": f"t >= {args.T_ctrl_min}",
                "points": int(traj_ctl.shape[0]),
            },
        },
        "orbit_estimates": {
            "z_peaks_uncontrolled": int(orbits_unc),
            "z_peaks_controlled": int(orbits_ctl),
        },
        "params": {
            "sigma": args.sigma,
            "rho": args.rho,
            "beta": args.beta,
            "dt": args.dt,
            "t_end_unc": args.t_end_unc,
            "t_end_ctl": args.t_end_ctl,
            "t_control": args.t_control,
            "eta": args.eta,
            "epsilon": args.epsilon,
            "x_eq": x_e,
            "y_eq": y_e,
            "z_eq": z_e,
        },
    }

    with open(os.path.join(args.out, "compare_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Resumo da comparação:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Figuras salvas em: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()