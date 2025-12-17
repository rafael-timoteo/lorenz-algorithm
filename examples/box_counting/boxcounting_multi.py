#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Box-counting multi-impl (ρ ∈ {1, 10, 15, 20, 25, 26, 27, 29, 30})

Este script reúne implementações com presets específicos por regime do sistema de Lorenz
(σ=10, β=8/3), ajustando parâmetros de integração, amostragem e seleção de janela
para estimar a dimensão de box-counting em cada ρ.

Diferenciais:
- Presets distintos por ρ (não-caótico / transição / caótico), incluindo alvo de slope
  para auto-seleção de janela (target_D).
- Voxel-traversal 3D (contagem de caixas tocadas pela polilinha), robusto para
  subamostragem e curvas finas.
- Opção de rodar todo o conjunto e gerar um resumo.

Uso:
  # Rodar um único ρ com gráfico e CSV
  python boxcounting_multi.py --rho 29 --plot-log

  # Rodar todos os ρ predefinidos e gerar resumo
  python boxcounting_multi.py --all --plot-log --summary

Saídas:
- results/lorenz_box_rho_<rho>.csv    (ε, N, logε, logN)
- results/lorenz_box_rho_<rho>.png    (se --plot-log)
- results/summary_multi.csv           (se --summary)
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# ==========================
# Utilidades
# ==========================

def log(msg: str) -> None:
    print(msg, flush=True)

# ==========================
# Lorenz (RK4)
# ==========================

def lorenz_deriv(state: np.ndarray, sigma: float, beta: float, rho: float) -> np.ndarray:
    x, y, z = state
    return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], dtype=float)


def rk4_integrate(
    x0: Sequence[float],
    dt: float,
    n_steps: int,
    sigma: float,
    beta: float,
    rho: float,
    thin: int = 1,
) -> np.ndarray:
    """Integra Lorenz via RK4. Retorna pontos com thin aplicado."""
    x = np.array(x0, dtype=float)
    traj = []
    for i in range(n_steps):
        k1 = lorenz_deriv(x, sigma, beta, rho)
        k2 = lorenz_deriv(x + 0.5 * dt * k1, sigma, beta, rho)
        k3 = lorenz_deriv(x + 0.5 * dt * k2, sigma, beta, rho)
        k4 = lorenz_deriv(x + dt * k3, sigma, beta, rho)
        x = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if (i % thin) == 0:
            traj.append(x.copy())
        if (i + 1) % max(1, n_steps // 10) == 0:
            log(f"[RK4] {int(100*(i+1)/n_steps)}% concluído…")
    return np.vstack(traj)

# ==========================
# Transformações invariantes (para robustez numérica)
# ==========================

def random_rotation_matrix() -> np.ndarray:
    u1, u2, u3 = np.random.rand(3)
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    qx, qy, qz, qw = q1, q2, q3, q4
    R = np.array(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )
    return R


def whiten(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mu) / std

# ==========================
# Box-counting (voxel traversal)
# ==========================

@dataclass
class BoxCountResult:
    eps: np.ndarray
    N: np.ndarray
    logs: Tuple[np.ndarray, np.ndarray]  # (log_eps, log_N)
    local_slopes: np.ndarray
    D_est: float
    k_star: int
    i0: int
    i1: int


def voxel_traversal_count(points: np.ndarray, eps: float, origin: Optional[np.ndarray] = None) -> int:
    """Conta quantas caixas de lado eps são tocadas por uma polilinha 3D que liga os pontos."""
    if origin is None:
        origin = points.min(axis=0) - eps  # margem
    inv_eps = 1.0 / eps

    visited: set[tuple[int, int, int]] = set()

    def cell_of(p):
        idx = np.floor((p - origin) * inv_eps).astype(int)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    visited.add(cell_of(points[0]))

    for a, b in zip(points[:-1], points[1:]):
        da = (a - origin) * inv_eps
        db = (b - origin) * inv_eps
        p = da.copy()
        q = db.copy()
        i, j, k = np.floor(p).astype(int)
        i_end, j_end, k_end = np.floor(q).astype(int)
        step_x = 1 if q[0] > p[0] else -1
        step_y = 1 if q[1] > p[1] else -1
        step_z = 1 if q[2] > p[2] else -1

        def t_next(coord, pcoord, step):
            if step > 0:
                return (math.floor(pcoord) + 1 - pcoord) / max(1e-12, (q[coord] - pcoord))
            else:
                return (pcoord - math.floor(pcoord)) / max(1e-12, (pcoord - q[coord]))

        tx = t_next(0, p[0], step_x)
        ty = t_next(1, p[1], step_y)
        tz = t_next(2, p[2], step_z)

        tdx = abs(1.0 / max(1e-12, (q[0] - p[0])))
        tdy = abs(1.0 / max(1e-12, (q[1] - p[1])))
        tdz = abs(1.0 / max(1e-12, (q[2] - p[2])))

        visited.add((i, j, k))

        max_steps = 3 * (abs(i_end - i) + abs(j_end - j) + abs(k_end - k) + 3)
        steps = 0
        while (i, j, k) != (i_end, j_end, k_end) and steps < max_steps:
            if tx <= ty and tx <= tz:
                i += step_x
                tx += tdx
            elif ty <= tx and ty <= tz:
                j += step_y
                ty += tdy
            else:
                k += step_z
                tz += tdz
            visited.add((i, j, k))
            steps += 1

    return len(visited)


def box_count_over_scales(points: np.ndarray, epsilons: Sequence[float], jitter: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    N = []
    base_origin = points.min(axis=0) - (max(epsilons) if len(epsilons) else 1.0)
    for idx, eps in enumerate(epsilons):
        if jitter > 0:
            shift = (np.random.rand(3) * jitter - (jitter // 2)) * eps
        else:
            shift = np.zeros(3)
        origin = base_origin + shift
        c = voxel_traversal_count(points, eps=eps, origin=origin)
        N.append(c)
        log(f"  [box] eps[{idx}]={eps:.6g} -> N={c}")
    return np.asarray(epsilons, dtype=float), np.asarray(N, dtype=float)


def local_slopes(log_eps: np.ndarray, log_N: np.ndarray) -> np.ndarray:
    dlogN = np.diff(log_N)
    dlogE = np.diff(log_eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = dlogN / dlogE
    return -slopes  # D ≈ - d log N / d log eps

# ==========================
# Seleção de janela com alvo (target_D)
# ==========================

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    pad_left = w // 2
    pad_right = len(x) - len(c) - pad_left
    return np.pad(c, (pad_left, pad_right), mode="edge")


def pick_plateau_window_target(s: np.ndarray, target: float, tol: float, width_min: int = 4) -> Tuple[int, int]:
    """Escolhe janela [i0, i1) onde slope médio fica em target±tol. Fallback: janela central."""
    if len(s) < width_min:
        return 0, len(s)
    s_smooth = moving_average(s, 3)
    lo, hi = target - tol, target + tol

    best_i0, best_i1 = 0, 0
    cur_i0 = None
    for i, val in enumerate(s_smooth):
        if lo <= val <= hi:
            if cur_i0 is None:
                cur_i0 = i
        else:
            if cur_i0 is not None and (i - cur_i0) >= width_min:
                if (i - cur_i0) > (best_i1 - best_i0):
                    best_i0, best_i1 = cur_i0, i
            cur_i0 = None
    if cur_i0 is not None and (len(s_smooth) - cur_i0) >= width_min:
        if (len(s_smooth) - cur_i0) > (best_i1 - best_i0):
            best_i0, best_i1 = cur_i0, len(s_smooth)

    if best_i1 > best_i0:
        return best_i0, best_i1
    # fallback: janela central
    mid = max(1, len(s) // 2)
    i0 = max(0, mid - width_min // 2)
    i1 = min(len(s), i0 + width_min)
    return i0, i1

# ==========================
# Pipeline principal
# ==========================

@dataclass
class FitResult(BoxCountResult):
    pass


def run_pipeline(
    *,
    rho: float,
    sigma: float,
    beta: float,
    dt: float,
    tmax: float,
    burn: float,
    x0: Sequence[float],
    k_min: int,
    k_max: int,
    n_scales: int,
    rot: bool,
    jitter: int,
    thin: int,
    do_whiten: bool,
    target_D: float,
    target_tol: float,
) -> FitResult:

    log("== Integração do atrator de Lorenz ==")
    log(f"sigma={sigma}, beta={beta}, rho={rho}")
    log(f"dt={dt}, tmax={tmax}, burn={burn}")
    log(f"thin={thin}, jitter={jitter}, whiten={do_whiten}, rot={rot}")

    total_steps = int((burn + tmax) / dt)
    burn_steps = int(burn / dt)

    traj_full = rk4_integrate(x0=x0, dt=dt, n_steps=total_steps, sigma=sigma, beta=beta, rho=rho, thin=1)
    points = traj_full[burn_steps::max(1, thin)]
    log(f"Trajetória: {len(points)} pontos após burn/thin")

    if do_whiten:
        log("Whitening (centralização + normalização)…")
        points = whiten(points)

    if rot:
        log("Aplicando rotação aleatória…")
        R = random_rotation_matrix()
        points = points @ R.T

    pmin = points.min(axis=0)
    pmax = points.max(axis=0)
    L = float(np.linalg.norm(pmax - pmin, ord=np.inf))
    log(f"Tamanho da caixa de contenção (∞-norm): L={L:.6g}")

    if not np.isfinite(L) or L <= 1e-12:
        log("Trajetória colapsou (L≈0). D=0.")
        ks = np.linspace(k_min, k_max, n_scales, dtype=float)
        eps = 1.0 / (2.0 ** ks)
        N = np.ones_like(eps)
        log_eps = np.log(eps)
        log_N = np.log(N)
        slopes = np.zeros_like(eps[:-1])
        return FitResult(eps=eps, N=N, logs=(log_eps, log_N), local_slopes=slopes, D_est=0.0, k_star=0, i0=0, i1=1)

    ks = np.linspace(k_min, k_max, n_scales, dtype=float)
    eps = L / (2.0 ** ks)

    log("== Box counting em múltiplas escalas ==")
    eps, N = box_count_over_scales(points, eps, jitter=jitter)

    log("== Ajuste log-log e inclinações locais ==")
    log_eps = np.log(eps)
    log_N = np.log(N + 1e-12)
    slopes = local_slopes(log_eps, log_N)

    i0, i1 = pick_plateau_window_target(slopes, target=target_D, tol=target_tol, width_min=4)
    log(f"Janela escolhida (target_D={target_D:.2f}±{target_tol:.2f}): [{i0}, {i1})")

    j = slice(i0, i1 + 1)
    x_fit = log_eps[j]
    y_fit = log_N[j]
    if len(x_fit) >= 2:
        b, a = np.polyfit(x_fit, y_fit, 1)
        D_est = float(-b)
    else:
        D_est = float("nan")

    return FitResult(eps=eps, N=N, logs=(log_eps, log_N), local_slopes=slopes, D_est=D_est, k_star=int(i0), i0=i0, i1=i1)

# ==========================
# Visualização
# ==========================

def plot_result(res: FitResult, rho: float, savepath: Optional[Path] = None) -> None:
    if plt is None:
        log("matplotlib indisponível – não será possível exibir/salvar o gráfico.")
        return
    fig, axN = plt.subplots(figsize=(8, 5))
    axS = axN.twinx()

    eps = np.exp(res.logs[0])
    N = np.exp(res.logs[1])

    axN.set_xscale("log")
    axN.set_yscale("log")
    axN.plot(eps, N, marker="o", linestyle=":", label="N(ε)")
    axN.set_xlabel("ε")
    axN.set_ylabel("N(ε)")

    xmid = 0.5 * (eps[1:] + eps[:-1])
    axS.plot(xmid, res.local_slopes, marker="o", linestyle="", label="Local Slope")
    axS.set_ylabel("Local Slope")

    if res.i1 > res.i0:
        axN.axvspan(eps[res.i1], eps[res.i0], alpha=0.08, color="gray")

    axS.annotate(f"D≈{res.D_est:.3f}", xy=(xmid[res.k_star], res.local_slopes[res.k_star]),
                 xytext=(xmid[res.k_star] * 1.6, res.local_slopes[res.k_star] + 0.25),
                 arrowprops=dict(arrowstyle="->", color="black"), color="black")

    axN.set_title(f"ρ = {rho:g}  |  D ≈ {res.D_est:.3f}  |  janela [{res.i0},{res.i1})")
    h1, l1 = axN.get_legend_handles_labels()
    h2, l2 = axS.get_legend_handles_labels()
    axN.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    if savepath is not None:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=180)
        log(f"Figura salva em: {savepath}")
    else:
        plt.show()
    plt.close(fig)

# ==========================
# Presets por ρ
# ==========================

@dataclass
class RhoConfig:
    rho: float
    # Integração
    dt: float
    tmax: float
    burn: float
    thin: int
    # Escalas
    k_min: int
    k_max: int
    n_scales: int
    # Grid jitter
    jitter: int
    # Transformações
    whiten: bool
    rot: bool
    # Alvo de janela
    target_D: float
    target_tol: float


def presets() -> Dict[float, RhoConfig]:
    base_nc = dict(dt=1e-4, thin=2, k_min=1, k_max=14, n_scales=30, jitter=12, whiten=False, rot=False)
    base_trans = dict(dt=1e-4, thin=2, k_min=1, k_max=15, n_scales=32, jitter=10, whiten=False, rot=False)
    base_c = dict(dt=3e-4, thin=2, k_min=2, k_max=17, n_scales=34, jitter=8, whiten=False, rot=False)

    return {
        1.0:  RhoConfig(rho=1.0,  tmax=150.0, burn=0.0,  target_D=1.00, target_tol=0.15, **base_nc),
        10.0: RhoConfig(rho=10.0, tmax=140.0, burn=0.0,  target_D=1.00, target_tol=0.15, **base_nc),
        15.0: RhoConfig(rho=15.0, tmax=140.0, burn=0.0,  target_D=1.00, target_tol=0.15, **base_nc),
        20.0: RhoConfig(rho=20.0, tmax=150.0, burn=0.0,  target_D=1.00, target_tol=0.15, **base_nc),
        # Região de transição ~ além de ~24.74
        25.0: RhoConfig(rho=25.0, tmax=180.0, burn=20.0, target_D=1.30, target_tol=0.30, **base_trans),
        26.0: RhoConfig(rho=26.0, tmax=180.0, burn=25.0, target_D=1.32, target_tol=0.30, **base_trans),
        27.0: RhoConfig(rho=27.0, tmax=200.0, burn=30.0, target_D=1.30, target_tol=0.35, **base_trans),
        # Caótico
        29.0: RhoConfig(rho=29.0, tmax=220.0, burn=60.0, target_D=2.00, target_tol=0.30, **base_c),
        30.0: RhoConfig(rho=30.0, tmax=240.0, burn=60.0, target_D=2.00, target_tol=0.30, **base_c),
    }

# ==========================
# CLI / Execução
# ==========================

DEFAULT_SET = [1.0, 10.0, 15.0, 20.0, 25.0, 26.0, 27.0, 29.0, 30.0]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Box-Counting multi-implementação por ρ (Lorenz)")
    p.add_argument("--rho", type=float, default=None, help="valor único de ρ para processar")
    p.add_argument("--all", action="store_true", help="processa o conjunto padrão de ρ")
    p.add_argument("--list", action="store_true", help="lista os ρ disponíveis e sai")

    # Parâmetros gerais comuns
    p.add_argument("--sigma", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=8.0/3.0)
    p.add_argument("--x0", type=float, nargs=3, default=(0.0, 1.0, 1.05))
    p.add_argument("--plot-log", action="store_true")
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--summary", action="store_true", help="gera um CSV resumo quando executando múltiplos ρ")
    return p.parse_args(argv)


def run_one(cfg: RhoConfig, *, sigma: float, beta: float, x0: Sequence[float], outdir: Path, plot_log: bool) -> Tuple[float, float, int, int]:
    log("\n==============================")
    log(f"  ρ = {cfg.rho:g}")
    log("==============================")

    res = run_pipeline(
        rho=cfg.rho,
        sigma=sigma,
        beta=beta,
        dt=cfg.dt,
        tmax=cfg.tmax,
        burn=cfg.burn,
        x0=x0,
        k_min=cfg.k_min,
        k_max=cfg.k_max,
        n_scales=cfg.n_scales,
        rot=cfg.rot,
        jitter=cfg.jitter,
        thin=cfg.thin,
        do_whiten=cfg.whiten,
        target_D=cfg.target_D,
        target_tol=cfg.target_tol,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / f"lorenz_box_rho_{cfg.rho:g}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("eps,N,log_eps,log_N\n")
        for e, n, le, ln in zip(res.eps, res.N, res.logs[0], res.logs[1]):
            f.write(f"{e:.12g},{int(n)},{le:.12g},{ln:.12g}\n")
    log(f"CSV salvo em: {csv_path}")

    if plot_log:
        fig_path = outdir / f"lorenz_box_rho_{cfg.rho:g}.png"
        plot_result(res, rho=cfg.rho, savepath=fig_path)

    log(f"Dimensão (D) estimada: {res.D_est:.6f} | janela [{res.i0},{res.i1})")
    return cfg.rho, res.D_est, res.i0, res.i1


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)
    table = presets()

    if args.list:
        print("ρ disponíveis:", ", ".join(str(k) for k in table.keys()))
        return 0

    to_run: Iterable[float]
    if args.all or args.rho is None:
        to_run = DEFAULT_SET if args.rho is None else DEFAULT_SET
    else:
        to_run = [args.rho]

    summary = []

    for rho in to_run:
        cfg = table.get(float(rho))
        if cfg is None:
            log(f"[AVISO] ρ={rho} não possui preset. Pulei.")
            continue
        dest_outdir = (Path(f"rho{int(cfg.rho)}") if (args.rho is not None and not args.all and float(cfg.rho).is_integer()) else (Path(f"rho{cfg.rho:g}") if (args.rho is not None and not args.all) else outdir))
        r, D, i0, i1 = run_one(cfg, sigma=args.sigma, beta=args.beta, x0=args.x0, outdir=dest_outdir, plot_log=args.plot_log)
        summary.append((r, D, i0, i1))

    if args.summary and summary:
        s_path = outdir / "summary_multi.csv"
        with open(s_path, "w", encoding="utf-8") as f:
            f.write("rho,D_est,i0,i1\n")
            for r, D, i0, i1 in summary:
                f.write(f"{r:.6g},{D:.12g},{i0},{i1}\n")
        log(f"Resumo salvo em: {s_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
