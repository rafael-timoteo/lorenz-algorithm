#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Box-counting (ρ=10, regime não-caótico)

Versão afinada para ρ=10 no sistema de Lorenz (σ=10, β=8/3 por padrão).
Para este parâmetro, a dinâmica converge para pontos de equilíbrio estáveis;
assim, a "geometria" relevante está no transiente (espirais em direção ao
atrator). Para estimar a dimensão de caixa dessa curva 3D, o script:

- Usa burn=0 (mantém o transiente) e recorta a cauda muito próxima do equilíbrio
  via parâmetros `--keep-head` e `--r-quantile-min`;
- Seleciona automaticamente uma janela-plateau com alvo D≈1.0;
- Utiliza travessia de voxels (segmentos) + jitters do grid com mediana;
- Evita escalas ruins: descarta ε muito grandes (saturação) e muito pequenas
  (subamostragem), baseando-se no passo espacial mediano.

Saídas:
- CSV: results/lorenz_box_rho_10.csv
- PNG (se --plot-log): results/lorenz_box_rho_10.png

Uso rápido:
  python boxcounting_rho10.py --plot-log
Mais robusto (mais lento):
  python boxcounting_rho10.py --jitters 7 --jitter-steps 12 --plot-log
Ajustes úteis:
  python boxcounting_rho10.py --keep-head 0.75 --r-quantile-min 0.15 --plot-log
"""
from __future__ import annotations
import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

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


def random_rotation_matrix() -> np.ndarray:
    # Rotação 3D aleatória via quaternions
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
# Pré-processamento: recorte do transiente tardio (quase colapsado)
# ==========================
def clip_transient(points: np.ndarray, keep_head: float, r_quantile_min: float) -> np.ndarray:
    """Mantém a parte inicial (head) e remove pontos muito próximos do equilíbrio.
    - keep_head: fração inicial da trajetória a manter (0<keep_head≤1)
    - r_quantile_min: remove pontos com ||x|| abaixo deste quantil (0–1)
    """
    n = len(points)
    if n == 0:
        return points
    head_max = max(2, int(n * max(0.05, min(1.0, keep_head))))
    pts_head = points[:head_max]
    r = np.linalg.norm(pts_head, axis=1)
    thr = np.quantile(r, max(0.0, min(1.0, r_quantile_min)))
    mask = r >= thr
    kept = pts_head[mask]
    if len(kept) < 10:
        kept = pts_head
    return kept

# ==========================
# Box counting (voxel traversal)
# ==========================
@dataclass
class BoxCountResult:
    eps: np.ndarray
    N: np.ndarray
    logs: Tuple[np.ndarray, np.ndarray]  # (log_eps, log_N)
    local_slopes: np.ndarray
    D_est: float
    k_star: int


def voxel_traversal_count(points: np.ndarray, eps: float, origin: Optional[np.ndarray] = None) -> int:
    if origin is None:
        origin = points.min(axis=0) - eps
    inv_eps = 1.0 / eps
    visited: set[tuple[int, int, int]] = set()

    def cell_of(p):
        idx = np.floor((p - origin) * inv_eps).astype(int)
        return (int(idx[0]), int(idx[1]), int(idx[2]))

    visited.add(cell_of(points[0]))

    for a, b in zip(points[:-1], points[1:]):
        da = (a - origin) * inv_eps
        db = (b - origin) * inv_eps
        p = da.copy(); q = db.copy()
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
                i += step_x; tx += tdx
            elif ty <= tx and ty <= tz:
                j += step_y; ty += tdy
            else:
                k += step_z; tz += tdz
            visited.add((i, j, k)); steps += 1

    return len(visited)


def box_count_over_scales(points: np.ndarray, epsilons: Sequence[float], jitter_steps: int = 0, jitters: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Conta N(ε) em várias escalas. Se jitters>1, usa mediana sobre origens deslocadas."""
    Ns = []
    base_origin = points.min(axis=0) - (max(epsilons) if len(epsilons) else 1.0)
    for idx, eps in enumerate(epsilons):
        counts = []
        for j in range(jitters):
            if jitter_steps > 0:
                shift = (np.random.rand(3) * jitter_steps - (jitter_steps // 2)) * eps
            else:
                shift = np.zeros(3)
            origin = base_origin + shift
            c = voxel_traversal_count(points, eps=eps, origin=origin)
            counts.append(c)
        c_med = int(np.median(counts))
        Ns.append(c_med)
        if jitters > 1:
            log(f"  [box] eps[{idx}]={eps:.6g} -> median N={c_med} (over {jitters} jitters)")
        else:
            log(f"  [box] eps[{idx}]={eps:.6g} -> N={c_med}")
    return np.asarray(epsilons, dtype=float), np.asarray(Ns, dtype=float)


def local_slopes(log_eps: np.ndarray, log_N: np.ndarray) -> np.ndarray:
    dlogN = np.diff(log_N); dlogE = np.diff(log_eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        slopes = dlogN / dlogE
    return -slopes

# ==========================
# Escalas úteis
# ==========================
def median_step(points: np.ndarray) -> float:
    deltas = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return float(np.median(deltas))


def useful_eps_mask(eps: np.ndarray, L: float, step_med: float) -> np.ndarray:
    eps_min = 2.0 * step_med
    eps_max = L / 6.0
    return (eps >= eps_min) & (eps <= eps_max)

# ==========================
# Janela-plateau (alvo ≈ 1.0)
# ==========================
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    pad_left = w // 2
    pad_right = len(x) - len(c) - pad_left
    return np.pad(c, (pad_left, pad_right), mode="edge")


def pick_plateau_window(s: np.ndarray, target_D: float = 1.0, width_min: int = 4, tol: float = 0.15) -> Tuple[int, int]:
    if len(s) < width_min:
        return 0, len(s)
    s_smooth = moving_average(s, 3)
    tmin, tmax = target_D - tol, target_D + tol
    best_i0, best_i1 = 0, 0
    cur_i0 = None
    for i, val in enumerate(s_smooth):
        if tmin <= val <= tmax:
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
    mid = len(s) // 2
    i0 = max(0, mid - width_min // 2)
    i1 = min(len(s), i0 + width_min)
    return i0, i1

# ==========================
# Pipeline
# ==========================
@dataclass
class FitResult(BoxCountResult):
    i0: int
    i1: int


def run_pipeline(
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
    jitter_steps: int,
    jitters: int,
    thin: int,
    do_whiten: bool,
    keep_head: float,
    r_quantile_min: float,
    force_mid_window: bool,
) -> FitResult:

    log("== Integração do atrator de Lorenz ==")
    log(f"sigma={sigma}, beta={beta}, rho={rho}")
    log(f"dt={dt}, tmax={tmax}, burn={burn}")
    log(f"x0={tuple(x0)} | thin={thin}")

    total_steps = int((burn + tmax) / dt)
    burn_steps = int(burn / dt)

    traj = rk4_integrate(x0=x0, dt=dt, n_steps=total_steps, sigma=sigma, beta=beta, rho=rho, thin=1)
    pts = traj[burn_steps::max(1, thin)]

    # Recorte do transiente tardio (quase colapsado no equilíbrio)
    pts = clip_transient(pts, keep_head=keep_head, r_quantile_min=r_quantile_min)
    log(f"Trajetória após clip: {len(pts)} pontos")

    if do_whiten:
        log("Whitening…")
        pts = whiten(pts)

    if rot:
        log("Aplicando rotação aleatória…")
        R = random_rotation_matrix(); pts = pts @ R.T

    pmin = pts.min(axis=0); pmax = pts.max(axis=0)
    L = float(np.linalg.norm(pmax - pmin, ord=np.inf))
    log(f"Tamanho de caixa (∞-norm): L={L:.6g}")
    if not np.isfinite(L) or L <= 1e-12:
        log("Degenerado (L≈0). D=0.")
        ks = np.linspace(k_min, k_max, n_scales, dtype=float)
        eps = 1.0 / (2.0 ** ks)
        N = np.ones_like(eps)
        log_eps = np.log(eps); log_N = np.log(N)
        slopes = np.zeros_like(eps[:-1])
        return FitResult(eps=eps, N=N, logs=(log_eps, log_N), local_slopes=slopes, D_est=0.0, k_star=0, i0=0, i1=1)

    ks = np.linspace(k_min, k_max, n_scales, dtype=float)
    eps_all = L / (2.0 ** ks)

    step_med = median_step(pts)
    mask = useful_eps_mask(eps_all, L, step_med)
    eps = eps_all[mask]
    if len(eps) < 4:
        log("Poucas escalas após recorte; usando todas as escalas como fallback.")
        eps = eps_all

    log("== Box counting em múltiplas escalas ==")
    eps, N = box_count_over_scales(pts, eps, jitter_steps=jitter_steps, jitters=jitters)

    log("== Ajuste log-log e inclinações locais ==")
    log_eps = np.log(eps)
    log_N = np.log(N + 1e-12)
    slopes = local_slopes(log_eps, log_N)

    if force_mid_window:
        mid = len(slopes) // 2
        i0 = max(0, mid - 2)
        i1 = min(len(slopes), i0 + 5)
        log(f"Janela central forçada: [{i0}, {i1})")
    else:
        i0, i1 = pick_plateau_window(slopes, target_D=1.0, width_min=4, tol=0.15)
        log(f"Janela-plateau escolhida: [{i0}, {i1}) (len={i1-i0}) alvo≈1.0")

    j = slice(i0, i1 + 1)
    x_fit = log_eps[j]; y_fit = log_N[j]
    if len(x_fit) >= 2:
        b, a = np.polyfit(x_fit, y_fit, 1)
        D_est = float(-b)
    else:
        D_est = float(np.nan)

    k_star = int(i0)
    return FitResult(eps=eps, N=N, logs=(log_eps, log_N), local_slopes=slopes, D_est=D_est, k_star=k_star, i0=i0, i1=i1)

# ==========================
# Visualização
# ==========================
def plot_fischer_style(res: FitResult, title: str = "", annotate: bool = True, savepath: Optional[Path] = None) -> None:
    if plt is None:
        log("matplotlib indisponível – sem gráfico."); return
    fig, axN = plt.subplots(figsize=(8, 5))
    axS = axN.twinx()

    eps = np.exp(res.logs[0]); N = np.exp(res.logs[1])

    axN.set_xscale("log"); axN.set_yscale("log")
    axN.plot(eps, N, marker="o", linestyle=":", label="N(ε)")
    axN.set_xlabel("ε"); axN.set_ylabel("N(ε)")

    xmid = 0.5 * (eps[1:] + eps[:-1])
    axS.plot(xmid, res.local_slopes, marker="o", linestyle="", label="Local Slope")
    axS.set_ylabel("Local Slope")

    if res.i1 > res.i0:
        xm0 = eps[res.i0]; xm1 = eps[res.i1]
        axN.axvspan(xm1, xm0, alpha=0.08, color="gray")

    if annotate and np.isfinite(res.D_est):
        axS.annotate(f"D≈{res.D_est:.3f}", xy=(xmid[res.k_star], res.local_slopes[res.k_star]),
                     xytext=(xmid[res.k_star] * 1.4, res.local_slopes[res.k_star] + 0.25),
                     arrowprops=dict(arrowstyle="->", color="black"), color="black")

    if title: axN.set_title(title)
    h1, l1 = axN.get_legend_handles_labels(); h2, l2 = axS.get_legend_handles_labels()
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
# CLI
# ==========================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Box-Counting 3D para Lorenz — versão ajustada para ρ=10 (não-caótico)")

    # Parâmetros do sistema
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--sigma", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=8.0 / 3.0)

    # Integração: passo pequeno, sem burn, longa o bastante para formar a espiral
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--tmax", type=float, default=180.0)
    p.add_argument("--burn", type=float, default=0.0)
    p.add_argument("--x0", type=float, nargs=3, default=(0.0, 1.0, 1.05))

    # Escalas (ε = L/2^k)
    p.add_argument("--k-min", type=int, default=1)
    p.add_argument("--k-max", type=int, default=14)
    p.add_argument("--n-scales", type=int, default=30)

    # Amostragem/robustez
    p.add_argument("--thin", type=int, default=1, help="subamostragem temporal (1=nenhum)")
    p.add_argument("--jitter-steps", type=int, default=10, help="desloca o grid em inteiros * ε por jitter")
    p.add_argument("--jitters", type=int, default=5, help="repetições por ε; usa mediana")
    p.add_argument("--rot", action="store_true", help="aplica rotação aleatória")
    p.add_argument("--whiten", action="store_true", help="centraliza/normaliza a nuvem antes da contagem")

    # Recorte do transiente tardio
    p.add_argument("--keep-head", type=float, default=0.75, help="fração inicial da trajetória a manter (0<..≤1)")
    p.add_argument("--r-quantile-min", type=float, default=0.10, help="remove pontos com ||x|| abaixo deste quantil (0–1)")

    # Janela de ajuste
    p.add_argument("--force-mid-window", action="store_true", help="ignora detecção de plateau e usa janela central")

    # Saída
    p.add_argument("--plot-log", action="store_true", help="gera o gráfico N(ε)+Local Slope e destaca a janela")
    p.add_argument("--outdir", type=str, default="results", help="diretório de saída para CSV/PNG")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    outdir = Path(args.outdir)

    res = run_pipeline(
        rho=args.rho,
        sigma=args.sigma,
        beta=args.beta,
        dt=args.dt,
        tmax=args.tmax,
        burn=args.burn,
        x0=args.x0,
        k_min=args.k_min,
        k_max=args.k_max,
        n_scales=args.n_scales,
        rot=bool(args.rot),
        jitter_steps=args.jitter_steps,
        jitters=args.jitters,
        thin=args.thin,
        do_whiten=bool(args.whiten),
        keep_head=args.keep_head,
        r_quantile_min=args.r_quantile_min,
        force_mid_window=bool(args.force_mid_window),
    )

    outdir.mkdir(parents=True, exist_ok=True)

    csv_path = outdir / f"lorenz_box_rho_{args.rho:g}.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("eps,N,log_eps,log_N\n")
        for e, n, le, ln in zip(res.eps, res.N, res.logs[0], res.logs[1]):
            f.write(f"{e:.12g},{int(n)},{le:.12g},{ln:.12g}\n")
    log(f"CSV salvo em: {csv_path}")

    if args.plot_log:
        fig_path = outdir / f"lorenz_box_rho_{args.rho:g}.png"
        title = f"ρ = {args.rho:g}  |  D ≈ {res.D_est:.3f}  |  janela [{res.i0},{res.i1})"
        plot_fischer_style(res, title=title, savepath=fig_path)

    log("\n==============================")
    log(" Resultado do ajuste (box counting 3D) ")
    log("==============================")
    log(f"Dimensão (D): {res.D_est:.6f}")
    log(f"Janela usada: [{res.i0}, {res.i1})  (k*={res.k_star})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())