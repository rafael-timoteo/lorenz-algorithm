#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Box-counting (ρ=5, ajustado)

Versão especializada do algoritmo de contagem de caixas para o sistema de Lorenz,
com parâmetros e lógica de ajuste pensados para o regime não-caótico (ρ=5).

O objetivo aqui é estimar corretamente D≈1 para a trajetória em espiral que
converge ao ponto de equilíbrio estável — usando janelas de ajuste estáveis
em escalas intermediárias, evitando as escalas muito grandes (saturação) e
as muito pequenas (colapso próximo ao ponto fixo).

Principais ajustes em relação à versão genérica:
- Parâmetros padrão voltados para ρ=5 (dt pequeno, muita amostragem, jitter e
  mais escalas).
- Seleção automática de uma "janela-plateau" onde o slope local ~ 1.0.
- Regressão linear apenas nessa janela para reduzir viés.
- Mantido o "voxel traversal" para cobrir segmentos (contagem mais fiel da curva).

Uso rápido:
  python boxcounting_rho5.py

Opções úteis:
  python boxcounting_rho5.py --plot-log  # salva gráfico log–log com slopes
  python boxcounting_rho5.py --tmax 120 --thin 1  # mais pontos (mais lento)

Saídas:
- CSV em results/lorenz_box_rho_5.csv
- PNG do gráfico (se --plot-log) em results/lorenz_box_rho_5.png
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
        # barra de progresso leve
        if (i + 1) % max(1, n_steps // 10) == 0:
            log(f"[RK4] {int(100*(i+1)/n_steps)}% concluído…")
    return np.vstack(traj)


def random_rotation_matrix() -> np.ndarray:
    # Rotação 3D aleatória (quaternions) — mantém topologia e dimensão.
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
# Box counting (voxel traversal)
# ==========================
@dataclass
class BoxCountResult:
    eps: np.ndarray
    N: np.ndarray
    logs: Tuple[np.ndarray, np.ndarray]  # (log_eps, log_N)
    local_slopes: np.ndarray
    D_est: float
    k_star: int  # índice do início da janela escolhida


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
# Seleção da janela de ajuste estável (plateau ~1.0)
# ==========================
def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < w:
        return x.copy()
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    # alinhar tamanhos (preenche extremidades com bordas)
    pad_left = w // 2
    pad_right = len(x) - len(c) - pad_left
    return np.pad(c, (pad_left, pad_right), mode="edge")


def pick_plateau_window(s: np.ndarray, width_min: int = 3, tol: float = 0.15) -> Tuple[int, int]:
    """Escolhe janela [i0, i1) onde o slope médio fica em ~1±tol. Fallback: janela central.
    s é a série de slopes locais (len = n_scales-1).
    """
    if len(s) < width_min:
        return 0, len(s)
    s_smooth = moving_average(s, 3)
    target_min, target_max = 1.0 - tol, 1.0 + tol

    best_i0, best_i1 = 0, 0
    cur_i0 = None
    for i, val in enumerate(s_smooth):
        if target_min <= val <= target_max:
            if cur_i0 is None:
                cur_i0 = i
        else:
            if cur_i0 is not None and (i - cur_i0) >= width_min:
                if (i - cur_i0) > (best_i1 - best_i0):
                    best_i0, best_i1 = cur_i0, i
            cur_i0 = None
    # fim da série
    if cur_i0 is not None and (len(s_smooth) - cur_i0) >= width_min:
        if (len(s_smooth) - cur_i0) > (best_i1 - best_i0):
            best_i0, best_i1 = cur_i0, len(s_smooth)

    if best_i1 > best_i0:
        return best_i0, best_i1
    # fallback: janela central de largura width_min
    mid = len(s) // 2
    i0 = max(0, mid - width_min // 2)
    i1 = min(len(s), i0 + width_min)
    return i0, i1


# ==========================
# Pipeline principal (ρ=5 por padrão)
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
    jitter: int,
    thin: int,
    do_whiten: bool,
    force_mid_window: bool,
) -> FitResult:

    log("== Integração do atrator de Lorenz ==")
    log(f"sigma={sigma}, beta={beta}, rho={rho}")
    log(f"dt={dt}, tmax={tmax}, burn={burn}")
    log(f"x0={tuple(x0)} | thin={thin}")

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
        log("Trajetória colapsou para um ponto (L≈0). Usando solução degenerada: D=0.")
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

    # Escolha da janela de ajuste
    if force_mid_window:
        # janela central de fallback
        mid = len(slopes) // 2
        i0 = max(0, mid - 2)
        i1 = min(len(slopes), i0 + 5)
        log(f"Janela forçada (central): [{i0}, {i1})")
    else:
        i0, i1 = pick_plateau_window(slopes, width_min=4, tol=0.15)
        log(f"Janela-plateau escolhida: [{i0}, {i1}) (len={i1-i0})")

    # Ajuste linear na janela escolhida: log N = a + b * log eps  =>  D = -b
    j = slice(i0, i1 + 1)  # +1 pois slopes tem n-1 pontos; regressão usa nós (eps/logs) correspondentes
    x_fit = log_eps[j]
    y_fit = log_N[j]
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
        log("matplotlib indisponível – não será possível exibir/salvar o gráfico.")
        return
    fig, axN = plt.subplots(figsize=(8, 5))
    axS = axN.twinx()

    eps = np.exp(res.logs[0])
    N = np.exp(res.logs[1])

    # N(eps): azul
    axN.set_xscale("log")
    axN.set_yscale("log")
    axN.plot(eps, N, marker="o", linestyle=":", color="tab:blue", label="N(ε)")
    axN.set_xlabel("ε")
    axN.set_ylabel("N(ε)", color="tab:blue")

    # Slopes locais: laranja (somente marcadores)
    xmid = 0.5 * (eps[1:] + eps[:-1])
    axS.plot(xmid, res.local_slopes, marker="o", linestyle="", color="tab:orange", label="Local Slope")
    axS.set_ylabel("Local Slope", color="tab:orange")

    # Destaca janela usada
    if res.i1 > res.i0:
        xm0 = eps[res.i0]
        xm1 = eps[res.i1]
        axN.axvspan(xm1, xm0, alpha=0.08, color="gray")  # atenção: eixo é log, mas axvspan aceita em coordenadas de dados

    if annotate and np.isfinite(res.D_est):
        axS.annotate(f"D≈{res.D_est:.3f}", xy=(xmid[res.k_star], res.local_slopes[res.k_star]),
                     xytext=(xmid[res.k_star] * 1.6, res.local_slopes[res.k_star] + 0.25),
                     arrowprops=dict(arrowstyle="->", color="black"), color="black")

    if title:
        axN.set_title(title)

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
# CLI
# ==========================
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Box-Counting 3D para Lorenz — versão ajustada para ρ=5")
    # Padrões afinados para ρ=5
    p.add_argument("--rho", type=float, default=5.0)
    p.add_argument("--sigma", type=float, default=10.0)
    p.add_argument("--beta", type=float, default=8.0 / 3.0)

    # Integração: dt pequeno, muito tempo para amostrar muitas voltas pré-colapso
    p.add_argument("--dt", type=float, default=1e-4)
    p.add_argument("--tmax", type=float, default=150.0)
    p.add_argument("--burn", type=float, default=0.0, help="mantenha 0 para não descartar a espiral pré-colapso")
    p.add_argument("--x0", type=float, nargs=3, default=(0.0, 1.0, 1.05))

    # Escalas: faixa mais ampla e densa
    p.add_argument("--k-min", type=int, default=1, help="menor expoente k (eps = L/2^k)")
    p.add_argument("--k-max", type=int, default=14, help="maior expoente k (eps = L/2^k)")
    p.add_argument("--n-scales", type=int, default=30, help="número de amostras entre k_min e k_max")

    # Opções extra
    p.add_argument("--thin", type=int, default=2, help="subamostragem da trajetória (1=nenhum)")
    p.add_argument("--jitter", type=int, default=12, help="desloca o grid em inteiros multiplicados por eps")
    p.add_argument("--rot", action="store_true", help="aplica uma rotação aleatória (default: não)")
    p.add_argument("--whiten", action="store_true", help="centraliza/normaliza a nuvem antes da contagem")
    p.add_argument("--force-mid-window", action="store_true", help="ignora detecção de plateau e usa janela central")
    p.add_argument("--plot-log", action="store_true", help="gera o gráfico N(ε) + Local Slope e destaca a janela")
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
        jitter=args.jitter,
        thin=args.thin,
        do_whiten=bool(args.whiten),
        force_mid_window=bool(args.force_mid_window),
    )

    outdir.mkdir(parents=True, exist_ok=True)

    # CSV individual
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