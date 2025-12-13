
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def lorenz_rhs(state, sigma, rho, beta, u=0.0):
    x, y, z = state
    dx = sigma * (y - x) + u
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz], dtype=float)

def rk4_step(f, state, dt, *f_args, **f_kwargs):
    k1 = f(state, *f_args, **f_kwargs)
    k2 = f(state + 0.5 * dt * k1, *f_args, **f_kwargs)
    k3 = f(state + 0.5 * dt * k2, *f_args, **f_kwargs)
    k4 = f(state + dt * k3, *f_args, **f_kwargs)
    return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

def simulate_tdfc_with_activation(x0, sigma, rho, beta, K, tau, dt, steps, t_ctrl):
    delay_steps = max(1, int(round(tau / dt)))
    buf = np.full(delay_steps, x0[0], dtype=float)
    head = 0
    state = np.array(x0, dtype=float)
    traj = np.zeros((steps, 3), dtype=float)
    u_hist = np.zeros(steps, dtype=float)
    t = np.arange(steps) * dt
    for i in range(steps):
        x = state[0]
        x_tau = buf[head]
        u = (K * (x_tau - x)) if (t[i] >= t_ctrl) else 0.0
        state = rk4_step(lorenz_rhs, state, dt, sigma, rho, beta, u=u)
        traj[i] = state
        u_hist[i] = u
        buf[head] = state[0]
        head += 1
        if head >= delay_steps:
            head = 0
    return t, traj, u_hist

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=10.0)
    ap.add_argument("--rho", type=float, default=28.0)
    ap.add_argument("--beta", type=float, default=8.0/3.0)
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--tmax", type=float, default=320.0)
    ap.add_argument("--t_ctrl", type=float, default=200.0)
    ap.add_argument("--K", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--x0", type=float, nargs=3, default=[0.0, 1.0, 1.05])
    ap.add_argument("--outdir", type=str, default="out")
    args = ap.parse_args()

    steps = int(round(args.tmax/args.dt))
    t, traj, u = simulate_tdfc_with_activation(args.x0, args.sigma, args.rho, args.beta,
                                               args.K, args.tau, args.dt, steps, args.t_ctrl)

    os.makedirs(args.outdir, exist_ok=True)

    # Left chaotic lobe up to ~215s (include slightly after control)
    mask_left = (t <= 215.0) & (traj[:,0] < 0.0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[mask_left,0], traj[mask_left,1], traj[mask_left,2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("Atrator (lobo esquerdo) — até t ≤ 215s")
    fig.savefig(os.path.join(args.outdir, "attractor_left_until215.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Post-control attractor (t >= t_ctrl)
    mask_post = (t >= args.t_ctrl)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[mask_post,0], traj[mask_post,1], traj[mask_post,2])
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title(f"Atrator pós-controle (t ≥ {args.t_ctrl:.1f}s)")
    fig.savefig(os.path.join(args.outdir, "attractor_post_control.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Reference x(t) with control line
    fig = plt.figure()
    plt.plot(t, traj[:,0], label="x(t)")
    plt.axvline(args.t_ctrl, linestyle="--", label="controle ON")
    plt.xlabel("Tempo"); plt.ylabel("x(t)"); plt.title("x(t) com t_ctrl marcado")
    plt.legend()
    fig.savefig(os.path.join(args.outdir, "xt_with_tctrl.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()
