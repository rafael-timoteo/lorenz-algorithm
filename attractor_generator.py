# attractor_generator.py
import matplotlib.pyplot as plt
import os

def plot_and_save_attractor(time_points, states, sigma, rho, beta, output_folder):
    """
    Plota o atrator de Lorenz em 3D e salva a imagem.
    """
    x_t = states[:, 0]
    y_t = states[:, 1]
    z_t = states[:, 2]

    fig_attractor = plt.figure(figsize=(10, 8))
    ax_attractor = fig_attractor.add_subplot(111, projection='3d')
    ax_attractor.plot(x_t, y_t, z_t, lw=0.5)
    ax_attractor.set_xlabel("X(t)")
    ax_attractor.set_ylabel("Y(t)")
    ax_attractor.set_zlabel("Z(t)")
    
    title = f"Atrator de Lorenz ($\\sigma={sigma}, \\rho={rho}, \\beta={beta:.2f}$)"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    filename = f"atrator_lorenz_rho{rho}.png"
    filepath = os.path.join(output_folder, filename)
    
    plt.savefig(filepath, dpi=1000)
    print(f"Atrator salvo em: {filepath}")
    
    plt.close(fig_attractor)