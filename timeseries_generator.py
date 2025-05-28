# timeseries_generator.py
import matplotlib.pyplot as plt
import os

def plot_and_save_timeseries(time_points, states, sigma, rho, beta, output_folder):
    """
    Plota as séries temporais das variáveis de estado e salva a imagem.
    """
    x_t = states[:, 0]
    y_t = states[:, 1]
    z_t = states[:, 2]

    fig_timeseries, axs_timeseries = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    fig_timeseries.suptitle(f"Séries Temporais ($\\sigma={sigma}, \\rho={rho}, \\beta={beta:.2f}$)", fontsize=16)

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

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    filename = f"series_temporais_rho{rho}.png"
    filepath = os.path.join(output_folder, filename)

    plt.savefig(filepath, dpi=150)
    print(f"Séries temporais salvas em: {filepath}")
    # plt.show()
    plt.close(fig_timeseries)