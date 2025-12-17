import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd

class LyapunovExponents:
    """
    Classe para cálculo de Expoentes de Lyapunov utilizando o método do mapa tangente.
    """
    
    def __init__(self):
        """Inicializa a classe LyapunovExponents."""
        pass
        
    def tangent_map_lyapunov_with_history(
        self,
        system_dynamics,
        jacobian_func,
        initial_state,
        t_span,
        dt,
        transient_time=0.0,
        rtol=1e-9,
        atol=1e-12,
        max_step=None,
    ):
        """
        Calcula o espectro de expoentes de Lyapunov e seu histórico temporal
        pelo método do mapa tangente. 

        A reortogonalização é feita a cada passo de tempo 'dt' usando a
        decomposição QR, que é uma implementação numérica do processo de
        Gram-Schmidt. 

        Args:
            system_dynamics (callable): Função que define as equações do sistema f(t, y).
            jacobian_func (callable): Função que calcula a matriz Jacobiana do sistema J(y). 
            initial_state (np.ndarray): Vetor de estado inicial.
            t_span (tuple): Intervalo de tempo da simulação (t_start, t_end).
            dt (float): Passo de tempo para a reortogonalização.
            transient_time (float, optional): Tempo inicial para descarte (regime transiente).
            rtol (float, optional): Tolerância relativa do integrador (solve_ivp).
            atol (float, optional): Tolerância absoluta do integrador (solve_ivp).
            max_step (float, optional): Passo máximo permitido pelo integrador.
                                        Se None, usa o próprio dt.

        Returns:
            tuple: Uma tupla contendo:
                - np.ndarray: Expoentes de Lyapunov finais (ordenados decrescentemente).
                - np.ndarray: Vetor de pontos de tempo (já sem o transiente).
                - np.ndarray: Matriz 2D com o histórico dos expoentes (ordenados a cada passo).
        """
        n = len(initial_state)
        t_start, t_end = t_span

        if max_step is None:
            max_step = dt

        # Define o sistema combinado (equações do sistema + equações do mapa tangente).
        # O sistema completo possui n + n*n equações. 
        def combined_system(t, y_flat):
            state_vector = y_flat[:n]
            phi_matrix = y_flat[n:].reshape((n, n))
            d_state_dt = system_dynamics(t, state_vector)
            J = jacobian_func(state_vector)
            d_phi_dt = J @ phi_matrix  # d(phi)/dt = J * phi
            return np.concatenate((d_state_dt, d_phi_dt.flatten()))

        # ------------------------------
        # 1) Fase transiente (queima)
        # ------------------------------
        if transient_time > 0.0:
            transient_span = (t_start, t_start + transient_time)
            sol = solve_ivp(
                system_dynamics,
                transient_span,
                initial_state,
                dense_output=True,
                method='RK45',
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            # Novo estado inicial já no atrator (aproximadamente)
            initial_state = sol.sol(transient_span[1])
            t_start = t_start + transient_time

        # ------------------------------
        # 2) Integração com mapa tangente
        # ------------------------------
        phi = np.identity(n)
        y_combined_initial = np.concatenate((initial_state, phi.flatten()))
        lyapunov_sum = np.zeros(n)
        
        current_time = t_start
        times = []
        lyapunov_history = []
        
        # Loop principal de integração e cálculo.
        while current_time < t_end:
            integration_span = (current_time, min(current_time + dt, t_end))
            sol = solve_ivp(
                combined_system,
                integration_span,
                y_combined_initial,
                method='RK45',
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            
            y_combined_final = sol.y[:, -1]
            current_state = y_combined_final[:n]
            phi_evolved = y_combined_final[n:].reshape((n, n))

            # Processo de reortogonalização de Gram-Schmidt (via Decomposição QR).
            Q, R = np.linalg.qr(phi_evolved)
            
            # As normas dos vetores para o cálculo dos expoentes são a diagonal de R.
            norms = np.abs(np.diag(R))
            
            # Evita log(0) se, por algum motivo numérico, algum valor for 0.
            norms[norms == 0] = 1e-300
            
            # Acumula a soma dos logaritmos das normas.
            lyapunov_sum += np.log(norms)

            # A nova matriz de perturbação para o próximo passo é a matriz ortonormal Q.
            y_combined_initial = np.concatenate((current_state, Q.flatten()))
            current_time = integration_span[1]

            # Armazena o histórico para a plotagem dos gráficos.
            times.append(current_time - t_start)
            # Calcula o expoente médio até o momento atual.
            elapsed_time = current_time - t_start
            current_exponents = lyapunov_sum / elapsed_time
            lyapunov_history.append(np.sort(current_exponents)[::-1])

        total_time = t_end - t_start
        final_lyapunov_exponents = lyapunov_sum / total_time
        
        return np.sort(final_lyapunov_exponents)[::-1], np.array(times), np.array(lyapunov_history)
    
    def compute_lorenz_spectrum(
        self,
        rho_values,
        initial_state=None,
        t_span=(0.0, 200.0),
        dt=0.05,
        transient_time=50.0,
        rtol=1e-9,
        atol=1e-12,
        max_step=None,
    ):
        """
        Calcula o espectro de Lyapunov para o sistema de Lorenz com diferentes valores de rho.
        
        Args:
            rho_values (list): Lista de valores do parâmetro rho para análise.
            initial_state (np.ndarray, optional): Vetor de estado inicial.
                                                  Default [0.0, 1.0, 1.05].
            t_span (tuple, optional): Intervalo de tempo da simulação.
            dt (float, optional): Passo de tempo para a reortogonalização.
            transient_time (float, optional): Tempo inicial para descarte.
            rtol (float, optional): Tolerância relativa do integrador (solve_ivp).
            atol (float, optional): Tolerância absoluta do integrador (solve_ivp).
            max_step (float, optional): Passo máximo permitido pelo integrador.
            
        Returns:
            dict: Dicionário com os resultados para cada valor de rho.
        """
        if initial_state is None:
            initial_state = np.array([0.0, 1.0, 1.05])
        
        sigma = 10.0
        beta = 8.0 / 3.0
        
        results = {}
        
        for r in rho_values:
            # Define funções do sistema de Lorenz com o rho atual
            def lorenz_system(t, y):
                x, y_coord, z = y
                dx_dt = sigma * (y_coord - x)
                dy_dt = x * (r - z) - y_coord
                dz_dt = x * y_coord - beta * z
                return np.array([dx_dt, dy_dt, dz_dt])

            def lorenz_jacobian(y):
                x, y_coord, z = y
                return np.array([
                    [-sigma,    sigma,      0.0],
                    [r - z,     -1.0,      -x  ],
                    [y_coord,   x,         -beta]
                ])
            
            final_exponents, times, history = self.tangent_map_lyapunov_with_history(
                lorenz_system,
                lorenz_jacobian,
                initial_state,
                t_span,
                dt,
                transient_time=transient_time,
                rtol=rtol,
                atol=atol,
                max_step=max_step,
            )
            results[r] = {'final': final_exponents, 'times': times, 'history': history}
            
        return results
    
    def create_spectrum_table(self, results, rho_values):
        """
        Cria uma tabela DataFrame com os valores dos expoentes de Lyapunov.
        
        Args:
            results (dict): Resultados do cálculo de expoentes de Lyapunov.
            rho_values (list): Lista de valores de rho.
            
        Returns:
            pd.DataFrame: Tabela com os valores dos expoentes.
        """
        table_data = {
            'ρ': rho_values,
            'λ1': [results[r]['final'][0] for r in rho_values],
            'λ2': [results[r]['final'][1] for r in rho_values],
            'λ3': [results[r]['final'][2] for r in rho_values]
        }
        return pd.DataFrame(table_data)

    def export_spectrum_data_to_csv(self, results, rho_values, filename='espectro_lyapunov_data.csv'):
        """
        Exporta los dados do espectro de Lyapunov para um arquivo CSV.
        
        Args:
            results (dict): Resultados do cálculo de expoentes de Lyapunov.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo CSV para salvar os dados.
        """
        all_data = pd.DataFrame()
        
        for r in rho_values:
            history = results[r]['history']
            times = results[r]['times']
            
            temp_df = pd.DataFrame({'time': times})
            
            for j in range(history.shape[1]):
                temp_df[f'lambda{j+1}_rho{r}'] = history[:, j]
                
            if all_data.empty:
                all_data = temp_df
            else:
                all_data = pd.merge(all_data, temp_df, on='time', how='outer')
        
        all_data = all_data.sort_values('time')
        all_data.to_csv(filename, index=False)
        return all_data
        
    def export_largest_exponent_data_to_csv(self, results, rho_values, filename='maior_expoente_lyapunov_data.csv'):
        """
        Exporta os dados do maior expoente de Lyapunov para um arquivo CSV.
        
        Args:
            results (dict): Resultados do cálculo de expoentes de Lyapunov.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo CSV para salvar os dados.
        """
        all_data = pd.DataFrame()
        
        for r in rho_values:
            history = results[r]['history']
            times = results[r]['times']
            
            temp_df = pd.DataFrame({'time': times})
            temp_df[f'lambda1_rho{r}'] = history[:, 0]
                
            if all_data.empty:
                all_data = temp_df
            else:
                all_data = pd.merge(all_data, temp_df, on='time', how='outer')
        
        all_data = all_data.sort_values('time')
        all_data.to_csv(filename, index=False)
        return all_data

    def plot_spectrum(self, results, rho_values, filename='Espectro_Lyapunov.png', csv_filename='espectro_lyapunov_data.csv'):
        """
        Gera e salva um gráfico do espectro de Lyapunov para diferentes valores de rho
        e exporta os dados para um arquivo CSV.
        
        Args:
            results (dict): Resultados do cálculo de expoentes de Lyapunov.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo para salvar o gráfico.
            csv_filename (str, optional): Nome do arquivo CSV para salvar os dados.
        """
        n_plots = len(rho_values)
        n_cols = 3
        n_rows = int(np.ceil(n_plots / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        fig.suptitle('Espectro de Lyapunov', fontsize=16)
        axes = axes.flatten()

        for i, r in enumerate(rho_values):
            ax = axes[i]
            history = results[r]['history']
            t = results[r]['times']
            for j in range(history.shape[1]):
                ax.plot(t, history[:, j], label=f'λ{j+1}')
            ax.set_title(f'ρ = {r}')
            ax.set_ylabel('λ [nats/s]')
            ax.set_xlabel('t [s]')
            ax.legend()
            ax.grid(True)
        
        for i in range(n_plots, len(axes)):
            fig.delaxes(axes[i])

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        self.export_spectrum_data_to_csv(results, rho_values, csv_filename)
        
    def plot_largest_exponent(self, results, rho_values, filename='Maior_Expoente_Lyapunov.png', csv_filename='maior_expoente_lyapunov_data.csv'):
        """
        Gera e salva um gráfico do maior expoente de Lyapunov para diferentes valores de rho
        e exporta os dados para um arquivo CSV.
        
        Args:
            results (dict): Resultados do cálculo de expoentes de Lyapunov.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo para salvar o gráfico.
            csv_filename (str, optional): Nome do arquivo CSV para salvar os dados.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        styles = ['-', '-', '-', ':']
        
        for i, r in enumerate(rho_values):
            history = results[r]['history']
            t = results[r]['times']
            ax.plot(t, history[:, 0], label=f'λ1: ρ = {r}', linestyle=styles[min(i, len(styles)-1)])

        ax.set_title('Maior Expoente de Lyapunov')
        ax.set_xlabel('t [s]')
        ax.set_ylabel('λ [nats/s]')
        ax.axhline(0, color='black', linewidth=0.8)
        ax.legend()
        ax.grid(True)
        ax.set_ylim(-1.5, 2.5)
        fig.savefig(filename)
        plt.close(fig)
        
        self.export_largest_exponent_data_to_csv(results, rho_values, csv_filename)

    def export_spectrum_table_to_csv(self, results, rho_values, filename='espectro_lyapunov_tabela.csv'):
        """
        Cria e exporta uma tabela com os valores finais dos expoentes de Lyapunov para um arquivo CSV.
        
        Args:
            results (dict): Resultados do cálculo de expoentes de Lyapunov.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo CSV para salvar a tabela.
            
        Returns:
            pd.DataFrame: Tabela com os valores dos expoentes.
        """
        df_table = self.create_spectrum_table(results, rho_values)
        df_table.to_csv(filename, index=False, float_format="%.4f")
        return df_table