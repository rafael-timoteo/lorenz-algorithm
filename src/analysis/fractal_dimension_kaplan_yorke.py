import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from lyapunov import LyapunovExponents

class FractalDimension:
    """
    Classe para cálculo da dimensão fractal usando os expoentes de Lyapunov.
    Utiliza a dimensão de Kaplan-Yorke (ou dimensão de Lyapunov).
    """
    
    def __init__(self):
        """Inicializa a classe FractalDimension."""
        self.lyapunov_calculator = LyapunovExponents()
        
    def calculate_kaplan_yorke_dimension(self, lyapunov_exponents):
        """
        Calcula a dimensão de Kaplan-Yorke (dimensão de Lyapunov) a partir dos expoentes de Lyapunov.
        
        A dimensão de Kaplan-Yorke é definida como:
        D_KY = j + (λ₁ + λ₂ + ... + λⱼ) / |λⱼ₊₁|
        
        onde j é o maior índice tal que a soma dos j primeiros expoentes é ainda positiva.
        
        Args:
            lyapunov_exponents (np.ndarray): Expoentes de Lyapunov.
            
        Returns:
            float: Dimensão de Kaplan-Yorke.
        """
        # Garante que os expoentes estão ordenados do maior para o menor
        sorted_exponents = np.sort(lyapunov_exponents)[::-1]
        
        # Para o caso de rho=28, devemos esperar que a soma dos dois primeiros expoentes seja positiva
        sum_top_two = sorted_exponents[0] + sorted_exponents[1]
        
        if sum_top_two > 0:
            j = 2
            dim_ky = j + sum_top_two / np.abs(sorted_exponents[2])
            return dim_ky
        
        # Se a soma dos dois primeiros não for positiva, verificar se pelo menos o primeiro é positivo
        if sorted_exponents[0] > 0:
            j = 1
            dim_ky = j + sorted_exponents[0] / np.abs(sorted_exponents[1])
            return dim_ky
        
        # Se nenhum expoente for positivo, retornar zero
        return 0.0
    
    def compute_dimensions_for_lorenz(self, rho_values, initial_state=None, t_span=(0, 100), dt=0.1, transient_time=10):
        """
        Calcula as dimensões fractais para o sistema de Lorenz com diferentes valores de rho.
        
        Args:
            rho_values (list): Lista de valores do parâmetro rho para análise.
            initial_state (np.ndarray, optional): Vetor de estado inicial.
            t_span (tuple, optional): Intervalo de tempo da simulação.
            dt (float, optional): Passo de tempo para a reortogonalização.
            transient_time (float, optional): Tempo inicial para descarte.
            
        Returns:
            dict: Dicionário com as dimensões fractais para cada valor de rho.
        """
        # Calcular os expoentes de Lyapunov
        results = self.lyapunov_calculator.compute_lorenz_spectrum(
            rho_values, initial_state, t_span, dt, transient_time
        )
        
        # Calcular as dimensões fractais
        dimensions = {}
        for r in rho_values:
            lyapunov_exponents = results[r]['final']
            dim_ky = self.calculate_kaplan_yorke_dimension(lyapunov_exponents)
            dimensions[r] = {
                'exponents': lyapunov_exponents,
                'dimension': dim_ky
            }
        
        return dimensions, results
    
    def create_dimensions_table(self, dimensions, rho_values):
        """
        Cria uma tabela DataFrame com os valores das dimensões fractais.
        
        Args:
            dimensions (dict): Dicionário com as dimensões fractais.
            rho_values (list): Lista de valores de rho.
            
        Returns:
            pd.DataFrame: Tabela com as dimensões fractais.
        """
        data = []
        for r in rho_values:
            exponents = dimensions[r]['exponents']
            dim = dimensions[r]['dimension']
            sum_top_two = exponents[0] + exponents[1]
            data.append({
                'ρ': r,
                'λ1': exponents[0],
                'λ2': exponents[1],
                'λ3': exponents[2],
                'λ1 + λ2': sum_top_two,
                'Dimensão_KY': dim
            })
        
        return pd.DataFrame(data)
    
    def export_dimensions_to_csv(self, dimensions, rho_values, filename='dimensoes_fractais.csv'):
        """
        Exporta as dimensões fractais calculadas para um arquivo CSV.
        
        Args:
            dimensions (dict): Dicionário com as dimensões fractais.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo CSV para salvar os dados.
            
        Returns:
            pd.DataFrame: Tabela com as dimensões fractais.
        """
        df = self.create_dimensions_table(dimensions, rho_values)
        df.to_csv(filename, index=False, float_format="%.4f")
        return df
    
    def print_kaplan_yorke_details(self, dimensions, rho):
        """
        Imprime os detalhes do cálculo da dimensão de Kaplan-Yorke para um valor específico de rho.
        
        Args:
            dimensions (dict): Dicionário com as dimensões fractais.
            rho (float): Valor de rho para o qual imprimir os detalhes.
        """
        if rho not in dimensions:
            print(f"Não há dados para rho = {rho}")
            return
            
        exponents = dimensions[rho]['exponents']
        dim_ky = dimensions[rho]['dimension']
        
        sum_top_two = exponents[0] + exponents[1]
        
        print(f"\nDimensão de Kaplan-Yorke para ρ = {rho}:")
        print(f"  Expoentes de Lyapunov: λ1 = {exponents[0]:.4f}, λ2 = {exponents[1]:.4f}, λ3 = {exponents[2]:.4f}")
        
        if sum_top_two > 0:
            print(f"  Soma dos dois maiores expoentes é positiva ({sum_top_two:.4f}).")
            print(f"  Dimensão de Kaplan-Yorke (D_KY) = 2 + ({exponents[0]:.4f} + {exponents[1]:.4f}) / |{exponents[2]:.4f}|")
            print(f"  ---> Dimensão Fractal Estimada = {dim_ky:.4f}")
        else:
            print(f"  A soma dos expoentes λ1 + λ2 = {sum_top_two:.4f} não é positiva como esperado.")
            if dim_ky > 0:
                j = 1  # Apenas o primeiro expoente é positivo
                print(f"  Dimensão de Kaplan-Yorke (D_KY) = {j} + ({exponents[0]:.4f}) / |{exponents[1]:.4f}|")
                print(f"  ---> Dimensão Fractal Estimada = {dim_ky:.4f}")
            else:
                print(f"  ---> Não foi possível calcular a dimensão fractal (todos os expoentes são negativos).")
    
    def plot_dimensions(self, dimensions, rho_values, filename='Dimensao_Fractal.png'):
        """
        Gera e salva um gráfico da dimensão fractal para diferentes valores de rho.
        
        Args:
            dimensions (dict): Dicionário com as dimensões fractais.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo para salvar o gráfico.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extrair dados para o gráfico
        dim_values = [dimensions[r]['dimension'] for r in rho_values]
        
        # Plotar a curva
        ax.plot(rho_values, dim_values, 'o-', linewidth=2, markersize=8, color='navy')
        
        # Personalizar o gráfico
        ax.set_title('Dimensão de Kaplan-Yorke vs. Parâmetro ρ', fontsize=16)
        ax.set_xlabel('Parâmetro ρ', fontsize=14)
        ax.set_ylabel('Dimensão de Kaplan-Yorke', fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Adicionar anotações para cada ponto
        for i, r in enumerate(rho_values):
            ax.annotate(f'{dim_values[i]:.2f}', 
                        xy=(r, dim_values[i]), 
                        xytext=(0, 10),
                        textcoords='offset points',
                        ha='center',
                        fontsize=10)
        
        # Definir limites adequados
        y_min = min(dim_values) - 0.2 if dim_values else 0
        y_max = max(dim_values) + 0.2 if dim_values else 3
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        fig.savefig(filename, dpi=150)
        plt.close(fig)

    def export_analysis_summary(self, dimensions, rho_values, filename='resumo_analise.csv', format='csv'):
        """
        Exporta um resumo da análise de dimensão fractal e expoentes de Lyapunov para um arquivo.
        
        Args:
            dimensions (dict): Dicionário com as dimensões fractais.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo para salvar o resumo.
            format (str, optional): Formato do arquivo ('csv' ou 'txt').
            
        Returns:
            bool: True se a exportação foi bem-sucedida, False caso contrário.
        """
        try:
            if format.lower() == 'csv':
                # Preparar dados para o DataFrame
                data = []
                for r in rho_values:
                    if r in dimensions:
                        dim = dimensions[r]['dimension']
                        exponents = dimensions[r]['exponents']
                        lambda1 = exponents[0]
                        state = "caótico" if lambda1 > 0 else "regular"
                        data.append({
                            'rho': r,
                            'dimensao': dim,
                            'lambda1': lambda1,
                            'lambda2': exponents[1],
                            'lambda3': exponents[2],
                            'estado': state
                        })
                
                # Criar e salvar DataFrame
                df = pd.DataFrame(data)
                df.to_csv(filename, index=False, float_format="%.4f")
                return True
            
            elif format.lower() == 'txt':
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("Resumo da Análise de Dimensão Fractal e Expoentes de Lyapunov\n")
                    f.write("==========================================================\n\n")
                    
                    # Cabeçalho da tabela
                    f.write(f"{'ρ':^8} | {'Dimensão':^10} | {'λ1':^10} | {'Estado':^10}\n")
                    f.write("-" * 45 + "\n")
                    
                    # Dados
                    for r in rho_values:
                        if r in dimensions:
                            dim = dimensions[r]['dimension']
                            exponents = dimensions[r]['exponents']
                            lambda1 = exponents[0]
                            state = "caótico" if lambda1 > 0 else "regular"
                            f.write(f"{r:8.1f} | {dim:10.4f} | {lambda1:10.4f} | {state:^10}\n")
                    
                    # Informações adicionais
                    f.write("\nNotas:\n")
                    f.write("- Um sistema é considerado caótico quando λ1 > 0\n")
                    f.write("- A dimensão de Kaplan-Yorke (D_KY) é calculada como j + (λ₁ + λ₂ + ... + λⱼ) / |λⱼ₊₁|\n")
                    f.write("  onde j é o maior índice tal que a soma dos j primeiros expoentes é positiva\n")
                return True
            
            else:
                print(f"Formato '{format}' não suportado. Use 'csv' ou 'txt'.")
                return False
                
        except Exception as e:
            print(f"Erro ao exportar resumo da análise: {e}")
            return False
    
    def export_analysis_summary_with_details(self, dimensions, rho_values, filename='analise_detalhada.txt'):
        """
        Exporta um resumo detalhado da análise, incluindo cálculos da dimensão de Kaplan-Yorke.
        
        Args:
            dimensions (dict): Dicionário com as dimensões fractais.
            rho_values (list): Lista de valores de rho.
            filename (str, optional): Nome do arquivo para salvar o resumo.
            
        Returns:
            bool: True se a exportação foi bem-sucedida, False caso contrário.
        """
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write("Análise Detalhada de Dimensão Fractal (Kaplan-Yorke)\n")
                f.write("=================================================\n\n")
                
                for r in sorted(rho_values):
                    if r in dimensions:
                        exponents = dimensions[r]['exponents']
                        dim_ky = dimensions[r]['dimension']
                        sum_top_two = exponents[0] + exponents[1]
                        
                        f.write(f"Dimensão de Kaplan-Yorke para ρ = {r}:\n")
                        f.write(f"  Expoentes de Lyapunov: λ1 = {exponents[0]:.4f}, λ2 = {exponents[1]:.4f}, λ3 = {exponents[2]:.4f}\n")
                        
                        if sum_top_two > 0:
                            f.write(f"  Soma dos dois maiores expoentes é positiva ({sum_top_two:.4f}).\n")
                            f.write(f"  Dimensão de Kaplan-Yorke (D_KY) = 2 + ({exponents[0]:.4f} + {exponents[1]:.4f}) / |{exponents[2]:.4f}|\n")
                            f.write(f"  ---> Dimensão Fractal Estimada = {dim_ky:.4f}\n")
                        else:
                            f.write(f"  A soma dos expoentes λ1 + λ2 = {sum_top_two:.4f} não é positiva.\n")
                            if dim_ky > 0:
                                j = 1  # Apenas o primeiro expoente é positivo
                                f.write(f"  Dimensão de Kaplan-Yorke (D_KY) = {j} + ({exponents[0]:.4f}) / |{exponents[1]:.4f}|\n")
                                f.write(f"  ---> Dimensão Fractal Estimada = {dim_ky:.4f}\n")
                            else:
                                f.write(f"  ---> Não foi possível calcular a dimensão fractal (todos os expoentes são negativos).\n")
                        
                        state = "caótico" if exponents[0] > 0 else "regular"
                        f.write(f"  Regime dinâmico: {state}\n\n")
                        f.write("-" * 60 + "\n\n")
                
                f.write("\nInterpretação:\n")
                f.write("- Um sistema é considerado caótico quando o maior expoente de Lyapunov (λ1) é positivo.\n")
                f.write("- Para o sistema de Lorenz, a transição para o caos ocorre em torno de ρ ≈ 24.\n")
                f.write("- A dimensão fractal aumenta com ρ no regime caótico, indicando maior complexidade.\n")
                f.write("- Para ρ = 28 (caso clássico de caos), a dimensão fractal é tipicamente próxima de 2.05.\n")
            
            return True
            
        except Exception as e:
            print(f"Erro ao exportar análise detalhada: {e}")
            return False
        
    def print_lyapunov_dimension_details(self, dimensions, rho):
        """
        Imprime os detalhes do cálculo da dimensão de Kaplan-Yorke no formato exato do 
        arquivo calculate_lyapunov_dimension.py.
        
        Args:
            dimensions (dict): Dicionário com as dimensões fractais.
            rho (float): Valor de rho para o qual imprimir os detalhes.
        """
        if rho not in dimensions:
            print(f"Não há dados para rho = {rho}")
            return
            
        exponents = dimensions[rho]['exponents']
        dim_ky = dimensions[rho]['dimension']
        
        # Ordenar expoentes do maior para o menor
        sorted_exponents = np.sort(exponents)[::-1]
        
        print("\nExpoentes de Lyapunov Finais:")
        print(f"  lambda_1 = {sorted_exponents[0]:.4f}")
        print(f"  lambda_2 = {sorted_exponents[1]:.4f}")
        print(f"  lambda_3 = {sorted_exponents[2]:.4f}")
        
        sum_top_two = sorted_exponents[0] + sorted_exponents[1]
        if sum_top_two > 0:
            j = 2
            print(f"\nSoma dos dois maiores expoentes é positiva ({sum_top_two:.4f}).")
            print(f"Dimensão de Kaplan-Yorke (D_ky) = {j} + ({sorted_exponents[0]:.4f} + {sorted_exponents[1]:.4f}) / |{sorted_exponents[2]:.4f}|")
            print(f"---> Dimensão Fractal Estimada = {dim_ky:.4f}")
        else:
            sum_first = sorted_exponents[0]
            if sum_first > 0:
                j = 1
                dim_ky_recalc = j + sum_first / np.abs(sorted_exponents[j])
                print(f"\nSoma do primeiro expoente é positiva ({sum_first:.4f}).")
                print(f"Dimensão de Kaplan-Yorke (D_ky) = {j} + ({sorted_exponents[0]:.4f}) / |{sorted_exponents[1]:.4f}|")
                print(f"---> Dimensão Fractal Estimada = {dim_ky_recalc:.4f}")
            else:
                print("\nNenhum expoente positivo encontrado. Impossível calcular a dimensão de Kaplan-Yorke.")

    def export_lyapunov_dimension_to_txt(self, dimensions, rho, filename='dimensao_lyapunov.txt'):
        """
        Exporta os detalhes do cálculo da dimensão de Kaplan-Yorke para um arquivo de texto
        no formato exato do cálculo em calculate_lyapunov_dimension.py.
        
        Args:
            dimensions (dict): Dicionário com as dimensões fractais.
            rho (float): Valor de rho para o qual exportar os detalhes.
            filename (str, optional): Nome do arquivo de texto.
            
        Returns:
            bool: True se a exportação foi bem-sucedida, False caso contrário.
        """
        try:
            if rho not in dimensions:
                print(f"Não há dados para rho = {rho}")
                return False
                
            exponents = dimensions[rho]['exponents']
            dim_ky = dimensions[rho]['dimension']
            
            # Ordenar expoentes do maior para o menor
            sorted_exponents = np.sort(exponents)[::-1]
            
            with open(filename, 'w') as f:
                f.write("Expoentes de Lyapunov Finais:\n")
                f.write(f"  lambda_1 = {sorted_exponents[0]:.4f}\n")
                f.write(f"  lambda_2 = {sorted_exponents[1]:.4f}\n")
                f.write(f"  lambda_3 = {sorted_exponents[2]:.4f}\n")
                
                sum_top_two = sorted_exponents[0] + sorted_exponents[1]
                if sum_top_two > 0:
                    j = 2
                    f.write(f"\nSoma dos dois maiores expoentes é positiva ({sum_top_two:.4f}).\n")
                    f.write(f"Dimensão de Kaplan-Yorke (D_ky) = {j} + ({sorted_exponents[0]:.4f} + {sorted_exponents[1]:.4f}) / |{sorted_exponents[2]:.4f}|\n")
                    f.write(f"---> Dimensão Fractal Estimada = {dim_ky:.4f}\n")
                else:
                    sum_first = sorted_exponents[0]
                    if sum_first > 0:
                        j = 1
                        dim_ky_recalc = j + sum_first / np.abs(sorted_exponents[j])
                        f.write(f"\nSoma do primeiro expoente é positiva ({sum_first:.4f}).\n")
                        f.write(f"Dimensão de Kaplan-Yorke (D_ky) = {j} + ({sorted_exponents[0]:.4f}) / |{sorted_exponents[1]:.4f}|\n")
                        f.write(f"---> Dimensão Fractal Estimada = {dim_ky_recalc:.4f}\n")
                    else:
                        f.write("\nNenhum expoente positivo encontrado. Impossível calcular a dimensão de Kaplan-Yorke.\n")
            
            return True
        
        except Exception as e:
            print(f"Erro ao exportar detalhes da dimensão de Lyapunov: {e}")
            return False