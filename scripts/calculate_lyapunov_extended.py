import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.lyapunov import LyapunovExponents

# Inicializa a classe
lyap = LyapunovExponents()

# Define os valores de rho solicitados: [1, 5, 10, 15, 20, 25, 26, 27, 28, 29, 30]
rho_values = [1, 5, 10, 15, 20, 25, 26, 27, 28, 29, 30]

# Condição inicial
initial_state = np.array([0.0, 1.0, 1.05])

# Parâmetros de simulação
t_span = (0, 100)  # Tempo de simulação de 100 segundos
dt = 0.1           # Passo de reortogonalização
transient_time = 20  # Tempo transiente para atingir o atrator

print("=" * 70)
print("CÁLCULO DE EXPOENTES DE LYAPUNOV - SISTEMA DE LORENZ")
print("=" * 70)
print(f"\nValores de ρ: {rho_values}")
print(f"Condição inicial: {initial_state}")
print(f"Intervalo de tempo: {t_span}")
print(f"Passo de reortogonalização: {dt} s")
print(f"Tempo transiente: {transient_time} s")
print("\nCalculando...\n")

# Calcula o espectro para todos os valores de rho
results = lyap.compute_lorenz_spectrum(
    rho_values=rho_values,
    initial_state=initial_state,
    t_span=t_span,
    dt=dt,
    transient_time=transient_time
)

# Cria e exibe a tabela de resultados
print("=" * 70)
print("RESULTADOS - EXPOENTES DE LYAPUNOV FINAIS")
print("=" * 70)
df_table = lyap.create_spectrum_table(results, rho_values)
print(df_table.to_string(index=False))

# Exporta a tabela para CSV
output_dir = 'output_plots/lyapunov_exponents/'
import os
os.makedirs(output_dir, exist_ok=True)

lyap.export_spectrum_table_to_csv(
    results, 
    rho_values, 
    f'{output_dir}espectro_lyapunov_tabela.csv'
)
print(f"\n✓ Tabela exportada: {output_dir}espectro_lyapunov_tabela.csv")

# Gera e salva o gráfico do espectro completo
lyap.plot_spectrum(
    results, 
    rho_values, 
    filename=f'{output_dir}Espectro_Lyapunov_Completo.png',
    csv_filename=f'{output_dir}espectro_lyapunov_data.csv'
)
print(f"✓ Gráfico salvo: {output_dir}Espectro_Lyapunov_Completo.png")
print(f"✓ Dados exportados: {output_dir}espectro_lyapunov_data.csv")

# Gera e salva o gráfico do maior expoente
lyap.plot_largest_exponent(
    results, 
    rho_values,
    filename=f'{output_dir}Maior_Expoente_Lyapunov.png',
    csv_filename=f'{output_dir}maior_expoente_lyapunov_data.csv'
)
print(f"✓ Gráfico salvo: {output_dir}Maior_Expoente_Lyapunov.png")
print(f"✓ Dados exportados: {output_dir}maior_expoente_lyapunov_data.csv")

# Análise detalhada
print("\n" + "=" * 70)
print("ANÁLISE DETALHADA POR VALOR DE ρ")
print("=" * 70)

for r in rho_values:
    exponents = results[r]['final']
    print(f"\nρ = {r:2d}:")
    print(f"  λ₁ = {exponents[0]:8.4f} nats/s")
    print(f"  λ₂ = {exponents[1]:8.4f} nats/s")
    print(f"  λ₃ = {exponents[2]:8.4f} nats/s")
    print(f"  Σλ = {sum(exponents):8.4f} nats/s")
    
    if exponents[0] > 0.01:
        behavior = "🔴 CAÓTICO (λ₁ > 0)"
    elif exponents[0] < -0.01:
        behavior = "🟢 ESTÁVEL (λ₁ < 0)"
    else:
        behavior = "🟡 QUASE-PERIÓDICO (λ₁ ≈ 0)"
    print(f"  Comportamento: {behavior}")

print("\n" + "=" * 70)
print("CÁLCULO CONCLUÍDO COM SUCESSO!")
print("=" * 70)
