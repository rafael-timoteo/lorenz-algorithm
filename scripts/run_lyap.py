import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from analysis.lyapunov import LyapunovExponents

rho_values = [1, 5, 10, 15, 20, 25, 26, 27, 28, 29, 30]

lyap = LyapunovExponents()

results = lyap.compute_lorenz_spectrum(
    rho_values,
    t_span=(0, 200),       # tempo total
    dt=0.05,               # intervalo de reortogonalização
    transient_time=50.0,   # queima inicial
    rtol=1e-9,
    atol=1e-12,
)

# Mostra tabela no terminal
table = lyap.create_spectrum_table(results, rho_values)
print(table)

# Gera gráficos
lyap.plot_spectrum(results, rho_values, filename="espectro.png")
lyap.plot_largest_exponent(results, rho_values, filename="lambda1.png")

# Exporta tabela para CSV
lyap.export_spectrum_table_to_csv(results, rho_values, filename="tabela_lyapunov.csv")