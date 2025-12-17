# Lorenz Algorithm

Framework Python para análise do sistema de Lorenz: simulação numérica, expoentes de Lyapunov, dimensões fractais, seções de Poincaré, controle de caos e sincronização.

## Instalação

### Requisitos
- Python 3.8+
- pip

### Configuração

```bash
git clone https://github.com/rafael-timoteo/lorenz-algorithm.git
cd lorenz-algorithm
pip install -r requirements.txt
```

## Estrutura do Projeto

```
lorenz-algorithm/
├── src/                          # Código fonte
│   ├── core/                     # Solver e equações
│   │   └── lorenz_solver.py
│   ├── analysis/                 # Lyapunov e dimensões fractais
│   │   ├── lyapunov.py
│   │   ├── fractal_dimension.py
│   │   └── fractal_dimension_kaplan_yorke.py
│   ├── visualization/            # Gráficos e plots
│   │   ├── attractor_generator.py
│   │   ├── timeseries_generator.py
│   │   ├── poincare_sections.py
│   │   └── poincare_sections_grad.py
│   └── control/                  # Controle de caos
│       └── chaos_control.py
├── scripts/                      # Scripts executáveis
│   ├── main.py                   # Pipeline completo
│   ├── run_lyap.py               # Análise de Lyapunov
│   ├── run_chaos_control.py      # Controle de caos
│   ├── calculate_lyapunov_extended.py
│   ├── compare_atratores_lorenz.py
│   └── plot_attractor_left_compare.py
├── examples/                     # Exemplos adicionais
│   ├── lorenz_sync.py            # Sincronização
│   ├── box_counting_alternative.py
│   └── box_counting/
├── output_plots/                 # Resultados gerados
├── requirements.txt
└── README.md
```

## Uso

### Execução Completa

```bash
cd scripts
python main.py
```

Gera todos os resultados em `output_plots/`:
- Atratores 3D para ρ ∈ {10, 15, 20, 28}
- Séries temporais
- Seções de Poincaré
- Expoentes de Lyapunov
- Dimensões fractais (box-counting e Kaplan-Yorke)

### Análise de Lyapunov

```bash
python run_lyap.py
```

Calcula o espectro de Lyapunov para múltiplos valores de ρ.

### Controle de Caos

```bash
python run_chaos_control.py
```

Simula controle de caos com ativação em t=200s.

### Uso Programático

```python
import sys
sys.path.append('../src')

from core import lorenz_system_equations, runge_kutta_4th_order_solver
from analysis import LyapunovExponents

# Simulação
sigma, rho, beta = 10.0, 28.0, 8.0/3.0
t, states = runge_kutta_4th_order_solver(
    lorenz_system_equations, 0.0, [0, 1, 20], 100.0, 0.01, sigma, rho, beta
)

# Lyapunov
lyap = LyapunovExponents()
results = lyap.compute_lorenz_spectrum([rho])
print(results[rho]['final'])
```

## Dependências

- numpy >= 1.21.0
- scipy >= 1.7.0
- matplotlib >= 3.4.0
- pandas >= 1.3.0
