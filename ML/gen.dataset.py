import numpy as np
import pandas as pd

def generate_polynomial_dataset(num_rows=10000, step=1):
    np.random.seed(42)

    # Geração dos valores de entrada
    x = np.arange(0, num_rows * step, step)

    # Coeficientes do polinômio de 3º grau
    a, b, c, d = 0.5, -2.0, 3.0, 10.0
    y = a * x**3 + b * x**2 + c * x + d

    # Uma segunda feature (constante aleatória)
    random_value = np.random.rand()

    # Criando DataFrame
    df = pd.DataFrame({'feature_1': x, 'feture_2': random_value, 'target': y})

    # Salvando para CSV
    df.to_csv('dataset.csv', index=False)
    print(f"dataset.csv criado com {len(df)} linhas (função polinomial de 3º grau).")

# Chamada da função
generate_polynomial_dataset()
