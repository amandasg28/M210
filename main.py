import numpy as np
from src.simplex import simplex

if __name__ == "__main__":
    # PPL Geléias
    # Coeficientes da função objetivo
    c = np.array([-5, -7])

    # Restrições lineares
    A = np.array([[3, 0], [0, 1.5], [0.25, 0.5]])
    b = np.array([250, 100, 50])

    # Execução
    solution, dual_prices, feasibility, optimal_profit = simplex(c, A, b)

    # Imprimindo os resultados
    print('Solução ótima:', solution)
    print('Preços-sombra (valores duais):', dual_prices)
    print('Viabilidade das restrições:', feasibility)
    print('Lucro ótimo:', optimal_profit)
