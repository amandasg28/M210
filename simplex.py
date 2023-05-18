import numpy as np

def simplex(c, A, b):
    m, n = A.shape

    # Adicionar variáveis de folga para as restrições
    A_extended = np.hstack((A, np.eye(m)))
    c_extended = np.hstack((c, np.zeros(m)))

    # Inicializar tabela Simplex
    tableau = np.zeros((m + 1, n + m + 1))
    tableau[:-1, :-1] = A_extended
    tableau[:-1, -1] = b
    tableau[-1, :-1] = c_extended

    while True:
        # Encontrar a coluna pivô (variável que entra na base)
        entering_var = np.argmin(tableau[-1, :-1])

        # Verificar se a solução é ótima
        if tableau[-1, entering_var] >= 0:
            break

        # Encontrar a linha pivô (variável que sai da base)
        ratios = tableau[:-1, -1] / tableau[:-1, entering_var]
        ratios[ratios <= 0] = np.inf  # Ignorar divisão por zero e valores negativos
        leaving_var = np.argmin(ratios)

        # Atualizar a tabela Simplex
        pivot = tableau[leaving_var, entering_var]
        tableau[leaving_var, :] /= pivot
        for i in range(m + 1):
            if i != leaving_var:
                factor = tableau[i, entering_var]
                tableau[i, :] -= factor * tableau[leaving_var, :]

    # Extrair a solução ótima e os preços-sombra
    solution = tableau[:-1, -1]
    dual_prices = tableau[-1, :-1]

    # Verificar a viabilidade das restrições
    feasibility = np.all(tableau[:-1, -1] >= 0)

    # Calcular o lucro ótimo
    optimal_profit = -tableau[-1, -1]

    return solution, dual_prices, feasibility, optimal_profit

# Definir os coeficientes da função objetivo
#c = np.array([-2, -3, -12, -1])
#c = np.array([-12, -60])
c = np.array([-3000, -5000])

# Definir as restrições lineares
#A = np.array([[1, 0, 1, 1], [0, 3, 0, 2], [2, 2, 4, 1], [0, 0, 1, 2]])
#b = np.array([25, 12, 15, 20])
#A = np.array([[6, 30], [6, 45], [6, 24]])
#b = np.array([2160, 1320, 900])
A = np.array([[0.5, 0.2], [0.25, 0.3], [0.25, 0.5]])
b = np.array([16, 11, 15])

# Executar o método Simplex
solution, dual_prices, feasibility, optimal_profit = simplex(c, A, b)

# Imprimir os resultados
print('Solução ótima:', solution)
print('Preços-sombra (valores duais):', dual_prices)
print('Viabilidade das restrições:', feasibility)
print('Lucro ótimo:', optimal_profit)