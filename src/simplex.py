# Importação da biblioteca
import numpy as np

# Implementação do método Simplex para resolver PPL
def simplex(c, A, b):
    m, n = A.shape # Determina as dimensões da matriz (m-linha/n-coluna)

    # Adiciona as variáveis lineares
    A_extended = np.hstack((A, np.eye(m))) # Adiciona as variáveis para as restrições
    c_extended = np.hstack((c, np.zeros(m))) # Adiciona os coeficientes das variáveis na função objetivo

    # Inicializa a tabela Simplex
    tableau = np.zeros((m + 1, n + m + 1)) # Cria matriz com zero de tamanho MxN e armazena as informações
    tableau[:-1, :-1] = A_extended # Representa as restrições
    tableau[:-1, -1] = b # Armazena os valores das restrições
    tableau[-1, :-1] = c_extended # Armazena os valores da função objetivo

    while True:
        # Encontra o índice da coluna pivô como o índice de menor valor na linha da função objetivo
        entering_var = np.argmin(tableau[-1, :-1])

        # Verificar se a solução é ótima
        if tableau[-1, entering_var] >= 0: # Se o valor da linha pivô e coluna pivô é >= 0
            break

        # Encontra o índice da linha pivô como o índice de menor valor das razões
        ratios = tableau[:-1, -1] / tableau[:-1, entering_var] # Calcula a razão entre a coluna pivô e as restrições
        ratios[ratios <= 0] = np.inf  # Define os valores das razões considerando apenas valores positivos
        leaving_var = np.argmin(ratios) # Encontra o índice da linha pivô

        # Atualiza a tabela Simplex
        pivot = tableau[leaving_var, entering_var] # Armazena o valor do elemento pivô
        tableau[leaving_var, :] /= pivot # Divide a linha pivô pelo elemento pivô
        for i in range(m + 1): # Percorre todas as linhas tabela
            if i != leaving_var: # Verifica se a linha atual não é a linha pivô
                factor = tableau[i, entering_var] # Armazena o coeficiente da coluna pivô
                tableau[i, :] -= factor * tableau[leaving_var, :] # Atualiza a linha atual

    # Extrai a solução ótima e os preços-sombra
    solution = tableau[:-1, -1]
    dual_prices = tableau[-1, :-1]

    # Verifica a viabilidade das restrições
    feasibility = np.all(tableau[:-1, -1] >= 0)

    # Calcula o lucro ótimo
    optimal_profit = tableau[-1, -1]

    return solution, dual_prices, feasibility, optimal_profit