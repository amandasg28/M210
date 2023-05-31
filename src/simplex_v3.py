import numpy as np

def simplex_duas_fases(c, A, b, maximize=True):
    """
    Implementação manual do método Simplex duas fases em Python.
    O PPL usado deve ser uma função de maximização e não pode haver restrições com sinal de maior ou igual. 
    Essa função foi gerada pelo ChatGPT utilizando os seguintes prompt:

    'Código em ppl em python usando método simplex'
    'Modifique o exemplo para fazer o método Simplex de otimização manualmente,
    recebendo os coeficientes da função objetivo e coeficientes de restrição e retornando a solução ótima,
    preço-sombra, viabilidade e lucro ótimo.

    Argumentos:
    c: array de coeficientes da função objetivo
    A: matriz de coeficientes das restrições
    b: array de valores das restrições
    maximize: booleano indicando se é um problema de maximização (padrão) ou minimização

    Retorna:
    solucao_otima: array com os valores ótimos das variáveis de decisão
    precos_sombra: array com os preços-sombra (multiplicadores de Lagrange) das restrições
    faixas_viabilidade: array com as faixas de viabilidade das restrições
    lucro: valor ótimo da função objetivo
    """

    m, n = A.shape  # número de restrições e variáveis de decisão

    # Etapa 1: Adicionar variáveis artificiais e transformar em um problema de minimização

    if maximize:
        c = -c  # converter em problema de minimização

    c_artificial = np.ones(m)  # vetor de coeficientes das variáveis artificiais
    c_total = np.concatenate((c, c_artificial), axis=None)  # vetor total de coeficientes

    A_total = np.concatenate((A, np.eye(m)), axis=1)  # matriz total de coeficientes das restrições

    # Etapa 2: Resolver o problema auxiliar

    # Construir a matriz aumentada
    tableau_auxiliar = np.concatenate((A_total, np.reshape(b, (m, 1))), axis=1)

    # Executar a fase 1 para obter uma solução viável básica
    solucao, tableau_auxiliar = fase_1(tableau_auxiliar, c_total, maximize)

    # Verificar se o problema auxiliar possui solução ótima
    if solucao is None:
        return None, None, None, None  # O problema é inviável

    # Etapa 3: Remover as variáveis artificiais e resolver o problema original

    # Remover as colunas correspondentes às variáveis artificiais do tableau auxiliar
    tableau_auxiliar = np.delete(tableau_auxiliar, np.arange(n, n + m), axis=1)

    # Obter os índices básicos do tableau auxiliar após a fase 1
    indices_basicos = np.where(tableau_auxiliar[:, :-1] == 0)[1]

    # Verificar se todas as variáveis artificiais foram eliminadas
    if np.any(indices_basicos >= n):
        return None, None, None, None  # O problema é inviável

    # Atualizar a matriz A e o vetor b para o problema original
    A_original = tableau_auxiliar[:, :n]
    b_original = tableau_auxiliar[:, -1]

    # Executar a fase 2 para obter a solução ótima
    solucao, tableau_final = fase_2(A_original, b_original, c, indices_basicos, maximize)

    # Etapa 4: Calcular os preços-sombra

    precos_sombra = calcula_precos_sombra(tableau_final, indices_basicos, maximize)

    # Etapa 5: Calcular as faixas de viabilidade

    faixas_viabilidade = calcula_faixas_viabilidade(tableau_final)

    # Etapa 6: Calcular o valor ótimo da função objetivo

    lucro = calcula_lucro(tableau_final, maximize)

    # Etapa 7: Retornar os resultados

    return solucao, precos_sombra, faixas_viabilidade, lucro


def fase_1(tableau, c, maximize, max_iter=1000):
    """
    Executa a fase 1 do algoritmo Simplex para encontrar uma solução viável básica inicial.

    Argumentos:
    tableau: matriz aumentada (tableau) do problema auxiliar
    c: array de coeficientes da função objetivo
    maximize: booleano indicando se é um problema de maximização
    max_iter: número máximo de iterações permitidas

    Retorna:
    solucao: array com os valores ótimos das variáveis de decisão
    tableau: tableau final após a fase 1
    """

    m, n = tableau.shape

    # Encontrar a coluna com o menor valor negativo no vetor b
    coluna_pivo = np.argmin(tableau[:, -1])

    iter_count = 0  # contador de iterações

    while tableau[coluna_pivo, -1] < 0:
        # Encontrar a linha com o menor valor não negativo no vetor b / coluna pivo
        linha_pivo = np.argmin(np.where(tableau[:, coluna_pivo] > 0, tableau[:, -1] / tableau[:, coluna_pivo], np.inf))

        # Verificar se o problema é ilimitado
        if np.all(tableau[:, coluna_pivo] <= 0):
            return None, None

        # Executar a operação de pivotamento
        tableau[linha_pivo, :] /= tableau[linha_pivo, coluna_pivo]
        for i in range(m):
            if i != linha_pivo:
                tableau[i, :] -= tableau[i, coluna_pivo] * tableau[linha_pivo, :]

        # Atualizar a coluna pivo
        tableau[:, coluna_pivo] = np.zeros(m)
        tableau[linha_pivo, coluna_pivo] = 1

        # Atualizar a coluna b
        tableau[:, -1] -= tableau[:, coluna_pivo] * tableau[linha_pivo, -1]

        # Encontrar a próxima coluna pivo
        coluna_pivo = np.argmin(tableau[:, -1])

        iter_count += 1  # incrementar o contador de iterações

        if iter_count >= max_iter:  # verificar se o número máximo de iterações foi atingido
            return None, None

    # Verificar se o problema auxiliar é inviável
    if maximize and tableau[coluna_pivo, -1] != 0:
        return None, None

    # Extrair a solução viável básica inicial
    indices_basicos = np.where(tableau[:, :-1] == 0)[1]
    solucao = np.zeros(n - 1)
    solucao[indices_basicos] = tableau[:, -1]

    return solucao, tableau


def fase_2(A, b, c, indices_basicos, maximize):
    """
    Executa a fase 2 do algoritmo Simplex para encontrar a solução ótima.

    Argumentos:
    A: matriz de coeficientes das restrições
    b: array de valores das restrições
    c: array de coeficientes da função objetivo
    indices_basicos: array com os índices básicos
    maximize: booleano indicando se é um problema de maximização

    Retorna:
    solucao: array com os valores ótimos das variáveis de decisão
    tableau: tableau final após a fase 2
    """

    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))

    # Preencher o tableau com os coeficientes das restrições
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b

    # Preencher a última linha do tableau com os coeficientes da função objetivo
    tableau[-1, :-1] = c

    while True:
        print("fase 2")
        # Encontrar a coluna pivo
        coluna_pivo = np.argmin(tableau[-1, :-1])

        # Verificar se a solução é ótima
        if maximize and tableau[-1, coluna_pivo] >= 0:
            break
        if not maximize and tableau[-1, coluna_pivo] <= 0:
            break

        # Encontrar a linha pivo
        linha_pivo = np.argmin(np.where(tableau[:-1, coluna_pivo] > 0, tableau[:-1, -1] / tableau[:-1, coluna_pivo], np.inf))

        # Executar a operação de pivotamento
        tableau[linha_pivo, :] /= tableau[linha_pivo, coluna_pivo]
        for i in range(m + 1):
            if i != linha_pivo:
                tableau[i, :] -= tableau[i, coluna_pivo] * tableau[linha_pivo, :]

        # Atualizar a coluna pivo
        tableau[:, coluna_pivo] = np.zeros(m + 1)
        tableau[linha_pivo, coluna_pivo] = 1

        # Atualizar a coluna b
        tableau[:, -1] -= tableau[:, coluna_pivo] * tableau[linha_pivo, -1]

        # Atualizar os índices básicos
        indices_basicos[linha_pivo] = coluna_pivo

    # Extrair a solução ótima
    solucao = np.zeros(n)
    solucao[indices_basicos] = tableau[:-1, -1]

    return solucao, tableau


def calcula_precos_sombra(tableau, indices_basicos, maximize):
    """
    Calcula os preços-sombra (multiplicadores de Lagrange) das restrições.

    Argumentos:
    tableau: tableau final após a fase 2
    indices_basicos: array com os índices básicos
    maximize: booleano indicando se é um problema de maximização

    Retorna:
    precos_sombra: array com os preços-sombra (multiplicadores de Lagrange) das restrições
    """

    m, n = tableau.shape
    precos_sombra = np.zeros(m - 1)

    for i in range(m - 1):
        if i in indices_basicos:
            coluna_pivo = np.where(tableau[i, :-1] != 0)[0][0]
            precos_sombra[i] = -tableau[-1, coluna_pivo] if maximize else tableau[-1, coluna_pivo]

    return precos_sombra


def calcula_faixas_viabilidade(tableau):
    """
    Calcula as faixas de viabilidade das restrições.

    Argumentos:
    tableau: tableau final após a fase 2

    Retorna:
    faixas_viabilidade: array com as faixas de viabilidade das restrições
    """

    m, n = tableau.shape
    faixas_viabilidade = np.zeros((m - 1, 2))

    for i in range(m - 1):
        if np.any(tableau[i, :-1] != 0):
            coluna_pivo = np.where(tableau[i, :-1] != 0)[0][0]
            faixas_viabilidade[i] = tableau[i, -1] / tableau[i, coluna_pivo], np.inf
        else:
            faixas_viabilidade[i] = -np.inf, np.inf

    return faixas_viabilidade


def calcula_lucro(tableau, maximize):
    """
    Calcula o valor ótimo da função objetivo.

    Argumentos:
    tableau: tableau final após a fase 2
    maximize: booleano indicando se é um problema de maximização

    Retorna:
    lucro: valor ótimo da função objetivo
    """

    lucro = tableau[-1, -1] if maximize else -tableau[-1, -1]
    return lucro

c = np.array([-2, -1])
A = np.array([[-2, -1], [-1, -3]])
b = np.array([-3, -4])

solucao_otima, precos_sombra, faixas_viabilidade, lucro = simplex_duas_fases(c, A, b, maximize=True)

print("Solução ótima:", solucao_otima)
print("Preços-sombra:", precos_sombra)
print("Faixas de viabilidade:", faixas_viabilidade)
print("Lucro:", lucro)
