class Simplex:
    """
    Classe que implementa o método Simplex básico, sem ser duas fases.
    Ou seja, o PPL usado deve ser uma função de maximização e não pode haver restrições com sinal de maior ou igual. 
    Essa é uma abordagem mais próxima do método utilizado em sala.
    """

    # Construtor da classe; inicializa a tabela Simplex vazia.
    def __init__(self):
        self.table = []

    # Recebe a função objetiva e a coloca na primeira linha da tabela
    def set_objective_function(self, obj_func: list):
        self.table.append(obj_func)

    # Recebe as restrições e as colocam 
    def add_restrictions(self, sa: list):
        self.table.append(sa)

    # Pega a columa pivô
    def get_entry_column(self) -> int:
        column_pivot = min(self.table[0])
        index = self.table[0].index(column_pivot)

        return index
    
    # Pega a linha da tabela que vai sair da iteração atual.
    def get_exit_line(self, entry_column: int) -> int:
        # Guardar os resultados da iteração para comparar depois
        results = {}

        # Iterando cada linha das restrições para encontrar o elemento pivô
        for line in range(len(self.table)):
            if line > 0: # Pulando a linha da função objetivo
                if self.table[line][entry_column] > 0: # Tratamento de divisão por zero
                    division = self.table[line][-1] / self.table[line][-1]
                    results[line] = division

        # Pega a linha a qual reside o menor resultado da divisão
        index = min(results, key=results.get)

        return index

    # Calculado a nova linha pivô
    def calculate_new_pivot_line(self, entry_column: int, exit_line: int) -> list:
        line = self.table[exit_line]
        pivot = line[entry_column]

        new_pivot_line = [value / pivot for value in line]

        return new_pivot_line
    
    # Faz o cálculo da nova linha, baseado na linha pivô
    def calculate_new_line(self, line: list, entry_column: int, pivot_line: list) -> list:
        # Calcula o elemento pivô
        pivot = line[entry_column] * -1

        result_line = [value * pivot for value in pivot_line]

        new_line = []

        for i in range(len(result_line)):
            sum_value = result_line[i] + line[i]
            new_line.append(sum_value)
        
        return new_line
    
    # Função para verificar se os coeficientes da função objetivo são negativos. Se sim, continuar as iterações do algoritmo.
    def is_negative(self) -> bool:
        negative = list(filter(lambda x: x < 0, self.table[0]))

        return True if len(negative) > 0 else False
    
    def calculate(self):
        # Calcula qual linha irá sair
        entry_column = self.get_entry_column()
        first_exit_line = self.get_exit_line(entry_column)
        pivot_line = self.calculate_new_pivot_line(entry_column, first_exit_line)

        # Troca a linha selecionada pela linha pivô
        self.table[first_exit_line] = pivot_line
        
        # Cria uma cópia da tabela para manipular os valores da mesma sem perder os dados antigos
        table_copy = self.table.copy()

        # Faz as iterações do algoritmo
        index = 0
        while index < len(self.table):
            if index != first_exit_line:
                line = table_copy[index]
                new_line = self.calculate_new_line(line, entry_column, pivot_line)
                self.table[index] = new_line
            
            index += 1
    
    def show_table(self):
        for i in range(len(self.table)):
            for j in range(len(self.table[0])):
                print (f"{self.table[i][j]}\t", end="")
            print()

    def solve(self):
        self.calculate()

        while self.is_negative():
            self.calculate()
        
        self.show_table()
    
    


