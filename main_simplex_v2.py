from src.simplex_v2 import Simplex

if __name__ == "__main__":
    """
    PPL: Sapateiro

        fo: Z = 5x + 2y
        sa:
            2x + y <= 6
            10x + 12y <= 60
            x, y => 0
    
    PPL (simplex):

        z - 5x - 2y = 0
            2x + y + f1 = 6
            10x + 12y + f2 = 60
    """

simplex = Simplex()

simplex.set_objective_function([1, -5, -2, 0, 0, 0])
simplex.add_restrictions([0, 2, 1, 1, 0, 6])
simplex.add_restrictions([0, 10, 12, 0, 12, 60])

simplex.solve()
