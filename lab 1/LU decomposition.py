from typing import List
from utility.matrix import Matrix

def lu_decomposition(system: Matrix):
    if system.calc_determinant() == 0 or system.matrix[0][0] == 0:
        raise ValueError('Разложение не работает для вырожденных матриц')

    L = Matrix(system.rows, system.columns, None, None, None, True)
    U = Matrix(system.rows, system.columns, None, None, None, False)

    for i in range(system.rows):
        for j in range(system.columns):
            if i <= j:
                U.matrix[i][j] = system.matrix[i][j] - sum(L.matrix[i][k] * U.matrix[k][j] for k in range(i))
            else:
                L.matrix[i][j] = (system.matrix[i][j] - sum(L.matrix[i][k] * U.matrix[k][j] for k in range(j))) / U.matrix[j][j]

    print(L, U, sep='\n\n')

def main():
    system = Matrix(4, 4, [[7, 8, 4, -6],
                                             [-1, 6, -2, -6],
                                             [2, 9, 6, -4],
                                             [5, 9, 1, 1]])
    lu_decomposition(system)

if __name__ == "__main__":
    main()