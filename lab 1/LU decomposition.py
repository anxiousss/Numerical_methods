from typing import List
from math import prod
from utility.matrix import Matrix


def back_substitution(system: Matrix) -> List[int | float]:
    X = [0 for _ in range(system.rows)]
    X[-1] = system.free_members[-1] / system.matrix[-1][-1]
    for i in range(system.rows - 1, -1, -1):
        X[i] = (system.free_members[i] - sum(system.matrix[i][j] * X[j]
                                             for j in range(i + 1, system.rows))) / system.matrix[i][i]
    return X

def direct_substitution(system: Matrix) -> List[int | float]:
    Y = [0 for _ in range(system.rows)]
    Y[0] = system.free_members[0]
    for i in range(1, system.rows):
        Y[i] = system.free_members[i] - sum(system.matrix[i][j] * Y[j] for j in range(0, i))

    return Y

def lu_decomposition(system: Matrix):
    if system.calc_determinant() == 0 or system.matrix[0][0] == 0:
        raise ValueError('Разложение не работает для вырожденных матриц')

    L = Matrix(system.rows, system.columns, None, system.free_members, None, True)
    U = Matrix(system.rows, system.columns, None, None, None, False)

    for i in range(system.rows):
        for j in range(system.columns):
            if i <= j:
                U.matrix[i][j] = system.matrix[i][j] - sum(L.matrix[i][k] * U.matrix[k][j] for k in range(i))
            else:
                L.matrix[i][j] = (system.matrix[i][j] - sum(L.matrix[i][k] * U.matrix[k][j] for k in range(j))) / U.matrix[j][j]

    return L, U

def inverse(L: Matrix, U: Matrix) -> Matrix:
    E = [[int(i == j) for j in range(L.rows)] for i in range(L.rows)]
    I = Matrix(L.rows, L.columns, [])
    for i in range(L.rows):
        L.free_members = E[i]
        Y = direct_substitution(L)
        U.free_members = Y
        I_column = back_substitution(U)
        I.matrix.append(I_column)

    I.transpose()
    return I

def main():
    system = Matrix(3, 3, [[7, 8, 4, -6],
                                             [-1, 6, -2, -6],
                                             [2, 9, 6, -4],
                                             [5, 9, 1, 1]], [-126, -42, -115, -67])
    L, U = lu_decomposition(system)
    """Y = direct_substitution(L)
    U.free_members = Y
    X = back_substitution(U)
    determinant = prod(L.matrix[i][i] for i in range(L.rows)) * prod(U.matrix[i][i] for i in range(U.rows))"""

    print(inverse(L, U).matrix)


if __name__ == "__main__":
    main()