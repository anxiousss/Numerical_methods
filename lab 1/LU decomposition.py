from typing import List, Tuple

import numpy as np

from utility.matrix import Matrix
from utility.functions import prod


def back_substitution(system: Matrix) -> List[float]:
    X = [0.0 for _ in range(system.rows)]
    X[-1] = system.free_members[-1] / system.matrix[-1][-1]
    for i in range(system.rows - 1, -1, -1):
        X[i] = (system.free_members[i] - sum(system.matrix[i][j] * X[j]
                                             for j in range(i + 1, system.rows))) / system.matrix[i][i]
    return X


def direct_substitution(system: Matrix) -> List[float]:
    Y = [0.0 for _ in range(system.rows)]
    Y[0] = system.free_members[0] / system.matrix[0][0]
    for i in range(1, system.rows):
        Y[i] = (system.free_members[i] - sum(system.matrix[i][j] * Y[j] for j in range(0, i))) / system.matrix[i][i]
    return Y


def lup_decomposition(system: Matrix) -> Tuple[Matrix, Matrix, List[int]]:
    n = system.rows
    A = system.copy()
    P = list(range(n))

    for k in range(n):
        # Поиск ведущего элемента
        max_index = k
        max_value = abs(A.matrix[k][k])
        for i in range(k + 1, n):
            if abs(A.matrix[i][k]) > max_value:
                max_value = abs(A.matrix[i][k])
                max_index = i

        if max_index != k:
            A.matrix[k], A.matrix[max_index] = A.matrix[max_index], A.matrix[k]
            P[k], P[max_index] = P[max_index], P[k]

        if abs(A.matrix[k][k]) < 1e-20:
            raise ValueError('Матрица вырождена')

        for i in range(k + 1, n):
            A.matrix[i][k] = A.matrix[i][k] / A.matrix[k][k]
            for j in range(k + 1, n):
                A.matrix[i][j] = A.matrix[i][j] - A.matrix[i][k] * A.matrix[k][j]

    L = Matrix(n, n, identity=True)
    U = Matrix(n, n)

    for i in range(n):
        for j in range(n):
            if i > j:
                L.matrix[i][j] = A.matrix[i][j]
            else:
                U.matrix[i][j] = A.matrix[i][j]

    return L, U, P


def apply_permutation(P: List[int], vector: List[float]) -> List[float]:
    """
    Применяет перестановку P к вектору.
    """
    n = len(P)
    result = [0.0] * n
    for i in range(n):
        result[i] = vector[P[i]]
    return result


def solve_lup(L: Matrix, U: Matrix, P: List[int], b: List[float]) -> List[float]:
    """
    Решает систему уравнений Ax = b используя LUP-разложение.
    """
    # Применяем перестановку к вектору b.
    b_permuted = apply_permutation(P, b)

    # Решаем Ly = Pb (прямая подстановка).
    L.free_members = b_permuted
    Y = direct_substitution(L)

    # Решаем Ux = y (обратная подстановка).
    U.free_members = Y
    X = back_substitution(U)

    return X


def inverse(L: Matrix, U: Matrix, P: List[int]) -> Matrix:
    """
    Вычисление обратной матрицы через LUP-разложение.
    """
    n = L.rows
    E = Matrix(n, n, identity=True)
    I = Matrix(n, n, [])

    for i in range(n):
        # Применяем перестановку к i-му столбцу единичной матрицы
        e_i = [E.matrix[j][i] for j in range(n)]
        permuted_e_i = apply_permutation(P, e_i)

        # Решаем Ly = Pe_i
        L_system = Matrix(n, n, L.matrix, permuted_e_i)
        Y = direct_substitution(L_system)

        # Решаем Ux = y
        U_system = Matrix(n, n, U.matrix, Y)
        I_column = back_substitution(U_system)
        I.matrix.append(I_column)

    I.transpose()
    return I


def determinant(L: Matrix, U: Matrix, P: List[int]) -> float:
    """
    Вычисление определителя через LUP-разложение.
    """
    n = L.rows
    det = prod(U.matrix[i][i] for i in range(n))

    # Учет перестановок (каждая перестановка меняет знак определителя.
    swaps = sum(1 if P[i] != i else 0 for i in range(n))
    if swaps % 2 == 1:
        det = -det

    return det


def main():
    system = Matrix(4, 4, [[7, 8, 4, -6],
                           [-1, 6, -2, -6],
                           [2, 9, 6, -4],
                           [5, 9, 1, 1]], [-126, -42, -115, -67])

    L, U, P = lup_decomposition(system)
    print("L matrix:")
    print(L)
    print("\nU matrix:")
    print(U)
    print(f"\nPermutation: {P}")

    solution = solve_lup(L, U, P, system.free_members)
    print(f"\nSolution (x): {solution}")

    det = determinant(L, U, P)
    print(f"\nDeterminant: {det}")

    inv = inverse(L, U, P)
    print("\nInverse matrix:")
    print(inv, end='\n\n\n')
    system2 = Matrix(4, 4, [[7, 8, 4, -6],
                           [-1, 6, -2, -6],
                           [2, 9, 6, -4],
                           [5, 9, 1, 1]], [-126, -42, -115, -67])
    print(system2 * inv, end='\n\n\n')



if __name__ == "__main__":
    main()
