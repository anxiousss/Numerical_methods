import math
from typing import List, Tuple, Any

import numpy as np

from utility.matrix import Matrix


def dot_product(a: List[int | float], b: List[int | float]) -> int | float:
    """
    Скалярное произведние векторов a и b.
    """
    return sum(a[i] * b[i] for i in range(len(a)))


def projection_operator(a: List[int | float], b: List[int | float]) -> List[int | float]:
    """Оператор проекции вектора a на вектор b"""
    coeff = dot_product(a, b) / dot_product(b, b)
    return [coeff * b[i] for i in range(len(b))]


def vector_add(a: List[int | float], b: List[int | float], sign: str) -> List[int | float]:
    """
    Сумма или разница двух векторов.
    """
    if sign == '+':
        return [a[i] + b[i] for i in range(len(a))]
    return [a[i] - b[i] for i in range(len(a))]


def calculate_norm(vector: List[int | float]) -> int | float:
    """
    Вычисление нормы вектора.
    """
    return sum(el ** 2 for el in vector) ** 0.5


def stop_condition(A: Matrix, accuracy: float, indexes: List[int]) -> bool:
    """
    Условие остановки работы QR алгоритма.
    """
    for i in range(A.rows):
        for j in range(A.columns):
            if ((abs(A.matrix[i][j]) >= accuracy and i > j and i in indexes)
                        or (abs(A.matrix[i][j]) == float('inf') and i < j)):
                return False
    return True


def qr_decomposition(system: Matrix) -> Tuple[Matrix, Matrix]:
    """
    QR разложение системы через процесс ортогонализации грамма-шмидта.
    Q - Ортонормированая матрица. Q^-1 = Q^T
    R - Верхнетреугольная матрица с СЗ на главной диагонали.
    """
    system.transpose()
    orthogonal = Matrix(system.rows, system.columns, identity=False)
    orthonormal = Matrix(system.rows, system.columns, identity=False)
    # Проделываем процесс грамма-шмидта для системы.
    for i in range(system.rows):
        total = [0 for _ in range(system.rows)]
        for j in range(i):
            term = projection_operator(system.matrix[i], orthogonal.matrix[j])
            total = vector_add(total, term, '-')
        orthogonal.matrix[i] = vector_add(system.matrix[i], total, '+')
        norm = calculate_norm(orthogonal.matrix[i])
        orthonormal.matrix[i] = [el / norm for el in orthogonal.matrix[i]]

    system.transpose()
    R = orthonormal * system
    orthonormal.transpose()
    return orthonormal, R


def qr_algorithm(A: Matrix, accuracy: float) -> set[str]:
    # Все матрицы A являются подобными
    current_A = A
    eigenvalues = set()
    arg_prev, arg_next = [], []
    while True:
        real_indexes = []
        for i in range(current_A.rows - 1):
            a, b, c, d = (current_A.matrix[i][i], current_A.matrix[i][i + 1],
                          current_A.matrix[i + 1][i], current_A.matrix[i + 1][i + 1])

            D = (a - d) ** 2 + 4 * b * c
            if D >= 0:
                real_indexes.append(i)
                eigenvalues.add(f'{current_A.matrix[i][i]}')
                eigenvalues.add(f'{current_A.matrix[i + 1][i + 1]}')
            else:
                alpha = -b / (2 * a)
                beta = abs(D) ** 0.5 / (2 * a)
                arg_next.append((alpha ** 2 + beta ** 2) ** 0.5)
                eigenvalues.add(f'{alpha} + {beta}j')
                eigenvalues.add(f'{alpha} - {beta}j')

        if stop_condition(current_A, accuracy, real_indexes) and all(abs(arg_prev[k] - arg_next[k]) < accuracy
                                                                     for k in range(math.ceil(current_A.rows / 2) // 2)):
            return eigenvalues

        arg_prev = arg_next
        arg_next.clear()
        Q, R = qr_decomposition(current_A)
        current_A = R * Q
        eigenvalues.clear()

def main():
    system = Matrix(3, 3, [[3, -7, -1], [-9, -8, 7], [5, 2, 2]])
    A = np.array([[3, -7, -1], [-9, -8, 7], [5, 2, 2]])
    w, v = np.linalg.eig(A)
    print(w)
    eigenvalues = qr_algorithm(system, 1e-10)
    print("Собственные значения:", eigenvalues)


if __name__ == "__main__":
    main()
