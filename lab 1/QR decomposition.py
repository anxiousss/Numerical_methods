import math
from typing import List, Tuple

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
            if ((abs(A.matrix[i][j]) >= accuracy and i > j and j in indexes)
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
    # Все матрицы A являются подобными.
    current_A = A
    # Множество собственных значений.
    eigenvalues = set()
    # Массивы для сравнения модуля комплесных чисел на соседних итерациях.
    mod_prev, mod_next = [], []
    while True:
        # столбцы в которых собственное значение действительно.
        real_indexes = []
        i = 0
        while i < current_A.rows:
            # Расчет собственных значений для блока 2x2.
            if i < current_A.rows - 1 and abs(current_A.matrix[i + 1][i]) > accuracy:
                a, b, c, d = (current_A.matrix[i][i], current_A.matrix[i][i + 1],
                              current_A.matrix[i + 1][i], current_A.matrix[i + 1][i + 1])

                trace = a + d
                det = a * d - b * c
                D = trace ** 2 - 4 * det
                if D >= 0:
                    real_indexes.append(i)
                    eigenvalues.add(f'{(trace + D ** 0.5) / 2}')
                    eigenvalues.add(f'{(trace - D ** 0.5) / 2}')
                else:
                    real_part = trace / 2
                    imag_part = math.sqrt(-D) / 2
                    eigenvalues.add(f"{real_part}+{imag_part}j")
                    eigenvalues.add(f"{real_part}-{imag_part}j")
                    mod_next.append((real_part ** 2 + imag_part ** 2) ** 0.5)

                i += 2

            else:
                eigenvalues.add(f'{current_A.matrix[i][i]}')
                i += 1

        # math.ceil(current_A.rows / 2) // 2 - количество комплексно сопряженных пар собвстенных значений.
        if stop_condition(current_A, accuracy, real_indexes) and (all(abs(mod_prev[k] - mod_next[k]) < accuracy
                                                                     for k in range(math.ceil(current_A.rows / 2) // 2))
                                                                     if len(mod_prev) != 0 else True):
            return eigenvalues

        mod_prev = mod_next
        mod_next.clear()
        Q, R = qr_decomposition(current_A)
        current_A = R * Q
        eigenvalues.clear()

def main():
    system = Matrix(3, 3, [[6, 5, -6], [4, -6, 9], [-6, 6, 1]])
    system2 = Matrix(3, 3, [[3, -7, -1], [-9, -8, 7], [5, 2, 2]])

    eigenvalues = qr_algorithm(system, 1e-10)
    print("Собственные значения:", *eigenvalues, sep='\t')
    eigenvalues = qr_algorithm(system2, 1e-10)
    print("Собственные значения:", *eigenvalues, sep='\t')



if __name__ == "__main__":
    main()
