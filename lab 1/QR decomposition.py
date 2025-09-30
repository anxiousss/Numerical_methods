from typing import List, Tuple

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


def stop_condition(A: Matrix, accuracy: float) -> bool:
    """
    Условие остановки работы QR алгоритма.
    """
    for i in range(A.rows):
        for j in range(A.columns):
            if (abs(A.matrix[i][j]) >= accuracy and i > j) or (abs(A.matrix[i][j]) == float('inf') and i < j):
                return False
    return True


def qr_decomposition(system: Matrix) -> Tuple[Matrix, Matrix]:
    system.transpose()
    orthogonal = Matrix(system.rows, system.columns, None, None, None, False)
    orthonormal = Matrix(system.rows, system.columns, None, None, None, False)
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


def qr_algorithm(A: Matrix, accuracy: float) -> List[int | float]:
    # Все матрицы A являются подобными
    current_A = A
    while True:
        Q, R = qr_decomposition(current_A)
        current_A = R * Q
        if stop_condition(current_A, accuracy):
            break
    return [current_A.matrix[i][i] for i in range(current_A.rows)]


def main():
    system = Matrix(3, 3, [[6, 5, -6], [4, -6, 9], [-6, 6, 1]])

    eigenvalues = qr_algorithm(system, 1e-10)
    print("Собственные значения:", eigenvalues)


if __name__ == "__main__":
    main()
