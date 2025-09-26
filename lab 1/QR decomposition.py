from typing import List, Tuple

import numpy as np

from utility.matrix import Matrix


def dot_product(a: List[int | float], b: List[int | float]) -> int | float:
    return sum(a[i] * b[i] for i in range(len(a)))


def projection_operator(a: List[int | float], b: List[int | float]) -> List[int | float]:
    coeff = dot_product(a, b) / dot_product(b, b)
    return [coeff * b[i] for i in range(len(b))]


def vector_add(a: List[int | float], b: List[int | float], sign: str) -> List[int | float]:
    if sign == '+':
        return [a[i] + b[i] for i in range(len(a))]
    return [a[i] - b[i] for i in range(len(a))]


def calculate_norm(vector: List[int | float]) -> int | float:
    return sum(el ** 2 for el in vector) ** 0.5


def stop_condition(A: Matrix, accuracy: float) -> bool:
    for i in range(A.rows):
        for j in range(A.columns):
            if (abs(A.matrix[i][j]) >= accuracy and i > j) or (abs(A.matrix[i][j]) == float('inf') and i < j):
                return False
    return True


def qr_decomposition(system: Matrix) -> Tuple[Matrix, Matrix]:
    system.transpose()
    orthogonal = Matrix(system.rows, system.columns, None, None, None, False)
    orthonormal = Matrix(system.rows, system.columns, None, None, None, False)
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
    while True:
        Q, R = qr_decomposition(A)
        A = R * Q
        if stop_condition(A, accuracy):
            break
    return [R.matrix[i][i] for i in range(R.rows)]


def main():
    system = Matrix(3, 3, [[6, 5, -6], [4, -6, 9], [-6, 6, 1]])
    A = np.array([[6, 5, -6], [4, -6, 9], [-6, 6, 1]])
    Q, R = np.linalg.qr(A)
    print(Q)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    print(eigenvalues)
    print(qr_algorithm(system, 1e-10))


if __name__ == "__main__":
    main()
