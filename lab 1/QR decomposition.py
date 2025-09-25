from typing import List

from utility.matrix import Matrix


def dot_product(a: List[int | float], b: List[int | float]) -> int | float:
    return sum(a[i] * b[i] for i in range(len(a)))


def projection_operator(a: List[int | float], b: List[int | float]) -> List[int | float]:
    coeff = dot_product(a, b) / dot_product(b, b)
    return [coeff * b[i] for i in range(len(b))]


def vector_add(a: List[int | float], b: List[int | float], sign: str) ->List[int | float]:
    if sign == '+':
        return [a[i] + b[i] for i in range(len(a))]
    return [a[i] - b[i] for i in range(len(a))]


def calculate_norm(vector: List[int | float]) -> int | float:
    return sum(el ** 2 for el in vector) ** 0.5

def qr_decomposition(system: Matrix) -> List[int | float]:
    system.transpose()
    orthogonal = Matrix(system.rows, system.columns, None, None, None, False)
    orthonormal = Matrix(system.rows, system.columns, None, None, None, False)
    for i in range(system.rows):
        total = [0 for _ in range(system.rows)]
        for j in range(i):
            term = projection_operator(orthogonal.matrix[j], system.matrix[i])
            total = vector_add(total, term, '-')
        orthogonal.matrix[i] = vector_add(system.matrix[i], total, '+')
        norm = calculate_norm(orthogonal.matrix[i])
        orthonormal.matrix[i] = [el / norm for el in orthogonal.matrix[i]]

    R = orthonormal * system
    return [R.matrix[i][i] for i in range(R.rows)]


def main():
    system = Matrix(2, 2, [[3, 1], [1, 2]])
    print(qr_decomposition(system))


if __name__ == "__main__":
    main()
