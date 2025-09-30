from typing import List

from utility.matrix import Matrix


def simple_iteration_method(system: Matrix, accuracy: float) -> List[int | float]:
    """
    Метод простой итерации.
    """
    # |a[i][i]| > сумма элементов под ним sum(a[i][j], j = 1...n, j != i)
    for i in range(system.rows):
        if abs(system.matrix[i][i]) <= sum(abs(system.matrix[i][j]) if i != j else 0 for j in range(system.columns)):
            raise ValueError('Не выполняется достаточное условие сходимости')

    if system.calc_determinant() == 0 or system.matrix[0][0] == 0:
        raise ValueError('Разложение не работает для вырожденных матриц.')

    alpha = Matrix(system.rows, system.columns, None)
    # alpha[i][j] 0 если i == j иначе -a[i][j] / a[i][i]
    alpha.matrix = [[0 if i == j else -system.matrix[i][j] / system.matrix[i][i]
                     for j in range(alpha.columns)] for i in range(alpha.rows)]
    # beta[i][j] = b[i] / a[i][i]
    beta = [system.free_members[i] / system.matrix[i][i] for i in range(system.rows)]
    X_prev, X_next = beta.copy(), beta.copy()
    diff = False
    while not diff:
        for i in range(system.rows):
            # x^(k+1) = alpha * x^k + beta
            # сумма произведений предыдущих иксов и элементов матрицы альфа + правая часть.
            X_next[i] = sum(alpha.matrix[i][l] * X_prev[l] if l != i else 0
                            for l in range(alpha.columns)) + beta[i]

        # Условие оканчания вычислений: |X[i]^(k + 1) - X[i]^k| < eps
        diff = all(abs(X_next[i] - X_prev[i]) < accuracy for i in range(system.rows))
        X_prev = X_next
        X_next = [0 for _ in range(system.rows)]

    return X_prev


def main():
    system = Matrix(4, 4, [[10, -1, -2, 5],
                           [4, 28, 7, 9],
                           [6, 5, -23, 4],
                           [1, 4, 5, -15]], [-99, 0, 67, 58])
    print(simple_iteration_method(system, 1e-20))


if __name__ == '__main__':
    main()
