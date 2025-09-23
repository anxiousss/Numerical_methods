from typing import List

from utility.matrix import Matrix


def simple_iteration_method(system: Matrix, accuracy: float) -> List[int | float]:
    if system.calc_determinant() == 0 or system.matrix[0][0] == 0:
        raise ValueError('Разложение не работает для вырожденных матриц')
    alpha = Matrix(system.rows, system.columns, None)
    alpha.matrix = [[0 if i == j else -system.matrix[i][j] / system.matrix[i][i]
                     for j in range(alpha.columns)] for i in range(alpha.rows)]
    beta = [system.free_members[i] / system.matrix[i][i] for i in range(system.rows)]
    X_prev, X_next = beta.copy(), beta.copy()
    diff = False
    c = 0
    while not diff:
        for i in range(system.rows):
            X_next[i] = sum(alpha.matrix[i][l] * X_prev[l] if l != i else 0
                            for l in range(alpha.columns)) + beta[i]
        diff = all(abs(X_next[i] - X_prev[i]) < accuracy for i in range(system.rows))
        X_prev = X_next
        X_next = [0 for _ in range(system.rows)]

    """for i in range(system.rows):
        if alpha.matrix[i][i] <= sum(abs(alpha.matrix[i][j]) if i != j else 0 for j in range(system.columns)):
            raise ValueError('SKILL ISSUE')"""

    return X_prev

def main():
    system = Matrix(4, 4, [[10, -1, -2, 5],
                                             [4, 28, 7, 9],
                                             [6, 5, -23, 4],
                                             [1, 4, 5, -15]], [-99, 0, 67, 58])
    print(simple_iteration_method(system, 1e-20))


if __name__ == '__main__':
    main()
