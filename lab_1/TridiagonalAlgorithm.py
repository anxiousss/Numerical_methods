from typing import List
from utility.matrix import Matrix


def tridiagonal_matrix_algorithm(system: Matrix) -> List[int]:
    """
    Метод прогонки для системы Ax = F. A - трехдиагональная матрица.
    A[i]X[i - 1] + B[i]X[i] + C[i + 1]X[i + 1] = F[i]
    """
    a, b, c = system.diagonals[2], system.diagonals[1], system.diagonals[0]
    P, Q = [], []
    for i in range(system.rows):
        P_i = -c[i] / b[i] if i == 0 else -c[i] / (b[i] + a[i] * P[i - 1])
        Q_i = system.free_members[i] / b[i] if i == 0 \
            else (system.free_members[i] - a[i] * Q[i - 1]) / (b[i] + a[i] * P[i - 1])

        P.append(P_i)
        Q.append(Q_i)

    X = []
    for i in range(system.rows - 1, -1, -1):
        x_i = Q[i] if i == system.rows - 1 else P[i] * X[system.rows - i - 2] + Q[i]
        X.append(x_i)

    return X[::-1]


def main():
    system = Matrix(5, 5, None, [30, -31, 108, -114, 124],
                    [[6, -7, 9, -6, 0], [-6, 10, 18, -17, 14], [0, 2, -8, 6, 9]])

    print(tridiagonal_matrix_algorithm(system))


if __name__ == '__main__':
    main()
