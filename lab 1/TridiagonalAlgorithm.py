from typing import List


from utility.matrix import Matrix

def tridiagonal_matrix_algorithm(system: Matrix) -> List[int]:
    a, b, c, P, Q = [], [], [], [], []
    for i in range(system.rows):
        a_i = 0 if i == 0 else system.matrix[i][i - 1]
        b_i = system.matrix[i][i]
        c_i = 0 if i == system.rows else system.matrix[i - 1][i]
        P_i = -c_i / b_i if i == 0 else -c_i / (b_i + a_i * P[i - 1])
        Q_i = system.free_members[i] / b_i if i == 0\
            else (system.free_members[i] - a_i * Q[i - 1]) / (b_i + a_i * P[i - 1])

        a.append(a_i);b.append(b_i);c.append(c_i);P.append(P_i);Q.append(Q_i)

    X = []
    for i in range(system.rows - 1, -1, -1):
        x_i = Q[i] if i == system.rows - 1 else P[i] * X[system.rows - i - 2] + Q[i]
        X.append(x_i)

    print(a, b, c, P, Q, X)
    return X[::-1]

def main():
    system = Matrix(5, 5, [[-6, 6, 0, 0, 0],
                                             [2, 10, -7, 0, 0],
                                             [0, -8, 18, 9, 0],
                                             [0, 0, 6, -17, -6],
                                             [0, 0, 0, 9, 14]], [30, -31, 108, -114, 124])


    tridiagonal_matrix_algorithm(system)

if __name__ == '__main__':
    main()