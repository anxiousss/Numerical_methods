from typing import List, Callable

from utility.matrix import Matrix
from lab_1.SeidelMethod import seidel_method
from lab_1.SimpleIterationMethod import simple_iteration_method
from lab_1.LU_decomposition import lup_decomposition, solve_lup

accuracy = 1e-20


def calculate_coefficients(X_values: List[int | float], Y_values: List[float | int ], dimension: int) -> Matrix:
    matrix = []
    free_members= []
    N = len(X_values)
    for i in range(dimension):
        """row = [sum(X_values[j] ** (degree + j) if j + degree != 0
                   else len(X_values) + 1 for j in range(N)) for _ in range(dimension)]
        free_members.append(sum(Y_values[j] * (X_values[j] ** (degree + j)) if j + degree != 0
                                else Y_values[j] for j in range(N)))"""

        row = []
        for j in range(dimension):
            term = sum(X_values[j] ** (j + k) for k in range(N)) if i != 0 else N + 1
            row.append(term)

        free_members.append(sum(Y_values[j] * (X_values[j] ** i) if i != 0 else Y_values[j] for j in range(N)))
        matrix.append(row)

    return Matrix(dimension, dimension, matrix, free_members)

def solve_system(system: Matrix) -> List[int | float]:
    L, U, P = lup_decomposition(system)
    return solve_lup(L, U, P, system.free_members)

def approximating_polynomial(X_values: List[int | float], Y_value: List[int | float], degree: int) -> List[int | float]:
    system = calculate_coefficients(X_values, Y_value, degree)
    polynominal_coefficients = solve_system(system)
    return polynominal_coefficients

def F(system_coefficients: List[int | float], x: int | float, dimension: int) -> int | float:
    return sum(system_coefficients[i] * (x ** i) for i in range(dimension))

def squared_errors_sum(F: Callable[[List[int | float], int | float, int], int | float],
                       system_coefficients: List[int | float], X_values: List[int | float],
                       Y_values: List[float | int ], dimension: int):
        return sum((F(system_coefficients, X_values[i], dimension) - Y_values[i]) ** 2 for i in range(dimension))



def main():
    X = [-0.7, -0.4, -0.1, 0.2, 0.5, 0.8]
    Y = [1.6462, 1.5823, 1.571,	1.5694,	1.5472,	1.4435]
    coefficients_first_degree = approximating_polynomial(X, Y, 2)
    print(f'Многочлен первой степени - {coefficients_first_degree[0]} + {coefficients_first_degree[1]}x')
    print(f'Квадратичное отклонение - {squared_errors_sum(F, coefficients_first_degree, X, Y, 2)}')
    coefficients_second_degree = approximating_polynomial(X, Y, 3)
    print(f'Многочлен второй степени - {coefficients_second_degree[0]} + {coefficients_second_degree[1]}x +{coefficients_second_degree[2]}x**2 ')
    print(f'Квадратичное отклонение - {squared_errors_sum(F, coefficients_second_degree, X, Y, 3)}')




if __name__ == '__main__':
    main()
