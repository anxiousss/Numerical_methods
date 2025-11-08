from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np

from utility.matrix import Matrix
from lab_1.LU_decomposition import lup_decomposition, solve_lup


def calculate_coefficients(X_values: List[float], Y_values: List[float], dimension: int) -> Matrix:
    """
    Вычисление коэфицентов системы.
    :param X_values: Прообразы узлов интерполяции.
    :param Y_values: Образы узлов интерполяции.
    :param dimension: Размерность матрицы.
    :return: Объект матрицы.
    """
    matrix = []
    free_members = []
    N = len(X_values)

    for k in range(dimension):
        row = []
        for i in range(dimension): 
            term = sum(x ** (k + i) for x in X_values)
            row.append(term)

        free_term = sum(Y_values[j] * (X_values[j] ** k) for j in range(N))
        free_members.append(free_term)
        matrix.append(row)

    return Matrix(dimension, dimension, matrix, free_members)

def approximating_polynomial(X_values: List[int | float], Y_values: List[int | float], degree: int) -> List[int | float]:
    """
    Решает систему МНК.
    :param X_values: Прообразы узлов интерполяции.
    :param Y_values: Образы узлов интерполяции.
    :param degree: Степень многочлена - 1.
    :return: Решение системы.
    """
    system = calculate_coefficients(X_values, Y_values, degree)
    L, U, P = lup_decomposition(system)
    return solve_lup(L, U, P, system.free_members)

def polynomial_value(coefficients: List[float], x: float) -> int | float:
    """
    Вычисляет значение многочлена в точке x.
    :param coefficients: Коэфициенты многочлена.
    :param x: Значение точки.
    :return: Значение многочлена в точке.
    """
    return sum(coef * (x ** i) for i, coef in enumerate(coefficients))


def squared_errors_sum(F: Callable[[List[int | float], int | float], int | float],
                       polynomial_coefficients: List[float], X_values: List[float],
                       Y_values: List[float], degree: int) -> float:
    """
    Вычисляет сумму квадратов ошибок.
    :param F: Функция вычисления многочлена в точке.
    :param polynomial_coefficients: Коэфициенты многочлена.
    :param X_values: Прообразы узлов интерполяции.
    :param Y_values: Образы узлов интерполяции.
    :param degree: Степень многочлена - 1.
    :return Сумма квадратов ошибки.
    """
    return sum((F(polynomial_coefficients, X_values[i]) - Y_values[i]) ** 2
               for i in range(degree))

def make_polynominal(coefficients: List[int | float]) -> str:
    """
    Составляет строку многочлен.
    :param coefficients: Коэфициенты многочлена.
    :return: Строка многочлен.
    """
    return ' '.join(f'{coeff}x^{i}' for i, coeff in enumerate(coefficients))




def main():
    X = [-0.7, -0.4, -0.1, 0.2, 0.5, 0.8]
    Y = [1.6462, 1.5823, 1.571,	1.5694,	1.5472,	1.4435]
    coefficients_first_degree = approximating_polynomial(X, Y, 2)
    print('Многочлен первой степени - ' + make_polynominal(coefficients_first_degree))
    print(f'Квадратичное отклонение - {squared_errors_sum(polynomial_value, coefficients_first_degree, X, Y, 2)}')
    coefficients_second_degree = approximating_polynomial(X, Y, 3)
    print('Многочлен второй степени - ' + make_polynominal(coefficients_second_degree))
    print(f'Квадратичное отклонение - {squared_errors_sum(polynomial_value, coefficients_second_degree, X, Y, 3)}')

    x_plot = np.linspace(min(X) - 0.1, max(X) + 0.1, 100)

    y_linear = [polynomial_value(coefficients_first_degree, x) for x in x_plot]
    y_quadratic = [polynomial_value(coefficients_second_degree, x) for x in x_plot]

    plt.figure(figsize=(10, 6))

    plt.scatter(X, Y, color='red', s=50, zorder=5, label='Табличные данные')

    plt.plot(x_plot, y_linear, 'b-', linewidth=2, label='Многочлен 1-й степени')

    plt.plot(x_plot, y_quadratic, 'g--', linewidth=2, label='Многочлен 2-й степени')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Аппроксимация методом наименьших квадратов')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    main()
