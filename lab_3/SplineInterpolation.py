from typing import List, Tuple

from utility.matrix import Matrix
from lab_1.TridiagonalAlgorithm import tridiagonal_matrix_algorithm


def make_system(X_values: List[int | float], F_values: List[int | float]) -> Tuple[List[int | float], Matrix]:
    """
    :param X_values: Прообразы узлов интерполяции.
    :param F_values: Образы узлов интерполяции.
    :return: Формирование линейной системы для решения.
    """
    n = len(X_values)
    lower_diagonal, main_diagonal, upper_diagonal, free_members = [], [], [], []
    h = [X_values[i] - X_values[i - 1] for i in range(1, n)]

    for i in range(1, n - 1):
        lower_diagonal.append(h[i - 1])
        main_diagonal.append(2 * (h[i - 1] + h[i]))
        upper_diagonal.append(h[i])
        free_members.append(6 * (
                (F_values[i + 1] - F_values[i]) / h[i] -
                (F_values[i] - F_values[i - 1]) / h[i - 1]
        ))

    if lower_diagonal:
        lower_diagonal[0] = 0
    if upper_diagonal:
        upper_diagonal[-1] = 0


    return (h, Matrix(
        rows = n - 2,
        columns = n - 2,
        free_members = free_members,
        diagonals = [upper_diagonal, main_diagonal, lower_diagonal]
    ))

def spline_coefficients(system: Matrix, F_values: List[int | float],
                        h: List[int | float]) -> List[Tuple[float, float, float, float]]:
    """
    Вычисление коэфициентов сплайна.
    :param system: Трехдиагональная система уравнений.
    :param F_values: Образы узлов интерполяции.
    :param h: Массив коэфициентов для вычислений.
    :return: Конечнын коэффициенты.
    """

    n = len(F_values)
    m_inner = tridiagonal_matrix_algorithm(system)

    m_full = [0.0] + m_inner + [0.0]

    coefficients = []

    for i in range(n - 1):
        a_i = F_values[i]
        b_i = (F_values[i + 1] - F_values[i]) / h[i] - h[i] * (2 * m_full[i] + m_full[i + 1]) / 6
        c_i = m_full[i] / 2
        d_i = (m_full[i + 1] - m_full[i]) / (6 * h[i])

        coefficients.append((a_i, b_i, c_i, d_i))

    return coefficients

def calc_spline(x: int | float, coefficients: List[Tuple[float, float, float, float]],
                X_values: List[int | float]) -> int | float:
    """
    Вычисляет значение кубического сплайна в точке.
    :param x: Точка для вычисления.
    :param coefficients: Коэфициенты для вычисления.
    :param X_values: Прообразы узлов интерполяции.
    :return: Образ интерполяции.
    """
    n = len(X_values)
    segment_index = -1
    for i in range(n - 1):
        if X_values[i] <= x <= X_values[i + 1]:
            segment_index = i
            break

    if segment_index == -1:
        if x < X_values[0]:
            segment_index = 0
        else:
            segment_index = n - 2

    a, b, c, d = coefficients[segment_index]
    left_border = X_values[segment_index]
    dx = x - left_border

    return a + b * dx + c * dx ** 2 + d * dx ** 3

def main():
    X = [-0.4, -0.1, 0.2, 0.5, 0.8]
    F = [1.5823, 1.5710, 1.5694, 1.5472, 1.4435]
    x = 0.1
    h, system = make_system(X, F)
    coefficients = spline_coefficients(system, F, h)
    point = calc_spline(x, coefficients, X)
    print(point)


if __name__ == '__main__':
    main()
