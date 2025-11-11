import numpy as np
import matplotlib.pyplot as plt
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
        free_members.append(3 * (
                (F_values[i + 1] - F_values[i]) / h[i] -
                (F_values[i] - F_values[i - 1]) / h[i - 1]
        ))

    if lower_diagonal:
        lower_diagonal[0] = 0
    if upper_diagonal:
        upper_diagonal[-1] = 0

    return (h, Matrix(
        rows=n - 2,
        columns=n - 2,
        free_members=free_members,
        diagonals=[upper_diagonal, main_diagonal, lower_diagonal]
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


def plot_spline_and_original(X: List[float], F: List[float], num_points: int = 1000):
    """
    Построение графика сплайна и изначальных отрезков.

    :param X: Узлы интерполяции (x-координаты)
    :param F: Значения функции в узлах (y-координаты)
    :param num_points: Количество точек для построения сплайна
    """
    h, system = make_system(X, F)
    coefficients = spline_coefficients(system, F, h)

    x_min, x_max = min(X), max(X)
    x_smooth = np.linspace(x_min, x_max, num_points)
    y_smooth = [calc_spline(x, coefficients, X) for x in x_smooth]

    plt.figure(figsize=(12, 8))

    plt.plot(X, F, 'o-', label='Исходные отрезки', linewidth=2, markersize=8, color='red', alpha=0.7)

    plt.plot(x_smooth, y_smooth, label='Кубический сплайн', linewidth=2, color='blue')

    plt.title('Интерполяция кубическим сплайном', fontsize=14)
    plt.xlabel('X', fontsize=12)
    plt.ylabel('F(X)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    for i, (x, y) in enumerate(zip(X, F)):
        plt.annotate(f'({x:.1f}, {y:.4f})', (x, y),
                     xytext=(5, 5), textcoords='offset points', fontsize=9)

    plt.tight_layout()
    plt.show()


def main():
    X = [-0.4, -0.1, 0.2, 0.5, 0.8]
    F = [1.5823, 1.5710, 1.5694, 1.5472, 1.4435]

    h, system = make_system(X, F)
    coefficients = spline_coefficients(system, F, h)
    x_point = 0.1
    point_value = calc_spline(x_point, coefficients, X)
    print(f"Значение сплайна в точке x={x_point}: {point_value}")

    plot_spline_and_original(X, F)


if __name__ == '__main__':
    main()