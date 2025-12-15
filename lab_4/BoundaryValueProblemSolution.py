from math import sin, log, tan, pi
from typing import Tuple, List, Callable

import matplotlib.pyplot as plt
import numpy as np


from lab_4.CauchyProblemSolution import runge_kutta_method, print_results, runge_romberg_error_estimation
from lab_1.TridiagonalAlgorithm import tridiagonal_matrix_algorithm
from utility.matrix import Matrix


def exact_solution(x: int | float) -> float:
    """
    Функция являющиеся точным решением оду.
    :param x: Прообраз.
    :return: Образ.
    """
    return sin(x) + 2 - sin(x) * log((1 + sin(x)) / (1 - sin(x)))


def f(x: int | float, y : int | float, z: int | float) -> int | float:
    """
    Задача Коши представленная в виде z' = f(x, y, z).
    :param x: Прообраз.
    :param y: Образ.
    :param z: Z = y'(x).
    :return: Значение z'.
    """
    return tan(x) * z - 2 * y


def shooting_method(initial_condition: Tuple[float, float, float, float], step: float, eps: float,
                    func: Callable[[float, float, float], float]) -> Tuple[List[float], List[float]]:
    """
    Метод стрельбы для краевой задачи.
    :param initial_condition: (a, alpha, b, beta).
        a - левая граница, alpha = y(a).
        b - правая граница, beta = y(b).
    :param step: шаг для метода Рунге-Кутты.
    :param eps: Точность для оканчания вычислений.
    :param func: правая часть ОДУ y'' = f(x, y, y').
    :return: массивы X и Y.
    """
    a, alpha, b, beta = initial_condition

    def residual(s: float) -> float:
        """Вычисляет невязку F(s) = y(b; s) - beta."""
        X, Y, _ = runge_kutta_method((a, b), step, (a, alpha, 0, s), func, 4)
        return Y[-1] - beta

    s0 = (beta - alpha) / (b - a)
    s1 = s0 * 1.1 if s0 != 0 else 0.1

    F0 = residual(s0)
    F1 = residual(s1)

    while True:
        if abs(F1) < eps:
            X, Y, _ = runge_kutta_method((a, b), step, (a, alpha, 0, s1), func, 4)
            return X, Y

        if abs(F1 - F0) < 1e-15:
            s_new = (s0 + s1) / 2
        else:
            s_new = s1 - F1 * (s1 - s0) / (F1 - F0)

        s0, s1 = s1, s_new
        F0, F1 = F1, residual(s_new)

def finite_difference_method(initial_condition: Tuple[float, float, float, float], N: int,
                              funcs: Tuple[Callable[[int | float], int | float],
                              Callable[[int | float], int | float], Callable[[int | float], int | float]]) \
                              -> Tuple[List[float], List[float]]:

    """
    Конечно-разностный метод для решения оду.
    :param initial_condition: (a, alpha, b, beta).
        a - левая граница, alpha = y(a).
        b - правая граница, beta = y(b).
    :param N: количество узлов.
    :param funcs: Функции коэффициентов около y', y, правая часть.
    :return: массивы X и Y.
    """

    a, alpha, b, beta = initial_condition
    h = (b - a) / N
    x = [a + i * h for i in range(N + 1)]

    p, q, f = funcs

    A = [0.0] * (N + 1)
    B = [0.0] * (N + 1)
    C = [0.0] * (N + 1)
    D = [0.0] * (N + 1)

    for i in range(1, N):
        xi = x[i]

        A[i] = 1 - (h / 2) * p(xi)
        B[i] = -2 + h ** 2 * q(xi)
        C[i] = 1 + (h / 2) * p(xi)
        D[i] = h ** 2 * f(xi)

    D[1] = D[1] - A[1] * alpha
    D[N - 1] = D[N - 1] - C[N - 1] * beta

    a_coef = A[2:N]
    b_coef = B[1:N]
    c_coef = C[1:N-1]
    F = D[1:N]

    a_coef = [0.0] + a_coef
    c_coef = c_coef + [0.0]

    system = Matrix(
        rows=N - 1,
        columns=N - 1,
        free_members=F,
        diagonals=[c_coef, b_coef, a_coef]
    )

    y_internal = tridiagonal_matrix_algorithm(system)

    y = [0.0] * (N + 1)
    y[0] = alpha
    y[N] = beta

    for i in range(1, N):
        y[i] = y_internal[i - 1]

    return x, y


def plot_comparison_graphs(X_shooting: List[float], Y_shooting: List[float],
                           X_finite: List[float], Y_finite: List[float],
                           exact_solution_func: Callable[[float], float],
                           title_suffix: str = "") -> None:
    """
    Построение графиков сравнения численных решений с точным решением.

    :param X_shooting: Массив x для метода стрельбы
    :param Y_shooting: Массив y для метода стрельбы
    :param X_finite: Массив x для конечно-разностного метода
    :param Y_finite: Массив y для конечно-разностного метода
    :param exact_solution_func: Функция точного решения
    :param title_suffix: Дополнительная строка для заголовков графиков
    """

    # Вычисление точных значений
    Y_exact_shooting = [exact_solution_func(x) for x in X_shooting]
    Y_exact_finite = [exact_solution_func(x) for x in X_finite]

    # Вычисление погрешностей
    errors_shooting = [abs(y_num - y_exact) for y_num, y_exact in zip(Y_shooting, Y_exact_shooting)]
    errors_finite = [abs(y_num - y_exact) for y_num, y_exact in zip(Y_finite, Y_exact_finite)]

    # 2. Детальный график сравнения
    plt.figure(figsize=(14, 6))

    plt.subplot(2, 2, 1)
    plt.plot(X_shooting, Y_exact_shooting, 'k-', linewidth=3, label='Точное решение', alpha=0.8)
    plt.plot(X_shooting, Y_shooting, 'b--', linewidth=1.5, label='Метод стрельбы')
    plt.plot(X_finite, Y_finite, 'r-.', linewidth=1.5, label='Конечно-разностный метод')
    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title(f'Сравнение решений y\'\' - tan(x)y\' + 2y = 0\n{title_suffix}')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.subplot(1, 2, 2)
    # График относительных погрешностей
    rel_errors_shooting = []
    for err, y_exact in zip(errors_shooting, Y_exact_shooting):
        if abs(y_exact) > 1e-10:
            rel_errors_shooting.append(err / abs(y_exact))
        else:
            rel_errors_shooting.append(0)

    rel_errors_finite = []
    for err, y_exact in zip(errors_finite, Y_exact_finite):
        if abs(y_exact) > 1e-10:
            rel_errors_finite.append(err / abs(y_exact))
        else:
            rel_errors_finite.append(0)

    plt.plot(X_shooting, rel_errors_shooting, 'b-', label='Метод стрельбы', alpha=0.7)
    plt.plot(X_finite, rel_errors_finite, 'r-', label='Конечно-разностный метод', alpha=0.7)
    plt.xlabel('x')
    plt.ylabel('Относительная погрешность')
    plt.title('Относительные погрешности методов')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')

    plt.tight_layout()
    plt.show()


def plot_solutions_comparison(X_shooting: List[float], Y_shooting: List[float],
                              X_finite: List[float], Y_finite: List[float],
                              exact_solution_func: Callable[[float], float],
                              title: str = "Сравнение методов решения ОДУ") -> None:
    """
    Построение графиков сравнения численных решений с точным решением.

    :param X_shooting: Массив x для метода стрельбы
    :param Y_shooting: Массив y для метода стрельбы
    :param X_finite: Массив x для конечно-разностного метода
    :param Y_finite: Массив y для конечно-разностного метода
    :param exact_solution_func: Функция точного решения
    :param title: Заголовок для графиков
    """

    Y_exact_shooting = [exact_solution_func(x) for x in X_shooting]
    Y_exact_finite = [exact_solution_func(x) for x in X_finite]

    errors_shooting = [abs(y_num - y_exact) for y_num, y_exact in zip(Y_shooting, Y_exact_shooting)]
    errors_finite = [abs(y_num - y_exact) for y_num, y_exact in zip(Y_finite, Y_exact_finite)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.plot(X_shooting, Y_exact_shooting, 'k-', linewidth=3, label='Точное решение', alpha=0.8)
    ax1.plot(X_shooting, Y_shooting, 'b--', linewidth=1.5, label='Метод стрельбы')
    ax1.plot(X_finite, Y_finite, 'r-.', linewidth=1.5, label='Конечно-разностный метод')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y(x)', fontsize=12)
    ax1.set_title(f'Сравнение решений\n{title}', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax1.scatter(X_finite, Y_finite, color='red', s=20, alpha=0.6, zorder=5)


    rel_errors_shooting = []
    for err, y_exact in zip(errors_shooting, Y_exact_shooting):
        if abs(y_exact) > 1e-10:
            rel_errors_shooting.append(err / abs(y_exact))
        else:
            rel_errors_shooting.append(0)

    rel_errors_finite = []
    for err, y_exact in zip(errors_finite, Y_exact_finite):
        if abs(y_exact) > 1e-10:
            rel_errors_finite.append(err / abs(y_exact))
        else:
            rel_errors_finite.append(0)

    ax2.plot(X_shooting, rel_errors_shooting, 'b-', label='Метод стрельбы', linewidth=1.5, alpha=0.8)
    ax2.plot(X_finite, rel_errors_finite, 'r-', label='Конечно-разностный метод', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('Относительная погрешность', fontsize=12)
    ax2.set_title('Относительные погрешности методов', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    max_rel_shooting = max(rel_errors_shooting) if max(rel_errors_shooting) > 0 else 0
    max_rel_finite = max(rel_errors_finite) if max(rel_errors_finite) > 0 else 0

    ax2.text(0.05, 0.95, f'Макс. отн. погр. (стрельба): {max_rel_shooting:.2e}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.1))

    ax2.text(0.05, 0.88, f'Макс. отн. погр. (разности): {max_rel_finite:.2e}',
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.1))

    plt.tight_layout()
    plt.show()


def main():
    initial_condition = (0, 2, pi / 6, 2.5 - 0.5 * log(3))


    step = 0.001
    eps = 1e-8
    X_shooting, Y_shooting = shooting_method(initial_condition, step, eps, f)
    X_2h, Y_2h = shooting_method(initial_condition, step * 2, eps, f)
    solutions = (X_shooting, Y_shooting, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(step, solutions, 4)
    print_results(X_shooting, Y_shooting, errors, exact_solution, "Метод стрельбы.")


    N = 100
    funcs = (lambda x: -tan(x), lambda x: 2, lambda x: 0)
    X_finite, Y_finite = finite_difference_method(initial_condition, N, funcs)
    X_2h_fd, Y_2h_fd = finite_difference_method(initial_condition, N * 2, funcs)
    solutions = (X_finite, Y_finite, X_2h_fd, Y_2h_fd)
    errors = runge_romberg_error_estimation(step, solutions, 2)
    print_results(X_finite, Y_finite, errors, exact_solution, "Конечно разностный метод.")

    plot_solutions_comparison(X_shooting, Y_shooting, X_finite, Y_finite,
                              exact_solution,
                              f"y'' - tan(x)y' + 2y = 0\nN={N}, шаг={step}")



    # Вычисление ошибок для статистики
    Y_exact_shooting = [exact_solution(x) for x in X_shooting]
    Y_exact_finite = [exact_solution(x) for x in X_finite]

    errors_shooting = [abs(y_num - y_exact) for y_num, y_exact in zip(Y_shooting, Y_exact_shooting)]
    errors_finite = [abs(y_num - y_exact) for y_num, y_exact in zip(Y_finite, Y_exact_finite)]


if __name__ == '__main__':
    main()
