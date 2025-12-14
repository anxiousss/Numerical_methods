from math import sin, log, tan, pi
from typing import Tuple, List, Callable

from lab_4.CauchyProblemSolution import runge_kutta_method, print_results, runge_romberg_error_estimation


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
        print(F1)

"""def finite_differrence_method(initial_condition: Tuple[float, float, float, float], N):
    a, alpha, b, beta = initial_condition
    h = (b - a) / N
    x_i = [a + i * h for i in range(0, N + 1)]
    y_i = []
    for i in range(0, N + 1):
        if i == 0:
            y_i.append(alpha)
        elif i == N:
            y_i.append(beta)
        else:
            y_i.append()"""


def main():
    initial_condition = (0, 2, pi / 6, 2.5 - 0.5 * log(3))
    step = 0.001
    eps = 1e-8
    X_h, Y_h = shooting_method(initial_condition, step, eps, f)
    X_2h, Y_2h = shooting_method(initial_condition, step * 2, eps, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation((0, pi / 6), step, solutions, 4)
    print_results(X_h, Y_h, errors, exact_solution, "Метод стрельбы.")



if __name__ == '__main__':
    main()
