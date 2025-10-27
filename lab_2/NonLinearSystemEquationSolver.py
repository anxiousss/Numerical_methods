from math import log10, log, sqrt

from typing import Callable, List, Tuple

from utility.functions import get_function_arity

from utility.matrix import Matrix

from lab_1.LU_decomposition import lup_decomposition, solve_lup



def first_equation(x: int | float, y: int | float) -> int | float:
    return x ** 2 - 2 * log10(y) - 1


def second_equation(x: int | float, y: int | float) -> int | float:
    return x ** 2 - 2 * x * y + 2


def jacobian(x: int | float, y: int | float) -> List[List[int | float]]:
    return [[2 * x, -2 / (y * log(10))],
             [2 * x - 2 * y, -2 * x]]


def eq_first_equation(x: int | float, y: int | float) -> int | float:
    return sqrt(2 * log10(y) + 1)


def eq_second_equation(x: int | float, y: int | float) -> int | float:
    return (x ** 2 + 2) / (2 * x)


def eq_jacobian(x: int | float, y: int | float) -> List[List[int | float]]:
    return [[0, 1 / (y * sqrt(log(10)) * sqrt(2 * log(x) + log(10)))],
            [0.5  - 1 / x ** 2, 0]]


def system_newton_method(jacobian: Callable[..., List[List[int | float]]],
                  initial_approximation: List[int | float], *equations: Callable[..., int | float],
                  accuracy: int | float = 1e-6) -> Tuple[List[int | float], int]:
    """
    :param jacobian: Якобиан функций системы.
    :param initial_approximation: Начальное приближение.
    :param equations: Функции системы.
    :param accuracy: Точность.
    :return: Решение системы и количество итераций.
    """

    dimension = get_function_arity(jacobian)
    x_prev = initial_approximation

    iterations = 0
    while True:
        system = Matrix(dimension, dimension, jacobian(*x_prev),
                        free_members=[-eq(*x_prev) for eq in equations])
        L, U, P = lup_decomposition(system)
        delta_x = solve_lup(L, U, P, system.free_members)
        x_next = [x0 + dx for x0, dx in zip(x_prev, delta_x)]

        delta_norm = max(abs(x0 - x1) for x0, x1 in zip(x_prev, x_next))
        residual_norm = max(abs(eq(*x_next)) for eq in equations)

        if delta_norm <= accuracy and residual_norm <= accuracy:
            return x_next, iterations

        x_prev = x_next
        iterations += 1


def system_simple_iteration_method(eq_jacobian: Callable[..., List[List[int | float]]],
                                   initial_approximation: List[int | float], *eq_equations: Callable[..., int | float],
                                    accuracy: int | float = 1e-6):
    """
    :param eq_jacobian: Якобиан эквивалетных функций системы.
    :param initial_approximation: Начальное приближение.
    :param eq_equations: Эквивалетные функции системы.
    :param accuracy: Точность.
    :return: Решение системы и количество итераций.
    """
    q = 0.1
    eps = 0.1
    left_borders = []
    right_borders = []
    dimension = len(initial_approximation)
    for i in range(dimension):
        left_borders.append(initial_approximation[i] - eps)
        right_borders.append(initial_approximation[i] + eps)

    space = left_borders.copy()
    iterations = 10000
    i = 0
    while i < iterations:
        val = Matrix.calculate_norm(eq_jacobian(*space))
        if 1 > q > val:
            q = val
        space = [v + 1/iterations for i, v in enumerate(space)]
        i += 1


    x_prev = initial_approximation
    iterations = 0
    while True:
        x_next = [eq(*x_prev) for eq in eq_equations]
        if all((q / (1 - q)) * abs(x0 - x1) < accuracy for x0, x1 in zip(x_prev, x_next)):
            return x_next, iterations

        x_prev = x_next
        iterations += 1


def main():
    solution, i = system_newton_method(jacobian, [0.5, 2],
                                    first_equation, second_equation, accuracy=1e-20)
    print(f'solution = {solution} iterations = {i}')
    solution, i = system_simple_iteration_method(eq_jacobian,[0.5, 2],
                                                 eq_first_equation, eq_second_equation, accuracy=1e-20)
    print(f'solution = {solution} iterations = {i}')


if __name__ == '__main__':
    main()
