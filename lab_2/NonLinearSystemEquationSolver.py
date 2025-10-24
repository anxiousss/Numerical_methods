from math import log10, log, sqrt, cos, exp, sin

from typing import Callable, List, Tuple

from utility.functions import get_function_arity

from utility.matrix import Matrix

from lab_1.LU_decomposition import lup_decomposition, solve_lup



# 20 x ** 2 - 2 * log10(y) - 1
# 18 4 * x - cos(y)
def first_equation(x: int | float, y: int | float) -> int | float:
    return 4 * x - cos(y)

# 20 x ** 2 - 2 * x * y + 2
# 18 4 * y - exp(x)
def second_equation(x: int | float, y: int | float) -> int | float:
    return 4 * y - exp(x)


def jacobian(x: int | float, y: int | float) -> List[List[int | float]]:
    return [[2 * x, -2 / (y * log(10))],
             [2 * x - 2 * y, -2 * x]]


# 20 sqrt(2 * log10(y) + 1)
# 18 cos(y) / 4
def eq_first_equation(x: int | float, y: int | float) -> int | float:
    return cos(y) / 4


# 20 (x ** 2 + 2) / (2 * x)
# 18 exp(x) / 4
def eq_second_equation(x: int | float, y: int | float) -> int | float:
    return exp(x) / 4

# 20 [[0, 1 / (y * sqrt(log(10)) * sqrt(2 * log(x) + log(10)))], [0.5  - 1 / x ** 2], 0]
# 18 [[5, sin(y)], [1 - exp(x), 4]]
def eq_derivative(x: int | float, y: int | float) -> List[List[int | float]]:
    return [[5, sin(y)], [1 - exp(x), 4]]


def system_newton_method(jacobian: Callable[..., List[List[int | float]]],
                  initial_approximation: List[int | float], *equations: Callable[..., int | float],
                  accuracy: int | float = 1e-6) -> Tuple[List[int | float], int]:

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


def system_simple_iteration_method(initial_approximation: List[int | float], *eq_equations: Callable[..., int | float],
                                    accuracy: int | float = 1e-6):

    x_prev = initial_approximation
    iterations = 0
    while True:
        x_next = [eq(*x_prev) for eq in eq_equations]
        if all(abs(x0 - x1) < accuracy for x0, x1 in zip(x_prev, x_next)):
            return x_next, iterations

        x_prev = x_next
        iterations += 1
        print(x_next, iterations)


def main():
    """solution, i = system_newton_method(jacobian, [0.5, 2],
                                    first_equation, second_equation, accuracy=1e-10)
    print(f'solution = {solution} iterations = {i}')"""
    solution, i = system_simple_iteration_method([1.0, 1.0],
                                                 eq_first_equation, eq_second_equation, accuracy=1e-10)
    print(f'solution = {solution} iterations = {i}')


if __name__ == '__main__':
    main()
