from typing import Callable, List
from math import sqrt



def y(x: int | float) -> int | float:
    """

    :param x:
    :return:
    """
    return sqrt(x) / (4 + 3 * x)


def test_y(x):
    return x / ((3 * x + 4)**2)

def midpoint_rule(func: Callable[[int | float], int | float], step: int | float,
                  a: int | float, b: int | float) -> int | float:
    """

    :param func:
    :param step:
    :param a:
    :param b:
    :return:
    """

    total = 0
    current_x = a
    while current_x < b:
        x0, x1 = current_x, current_x + step
        term = step * func((x0 + x1) / 2)
        total += term
        current_x += step

    return total

def trapezoidal_rule(func: Callable[[int | float], int | float], step: int | float,
                     a: int | float, b: int | float) -> int | float:
    """

    :param func:
    :param step:
    :param a:
    :param b:
    :return:
    """

    total = 0
    current_x = a
    while current_x < b:
        f0, f1 = func(current_x), func(current_x + step)
        term = step * (f0 + f1)
        total += term
        current_x += step

    return total / 2


def simpsons_rule(func: Callable[[int | float], int | float], step: int | float,
                  a: int | float, b: int | float) -> int | float:
    n = int((b - a) / step)
    if n % 2 != 0:
        n -= 1
        step = (b - a) / n

    total = func(a) + func(b)
    current_x = a + step

    for i in range(1, n, 2):
        total += 4 * func(current_x)
        current_x += 2 * step

    current_x = a + 2 * step
    for i in range(2, n, 2):
        total += 2 * func(current_x)
        current_x += 2 * step

    return total * step / 3

def runge_romberg_richardson_method(exact_value: int | float, step_value: int | float,
                                    half_step_value: int | float, accuracy_order: int | float) -> int | float:
    """

    :param exact_value:
    :param step_value:
    :param half_step_value:
    :param accuracy_order:
    :return:
    """

    return half_step_value + (half_step_value - step_value) / (2 ** accuracy_order - 1)



def main():
    # a = 1
    # b = 5
    # steps = [1.0, 0.5]
    exact_value = -0.16474
    step_value = simpsons_rule(test_y, 0.5, -1.0, 1.0)
    half_step_value = simpsons_rule(test_y, 0.25, -1.0, 1.0)
    estimate = runge_romberg_richardson_method(exact_value, step_value, half_step_value, 4)
    print(exact_value - estimate)



if __name__ == '__main__':
    main()
