from math import tan, cos, sqrt

from typing import Callable, Tuple


def equation(x: int | float) -> int | float:
    return tan(x) - 5 * x ** 2 + 1


def deriative(x: int | float) -> int | float:
    return 1 / (cos(x) ** 2) - 10 * x


def second_deriative(x: int | float) -> int | float:
    return 2 * tan(x) * (1 / cos(x)) ** 2


def eq_equation(x: int | float) -> int | float:
    return sqrt((tan(x) + 1) / 5)


def eq_deriative(x: int | float) -> int | float:
    return (1 / cos(x) ** 2) /  (2 * sqrt(5) * sqrt(tan(x) + 1))


def simple_iteration_method(eq_equation: Callable[[int | float], int | float],
                            eq_deriative: Callable[[int | float], int | float],
                            a: int | float,
                            b: int | float,
                            accuracy: float) -> Tuple[float, int]:


    x_prev = (a + b) / 2
    q = max(eq_deriative(a), eq_deriative(b))
    k = 0
    while True:
        x_next = eq_equation(x_prev)
        if (q / (1 - q)) * abs(x_next - x_prev) <= accuracy:
            return x_prev, k
        x_prev = x_next
        k += 1


def newton_method(equation: Callable[[int | float], int | float],
                  deriative: Callable[[int | float], int | float],
                  second_deriative: Callable[[int | float], int | float],
                  a: int | float, b: int | float,
                  accuracy: float) -> Tuple[float, int]:

    """
    Метод Ньютона работает если точка x0 выбрана так, что f(x0) * f''(x0) > 0, то начатая с нее последовательность xk
    k = 0, 1, 2 ... определяемая методом Ньютона, монотонно сходится к корню x;


    :param equation: Изначальное уравнение.
    :param deriative: Первая производная.
    :param second_deriative: Вторая Производная.
    :param a: Левая граница отрезка.
    :param b: Правая граница отрезка.
    :param accuracy: Точность.
    :return: Корень уравнения и количество итераций.
    """
    initial_approximation = (a + b) / 2

    if equation(initial_approximation) * second_deriative(initial_approximation) <= 0:
        raise ValueError('Условие сходимости не выполняется.')

    x_prev = initial_approximation
    k = 0
    while True:
        x_next = x_prev - equation(x_prev) / deriative(x_prev)
        k += 1
        if abs(x_next - x_prev) < accuracy:
            return x_next, k
        x_prev = x_next


def main():
    root, iteration = newton_method(equation, deriative, second_deriative, 0.2, 0.6, 1e-20)
    print(root, iteration)
    root, iteration = simple_iteration_method(eq_equation, eq_deriative, 0.2, 0.6, 1e-20)
    print(root, iteration)


if __name__ == '__main__':
    main()
