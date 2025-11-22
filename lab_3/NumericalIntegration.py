from typing import Callable, List
from math import sqrt


def y(x: int | float) -> int | float:
    """
    Подыинтегральная функция.
    :param x: Аргумент функции.
    :return: Образ.
    """
    return sqrt(x) / (4 + 3 * x)

def midpoint_rule(func: Callable[[int | float], int | float], step: int | float,
                  a: int | float, b: int | float) -> int | float:
    """
    Вычисляет определенный интеграл от a до b для функции func методом прямоугольников.
    :param func: Функция для которой вычисляется интеграл.
    :param step: Шаг интегрирования.
    :param a: Начало отрезка.
    :param b: Конец отрезка.
    :return: Приблизительное значение интгерала.
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
    Вычисляет определенный интеграл от a до b для функции func методом трапеций.
    :param func: Функция для которой вычисляется интеграл.
    :param step: Шаг интегрирования.
    :param a: Начало отрезка.
    :param b: Конец отрезка.
    :return: Приблизительное значение интгерала.
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
    """
    Вычисляет определенный интеграл от a до b для функции func методом Симпсона.
    :param func: Функция для которой вычисляется интеграл.
    :param step: Шаг интегрирования.
    :param a: Начало отрезка.
    :param b: Конец отрезка.
    :return: Приблизительное значение интгерала.
    """
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

def runge_romberg_richardson_method(step_value: int | float,
                                    half_step_value: int | float, accuracy_order: int | float) -> int | float:
    """
    Вычисляет уточненное значение интеграла.
    :param step_value: Значение с шагом h.
    :param half_step_value: Значение с шагом h/2.
    :param accuracy_order: Порядок точности.
    :return: Уточненное значение.
    """

    return half_step_value + (half_step_value - step_value) / (2 ** accuracy_order - 1)


def main():
    exact_value = 0.53122
    a = 1
    b = 5
    h1 = 1.0
    h2 = 0.5

    methods = [
        ("Метод прямоугольников", midpoint_rule),
        ("Метод трапеций", trapezoidal_rule),
        ("Метод Симпсона", simpsons_rule)
    ]

    print("Вычисление интеграла ∫(√x/(4+3x))dx от 1 до 5")
    print("Точное значение: 0.53122")
    print("-" * 85)
    print(f"{'Метод':<25} | {'h = 1.0':<12} | {'h = 0.5':<12} | {'Уточненное':<12} | {'Погрешность':<12}")
    print("-" * 85)

    for method_name, method_func in methods:
        if method_name == "Метод Симпсона":
            accuracy_order = 4
        else:
            accuracy_order = 2

        value_h1 = method_func(y, h1, a, b)
        value_h2 = method_func(y, h2, a, b)

        refined_value = runge_romberg_richardson_method(
            value_h1, value_h2, accuracy_order
        )

        error = abs(refined_value - exact_value)

        print(f"{method_name:<25} | {value_h1:<12.6f} | {value_h2:<12.6f} | "
              f"{refined_value:<12.6f} | {error:<12.2e}")

    print("-" * 85)


if __name__ == '__main__':
    main()
