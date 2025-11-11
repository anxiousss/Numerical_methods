from math import acos, prod
from typing import List, Callable


def f(x: int | float) -> float:
    """
    Целевая функция y = f(x).
    """
    return acos(x) + x

def get_y_values(func: Callable[[int | float], float], X_values: List[int | float]) -> List[int | float]:
    """
    Получение Y_i из списка X_i.
    :param func: Целевая функция.
    :param X_values:  Массив прообразов.
    :return: Массив образов Y.
    """
    return [func(x) for x in X_values]

def lagrange_interpolation_polynominal(x: float, X_values: List[int | float],
                         Y_values: List[int | float]) -> float:
    """
    Вычисление интерполяционного многочлена Лагранжа в точке x.
    :param x: Значение точки в которой надо вычислить функцию.
    :param X_values: Массив прообразов.
    :param Y_values: Массив образов.
    :return Приблизительный образ y в точке f(x).
    """
    n = len(X_values)
    return sum(y * prod((x - X_values[j]) / (X_values[i] - X_values[j]) if i != j else 1
                     for j in range(n)) for i, y in enumerate(Y_values))

def divided_difference(func: Callable[[int | float], float], X_values: List[int | float]) -> float:
    """
    Вычисление разделённой разности.
    :param func: Целевая функция.
    :param X_values: Массив прообразов.
    :return: Значение разделенной разности для дискретного набора точек.
    """

    if len(X_values) == 1:
        return func(X_values[0])

    if len(X_values) == 2:
        return (func(X_values[1]) - func(X_values[0])) / (X_values[1] - X_values[0])
    return (divided_difference(func, X_values[1:]) - divided_difference(func, X_values[:-1])) / (
                X_values[-1] - X_values[0])

def newton_interpolation_polynominal(x: int |float, func: Callable[[int | float], float],
                                     diff_func: Callable[[Callable[[int | float], float], List[int | float]], float],
                                     X_values: List[int | float]) -> float:
    """
    Вычисление интерполяционного многочлена Ньютона в точке x.
    :param x: Значение точки в которой надо вычислить функцию.
    :param func: Целевая функция.
    :param diff_func: Функция разделенной разности.
    :param X_values: Массив прообразов.
    :return: Приблизительный образ y в точке f(x).
    """
    n = len(X_values)
    return sum(prod((x - X_values[j]) for j in range(i)) *
                                   diff_func(func, X_values[:i + 1]) for i in range(n))

def fault_estimate(x: float, func: Callable[[int | float], float], possible_value: float) -> float:
    """
    Оценка погрешности вычислений.
    :param x: Значение точки в которой надо вычислить функцию.
    :param func: Целевая функция.
    :param possible_value: Возможное значение точки из интерполяции.
    :return: Оценка погрешности.
    """
    return abs(func(x) - possible_value)

def main():
    X1 = [-0.4, -0.1, 0.2, 0.5]
    X2 = [-0.4, 0, 0.2, 0.5]
    x = 0.1
    Y1 = get_y_values(f, X1)
    Y2 = get_y_values(f, X2)
    lagrange_val1 = lagrange_interpolation_polynominal(x, X1, Y1)
    newton_val1 = newton_interpolation_polynominal(x, f, divided_difference, X1)
    estimate_lagrange1 = fault_estimate(x, f, lagrange_val1)
    estimate_newton1 = fault_estimate(x, f, newton_val1)
    print('Значения многочленов и оценка погрешности для массива X1')
    print(f'Значение многочлена Лагранжа = {lagrange_val1}. Значение многочлена Ньютона = {newton_val1}')
    print(f'Оценка погрешности многочлена Лагранжа = {estimate_lagrange1}. '
          f'Оценка погрености многочлена Ньютона = {estimate_newton1}', end='\n\n')

    lagrange_val2 = lagrange_interpolation_polynominal(x, X2, Y2)
    newton_val2 = newton_interpolation_polynominal(x, f, divided_difference, X2)
    estimate_lagrange2 = fault_estimate(x, f, lagrange_val2)
    estimate_newton2 = fault_estimate(x, f, newton_val2)
    print('Значения многочленов и оценка погрешности для массива X2')
    print(f'Значение многочлена Лагранжа = {lagrange_val2}. Значение многочлена Ньютона = {newton_val2}')
    print(f'Оценка погрешности многочлена Лагранжа = {estimate_lagrange2}. '
          f'Оценка погрености многочлена Ньютона = {estimate_newton2}')


if __name__ == '__main__':
    main()
