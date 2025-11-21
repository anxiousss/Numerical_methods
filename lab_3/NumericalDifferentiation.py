from typing import List, Tuple


def boundary_defining(X: List[int | float], Y: List[int | float],
                      x: int | float) -> List[Tuple[int | float, int | float, int | float]]:
    """
    :param X:
    :param Y:
    :param x:
    :return:
    """

    if x < X[0] or x > X[-1]:
        raise ValueError('Точка вне границ X')

    for i in range(len(X) - 2):
        x1, x2, x3 = X[i], X[i + 1], X[i + 2]

        if x1 <= x <= x2:
            y1, y2, y3 = Y[i], Y[i + 1], Y[i + 2]
            return [(x1, x2, x3), (y1, y2, y3)]


def first_derivative(values: List[Tuple[int | float, int | float, int | float]], x: int | float) -> int | float:
    """

    :param values:
    :param x:
    :return:
    """
    x1, x2, x3 = values[0]
    y1, y2, y3 = values[1]

    return (y2 - y1) / (x2 - x1) + (((y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)) / (x3 - x1)) * (2 * x - x1 - x2)


def second_deriaitive(values: List[Tuple[int | float, int | float, int | float]]) -> int | float:
    """

    :param values:
    :return:
    """
    x1, x2, x3 = values[0]
    y1, y2, y3 = values[1]

    return 2 * (((y3 - y2) / (x3 - x2) - (y2 - y1) / (x2 - x1)) / (x3 - x1))


def main():
    X = [-1.0, 0.0, 1.0, 2.0, 3.0]
    Y = [1.3562, 1.5708, 1.7854, 2.4636, 3.3218]
    x = 1.0
    values = boundary_defining(X, Y, x)
    fd = first_derivative(values, x)
    sd = second_deriaitive(values)

    print(f'Первая производная = {fd}')
    print(f'Вторая производная = {sd}')


if __name__ == '__main__':
    main()
