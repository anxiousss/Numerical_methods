from math import tan, cos

from typing import Callable, Tuple


def equation(x: int | float) -> int | float:
    return tan(x) - 5 * x ** 2 + 1


def deriative(x: int | float) -> int | float:
    return 1 / (cos(x) ** 2) - 10 * x


def newton_method(equation: Callable[[int | float], int | float],
                  deriative: Callable[[int | float], int | float],
                  initial_approximation: int | float,
                  accuracy: float) -> Tuple[float, int]:

    x_prev = initial_approximation
    k = 0
    while True:
        x_next = x_prev - equation(x_prev) / deriative(x_prev)
        k += 1
        if abs(x_next - x_prev) < accuracy:
            return x_next, k
        x_prev = x_next





def main():
    root, iteration = newton_method(equation, deriative, 0.4, 1e-10)
    print(root, iteration)


if __name__ == '__main__':
    main()
