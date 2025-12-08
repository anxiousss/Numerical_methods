from typing import Tuple, List, Callable


def exact_solution(x: int | float) -> int | float:
    """
    Функция являющиеся точным решением оду.
    :param x: Прообраз.
    :return: Образ.
    """
    return x ** 1.5 if x >= 0 else (-x) ** 1.5

def f(x: int | float, y : int | float, z: int | float) -> int | float:
    """
    Задача Коши представленная в виде z' = f(x, y, z).
    :param x: Прообраз.
    :param y: Образ.
    :param z: Z = y'(x).
    :return: Значение z'.
    """
    return (-0.5 * z + 0.75 * y) / (x ** 2 - x)


def euler_method(space: Tuple[int | float, int | float], h: float | int,
                 initial_condition: Tuple[int | float, int | float, int | float, int | float],
                 func: Callable[[int | float, int | float, int | float], int | float]) \
        -> Tuple[List[int | float], List[int | float]]:

    """
    Метод Эйлера для оду 2 порядка.
    :param space: Отрезок исков на котором рассматривается решении.
    :param h: Шаг алгоритма.
    :param initial_condition: Начальное услоивие.
    :param func: Задача коши представленная в виде z' = f(x, y, z).
    :return: Два массива точек X, Y.
    """

    x0, y0, _, z0 = initial_condition
    a, b = space

    x_values = [x0]
    y_values = [y0]

    n_steps = int(round((b - a) / h))

    x_current = x0
    y_current = y0
    z_current = z0

    for _ in range(n_steps):
        if x_current + h > b + 1e-12:
            break

        y_next = y_current + h * z_current
        z_next = z_current + h * func(x_current, y_current, z_current)
        x_next = x_current + h

        x_values.append(x_next)
        y_values.append(y_next)

        x_current, y_current, z_current = x_next, y_next, z_next

    return x_values, y_values

def euler_cauchy_method(space: Tuple[int | float, int | float], h: float | int,
                 initial_condition: Tuple[int | float, int | float, int | float, int | float],
                 func: Callable[[int | float, int | float, int | float], int | float]) \
        -> Tuple[List[int | float], List[int | float]]:
    """
    Метод Эйлера-Коши для оду 2 порядка.
    :param space: Отрезок исков на котором рассматривается решении.
    :param h: Шаг алгоритма.
    :param initial_condition: Начальное услоивие.
    :param func: Задача коши представленная в виде z' = f(x, y, z).
    :return: Два массива точек X, Y.
    """


    x0, y0, _, z0 = initial_condition
    a, b = space

    x_values = [x0]
    y_values = [y0]

    n_steps = int(round((b - a) / h))

    x_current = x0
    y_current = y0
    z_current = z0

    for _ in range(n_steps):
        if x_current + h > b + 1e-12:
            break

        y_tilde = y_current + h * z_current
        z_tilde = z_current + h * func(x_current, y_current, z_current)

        y_next = y_current + h * 0.5 * (z_current + z_tilde)
        z_next = z_current + h * 0.5 * (
                func(x_current, y_current, z_current) +
                func(x_current + h, y_tilde, z_tilde)
        )
        x_next = x_current + h

        x_values.append(x_next)
        y_values.append(y_next)

        x_current, y_current, z_current = x_next, y_next, z_next

    return x_values, y_values

def improved_euler_method(space: Tuple[int | float, int | float], h: float | int,
                 initial_condition: Tuple[int | float, int | float, int | float, int | float],
                 func: Callable[[int | float, int | float, int | float], int | float]) \
        -> Tuple[List[int | float], List[int | float]]:
    """
    Улучшенный метод Эйлера для оду 2 порядка.
    :param space: Отрезок исков на котором рассматривается решении.
    :param h: Шаг алгоритма.
    :param initial_condition: Начальное услоивие.
    :param func: Задача коши представленная в виде z' = f(x, y, z).
    :return: Два массива точек X, Y.
    """

    x0, y0, _, z0 = initial_condition
    a, b = space

    x_values = [x0]
    y_values = [y0]

    n_steps = int(round((b - a) / h))

    x_current = x0
    y_current = y0
    z_current = z0

    for _ in range(n_steps):
        if x_current + h > b + 1e-12:
            break

        y_half = y_current + (h / 2) * z_current
        z_half = z_current + (h / 2) * func(x_current, y_current, z_current)
        x_half = x_current + h / 2

        y_next = y_current + h * z_half
        z_next = z_current + h * func(x_half, y_half, z_half)
        x_next = x_current + h

        x_values.append(x_next)
        y_values.append(y_next)

        x_current, y_current, z_current = x_next, y_next, z_next

    return x_values, y_values

def runge_kutta_method(space: Tuple[int | float, int | float], h: float | int,
                 initial_condition: Tuple[int | float, int | float, int | float, int | float],
                 func: Callable[[int | float, int | float, int | float], int | float], p: int) \
        -> Tuple[List[int | float], List[int | float], List[int | float]]:

    """
    Метод Рунге-Кутты для оду 2 порядка.
    :param space: Отрезок исков на котором рассматривается решении.
    :param h: Шаг алгоритма.
    :param initial_condition: Начальное услоивие.
    :param func: Задача коши представленная в виде z' = f(x, y, z).
    :param p: Порядок метода.
    :return: Два массива точек X, Y.
    """

    x0, y0, _, z0 = initial_condition
    a, b = space

    x_values = [x0]
    y_values = [y0]
    z_values = [z0]

    x_current, y_current, z_current = x0, y0, z0
    n_steps = int(round((b - a) / h))

    for _ in range(n_steps):
        if x_current + h > b + 1e-12:
            break

        K1_y = h * z_current
        K1_z = h * func(x_current, y_current, z_current)

        if p == 3:
            K2_y = h * (z_current + K1_z / 3)
            K2_z = h * func(x_current + h / 3,
                            y_current + K1_y / 3,
                            z_current + K1_z / 3)

            K3_y = h * (z_current + (K2_z * 2) / 3 )
            K3_z = h * func(x_current + (h * 2) / 3,
                            y_current + (K2_y * 2) / 3,
                            z_current + (K2_z * 2) / 3)

            y_next = y_current + (K1_y + 3 * K3_y) / 4
            z_next = z_current + (K1_z + 3 * K3_z) / 4

        elif p == 4:


            K2_y = h * (z_current + K1_z / 2)
            K2_z = h * func(x_current + h / 2,
                            y_current + K1_y / 2,
                            z_current + K1_z / 2)

            K3_y = h * (z_current + K2_z / 2)
            K3_z = h * func(x_current + h / 2,
                            y_current + K2_y / 2,
                            z_current + K2_z / 2)

            K4_y = h * (z_current + K3_z)
            K4_z = h * func(x_current + h,
                            y_current + K3_y,
                            z_current + K3_z)

            y_next = y_current + (K1_y + 2 * K2_y + 2 * K3_y + K4_y) / 6
            z_next = z_current + (K1_z + 2 * K2_z + 2 * K3_z + K4_z) / 6

        x_next = x_current + h

        x_values.append(x_next)
        y_values.append(y_next)
        z_values.append(z_next)

        x_current, y_current, z_current = x_next, y_next, z_next

    return x_values, y_values, z_values

def adams_bashforth_moulton_method(space: Tuple[int | float, int | float], h: float | int,
                 initial_condition: Tuple[int | float, int | float, int | float, int | float],
                 func: Callable[[int | float, int | float, int | float], int | float]) \
        -> Tuple[List[int | float], List[int | float]]:
    """
    Метод Адамса-Бэшфортса-Моултона для решения оду 2 порядка.
    :param space: Отрезок исков на котором рассматривается решении.
    :param h: Шаг алгоритма.
    :param initial_condition: Начальное услоивие.
    :param func: Задача коши представленная в виде z' = f(x, y, z).
    :return: Два массива точек X, Y.
    """
    x0, y0, _,  z0 = initial_condition
    a, b = space

    x_rk, y_rk, z_rk = runge_kutta_method(space, h, initial_condition, func, 4)

    X = x_rk[:4]
    Y = y_rk[:4]
    Z = z_rk[:4]
    F = [func(x, y, z) for x, y, z in zip(X, Y, Z)]

    n_steps = int(round((b - a) / h))

    for k in range(3, n_steps):
        if X[-1] + h > b + 1e-12:
            break

        x_new = X[k] + h

        y_new = Y[k] + h / 24 * (55 * Z[k] - 59 * Z[k - 1] + 37 * Z[k - 2] - 9 * Z[k - 3])
        z_new_pred = Z[k] + h / 24 * (55 * F[k] - 59 * F[k - 1] + 37 * F[k - 2] - 9 * F[k - 3])
        f_new_pred = func(x_new, y_new, z_new_pred)

        z_new = Z[k] + h / 24 * (9 * f_new_pred + 19 * F[k] - 5 * F[k - 1] + F[k - 2])

        X.append(x_new)
        Y.append(y_new)
        Z.append(z_new)
        F.append(func(x_new, y_new, z_new))

    return X, Y

def runge_romberg_error_estimation(space: Tuple[float, float], h: float,
                                   solutions: Tuple[List[float], List[float], List[float], List[float]], p: int = 1
                                   ) -> List[int | float]:

    """
    Оценка погрешности методом Рунге-Ромберга.
    :param space: Отрезок исков на котором рассматривается решении.
    :param h: Шаг алгоритма.
    :param solutions: Решения оду с шагами h и 2h.
    :param p: Порядок точности.
    :return: Возврашет список оценок главного члена погрешности.
    """
    x_h, y_h, x_2h, y_2h = solutions

    results = []

    for i, x_val in enumerate(x_2h):
        idx_h = int(round((x_val - space[0]) / h))

        if 0 <= idx_h < len(y_h) and abs(x_h[idx_h] - x_val) < 1e-10:
            y_h_val = y_h[idx_h]
            y_2h_val = y_2h[i]

            R_h = (y_h_val - y_2h_val) / (2 ** p - 1)

            results.append((x_val, y_h_val, R_h))

    return results


def print_results(X: List[int | float],  Y: List[int | float],
                  runge_estimates: List[int | float], method_name: str) -> None:
    """
    Вывод резульататов методов и сравнение с точной функцией.
    :param X: Массив Прообразов.
    :param Y: Массив прообразов.
    :param runge_estimates: Оценка погрешности методом Рунге-Ромберга.
    :param method_name: Имя метода.
    :return:
    """
    runge_dict = {x: (y_h, R_h) for x, y_h, R_h in runge_estimates}

    exact_error_data = []

    for x, y_approx in zip(X, Y):
        y_exact = exact_solution(x)
        abs_error = abs(y_approx - y_exact)
        rel_error = abs_error / abs(y_exact) if abs(y_exact) > 1e-12 else 0

        exact_error_data.append((x, y_approx, abs_error, rel_error))

    print(f"\n{method_name:^100}")
    print("=" * 120)
    print(f"{'x':<8} {'y_approx':<12} {'y_exact':<12} {'Abs Error':<15} "
          f"{'Rel Error':<15} {'R_h':<15} {'Ratio':<15} {'Error Type':<15}")
    print("-" * 120)

    for i, (x, y_approx) in enumerate(zip(X, Y)):
        y_exact = exact_solution(x)
        abs_error = abs(y_approx - y_exact)
        rel_error = abs_error / abs(y_exact) if abs(y_exact) > 1e-12 else 0

        if x in runge_dict:
            y_h, R_h = runge_dict[x]
            runge_abs = abs(R_h)
            ratio = abs_error / runge_abs if runge_abs > 0 else float('inf')

            error_type = "UNDER" if ratio < 0.5 else "OVER" if ratio > 2.0 else "GOOD"

            print(f"{x:<8.2f} {y_approx:<12.6f} {y_exact:<12.6f} "
                  f"{abs_error:<15.6e} {rel_error:<15.6e} {R_h:<15.6e} "
                  f"{ratio:<15.3f} {error_type:<15}")
        else:
            print(f"{x:<8.2f} {y_approx:<12.6f} {y_exact:<12.6f} "
                  f"{abs_error:<15.6e} {rel_error:<15.6e} {'N/A':<15} {'N/A':<15} {'N/A':<15}")

    print()


def main():
    # y(2)=2^1.5, y'(2)=1.5√2.
    initial_condition = (2, 2 ** 1.5, 2, 1.5 * (2 ** 0.5))
    space = (2, 3)
    h = 0.1
    X_h, Y_h = euler_method(space, h, initial_condition, f)
    X_2h, Y_2h = euler_method(space, 2 * h, initial_condition, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions)
    print_results(X_h, Y_h, errors, "Явный метод Эйлера.")

    X_h, Y_h = euler_cauchy_method(space, h, initial_condition, f)
    X_2h, Y_2h = euler_cauchy_method(space, 2 * h, initial_condition, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, 2)
    print_results(X_h, Y_h, errors, "Метод Эйлера-Коши.")

    X_h, Y_h = improved_euler_method(space, h, initial_condition, f)
    X_2h, Y_2h = improved_euler_method(space, 2 * h, initial_condition, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, 2)
    print_results(X_h, Y_h, errors, "Улучшенный метод Эйлера.")

    X_h, Y_h, _ = runge_kutta_method(space, h, initial_condition, f, 3)
    X_2h, Y_2h, _ = runge_kutta_method(space, 2 * h, initial_condition, f, 3)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, 3)
    print_results(X_h, Y_h, errors, "Метод Рунге-Кутты 3 порядка.")

    X_h, Y_h, _ = runge_kutta_method(space, h, initial_condition, f, 4)
    X_2h, Y_2h, _ = runge_kutta_method(space, 2 * h, initial_condition, f, 4)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, 4)
    print_results(X_h, Y_h, errors, "Метод Рунге-Кутты 4 порядка.")

    X_h, Y_h = adams_bashforth_moulton_method(space, h, initial_condition, f)
    X_2h, Y_2h = adams_bashforth_moulton_method(space, 2 * h, initial_condition, f)
    solutions = (X_h, Y_h, X_2h, Y_2h)
    errors = runge_romberg_error_estimation(space, h, solutions, 4)
    print_results(X_h, Y_h, errors, "Метод Адамса-Бэшфортса-Моултона.")



if __name__ == '__main__':
    main()
