"""
Лабораторная работа № 1. Вариант 11.
Итерационные методы вычисления √a, ∛b, ln c.

Автор: Колесный Степан, группа БИ2-1.

Исходные данные варианта 11:
    a = 29, b = 70, c = 50, eps = 1e-4,
    критерий остановки: Р (по разности соседних приближений),
    начальное приближение: x0 = 1,
    дополнительное требование: Д3 (нормализация аргумента логарифма).

Запуск:  python lab1_variant11.py
"""
import math

# ---------------------------------------------------------------- параметры
A, B, C = 29, 70, 50
EPS = 1e-4
X0 = 1.0
N_MAX = 100
N_MAX_LN = 10_000


# ------------------------------------------------------------ квадратный корень
def sqrt_newton(a, eps=EPS, x0=X0, n_max=N_MAX, trace=None):
    """Квадратный корень методом Ньютона (формула Герона).

    Возвращает (результат, число итераций).
    Если передан список trace, в него добавляются кортежи (n, x_n, |x_n - x_{n-1}|).
    """
    if a < 0:
        raise ValueError("Подкоренное выражение должно быть неотрицательным")
    if a == 0:
        return 0.0, 0
    x = float(x0)
    for n in range(1, n_max + 1):
        x_new = 0.5 * (x + a / x)
        if trace is not None:
            trace.append((n, x_new, abs(x_new - x)))
        if abs(x_new - x) < eps:          # критерий Р
            return x_new, n
        x = x_new
    raise RuntimeError("Точность не достигнута за N_MAX итераций")


# ------------------------------------------------------------ кубический корень
def cbrt_newton(b, eps=EPS, x0=X0, n_max=N_MAX, trace=None):
    """Кубический корень методом Ньютона (с учётом знака аргумента)."""
    if b == 0:
        return 0.0, 0
    sign = -1 if b < 0 else 1
    b = abs(b)                            # метод применяется к модулю
    x = float(x0)
    for n in range(1, n_max + 1):
        x_new = (2 * x + b / (x * x)) / 3
        if trace is not None:
            trace.append((n, sign * x_new, abs(x_new - x)))
        if abs(x_new - x) < eps:          # критерий Р
            return sign * x_new, n
        x = x_new
    raise RuntimeError("Точность не достигнута за N_MAX итераций")


# ------------------------------------------------- натуральный логарифм (ряд)
def ln_series(c, eps=EPS, n_max=N_MAX_LN):
    """ln c = 2 (y + y^3/3 + y^5/5 + ...), y = (c - 1)/(c + 1).

    Без нормализации аргумента. Возвращает (результат, число членов ряда).
    """
    if c <= 0:
        raise ValueError("Аргумент логарифма должен быть положительным")
    y = (c - 1) / (c + 1)
    y2 = y * y
    power = y                              # текущая степень y^(2k+1)
    total = 0.0
    for k in range(n_max):
        term = 2 * power / (2 * k + 1)
        if abs(term) < eps:
            return total, k
        total += term
        power *= y2
    raise RuntimeError("Точность не достигнута за n_max итераций")


# ------------------------------------- Д3: логарифм с нормализацией аргумента
def normalize(c):
    """Представляет c = m * 2^k, m in [0.5, 1). Возвращает (m, k).

    Используются только умножение и деление на 2 (без math.log / frexp).
    """
    if c <= 0:
        raise ValueError("Аргумент логарифма должен быть положительным")
    m, k = float(c), 0
    while m >= 1:
        m /= 2
        k += 1
    while m < 0.5:
        m *= 2
        k -= 1
    return m, k


def ln_normalized(c, eps=EPS, n_max=N_MAX_LN):
    """ln c = ln m + k ln 2, оба логарифма считаются рядом при |y| <= 1/3.

    Возвращает (результат, суммарное число членов ряда, m, k).
    """
    m, k = normalize(c)
    ln_m, n_m = ln_series(m, eps, n_max)
    if k != 0:
        ln2, n_2 = ln_series(2.0, eps, n_max)
    else:
        ln2, n_2 = 0.0, 0
    return ln_m + k * ln2, n_m + n_2, m, k


# -------------------------------------------------------------------- вывод
def table_row(name, arg, val, ref, n):
    return (f"{name:<10}{arg:>10}{val:>16.10f}{ref:>16.10f}"
            f"{abs(val - ref):>14.2e}{n:>10}")


def print_trace(title, trace, ref):
    print(f"\n{title}")
    print(f"{'n':>3}{'x_n':>18}{'|x_n - x_(n-1)|':>20}{'|x_n - эталон|':>18}")
    for n, x, d in trace:
        print(f"{n:>3}{x:>18.10f}{d:>20.3e}{abs(x - ref):>18.3e}")


def main():
    # эталонные значения (допустимо использовать math только здесь)
    ref_sqrt = math.sqrt(A)
    ref_cbrt = B ** (1 / 3)
    ref_ln = math.log(C)

    print(f"Вариант 11: a = {A}, b = {B}, c = {C}, eps = {EPS}, x0 = {X0}\n")

    v_sqrt, n_sqrt = sqrt_newton(A)
    v_cbrt, n_cbrt = cbrt_newton(B)
    v_ln, n_ln = ln_series(C)
    v_lnn, n_lnn, m, k = ln_normalized(C)

    print(f"{'Функция':<10}{'Аргумент':>10}{'Результат':>16}"
          f"{'Эталон':>16}{'Погрешность':>14}{'Итераций':>10}")
    print(table_row("sqrt", A, v_sqrt, ref_sqrt, n_sqrt))
    print(table_row("cbrt", B, v_cbrt, ref_cbrt, n_cbrt))
    print(table_row("ln", C, v_ln, ref_ln, n_ln))
    print(table_row("ln (норм.)", C, v_lnn, ref_ln, n_lnn))

    # ---- Д3: сравнение логарифма без нормализации и с нормализацией
    y_plain = (C - 1) / (C + 1)
    y_norm = (m - 1) / (m + 1)
    print("\nД3. Нормализация аргумента логарифма")
    print(f"  c = {C} = m * 2^k,  m = {m},  k = {k}")
    print(f"  Без нормализации: y = {y_plain:.6f},  членов ряда: {n_ln}")
    print(f"  С нормализацией:  y(m) = {y_norm:.6f}, y(2) = {1/3:.6f}")
    ln_m, n_m = ln_series(m)
    ln_2, n_2 = ln_series(2.0)
    print(f"    ln m   = {ln_m:.10f}  (членов: {n_m})")
    print(f"    ln 2   = {ln_2:.10f}  (членов: {n_2})")
    print(f"    ln c   = ln m + k*ln 2 = {v_lnn:.10f}")
    print(f"  Итого членов ряда: {n_lnn}; выигрыш: {n_ln / n_lnn:.2f} раза")

    # ---- таблицы итераций (для анализа сходимости)
    tr = []
    sqrt_newton(A, trace=tr)
    print_trace("Итерации для sqrt(29), x0 = 1", tr, ref_sqrt)
    tr = []
    cbrt_newton(B, trace=tr)
    print_trace("Итерации для cbrt(70), x0 = 1", tr, ref_cbrt)

    # ---- проверка условия Д3 для разных c (нормализация даёт выигрыш)
    print("\nСравнение числа членов ряда для разных c (eps = 1e-4)")
    print(f"{'c':>8}{'без норм.':>12}{'с норм.':>10}")
    for c in (2, 5, 10, 50, 100, 1000):
        _, n1 = ln_series(c)
        _, n2, _, _ = ln_normalized(c)
        print(f"{c:>8}{n1:>12}{n2:>10}")


if __name__ == "__main__":
    main()
