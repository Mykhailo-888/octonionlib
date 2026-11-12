from dataclasses import dataclass
from numbers import Real
from typing import ClassVar, Tuple, Dict


@dataclass(frozen=True)
class Octonion:
    a: Tuple[float, float, float, float, float, float, float, float]

    # Cayley-Dickson convention:
    # (a,b)(c,d) = (ac-conj(d)b, da+b conj(c)).
    _triples: ClassVar = (
        (1, 2, 3), (1, 4, 5), (1, 7, 6),
        (2, 4, 6), (2, 5, 7),
        (3, 4, 7), (3, 6, 5)
    )
    _mul_table: ClassVar[Dict[Tuple[int, int], Tuple[int, int]]] = {}

    for (i, j, k) in _triples:
        _mul_table[(i, j)] = (+1, k)
        _mul_table[(j, k)] = (+1, i)
        _mul_table[(k, i)] = (+1, j)
        _mul_table[(j, i)] = (-1, k)
        _mul_table[(k, j)] = (-1, i)
        _mul_table[(i, k)] = (-1, j)

    for t in range(1, 8):
        _mul_table[(t, t)] = (-1, 0)

    def __post_init__(self):
        values = tuple(self.a)
        if len(values) != 8:
            raise ValueError("Octonion requires exactly 8 components")
        if any(not isinstance(value, Real) for value in values):
            raise TypeError("Octonion components must be real numbers")
        object.__setattr__(
            self, "a", tuple(float(value) for value in values)
        )

    @staticmethod
    def real(x: float) -> "Octonion":
        return Octonion((x, 0, 0, 0, 0, 0, 0, 0))

    @staticmethod
    def basis(i: int, coef: float = 1.0) -> "Octonion":
        if (
            not isinstance(i, int)
            or isinstance(i, bool)
            or not (1 <= i <= 7)
        ):
            raise ValueError("basis index must be 1..7")
        a = [0.0] * 8
        a[i] = coef
        return Octonion(tuple(a))

    def __repr__(self) -> str:
        a0, a1, a2, a3, a4, a5, a6, a7 = self.a
        parts = [f"{a0}"]
        for i, ai in enumerate(
            (a1, a2, a3, a4, a5, a6, a7), start=1
        ):
            if ai != 0:
                parts.append(f"{ai}e{i}")
        return " + ".join(parts)

    def __add__(self, other: "Octonion") -> "Octonion":
        if not isinstance(other, Octonion):
            return NotImplemented
        return Octonion(tuple(x + y for x, y in zip(self.a, other.a)))

    def __sub__(self, other: "Octonion") -> "Octonion":
        if not isinstance(other, Octonion):
            return NotImplemented
        return Octonion(tuple(x - y for x, y in zip(self.a, other.a)))

    def __neg__(self) -> "Octonion":
        return Octonion(tuple(-x for x in self.a))

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Octonion(tuple(x * other for x in self.a))
        if not isinstance(other, Octonion):
            return NotImplemented

        a0, a1, a2, a3, a4, a5, a6, a7 = self.a
        b0, b1, b2, b3, b4, b5, b6, b7 = other.a
        scalar = a0 * b0 - (
            a1 * b1 + a2 * b2 + a3 * b3 + a4 * b4
            + a5 * b5 + a6 * b6 + a7 * b7
        )

        c = [0.0] * 8
        for i in range(1, 8):
            c[i] += a0 * other.a[i] + b0 * self.a[i]

        for i in range(1, 8):
            ai = self.a[i]
            if ai == 0:
                continue
            for j in range(1, 8):
                bj = other.a[j]
                if bj == 0:
                    continue
                s, k = Octonion._mul_table[(i, j)]
                if k == 0:
                    # Already included in scalar above.
                    continue
                c[k] += s * ai * bj

        return Octonion((
            scalar, c[1], c[2], c[3], c[4], c[5], c[6], c[7]
        ))

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return Octonion(tuple(other * x for x in self.a))
        return NotImplemented

    def conj(self) -> "Octonion":
        a0, a1, a2, a3, a4, a5, a6, a7 = self.a
        return Octonion((a0, -a1, -a2, -a3, -a4, -a5, -a6, -a7))

    def norm2(self) -> float:
        return sum(x * x for x in self.a)

    def norm(self) -> float:
        from math import sqrt
        return sqrt(self.norm2())

    def inv(self) -> "Octonion":
        n2 = self.norm2()
        if n2 == 0:
            raise ZeroDivisionError(
                "Octonion inverse undefined for zero norm"
            )
        return (1.0 / n2) * self.conj()

    def almost_eq(self, other: "Octonion", tol: float = 1e-9) -> bool:
        if not isinstance(other, Octonion):
            return False
        if tol < 0:
            raise ValueError("tolerance must be non-negative")
        return all(
            abs(x - y) <= tol for x, y in zip(self.a, other.a)
        )

if __name__ == "__main__":
    x = Octonion((1, 2, 0, 0, 0, 0, 0, 0))
    y = Octonion((3, 1, 0, 0, 0, 0, 0, 0))

    print("=== Калькулятор октоніонів ===")
    print("x =", x)
    print("y =", y)
    print()

    print("Додавання: x + y =", x + y)
    print("Віднімання: x - y =", x - y)
    print("Множення: x * y =", x * y)
    print("Множення на число: x * 2 =", x * 2)
    print()

    print("Норма x =", x.norm())
    print("Спряжене x =", x.conj())
    print("Обернене x =", x.inv())
