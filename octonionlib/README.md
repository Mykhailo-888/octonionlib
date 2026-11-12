# OctonionLib

A pure-Python library for octonion arithmetic, with no third-party
runtime dependencies. Requires Python 3.9 or newer.

An octonion has eight real components:
a0 + a1*e1 + a2*e2 + ... + a7*e7.

## Features

- Addition, subtraction and negation
- Octonion multiplication and scalar multiplication
- Conjugation: `.conj()`
- Norm and squared norm: `.norm()` and `.norm2()`
- Inverse of a nonzero octonion: `.inv()`
- Approximate comparison: `.almost_eq(other, tol=1e-9)`
- Validation of exactly eight real components

## Installation from source

From the repository root:

```bash
python -m pip install ./octonionlib
```

## Example

```python
from octonionlib import Octonion

e1 = Octonion.basis(1)
e2 = Octonion.basis(2)
e3 = Octonion.basis(3)
e4 = Octonion.basis(4)
e7 = Octonion.basis(7)

assert e1 * e1 == Octonion.real(-1)
assert e1 * e2 == e3
assert e2 * e1 == -e3

# Non-associativity: parentheses change the result.
assert (e1 * e2) * e4 == e7
assert e1 * (e2 * e4) == -e7

# e1, e2 and e3 form an associative quaternion subalgebra.
assert (e1 * e2) * e3 == Octonion.real(-1)
assert e1 * (e2 * e3) == Octonion.real(-1)

x = Octonion((1, 2, 3, 0, 0, 0, 0, 0))
assert (x * x.conj()).almost_eq(Octonion.real(x.norm2()))
assert (x * x.inv()).almost_eq(Octonion.real(1))
```

## Multiplication convention

The oriented Fano triples are:

- (1, 2, 3)
- (1, 4, 5)
- (1, 7, 6)
- (2, 4, 6)
- (2, 5, 7)
- (3, 4, 7)
- (3, 6, 5)

For each triple (i, j, k), ei*ej = ek. Cyclic permutations preserve
the sign; reversing the order negates it. Each imaginary basis unit
squares to -1.

This matches the quaternion-pair Cayley-Dickson convention:

(a, b)*(c, d) = (a*c - conj(d)*b, d*a + b*conj(c)).

Octonions are noncommutative and nonassociative, but alternative:
(x*x)*y = x*(x*y) and (y*x)*x = y*(x*x).

## Tests

From the repository root:

```bash
cd octonionlib
python -m unittest discover -s tests -v
```

The tests check all 64 basis products against an independent
quaternion-pair implementation, plus 100 seeded random cases.
They cover inverses, conjugation, norm composition, distributivity,
alternativity, nonassociativity and input validation.

## Numerical limitations

Components are stored as Python floats. Use `.almost_eq()` when
comparing computed floating-point results. The implementation is
intended for ordinary finite inputs and does not provide arbitrary
precision or overflow protection.

## License

MIT, as declared in pyproject.toml.
