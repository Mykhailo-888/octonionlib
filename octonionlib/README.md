# OctonionLib

A pure Python library for working with **octonions** (8-dimensional hypercomplex numbers).

Lightweight and dependency-free implementation focused exclusively on octonions, using the standard multiplication table based on the Fano plane.

## Features

- `Octonion` class with components (a0, a1, ..., a7)
- Arithmetic operations: `+`, `-`, `*` (non-commutative and non-associative), scalar multiplication
- Conjugate: `.conj()`
- Norm: `.norm()` and `.norm2()`
- Inverse: `.inv()` (for non-zero octonions)
- Approximate equality: `.almost_eq()` for floating-point comparisons
- Nice string representation (e.g., `1.0 + 2.0e1 + 3.0e3`)
- Tests demonstrating non-associativity

## Installation

Just copy `octonionlib/octonion.py` into your project, or (soon) `pip install octonionlib` after publishing to PyPI.

## Quick Example

```python
from octonionlib import Octonion

e1 = Octonion.basis(1)
e2 = Octonion.basis(2)
e3 = Octonion.basis(3)

print(e1 * e2)   # → +1.0e3  (equals e3)
print(e2 * e1)   # → -1.0e3  (anti-commutative)

# Non-associativity
print((e1 * e2) * e3)  # → -1.0e1
print(e1 * (e2 * e3))  # → +1.0e1
## Octonion Multiplication Rules

Multiplication is defined using the **Fano plane** (mnemonic for basis elements e₁–e₇):

![Fano Plane 1](https://upload.wikimedia.org/wikipedia/commons/9/92/Fano_plane.svg)

![Fano Plane 2](http://theoryofeverything.org/MyToE/wp-content/uploads/2013/06/FanoLegendre.jpg)

![Fano Plane 3](http://theoryofeverything.org/wordpress/wp-content/uploads/2012/11/ToE_Demonstration-123a2.png)

Full multiplication table for basis elements e₁–e₇:

![Octonion Multiplication Table](https://upload.wikimedia.org/wikipedia/commons/a/a3/Octonion-Multiplication-Table.png)

![Colored Multiplication Table](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f7/Spacetime_Algebra_Octonion_Multiplication_Tables.svg/800px-Spacetime_Algebra_Octonion_Multiplication_Tables.svg.png)
