import random
import unittest

from octonionlib import Octonion


def qmul(a, b):
    w, x, y, z = a
    v, p, q, r = b
    return (
        w*v-x*p-y*q-z*r,
        w*p+x*v+y*r-z*q,
        w*q-x*r+y*v+z*p,
        w*r+x*q-y*p+z*v,
    )


def qconj(a):
    return (a[0], -a[1], -a[2], -a[3])


def reference_product(a, b):
    ac = qmul(a[:4], b[:4])
    db = qmul(qconj(b[4:]), a[4:])
    da = qmul(b[4:], a[:4])
    bc = qmul(a[4:], qconj(b[:4]))
    return tuple(x-y for x, y in zip(ac, db)) + tuple(
        x+y for x, y in zip(da, bc)
    )


class TestOctonion(unittest.TestCase):
    def test_all_basis_products(self):
        basis = [Octonion.real(1)] + [
            Octonion.basis(i) for i in range(1, 8)
        ]
        for i, x in enumerate(basis):
            for j, y in enumerate(basis):
                with self.subTest(i=i, j=j):
                    self.assertEqual(
                        (x*y).a, reference_product(x.a, y.a)
                    )

    def test_non_associativity(self):
        x, y, z = (Octonion.basis(i) for i in (1, 2, 4))
        self.assertEqual((x*y)*z, Octonion.basis(7))
        self.assertEqual(x*(y*z), -Octonion.basis(7))

    def test_quaternion_subalgebra(self):
        x, y, z = (Octonion.basis(i) for i in (1, 2, 3))
        self.assertEqual((x*y)*z, Octonion.real(-1))
        self.assertEqual(x*(y*z), Octonion.real(-1))

    def test_general_identities(self):
        rng = random.Random(42)
        one = Octonion.real(1)
        for sample in range(100):
            x, y, z = [
                Octonion(tuple(rng.uniform(-2, 2) for _ in range(8)))
                for _ in range(3)
            ]
            with self.subTest(sample=sample):
                self.assertTrue(
                    (x*y).almost_eq(Octonion(reference_product(x.a, y.a)))
                )
                self.assertTrue(
                    (x*x.conj()).almost_eq(Octonion.real(x.norm2()))
                )
                self.assertTrue((x*x.inv()).almost_eq(one))
                self.assertTrue((x.inv()*x).almost_eq(one))
                self.assertTrue((x*y).conj().almost_eq(y.conj()*x.conj()))
                self.assertAlmostEqual(
                    (x*y).norm2(), x.norm2()*y.norm2(), places=9
                )
                self.assertTrue(((x*x)*y).almost_eq(x*(x*y)))
                self.assertTrue(((y*x)*x).almost_eq(y*(x*x)))
                self.assertTrue((x*(y+z)).almost_eq(x*y+x*z))
                self.assertTrue(((x+y)*z).almost_eq(x*z+y*z))

    def test_add_subtract_and_scalar(self):
        x = Octonion(tuple(range(8)))
        self.assertEqual(x-x, Octonion.real(0))
        self.assertEqual(x+x, 2*x)
        self.assertEqual(x*2, 2*x)
        self.assertAlmostEqual(x.norm()**2, x.norm2())

    def test_invalid_components(self):
        for size in (0, 7, 9):
            with self.assertRaises(ValueError):
                Octonion((0,)*size)
        for value in ("1", None, 1j):
            with self.assertRaises(TypeError):
                Octonion((value, 0, 0, 0, 0, 0, 0, 0))

    def test_zero_inverse(self):
        with self.assertRaises(ZeroDivisionError):
            Octonion.real(0).inv()

    def test_invalid_basis(self):
        for index in (0, 8, -1, 1.5, True):
            with self.assertRaises(ValueError):
                Octonion.basis(index)


if __name__ == "__main__":
    unittest.main()
