import unittest
from octonionlib import Octonion

class TestOctonion(unittest.TestCase):
    def test_basic(self):
        e1 = Octonion.basis(1)
        e2 = Octonion.basis(2)
        e3 = Octonion.basis(3)
        self.assertTrue((e1*e1).almost_eq(Octonion.real(-1)))
        self.assertTrue((e1*e2).almost_eq(e3))
        self.assertTrue((e2*e1).almost_eq(-1*e3))

    def test_conjugate_norm(self):
        x = Octonion((1,2,3,0,0,0,0,0))
        self.assertTrue((x*x.conj()).almost_eq(Octonion.real(x.norm2())))

    def test_inverse(self):
        x = Octonion((1,1,0,0,0,0,0,0))
        invx = x.inv()
        self.assertTrue((x*invx).almost_eq(Octonion.real(1.0)))

    def test_non_associativity(self):
        e1 = Octonion.basis(1)
        e2 = Octonion.basis(2)
        e3 = Octonion.basis(3)
        left = (e1*e2)*e3
        right = e1*(e2*e3)
        self.assertFalse(left.almost_eq(right))

if __name__ == "__main__":
    unittest.main()
