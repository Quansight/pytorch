import torch
import numpy as np
import unittest
import operator
from common_utils import TestCase

class SubTensor(torch.Tensor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._inner = None

    @property
    def inner(self):
        return self._inner
    
    @inner.setter
    def inner(self, value):
        self._inner = value

    def __torch_finalize__(self, obj: "SubTensor"):
        self.inner = obj.inner

class TestMethodPassthrough(TestCase):
    def test_indexing(self):
        t = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        s = SubTensor(t)
        s.inner = "random_string"
        
        # Checks for type
        self.assert_(isinstance(s[0], SubTensor))
        self.assert_(isinstance(s[1:], SubTensor))
        self.assert_(isinstance(s[:], SubTensor))
        self.assert_(isinstance(s[...], SubTensor))
        self.assert_(isinstance(s[..., 0], SubTensor))
        self.assert_(isinstance(s[..., 1:], SubTensor))

        # Checks for equality
        self.assertEqual(t[0], s[0])
        self.assertEqual(t[1:], s[1:])
        self.assertEqual(t[:], s[:])
        self.assertEqual(t[...], s[...])
        self.assertEqual(t[..., 0], s[..., 0])
        self.assertEqual(t[..., 1:], s[..., 1:])

        # Checks for preservation of inner data
        self.assertEqual(s[0].inner, s.inner)
        self.assertEqual(s[1:].inner, s.inner)
        self.assertEqual(s[:].inner, s.inner)
        self.assertEqual(s[...].inner, s.inner)
        self.assertEqual(s[..., 0].inner, s.inner)
        self.assertEqual(s[..., 1:].inner, s.inner)

    def test_binary_operators(self):
        operators = [
            operator.add,
            operator.sub,
            operator.mul,
            operator.truediv,
            operator.floordiv,
            operator.matmul,
            operator.le,
            operator.ge,
            operator.lt,
            operator.gt,
            operator.eq,
            operator.ne,
            operator.and_,
            operator.or_,
            operator.mod,
        ]

        t = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        s = SubTensor(t)
        s.inner = "random_string"

        for op in operators:
            base = op(t, t)

            # Equality
            self.assertEqual(op(s, s), base)
            self.assertEqual(op(s, t), base)
            self.assertEqual(op(t, s), base)

            # Typecheck
            self.assert_(isinstance(op(s, s), SubTensor))
            self.assert_(isinstance(op(s, t), SubTensor))
            self.assert_(isinstance(op(t, s), SubTensor))

            # Inner data
            self.assertEqual(op(s, s).inner, s.inner)
            self.assertEqual(op(s, t).inner, s.inner)
            self.assertEqual(op(t, s).inner, s.inner)
    
    def test_unary_operators(self):
        operators = [
            operator.neg,
            operator.pos,
            operator.inv
        ]

        t = torch.arange(9, dtype=torch.float32).reshape(3, 3)
        s = SubTensor(t)
        s.inner = "random_string"

        for op in operators:
            base = op(t)
            self.assertEqual(op(s), base)
            self.assert_(isinstance(op(s), SubTensor))
            self.assertEqual(op(s).inner, s.inner)

if __name__ == '__main__':
    unittest.main()
