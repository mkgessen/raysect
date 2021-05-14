from raysect.core.sandbox.testing import Tester
import numpy as np

narr = np.arange(27, dtype=np.dtype("i")).reshape((3, 3, 3))
t = Tester()
# t.call_foo(np.array(range(0,10)))
# print(t.gimme_numbers())
res = t.sum(np.array(range(0,10)), np.array(range(20,30)))
print(res)
