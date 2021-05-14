from numpy cimport ndarray
cimport numpy as np
import numpy as np

cdef class Tester:
    cdef void foo(self, object num_list):
        self.num_list = np.array(num_list, dtype=np.float64)

    cdef double bar(self, double x, double y):
        print(f"hello world with {x} {y}")
        return x * y

    cdef ndarray _sum(self, ndarray a1, ndarray a2):
        return a1 + a2

    def call_foo(self, array_in):
        self.foo(array_in)

    def call_bar(self, double x, double y):
        return self.bar(x, y)

    def gimme_numbers(self):
        return self.num_list

    def sum(self, a1, a2):
        return self._sum(a1, a2)
