from numpy cimport ndarray

cdef class Tester:
    cdef:
        #double[::1] num_list
        ndarray num_list
    cdef void foo(self, object num_list)
    cdef double bar(self, double x, double y)
    cdef ndarray _sum(self, ndarray a1, ndarray a2)
