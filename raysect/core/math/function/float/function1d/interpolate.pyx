cimport cython
cimport numpy as np
import numpy as np

from raysect.core.math.cython.interpolation.linear cimport linear1d
from raysect.core.math.cython.utility cimport find_index
# from raysect.core.math.interpolation cimport InterpolationType, ExtrapolationType


cdef class Interpolate1D(Function1D):
    def __init__(self, object x, object f, str interpolation_type):
        # todo check dimensions, monotonicity
        x = np.array(x, dtype=np.float64)
        f = np.array(f, dtype=np.float64)

        if x.ndim != 1:
            raise ValueError(f"The x array must be 1D. Got {x.shape}")

        if f.ndim != 1:
            raise ValueError(f"The x array must be 1D. Got {f.shape}")

        print(x.shape, f.shape)
        if x.shape != f.shape:
            raise ValueError(f"Shape mismatch between x array ({x.shape}) and f array ({f.shape})")

        if (np.diff(x) <= 0).any():
            raise ValueError("The x array must be monotonically increasing.")

        if interpolation_type == 'linear':
            self._impl = Interpolate1DLinear(x, f)
        elif interpolation_type == 'cubic':
            self._impl = Interpolate1DCubic
        else:
            raise ValueError(f"Interpolation type {interpolation_type} not supported")

    cdef double evaluate(self, double x) except? -1e999:
        return self._impl.evaluate(x)

    @property
    def domain(self):
        return self._impl.domain


cdef class Interpolate1DLinear(Interpolate1D):
    def __init__(self, object x, object f):
        self._x = np.array(x, dtype=np.float64)
        self._f = np.array(f, dtype=np.float64)

    cdef double evaluate(self, double x) except? -1e999:
        cdef int index = find_index(self._x, x)
        return linear1d(self._x[index], self._x[index+1], self._f[index], self._f[index+1], x)

    @property
    def domain(self):
        return np.min(self._x), np.max(self._x), np.min(self._f), np.max(self._f)


cdef class Interpolate1DCubic(Interpolate1D):
    def __init__(self, object x, object f):
        self._x = np.array(x, dtype=np.float64)
        self._f = np.array(f, dtype=np.float64)
        raise NotImplementedError(f"{self.__class__} not implemented")

    cdef double evaluate(self, double x) except? -1e999:
        raise NotImplementedError(f"{self.__class__} not implemented")

    @property
    def domain(self):
        raise NotImplementedError(f"{self.__class__} not implemented")


cdef class Extrapolator1D(Function1D):
    def __init__(self, str extrapolation_type, double range):
        self._range = range

        if extrapolation_type == 'nearest':
            self._impl = Extrapolator1DNearest(range)

    cdef double evaluate(self, double x) except? -1e999:
        raise NotImplementedError(f"{self.__class__} not implemented")

cdef class ExtrapolatorNone(Extrapolator1D):
    def __init__(self):
        pass

cdef class Extrapolator1DNearest(Extrapolator1D):
    def __init__(self):
        pass

cdef class Extrapolator1DLinear(Extrapolator1D):
    def __init__(self):
        pass

cdef class Extrapolator1DQuadratic(Extrapolator1D):
    def __init__(self):
        pass