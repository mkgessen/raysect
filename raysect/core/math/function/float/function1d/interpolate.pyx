cimport cython
cimport numpy as np
import numpy as np

from raysect.core.math.cython.interpolation.linear cimport linear1d
from raysect.core.math.cython.utility cimport find_index, lerp
from numpy cimport ndarray
# from raysect.core.math.interpolation cimport InterpolationType, ExtrapolationType


cdef class Interpolate1D(Function1D):
    def __init__(self, object x, object f, str interpolation_type, Extrapolator1D extrapolator):
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
            self._impl = Interpolate1DLinear(x, f, extrapolator)
        elif interpolation_type == 'cubic':
            self._impl = Interpolate1DCubic()
        else:
            raise ValueError(f"Interpolation type {interpolation_type} not supported")

    cdef double evaluate(self, double x) except? -1e999:
        return self._impl.evaluate(x)

    @property
    def domain(self):
        return self._impl.domain


cdef class Interpolate1DLinear(Interpolate1D):
    def __init__(self, object x, object f, Extrapolator1D extrapolator):
        self._x = np.array(x, dtype=np.float64)
        self._f = np.array(f, dtype=np.float64)
        self._extrapolator = extrapolator
        self._extrapolator._impl._x = self._x
        self._extrapolator._impl._f = self._f
        print(type(self._extrapolator._x))

    cdef double evaluate(self, double x) except? -1e999:
        cdef int index = find_index(self._x, x)
        cdef int nx = self._x.shape[0]

        if 0 <= index < nx - 1:
            return linear1d(self._x[index], self._x[index+1], self._f[index], self._f[index+1], x)
        elif index == -1:
            print("extrapolating down")
            return self._extrapolator._extrapolate(x, 0, 0, self._x[0])
        elif index == nx - 1:
            print("extrapolating up")
            return self._extrapolator._extrapolate(x, 0, nx - 2, self._x[nx - 1])
        # if index == -1 or index == len(self._x) -1:
        #     self._extrapolator.evaluate()
        # else:
        #     return linear1d(self._x[index], self._x[index+1], self._f[index], self._f[index+1], x)

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
            print("using nearest")
            self._impl = Extrapolator1DNearest(range)
        elif extrapolation_type == 'none':
            self._impl = ExtrapolatorNone()

    cdef double evaluate(self, double x) except? -1e999:
        raise NotImplementedError(f"{self.__class__} not implemented")

    cdef double _extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        return self._impl._extrapolate(px, order, index, rx)

cdef class ExtrapolatorNone(Extrapolator1D):
    def __init__(self):
        pass

    cdef double _extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        raise ValueError(f"Extrapolation not available. Interpolate within function range {np.min(self._x)}-{np.max(self._x)}")

cdef class Extrapolator1DNearest(Extrapolator1D):
    def __init__(self, range):
        self._range = range
        print("nearest created")

    cdef double _extrapolate(self, double px, int order, int index, double rx) except? -1e999:
        print(f"_extrapolate px:{px}, order:{order}, index:{index}, rx:{rx}")
        print(type(self._x), type(self._f))
        return lerp(self._x[index], self._x[index + 1], self._f[index], self._f[index + 1], px)

#
# cdef class Extrapolator1DLinear(Extrapolator1D):
#     def __init__(self):
#         pass
#
# cdef class Extrapolator1DQuadratic(Extrapolator1D):
#     def __init__(self):
#         pass