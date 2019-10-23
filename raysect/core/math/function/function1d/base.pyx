# cython: language_level=3

# Copyright (c) 2014-2018, Dr Alex Meakins, Raysect Project
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright notice,
#        this list of conditions and the following disclaimer.
#
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#
#     3. Neither the name of the Raysect Project nor the names of its
#        contributors may be used to endorse or promote products derived from
#        this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import numbers
cimport cython
from libc.math cimport floor


cdef class Function1D:
    """
    Cython optimised class for representing an arbitrary 1D function.

    Using __call__() in cython is slow. This class provides an overloadable
    cython cdef evaluate() method which has much less overhead than a python
    function call.

    For use in cython code only, this class cannot be extended via python.

    To create a new function object, inherit this class and implement the
    evaluate() method. The new function object can then be used with any code
    that accepts a function object.
    """

    cdef double evaluate(self, double x) except? -1e999:
        raise NotImplementedError("The evaluate() method has not been implemented.")

    def __call__(self, double x):
        """ Evaluate the function f(x)

        :param float x: function parameter x
        :rtype: float
        """
        return self.evaluate(x)

    def __add__(object a, object b):
        if isinstance(a, Function1D):
            if isinstance(b, Function1D):
                # a() + b()
                return AddFunction1D(<Function1D> a, <Function1D> b)
            elif isinstance(b, numbers.Real):
                # a() + B -> B + a()
                return AddScalar1D(<double> b, <Function1D> a)
        elif isinstance(a, numbers.Real):
            if isinstance(b, Function1D):
                # A + b()
                return AddScalar1D(<double> a, <Function1D> b)
        return NotImplemented

    def __sub__(object a, object b):
        if isinstance(a, Function1D):
            if isinstance(b, Function1D):
                # a() - b()
                return SubtractFunction1D(<Function1D> a, <Function1D> b)
            elif isinstance(b, numbers.Real):
                # a() - B -> -B + a()
                return AddScalar1D(-(<double> b), <Function1D> a)
        elif isinstance(a, numbers.Real):
            if isinstance(b, Function1D):
                # A - b()
                return SubtractScalar1D(<double> a, <Function1D> b)
        return NotImplemented

    def __mul__(a, b):
        if isinstance(a, Function1D):
            if isinstance(b, Function1D):
                # a() * b()
                return MultiplyFunction1D(<Function1D> a, <Function1D> b)
            elif isinstance(b, numbers.Real):
                # a() * B -> B * a()
                return MultiplyScalar1D(<double> b, <Function1D> a)
        elif isinstance(a, numbers.Real):
            if isinstance(b, Function1D):
                # A * b()
                return MultiplyScalar1D(<double> a, <Function1D> b)
        return NotImplemented

    @cython.cdivision(True)
    def __truediv__(a, b):
        cdef double v
        if isinstance(a, Function1D):
            if isinstance(b, Function1D):
                # a() / b()
                return DivideFunction1D(<Function1D> a, <Function1D> b)
            elif isinstance(b, numbers.Real):
                # a() / B -> 1/B * a()
                v = <double> b
                if v == 0.0:
                    raise ZeroDivisionError("Scalar used as the denominator of the division is zero valued.")
                return MultiplyScalar1D(1/v, <Function1D> a)
        elif isinstance(a, numbers.Real):
            if isinstance(b, Function1D):
                # A * b()
                return DivideScalar1D(<double> a, <Function1D> b)
        return NotImplemented

    def __mod__(a, b):
        cdef double v
        if isinstance(a, Function1D):
            if isinstance(b, Function1D):
                # a() % b()
                return ModuloFunction1D(<Function1D> a, <Function1D> b)
            elif isinstance(b, numbers.Real):
                # a() % B
                v = <double> b
                if v == 0.0:
                    raise ZeroDivisionError("Scalar used as the divisor of the division is zero valued.")
                return ModuloFunctionScalar1D(<Function1D> a, v)
        elif isinstance(a, numbers.Real):
            if isinstance(b, Function1D):
                # A % b()
                return ModuloScalarFunction1D(<double> a, <Function1D> b)
        return NotImplemented

    def __neg__(self):
        return MultiplyScalar1D(-1, self)

    def __pow__(a, b, c):
        if c is not None:
            # Optimised impolementation of pow(a, b, c) not available: fall back
            # to general implementation
            return (a ** b) % c
        if isinstance(a, Function1D):
            if isinstance(b, Function1D):
                # a() ** b()
                return PowFunction1D(<Function1D> a, <Function1D> b)
            elif isinstance(b, numbers.Real):
                # a() ** b
                return PowFunctionScalar1D(<Function1D> a, <double> b)
        elif isinstance(a, numbers.Real):
            if isinstance(b, Function1D):
                # a ** b()
                return PowScalarFunction1D(<double> a, <Function1D> b)
        return NotImplemented


cdef class PythonFunction1D(Function1D):
    """
    Wraps a python callable object with a Function1D object.

    This class allows a python object to interact with cython code that requires
    a Function1D object. The python object must implement __call__() expecting
    one argument.

    This class is intended to be used to transparently wrap python objects that
    are passed via constructors or methods into cython optimised code. It is not
    intended that the users should need to directly interact with these wrapping
    objects. Constructors and methods expecting a Function1D object should be
    designed to accept a generic python object and then test that object to
    determine if it is an instance of Function1D. If the object is not a
    Function1D object it should be wrapped using this class for internal use.

    See also: autowrap_function1d()

    :param object function: the python function to wrap, __call__() function must be
    implemented on the object.
    """
    def __init__(self, object function):
        self.function = function

    cdef double evaluate(self, double x) except? -1e999:
        return self.function(x)


cdef Function1D autowrap_function1d(object function):
    """
    Automatically wraps the supplied python object in a PythonFunction1D object.

    If this function is passed a valid Function1D object, then the Function1D
    object is simply returned without wrapping.

    This convenience function is provided to simplify the handling of Function1D
    and python callable objects in constructors, functions and setters.
    """

    if isinstance(function, Function1D):
        return <Function1D> function
    else:
        return PythonFunction1D(function)


cdef class AddFunction1D(Function1D):
    """
    A Function1D class that implements the addition of the results of two Function1D objects: f1() + f2()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function1D object.

    :param Function1D function1: A Function1D object.
    :param Function1D function2: A Function1D object.
    """
    def __init__(self, Function1D function1, Function1D function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) + self._function2.evaluate(x)


cdef class SubtractFunction1D(Function1D):
    """
    A Function1D class that implements the subtraction of the results of two Function1D objects: f1() - f2()

    This class is not intended to be used directly, but rather returned as the result of a __sub__() call on a
    Function1D object.

    :param Function1D function1: A Function1D object.
    :param Function1D function2: A Function1D object.
    """
    def __init__(self, Function1D function1, Function1D function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) - self._function2.evaluate(x)


cdef class MultiplyFunction1D(Function1D):
    """
    A Function1D class that implements the multiplication of the results of two Function1D objects: f1() * f2()

    This class is not intended to be used directly, but rather returned as the result of a __mul__() call on a
    Function1D object.

    :param Function1D function1: A Function1D object.
    :param Function1D function2: A Function1D object.
    """
    def __init__(self, function1, function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        return self._function1.evaluate(x) * self._function2.evaluate(x)


cdef class DivideFunction1D(Function1D):
    """
    A Function1D class that implements the division of the results of two Function1D objects: f1() / f2()

    This class is not intended to be used directly, but rather returned as the result of a __truediv__() call on a
    Function1D object.

    :param Function1D function1: A Function1D object.
    :param Function1D function2: A Function1D object.
    """
    def __init__(self, function1, function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef double denominator = self._function2.evaluate(x)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._function1.evaluate(x) / denominator


cdef class ModuloFunction1D(Function1D):
    """
    A Function1D class that implements the modulo of the results of two Function1D objects: f1() % f2()

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function1D object.

    :param Function1D function1: A Function1D object.
    :param Function1D function2: A Function1D object.
    """
    def __init__(self, function1, function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef double divisor = self._function2.evaluate(x)
        if divisor == 0.0:
            raise ZeroDivisionError("Function used as the divisor of the modulo returned a zero value.")
        return self._function1.evaluate(x) % divisor


cdef class PowFunction1D(Function1D):
    """
    A Function1D class that implements the pow() operator on two Function1D objects.

    This class is not intended to be used directly, but rather returned as the result of a __pow__() call on a
    Function1D object.

    :param Function1D function1: A Function1D object.
    :param Function1D function2: A Function1D object.
    """
    def __init__(self, function1, function2):
        self._function1 = autowrap_function1d(function1)
        self._function2 = autowrap_function1d(function2)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double base, exponent
        base = self._function1.evaluate(x)
        exponent = self._function2.evaluate(x)
        if base < 0 and floor(exponent) != exponent:  # Would return a complex value rather than double
            raise ValueError("Negative base and non-integral exponent is not supported")
        if base == 0 and exponent < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return base ** exponent


cdef class AddScalar1D(Function1D):
    """
    A Function1D class that implements the addition of scalar and the result of a Function1D object: K + f()

    This class is not intended to be used directly, but rather returned as the result of an __add__() call on a
    Function1D object.

    :param float value: A double value.
    :param Function1D function: A Function1D object.
    """
    def __init__(self, double value, Function1D function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value + self._function.evaluate(x)


cdef class SubtractScalar1D(Function1D):
    """
    A Function1D class that implements the subtraction of scalar and the result of a Function1D object: K - f()

    This class is not intended to be used directly, but rather returned as the result of an __sub__() call on a
    Function1D object.

    :param double value: A double value.
    :param Function1D function: A Function1D object.
    """
    def __init__(self, double value, Function1D function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value - self._function.evaluate(x)


cdef class MultiplyScalar1D(Function1D):
    """
    A Function1D class that implements the multiplication of scalar and the result of a Function1D object: K * f()

    This class is not intended to be used directly, but rather returned as the result of an __mul__() call on a
    Function1D object.

    :param float value: A double value.
    :param Function1D function: A Function1D object.
    """
    def __init__(self, double value, Function1D function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        return self._value * self._function.evaluate(x)


cdef class DivideScalar1D(Function1D):
    """
    A Function1D class that implements the division of scalar and the result of a Function1D object: K / f()

    This class is not intended to be used directly, but rather returned as the result of an __div__() call on a
    Function1D object.

    :param float value: A double value.
    :param Function1D function: A Function1D object.
    """
    def __init__(self, double value, Function1D function):
        self._value = value
        self._function = autowrap_function1d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef double denominator = self._function.evaluate(x)
        if denominator == 0.0:
            raise ZeroDivisionError("Function used as the denominator of the division returned a zero value.")
        return self._value / denominator


cdef class ModuloScalarFunction1D(Function1D):
    """
    A Function1D class that implements the modulo of scalar and the result of a Function1D object: K % f()

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function1D object.

    :param float value: A double value.
    :param Function1D function: A Function1D object.
    """
    def __init__(self, double value, Function1D function):
        self._value = value
        self._function = autowrap_function1d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        cdef double divisor = self._function.evaluate(x)
        if divisor == 0.0:
            raise ZeroDivisionError("Function used as the divisor of the modulo returned a zero value.")
        return self._value % divisor


cdef class ModuloFunctionScalar1D(Function1D):
    """
    A Function1D class that implements the modulo of the result of a Function1D object and a scalar: f() % K

    This class is not intended to be used directly, but rather returned as the result of a __mod__() call on a
    Function1D object.

    :param Function1D function: A Function1D object.
    :param float value: A double value.
    """
    def __init__(self, Function1D function, double value):
        if value == 0:
            raise ValueError("Divisor cannot be zero")
        self._value = value
        self._function = autowrap_function1d(function)

    @cython.cdivision(True)
    cdef double evaluate(self, double x) except? -1e999:
        return self._function.evaluate(x) % self._value


cdef class PowScalarFunction1D(Function1D):
    """
    A Function1D class that implements the pow of scalar and the result of a Function1D object: K ** f()

    This class is not intended to be used directly, but rather returned as the result of an __pow__() call on a
    Function1D object.

    :param float value: A double value.
    :param Function1D function: A Function1D object.
    """
    def __init__(self, double value, Function1D function):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double exponent = self._function.evaluate(x)
        if self._value < 0 and floor(exponent) != exponent:
            raise ValueError("Negative base and non-integral exponent is not supported")
        if self._value == 0 and exponent < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return self._value ** exponent


cdef class PowFunctionScalar1D(Function1D):
    """
    A Function1D class that implements the pow of the result of a Function1D object and a scalar: f() ** K

    This class is not intended to be used directly, but rather returned as the result of an __pow__() call on a
    Function1D object.

    :param Function1D function: A Function1D object.
    :param float value: A double value.
    """
    def __init__(self, Function1D function, double value):
        self._value = value
        self._function = autowrap_function1d(function)

    cdef double evaluate(self, double x) except? -1e999:
        cdef double base = self._function.evaluate(x)
        if base < 0 and floor(self._value) != self._value:
            raise ValueError("Negative base and non-integral exponent is not supported")
        if base == 0 and self._value < 0:
            raise ZeroDivisionError("0.0 cannot be raised to a negative power")
        return base ** self._value
