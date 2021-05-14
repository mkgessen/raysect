import numpy as np
from raysect.core.math.function.float.function1d.interpolate import Interpolate1D, Extrapolator1D
from raysect.core.math.function.float.function1d.cmath import Sin1D, Cos1D
from raysect.core.math.interpolation import InterpolationType
import matplotlib.pyplot as plt

x = np.linspace(0, 3, num=10)
func = np.sin(x)

plt.plot(x, func)


sin_interp = Interpolate1D(x, func, 'linear', extrapolator=Extrapolator1D('nearest', 2.0))

sin_interp2 = Interpolate1D(x, func, 'linear', extrapolator=Extrapolator1D('linear', 2.0))

result = []
result2 = []

test_scale = np.linspace(0, 5, num=20)
for t in test_scale:
    print(sin_interp(t), t)
    result.append(sin_interp(t))
    result2.append(sin_interp2(t))

print(sin_interp.domain) # x_min, x_max, f_min, f_max
plt.scatter(test_scale, result, color='r')
plt.scatter(test_scale, result2, color='g')

plt.show()