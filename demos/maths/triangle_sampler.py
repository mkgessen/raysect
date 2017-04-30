
import matplotlib.pyplot as plt
from raysect.core.math import TriangleSampler, Point3D


def plot_samples(samples):
    plt.figure()
    for s in samples:
        plt.plot(s.x, s.y, 'k.')

    plt.figure()
    for s in samples:
        plt.plot(s.x, s.z, 'r.')


ts1 = TriangleSampler(Point3D(0, 0, 0), Point3D(1, 0, 0), Point3D(0, 1, 0))
samples = ts1.sample(1000)
plot_samples(samples)

ts2 = TriangleSampler(Point3D(-1, -1, 0), Point3D(0, 1, 0), Point3D(1, 0.1, 0))
samples = ts2.sample(1000)
plot_samples(samples)

ts3 = TriangleSampler(Point3D(-1, -1, -0.5), Point3D(0, 1, 1), Point3D(1, 0.1, 0))
samples = ts3.sample(1000)
plot_samples(samples)


plt.show()
