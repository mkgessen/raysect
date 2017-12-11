# Copyright (c) 2016, Dr Alex Meakins, Raysect Project
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

import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt

cimport cython
from raysect.optical cimport Spectrum
from raysect.optical.observer.pipeline.spectral.power cimport SpectralPowerPipeline0D, SpectralPowerPipeline2D
from raysect.optical.observer.base.slice cimport SpectralSlice
from raysect.core.math cimport StatsArray1D
from raysect.optical.observer.base.processor cimport PixelProcessor


_DEFAULT_PIPELINE_NAME = "Spectral Radiance Pipeline"
_DISPLAY_DPI = 100
_DISPLAY_SIZE = (800 / _DISPLAY_DPI, 600 / _DISPLAY_DPI)


cdef class SpectralRadiancePipeline0D(SpectralPowerPipeline0D):
    """
    A basic spectral radiance pipeline for 0D observers (W/str/m^2/nm).

    The mean spectral radiance for the observer is stored along with the associated
    error on each wavelength bin.

    Spectral values and errors are available through the self.frame attribute.

    :param bool accumulate: Whether to accumulate samples with subsequent calls
      to observe() (default=True).
    :param str name: User friendly name for this pipeline.
    """

    def __init__(self, bint accumulate=True, str name=None, bint display_progress=True):
        name = name or _DEFAULT_PIPELINE_NAME
        super().__init__(accumulate=accumulate, name=name, display_progress=display_progress)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int slice_id):
        return SpectralRadiancePixelProcessor(self._spectral_slices[slice_id])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _render_display(self):

        cdef:
            np.ndarray errors
            double[::1] errors_mv
            int i

        errors = np.empty(self.samples.length)
        errors_mv = errors
        for i in range(self.samples.length):
            errors_mv[i] = self.samples.error(i)

        # create a fresh figure if the existing figure window has gone missing
        if not self._display_figure or not plt.fignum_exists(self._display_figure.number):
            self._display_figure = plt.figure(facecolor=(1, 1, 1), figsize=_DISPLAY_SIZE, dpi=_DISPLAY_DPI)
        fig = self._display_figure

        # set window title
        fig.canvas.set_window_title(self.name)

        fig.clf()
        plt.plot(self.wavelengths, self.samples.mean[:], color=(0, 0, 1))
        plt.plot(self.wavelengths, self.samples.mean[:] + errors[:], color=(0.685, 0.685, 1.0))
        plt.plot(self.wavelengths, self.samples.mean[:] - errors[:], color=(0.685, 0.685, 1.0))
        plt.title('{}'.format(self.name))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Spectral Radiance (W/str/m^2/nm)')
        fig.canvas.draw_idle()
        plt.show()

    cpdef Spectrum to_spectrum(self):
        """
        Returns the mean spectral radiance in a Spectrum() object.
        """

        cdef Spectrum spectrum

        if not self.samples:
            raise ValueError("No spectrum has been observed.")
        spectrum = Spectrum(self.min_wavelength, self.max_wavelength, self.bins)
        spectrum.samples[:] = self.samples.mean[:]
        return spectrum


cdef class SpectralRadiancePipeline2D(SpectralPowerPipeline2D):
    """
    A basic spectral radiance pipeline for 2D observers (W/str/m^2/nm).

    The mean spectral radiance for each pixel is stored along with the associated
    error on each wavelength bin in a 2D frame object.

    Spectral values and errors are available through the self.frame attribute.

    :param bool accumulate: Whether to accumulate samples with subsequent calls
      to observe() (default=True).
    :param str name: User friendly name for this pipeline.
    """

    def __init__(self, bint accumulate=True, str name=None):
        name = name or _DEFAULT_PIPELINE_NAME
        super().__init__(accumulate=accumulate, name=name)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef PixelProcessor pixel_processor(self, int x, int y, int slice_id):
        return SpectralRadiancePixelProcessor(self._spectral_slices[slice_id])

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def display_pixel(self, int x, int y):

        cdef:
            np.ndarray errors
            double[::1] errors_mv
            int i

        errors = np.empty(self.frame.nz)
        errors_mv = errors
        for i in range(self.frame.nz):
            errors_mv[i] = self.frame.error(x, y, i)

        plt.figure()
        plt.plot(self.wavelengths, self.frame.mean[x, y, :], color=(0, 0, 1))
        plt.plot(self.wavelengths, self.frame.mean[x, y, :] + errors[:], color=(0.685, 0.685, 1.0))
        plt.plot(self.wavelengths, self.frame.mean[x, y, :] - errors[:], color=(0.685, 0.685, 1.0))
        plt.title('{} - Pixel ({}, {})'.format(self.name, x, y))
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Spectral Radiance (W/str/m^2/nm)')
        plt.draw()
        plt.show()

    cpdef Spectrum to_spectrum(self, int x, int y):
        """
        Returns the mean spectral radiance of pixel (x, y) in a Spectrum() object.
        """

        cdef Spectrum spectrum

        if not self.samples:
            raise ValueError("No spectrum has been observed.")
        spectrum = Spectrum(self.min_wavelength, self.max_wavelength, self.bins)
        spectrum.samples[:] = self.samples.mean[:]
        return spectrum


cdef class SpectralRadiancePixelProcessor(PixelProcessor):
    """
    PixelProcessor that stores the spectral radiance observed by each pixel.
    """

    def __init__(self, SpectralSlice slice):
        self.bins = StatsArray1D(slice.bins)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef object add_sample(self, Spectrum spectrum, double etendue):

        cdef int index
        for index in range(self.bins.length):
            self.bins.add_sample(index, spectrum.samples_mv[index])

    cpdef tuple pack_results(self):
        return self.bins.mean, self.bins.variance

