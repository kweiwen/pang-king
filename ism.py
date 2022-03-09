# import numpy as np
#
# def sim_microphone(pos, angle, mtype):
#     #  Polar Pattern         alpha
#     #  ---------------------------
#     #  Bidirectional         0
#     #  Hypercardioid         0.25
#     #  Cardioid              0.5
#     #  Subcardioid           0.75
#     #  Omnidirectional       1
#
#     if mtype in 'bcsh':
#         if mtype == 'b':
#             rho = 0
#         elif mtype == 'h':
#             rho = 0.25
#         elif mtype == 'c':
#             rho = 0.5
#         elif mtype == 's':
#             rho = 0.75
#         else:
#             rho = 1
#
#         x = pos[0]
#         y = pos[1]
#         z = pos[2]
#
#         dist = np.sqrt(np.sum(np.power(pos, 2)))
#         vartheta = np.arccos(z / dist)
#         varphi = np.arctan2(y, x)
#         gain = np.sin(np.pi * 0.5 - angle[1]) * np.sin(vartheta) * np.cos(angle[0] - varphi) + np.cos(np.pi * 0.5 - angle[1]) * np.cos(vartheta)
#         gain = rho + (1 - rho) * gain
#         return gain
#
#     else:
#         return 1
#
# def computeRIR(c, fs, rr, nMicrophones, nSamples, ss, LL, beta, microphone_type, nOrder, angle, isHighPassFilter):
#     imp = np.zeros([nSamples, nMicrophones], dtype=np.double)
#
#     # frequency cut-off of idea low pass filter
#     Fc = 1
#
#     # width of FIR taps
#     width = 2 * round(0.004 * fs) + 1
#     width_half = round(0.004 * fs)
#     time_width = np.arange(width)
#
#     cTs = c / fs
#
#     hann_window = 0.5 * (1 + np.cos(np.linspace(-np.pi, np.pi, width)))
#
#     Rm = np.zeros(3)
#     Rp = np.zeros(3)
#
#     s = ss / cTs
#     L = LL / cTs
#
#     for idxMicrophone in range(nMicrophones):
#         r = rr.T[idxMicrophone] / cTs
#
#         n1 = int(np.ceil(nSamples / (2 * L[0])))
#         n2 = int(np.ceil(nSamples / (2 * L[1])))
#         n3 = int(np.ceil(nSamples / (2 * L[2])))
#
#         # Generate room impulse response
#         for mx in range(-n1, n1 + 1, 1):
#             Rm[0] = 2 * mx * L[0]
#
#             for my in range(-n2, n2 + 1, 1):
#                 Rm[1] = 2 * my * L[1]
#
#                 for mz in range(-n3, n3 + 1, 1):
#                     Rm[2] = 2 * mz * L[2]
#
#                     for q in range(2):
#                         Rp[0] = (1 - 2 * q) * s[0] - r[0]
#                         ISM_X = Rm[0] + (1 - 2 * q) * s[0]
#                         beta_x1 = np.power(beta[0], np.abs(mx - q))
#                         beta_x2 = np.power(beta[1], np.abs(mx))
#
#                         for j in range(2):
#                             Rp[1] = (1 - 2 * j) * s[1] - r[1]
#                             ISM_Y = Rm[1] + (1 - 2 * j) * s[1]
#                             beta_y1 = np.power(beta[2], np.abs(my - j))
#                             beta_y2 = np.power(beta[3], np.abs(my))
#
#                             for k in range(2):
#                                 Rp[2] = (1 - 2 * k) * s[2] - r[2]
#                                 ISM_Z = Rm[2] + (1 - 2 * k) * s[2]
#                                 beta_z1 = np.power(beta[4], np.abs(mz - k))
#                                 beta_z2 = np.power(beta[5], np.abs(mz))
#
#                                 order = abs(2 * mx - q) + abs(2 * my - j) + abs(2 * mz - k)
#                                 dist_in_sample = np.sqrt(np.sum(np.power(Rm+Rp, 2)))
#
#                                 if order <= nOrder or nOrder == -1:
#                                     fdist_in_sample = np.floor(dist_in_sample)
#
#                                     if fdist_in_sample < nSamples:
#                                         mic_gain = sim_microphone(Rm+Rp, angle, microphone_type[0])
#                                         dist = dist_in_sample * cTs
#                                         gain = 1 * beta_x1 * beta_x2 * beta_y1 * beta_y2 * beta_z1 * beta_z2 / (4 * np.pi * dist)
#
#                                         t = (time_width - width_half) - (dist_in_sample - fdist_in_sample)
#                                         LPI = hann_window * np.sinc(Fc * t)
#
#                                         startPosition = int(fdist_in_sample - width_half)
#
#                                         for n in range(width):
#                                             if startPosition + n >= 0 and startPosition + n < nSamples:
#                                                 imp[startPosition + n][idxMicrophone] += gain * LPI[n]
#
#         # # HIGH Pass FILTER
#         # W = 2 * np.pi * 100 / fs
#         # R1 = np.exp(-W)
#         # B1 = 2 * R1 * np.cos(W)
#         # B2 = -R1 * R1
#         # A1 = -(1 + R1)
#         #
#         # #  'Original' high-pass filter as proposed by Allen and Berkley.
#         # if isHighPassFilter == 1:
#         #     Y = [0] * 3
#         #
#         #     for idx in range(nSamples):
#         #         X0 = imp[idx][idxMicrophone]
#         #         Y[2] = Y[1]
#         #         Y[1] = Y[0]
#         #         Y[0] = B1 * Y[1] + B2 * Y[2] + X0
#         #         imp[idx][idxMicrophone] = Y[0] + A1 * Y[1] + R1 * Y[2]
#     return imp
#
# def rir_generator(c, samplingRate, micPositions, srcPosition, LL, **kwargs):
#     if type(LL) is not np.array:
#         LL = np.array(LL, ndmin=2)
#     if LL.shape[0] == 1:
#         LL = np.transpose(LL)
#
#     if type(micPositions) is not np.array:
#         micPositions = np.array(micPositions, ndmin=2)
#     if type(srcPosition) is not np.array:
#         srcPosition = np.array(srcPosition, ndmin=2)
#
#     """Passing beta"""
#     beta = np.zeros([6, 1], dtype=np.double)
#     if 'beta' in kwargs:
#         betaIn = kwargs['beta']
#         if type(betaIn) is not np.array:
#             betaIn = np.transpose(np.array(betaIn, ndmin=2))
#         if (betaIn.shape[0]) == 6:
#             beta = betaIn
#             V = LL[0] * LL[1] * LL[2]
#             alpha = ((1 - pow(beta[0], 2)) + (1 - pow(beta[1], 2))) * LL[0] * LL[2] + ((1 - pow(beta[2], 2)) + (1 - pow(beta[3], 2))) * LL[1] * LL[2] + ((1 - pow(beta[4], 2)) + (1 - pow(beta[5], 2))) * LL[0] * LL[1]
#             reverberation_time = 24 * np.log(10.0) * V / (c * alpha)
#             if (reverberation_time < 0.128):
#                 reverberation_time = 0.128
#         else:
#             reverberation_time = betaIn
#             if (reverberation_time != 0):
#                 V = LL[0] * LL[1] * LL[2]
#                 S = 2 * (LL[0] * LL[2] + LL[1] * LL[2] + LL[0] * LL[1])
#                 alfa = 24 * V * np.log(10.0) / (c * S * reverberation_time)
#                 if alfa > 1:
#                     raise ValueError("Error: The reflection coefficients cannot be calculated using the current room parameters, i.e. room size and reverberation time.\n Please specify the reflection coefficients or change the room parameters.")
#                 beta = np.zeros([6, 1])
#                 beta += np.sqrt(1 - alfa) * -1
#             else:
#                 beta = np.zeros([6, 1])
#     else:
#         raise ValueError("Error: Specify either RT60 (ex: beta=0.4) or reflection coefficients (beta=[0.3,0.2,0.5,0.1,0.1,0.1])")
#
#     """Number of samples: Default T60 * Fs"""
#     if 'nsample' in kwargs:
#         nsamples = kwargs['nsample']
#     else:
#         nsamples = int(reverberation_time * samplingRate)
#
#     """Mic type: Default omnidirectional"""
#     m_type = 'omnidirectional'
#     if 'mtype' in kwargs:
#         m_type = kwargs['mtype']
#     if m_type == 'bidirectional':
#         mtype = 'b'
#     if m_type == 'cardioid':
#         mtype = 'c'
#     if m_type == 'subcardioid':
#         mtype = 's'
#     if m_type == 'hypercardioid':
#         mtype = 'h'
#     if m_type == 'omnidirectional':
#         mtype = 'o'
#
#     """Reflection order: Default -1"""
#     order = -1
#     if 'order' in kwargs:
#         order = kwargs['order']
#         if order<-1:
#             raise ValueError("Invalid input: reflection order should be > -1")
#
#     """Room dimensions: Default 3"""
#     dim = 3
#     if 'dim' in kwargs:
#         dim = kwargs['dim']
#         if dim not in [2,3]:
#             raise ValueError("Invalid input: 2 or 3 dimensions expected")
#         if dim == 2:
#             beta[4] = 0
#             beta[5] = 0
#
#     """Orientation"""
#     angle = np.zeros([2, 1], dtype=np.double)
#     if 'orientation' in kwargs:
#         orientation = kwargs['orientation']
#         if type(orientation) is not np.array:
#             orientation = np.array(orientation, ndmin=2)
#         if orientation.shape[1] == 1:
#             angle[0] = orientation[0]
#         else:
#             angle[0] = orientation[0, 0]
#             angle[1] = orientation[0, 1]
#
#     """hp_filter enable"""
#     isHighPassFilter = 1
#     if 'hp_filter' in kwargs:
#         isHighPassFilter = kwargs['hp_filter']
#
#     numMics = micPositions.shape[0]
#
#     roomDim = np.ascontiguousarray(LL.astype('double'), dtype=np.double)
#     micPos = np.ascontiguousarray(np.transpose(micPositions).astype('double'), dtype=np.double)
#     srcPos = np.ascontiguousarray(np.transpose(srcPosition).astype('double'), dtype=np.double)
#
#     imp = computeRIR(c, samplingRate, micPos, numMics, nsamples, srcPos, roomDim, beta, mtype, order, angle, isHighPassFilter)
#     return imp.T

from collections import defaultdict
import numpy as np
import math
from scipy import signal, interpolate

def octave_bands(fc=1000, third=False, start=0.0, n=8):
    """
    Create a bank of octave bands

    Parameters
    ----------
    fc : float, optional
        The center frequency
    third : bool, optional
        Use third octave bands (default False)
    start : float, optional
        Starting frequency for octave bands in Hz (default 0.)
    n : int, optional
        Number of frequency bands (default 8)
    """

    div = 1
    if third:
        div = 3

    # Octave Bands
    fcentre = fc * (
        2.0 ** (np.arange(start * div, (start + n) * div - (div - 1)) / div)
    )
    fd = 2 ** (0.5 / div)
    bands = np.array([[f / fd, f * fd] for f in fcentre])

    return bands, fcentre

class ISM:

    def __init__(self):
        pass

    def createMultiBands(self, base_frequency = 125, n_fft = 512):
        self.base_freq = base_frequency
        self.n_fft = n_fft

        # compute the number of bands
        self.n_bands = math.floor(np.log2(self.fs / base_frequency))

        self.bands, self.centers = octave_bands(fc=self.base_freq, n=self.n_bands, third=False)

        self._make_filters()

    def _make_filters(self):
        """
        Create the band-pass filters for the octave bands

        Parameters
        ----------
        order: int, optional
            The order of the IIR filters (default: 8)
        output: {'ba', 'zpk', 'sos'}
            Type of output: numerator/denominator ('ba'), pole-zero ('zpk'), or
            second-order sections ('sos'). Default is 'ba'.

        Returns
        -------
        A list of callables that will each apply one of the band-pass filters
        """

        """
        filter_bank = bandpass_filterbank(
            self.bands, fs=self.fs, order=order, output=output
        )

        return [lambda sig: sosfiltfilt(bpf, sig) for bpf in filter_bank]
        """

        # This seems to work only for Octave bands out of the box
        centers = self.centers
        n = len(self.centers)

        new_bands = [[centers[0] / 2, centers[1]]]
        for i in range(1, n - 1):
            new_bands.append([centers[i - 1], centers[i + 1]])
        new_bands.append([centers[-2], self.fs / 2])

        n_freq = self.n_fft // 2 + 1
        freq_resp = np.zeros((n_freq, n))
        freq = np.arange(n_freq) / self.n_fft * self.fs

        for b, (band, center) in enumerate(zip(new_bands, centers)):
            lo = np.logical_and(band[0] <= freq, freq < center)
            freq_resp[lo, b] = 0.5 * (1 + np.cos(2 * np.pi * freq[lo] / center))

            if b != n - 1:
                hi = np.logical_and(center <= freq, freq < band[1])
                freq_resp[hi, b] = 0.5 * (1 - np.cos(2 * np.pi * freq[hi] / band[1]))
            else:
                hi = center <= freq
                freq_resp[hi, b] = 1.0

        filters = np.fft.fftshift(
            np.fft.irfft(freq_resp, n=self.n_fft, axis=0),
            axes=[0],
        )

        # remove the first sample to make them odd-length symmetric filters
        self.filters = filters[1:, :]

    def get_bw(self):
        """Returns the bandwidth of the bands"""
        return np.array([b2 - b1 for b1, b2 in self.bands])

    def resample(self, coeffs=0.0, center_freqs=None, interp_kind="linear"):
        """
        Takes as input a list of values with optional corresponding center frequency.
        Returns a list with the correct number of octave bands. Interpolation and
        extrapolation are used to fill in the missing values.

        Parameters
        ----------
        coeffs: list
            A list of values to use for the octave bands
        center_freqs: list, optional
            The optional list of center frequencies
        interp_kind: str
            Specifies the kind of interpolation as a string (‘linear’,
            ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’,
            ‘next’, where ‘zero’, ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a
            spline interpolation of zeroth, first, second or third order;
            ‘previous’ and ‘next’ simply return the previous or next value of
            the point) or as an integer specifying the order of the spline
            interpolator to use. Default is ‘linear’.
        """

        if not isinstance(coeffs, (list, np.ndarray)):
            # when the parameter is a scalar just do flat extrapolation
            ret = [coeffs] * self.n_bands

        if len(coeffs) == 1:
            ret = coeffs * int(self.n_bands)

        else:
            # by default infer the center freq to be the low ones
            if center_freqs is None:
                center_freqs = self.centers[: len(coeffs)]

            # create the interpolator in log domain
            interpolator = interpolate.interp1d(
                np.log2(center_freqs),
                coeffs,
                fill_value="extrapolate",
                kind=interp_kind,
            )
            ret = interpolator(np.log2(self.centers))

            # now clip between 0. and 1.
            ret[ret < 0.0] = 0.0
            ret[ret > 1.0] = 1.0

        return ret

    def analysis(self, x, band=None):
        """
        Process a signal x through the filter bank

        Parameters
        ----------
        x: ndarray (n_samples)
            The input signal

        Returns
        -------
        ndarray (n_samples, n_bands)
            The input signal filters through all the bands
        """

        if band is None:
            bands = range(self.filters.shape[1])
        else:
            bands = [band]

        output = np.zeros((x.shape[0], len(bands)), dtype=x.dtype)

        for i, b in enumerate(bands):
            output[:, i] = signal.fftconvolve(x, self.filters[:, b], mode="same")

        if output.shape[1] == 1:
            return output[:, 0]
        else:
            return output

    def defineSystem(self, fs, velocity, sample, time = 0.004):
        self.fs = fs
        self.velocity = velocity
        self.width = int(2 * np.round(time * fs) + 1)
        self.width_half = int(np.round(time * fs))
        self.time_width = np.arange(self.width)
        self.cTs = velocity / fs
        self.hann_window = 0.5 * (1 + np.cos(np.linspace(-np.pi, np.pi, self.width)))
        self.nSamples = sample
        self.cluster = defaultdict(list)

    def createRoom(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.volume = self.x * self.y * self.z
        self.surface = 2 * (self.x * self.y + self.x * self.z + self.y * self.z)
        self.dimension = np.array([x, y, z])
        self.l = np.array([x, y, z]) / self.cTs

    def addMicrophone(self, x, y, z):
        self.mircophone = np.array([x, y, z])
        self.r = np.array([x, y, z]) / self.cTs

    def addSource(self, x, y, z):
        self.source = np.array([x, y, z])
        self.s = np.array([x, y, z]) / self.cTs

    def createMaterialByCoefficient(self, x1, x2, y1, y2, z1, z2):
        # define reflection coefficient
        self.beta = np.array([x1, x2, y1, y2, z1, z2])

        # Sabin-Franklin’s formula
        # RT_60 = \frac { 24 * ln(10) * V } { c * \sum_{i = 1}^{6} S_i * (1 - beta_i^{2}) }
        alpha = (1 - np.power(x1, 2)) * self.y * self.z + \
                (1 - np.power(x2, 2)) * self.y * self.z + \
                (1 - np.power(y1, 2)) * self.x * self.z + \
                (1 - np.power(y2, 2)) * self.x * self.z + \
                (1 - np.power(z1, 2)) * self.x * self.y + \
                (1 - np.power(z2, 2)) * self.x * self.y
        if (np.sum(alpha) > 0):
            self.reverberation = 24 * np.log(10.0) * self.volume / (self.velocity * alpha)

    def createMaterialByTime(self, reverberation_time):
        if (reverberation_time != 0):
            alpha = 24 * np.log(10.0) * self.volume / (self.velocity * self.surface * reverberation_time)
            beta = np.sqrt(1 - alpha)
            self.beta = np.full(6, beta * -1)
            return self.beta

    def computeISM(self):
        n1 = int(np.ceil(self.nSamples / (2 * self.l[0])))
        n2 = int(np.ceil(self.nSamples / (2 * self.l[1])))
        n3 = int(np.ceil(self.nSamples / (2 * self.l[2])))

        Rm = np.zeros(3)
        Rp = np.zeros(3)

        # Generate room impulse response
        for mx in range(-n1, n1 + 1, 1):
            Rm[0] = 2 * mx * self.l[0]

            for my in range(-n2, n2 + 1, 1):
                Rm[1] = 2 * my * self.l[1]

                for mz in range(-n3, n3 + 1, 1):
                    Rm[2] = 2 * mz * self.l[2]

                    for q in range(2):
                        Rp[0] = (1 - 2 * q) * self.s[0] - self.r[0]
                        ISM_X = Rm[0] + (1 - 2 * q) * self.s[0]
                        beta_x1 = np.power(self.beta[0], np.abs(mx - q))
                        beta_x2 = np.power(self.beta[1], np.abs(mx))

                        for j in range(2):
                            Rp[1] = (1 - 2 * j) * self.s[1] - self.r[1]
                            ISM_Y = Rm[1] + (1 - 2 * j) * self.s[1]
                            beta_y1 = np.power(self.beta[2], np.abs(my - j))
                            beta_y2 = np.power(self.beta[3], np.abs(my))

                            for k in range(2):
                                Rp[2] = (1 - 2 * k) * self.s[2] - self.r[2]
                                ISM_Z = Rm[2] + (1 - 2 * k) * self.s[2]
                                beta_z1 = np.power(self.beta[4], np.abs(mz - k))
                                beta_z2 = np.power(self.beta[5], np.abs(mz))

                                order = abs(2 * mx - q) + abs(2 * my - j) + abs(2 * mz - k)
                                dist_in_sample = np.sqrt(np.sum(np.power(Rm + Rp, 2)))
                                fdist_in_sample = np.floor(dist_in_sample)

                                temp = dict()
                                temp['dist'] = dist_in_sample * self.cTs
                                temp['start'] = int(fdist_in_sample - self.width_half)
                                temp['center'] = fdist_in_sample
                                temp['end'] = int(fdist_in_sample + self.width_half)
                                temp['beta'] = np.array([beta_x1, beta_x2, beta_y1, beta_y2, beta_z1, beta_z2])
                                temp['image_source'] = np.array([ISM_X, ISM_Y, ISM_Z]) * self.cTs
                                temp['q'] = q
                                temp['j'] = j
                                temp['k'] = k
                                temp['Rm'] = Rm * self.cTs
                                temp['Rp'] = Rp * self.cTs

                                self.cluster[order].append(temp)

    def computeRIR(self):
        bws = self.get_bw()
        cluster_size = len(self.cluster)
        imp = np.zeros((self.nSamples, cluster_size))

        for b, bw in enumerate(bws):
            print("current band: ", b)
            for order in range(cluster_size):
                for index, block in enumerate(self.cluster[order]):
                    dist_in_sample = self.cluster[order][index]['dist'] / self.cTs
                    fdist_in_sample = int(dist_in_sample)
                    dist = self.cluster[order][index]['dist']
                    beta = self.cluster[order][index]['beta']
                    startPosition = self.cluster[order][index]['start']

                    if fdist_in_sample < self.nSamples:
                        beta_sb = beta.T[b]
                        gain = beta_sb[0] * beta_sb[1] * beta_sb[2] * beta_sb[3] * beta_sb[4] * beta_sb[5] / (4 * np.pi * dist)

                        t = (self.time_width - self.width_half) - (dist_in_sample - fdist_in_sample)
                        LPI = self.hann_window * np.sinc(t)
                        sub_band = self.analysis(gain * LPI, b)
                        for n in range(self.width):
                            if startPosition + n >= 0 and startPosition + n < self.nSamples:
                                imp[startPosition + n][order] = imp[startPosition + n][order] + sub_band[n]
        return imp