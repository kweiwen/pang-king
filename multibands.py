
class MultiBands:

    def __init__(self, base_frequency = 125, n_fft = 512):
        self.base_freq = base_frequency
        self.n_fft = n_fft

        # compute the number of bands
        self.n_bands = math.floor(np.log2(self.fs / base_frequency))

        self.bands, self.centers = self.octave_bands(fc=self.base_freq, n=self.n_bands, third=False)

        self._make_filters()

    def octave_bands(self, fc=1000, third=False, start=0.0, n=8):
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