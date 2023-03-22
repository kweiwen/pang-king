# function [level, A, N] = decayFitNet2InitialLevel(T, A, N, normalization, fs, rirLen, fBands)
# % convert decayFitNet estimation to initial level as used in FDNs
# %
# % (c) Sebastian J. Schlecht, Saturday, 28 January 2023
#
# % Denormalize the amplitudes of the EDC
# A = A .* normalization;
# N = N .* normalization;
#
# % Estimate the energy of the octave filters
# rirFBands = octaveFiltering([1; zeros(fs,1)], fs, fBands);
# bandEnergy = sum(rirFBands.^2,1);
#
# % Cumulative energy is a geometric series of the gain per sample
# gainPerSample = db2mag(RT602slope(T, fs));
# decayEnergy = 1 ./ (1 - gainPerSample.^2);
#
# % initial level
# level = sqrt( A ./ bandEnergy ./ decayEnergy * rirLen);
# % there is an offset because, the FDN is not energy normalized
# % The rirLen factor is due to the normalization in schroederInt (in DecayFitNet)
#
# end

import numpy as np
import sys
from scipy.signal import butter, sosfilt, zpk2sos
import matplotlib.pyplot as plt
def db2mag(input_data):
    return 10**(input_data/20.0)

def RT602slope(RT60, fs):
    if np.any(RT60 == 0):
        RT60[RT60 == 0] = sys.float_info.epsilon

    slope = -60.0 / (RT60 * fs)
    return slope

def octaveFiltering(inputSignal, fs, fBands):
    numBands = len(fBands)
    outBands = np.zeros((len(inputSignal), numBands))

    for bIdx in range(numBands):
        # Determine IIR filter coefficients for this band
        if fBands[bIdx] == 0:
            # Lowpass band below lowest octave band
            fCutoff = (1/np.sqrt(2))*fBands[bIdx+1]
            z, p, k = butter(5, fCutoff/fs*2, btype='low', output='zpk')
        elif fBands[bIdx] == fs/2:
            # Highpass band above highest octave band
            fCutoff = np.sqrt(2)*fBands[bIdx-1]
            z, p, k = butter(5, fCutoff/fs*2, btype='high', output='zpk')
        else:
            thisBand = fBands[bIdx] * np.array([1/np.sqrt(2), np.sqrt(2)])
            z, p, k = butter(5, thisBand/fs*2, btype='bandpass', output='zpk')

        print('bIdx: ', bIdx)
        print('z   : ', z)
        print('p   : ', p)
        print('k   : ', k)
        # Zero phase filtering
        sos = zpk2sos(z, p, k)
        outBands[:, bIdx] = sosfilt(sos, inputSignal)

    return outBands

def decayFitNet2InitialLevel(T, A, N, normalization, fs, rirLen, fBands):
    A_norm = A * normalization
    N_norm = N * normalization

    impulse = np.zeros(fs + 1)
    impulse[0] = 1
    rirFBands = octaveFiltering(impulse, fs, fBands)
    bandEnergy = np.sum(rirFBands * rirFBands, 1)

    gainPerSample = db2mag(RT602slope(T, fs))
    decayEnergy = 1 / (1 - gainPerSample ** 2)

    level = np.sqrt( A_norm / bandEnergy / decayEnergy * rirLen)

    return level, A_norm, N_norm


if __name__ == "__main__":
    fs = 48000
    # impulse = np.zeros(fs+1)
    impulse = np.random.ranf(fs + 1)
    fbands = [0, 500, 1000, 2000, 4000, 8000, 16000, fs/2]
    RIRs = octaveFiltering(impulse, fs, fbands).T

    for RIR in RIRs:
        magnitude = np.abs(np.fft.rfft(RIR))
        fn = np.log10(magnitude/len(RIR)) * 20
        plt.plot(fn)
    plt.show()







