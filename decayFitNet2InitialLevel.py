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

        # print('bIdx: ', bIdx)
        # print('z   : ', z)
        # print('p   : ', p)
        # print('k   : ', k)
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
    bandEnergy = np.sum(rirFBands * rirFBands, 0)

    gainPerSample = db2mag(RT602slope(T, fs))
    decayEnergy = 1 / (1 - gainPerSample ** 2)

    level = np.sqrt( A_norm / bandEnergy / decayEnergy * rirLen)

    return level, A_norm, N_norm

def graphicEQ(centerOmega, shelvingOmega, R, gaindB):
    numFreq = len(centerOmega) + len(shelvingOmega) + 1
    assert len(gaindB) == numFreq
    SOS = np.zeros((numFreq, 6))

    for band in range(numFreq):
        if band == 0:
            b = [1, 0, 0] * db2mag(gaindB[band])
            a = [1, 0, 0]
        elif band == 1:
            b, a = shelvingFilter(shelvingOmega[0], db2mag(gaindB[band]), 'low')
        elif band == numFreq - 1:
            b, a = shelvingFilter(shelvingOmega[1], db2mag(gaindB[band]), 'high')
        else:
            Q = np.sqrt(R) / (R - 1)
            b, a = bandpassFilter(centerOmega[band - 2], db2mag(gaindB[band]), Q)

        sos = np.hstack((b, a))

        SOS[band, :] = sos
    return SOS


def shelvingFilter(omegaC, gain, Q):
    b = np.zeros(3)
    a = np.zeros(3)

    t = np.tan(omegaC / 2)
    t2 = t ** 2
    g2 = gain ** 0.5
    g4 = gain ** 0.25

    b[0] = g2 * t2 + np.sqrt(2) * t * g4 + 1
    b[1] = 2 * g2 * t2 - 2
    b[2] = g2 * t2 - np.sqrt(2) * t * g4 + 1

    a[0] = g2 + np.sqrt(2) * t * g4 + t2
    a[1] = 2 * t2 - 2 * g2
    a[2] = g2 - np.sqrt(2) * t * g4 + t2

    b = g2 * b

    if type == 'low':
        pass
    elif type == 'high':
        tmp = b
        b = a * gain
        a = tmp

    return b, a


def bandpassFilter(omegaC, gain, Q):
    b = np.zeros(3)
    a = np.zeros(3)

    bandWidth = omegaC / Q
    t = np.tan(bandWidth / 2)

    b[0] = np.sqrt(gain) + gain * t
    b[1] = -2 * np.sqrt(gain) * np.cos(omegaC)
    b[2] = np.sqrt(gain) - gain * t

    a[0] = np.sqrt(gain) + t
    a[1] = -2 * np.sqrt(gain) * np.cos(omegaC)
    a[2] = np.sqrt(gain) - t

    return b, a


if __name__ == "__main__":
    fs = 48000
    fBands = [0, 500, 1000, 2000, 4000, 8000, 16000, fs/2]
    rirLen = fs

    impulse = np.random.ranf(fs + 1)
    # impulse = np.zeros(fs+1)
    # RIRs = octaveFiltering(impulse, fs, fBands).T
    #
    # for RIR in RIRs:
    #     magnitude = np.abs(np.fft.rfft(RIR))
    #     fn = np.log10(magnitude/len(RIR)) * 20
    #     plt.plot(fn)
    # plt.show()

    normalization = np.random.ranf(8)
    T = np.random.ranf(8)
    A = np.random.ranf(8)
    N = np.random.ranf(8)
    level, A_norm, N_norm = decayFitNet2InitialLevel(T, A, N, normalization, fs, rirLen, fBands)

    # print(level)
    # print(A_norm)
    # print(N_norm)
