"""
Function Description: Calculate EnergyDecayRelief (EDR)and EnergyDecayCurve (EDC)
Function Input:  IR files or generated IR data and Sample Rate
"""
import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import wavio
from mpl_toolkits.mplot3d import Axes3D

def nextpow2(n):
    return np.ceil(np.log2(np.abs(n))).astype('long')

def _Integral(array):
    return np.flipud(np.cumsum(np.flipud(array)))

# calculate EDR
def _EnergyDecayRelief(inpath, fs=48000):
    if type(inpath) == str:
        w = wavio.read(inpath)
        data = w.data[:, 0] / (2 ** (w.sampwidth * 8 - 1) - 1)
        fs = w.rate
    else:
        data = inpath

    # Settings and Calculating STFT
    frameSizeMs = 20  # minimum frame length, in ms
    overlap = 0.75  # fraction of frame overlapping
    minFrameLen = fs * frameSizeMs / 1000
    frameLenPow = nextpow2(minFrameLen)
    frameLen = 2 ** frameLenPow  # frame length = fft size
    F, T, B = signal.stft(data, fs, 'hann', frameLen, overlap * frameLen)

    # Calculating Energy
    B_energy = np.multiply(B, B.conjugate())
    B_EDR = np.zeros((np.size(B, 0), np.size(B, 1)))
    for i in range(0, np.size(B, 0)):
        B_EDR[i, :] = _Integral(B_energy[i, :])
    return T, F, B_EDR

# calculate EDC
def _EnergyDecayCurve(inpath, fs=48000):
    if type(inpath) == str:
        w = wavio.read(inpath)
        data = w.data[:, 0] / (2 ** (w.sampwidth * 8 - 1) - 1)
        fs = w.rate
    else:
        data = inpath

    SampleNumber = len(data)
    total_time = SampleNumber / fs * 1000
    T = np.linspace(0, total_time, SampleNumber)
    B_EDC = _Integral(np.square(data))
    return T, B_EDC

"""
Example: Input a IR File
"""
if __name__ == "__main__":
    path = 'D:/lixin25/Download/Eric Leung Acoustic Guitar IRs/ACgtr IRs/Ac Gtr Verb st.wav'

    # Plot EDC
    T, B_EDC = _EnergyDecayCurve(path)

    B_EDCdb = 10.0 * np.log10(B_EDC)  # Convert to Decibel Representation
    offset = np.max(B_EDCdb)
    B_EDCdbN = B_EDCdb - offset  # Normalized
    minPlotDB = -140  # Set Threshold
    for i in range(len(B_EDCdbN)):
        if (B_EDCdbN[i] < minPlotDB):
            B_EDCdbN[i] = minPlotDB

    plt.plot(T, B_EDCdbN)
    plt.grid()
    plt.xlabel('time(ms)')
    plt.ylabel('EDC(dB)')
    plt.title('Energy Decay Curve of IR')
    plt.show()

    # Plot EDR
    T, F, B_EDR = _EnergyDecayRelief(path)

    B_EDRdb = 10 * np.log10(B_EDR)  # Convert to Decibel Representation
    offset = np.max(B_EDRdb)
    B_EDRdbN = B_EDRdb - offset  # Normalized
    B_EDRdbN_trunc = B_EDRdbN
    minPlotDB = -70  # Set Threshold
    for i in range(0, np.size(B_EDRdbN, 0)):
        for j in range(0, np.size(B_EDRdbN, 1)):
            if (B_EDRdbN_trunc[i, j] < minPlotDB):
                B_EDRdbN_trunc[i, j] = minPlotDB

    X, Y = np.meshgrid(T * 1000, F / 1000)
    Z = B_EDRdbN_trunc
    c = plt.pcolormesh(X, Y, Z, cmap='jet')  # Two-dimensional
    plt.colorbar(c)
    plt.title(' Energy Decay Relief of IR')
    plt.ylabel('frequency(kHz)')
    plt.xlabel('time(ms)')
    plt.show()

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(X, Y, Z, 200, cmap='jet')  # Three-dimensional
    ax.set_ylabel('frequency/kHz')
    ax.set_xlabel('time(ms)')
    ax.set_zlabel('Magnitude (dB)')
    ax.set_title('Energy Decay Relief of IR')
    plt.show()