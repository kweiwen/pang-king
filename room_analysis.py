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

def PreProcessData(xn, frameSize, stepSize):
    padding = len(xn) % stepSize
    if(padding == 0):
        temp = xn
    else:
        temp = np.append(xn, np.zeros(stepSize - padding))

    data = np.append(np.zeros(frameSize - stepSize), temp)

    iteration = (len(data) // stepSize) - (frameSize // stepSize) + 1

    dataSet = np.zeros([iteration, frameSize])

    for i in range(iteration):
        start = i * stepSize
        end = start + frameSize
        dataSet[i] = data[start:end]

    return dataSet, data, len(data), iteration

# calculate EDR
def _EnergyDecayRelief(dataSet, iteration, N):
    fftSet = np.zeros([N, iteration])
    datalen = len(dataSet[0])
    winfunc = np.hanning(datalen)
    for i in range(iteration):
        dataSet[i] *= winfunc
        fftSet[:, i] = np.fft.fft(dataSet[i], N)

    B_Amp = np.abs(fftSet) / N * 2
    B_AmpSum = np.zeros([int(N / 2) + 1, iteration])

    for i in range(np.size(B_AmpSum, 0)):
        B_AmpSum[i, :] = _Integral(B_Amp[i, :])
    B_EDR = B_AmpSum ** 2
    return B_EDR

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
    w = wavio.read(path)
    IRdata = w.data[:, 0] / (2 ** (w.sampwidth * 8 - 1) - 1)
    fs = w.rate
    # Plot EDC
    T, B_EDC = _EnergyDecayCurve(IRdata)

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
    frameSizeMs = 20
    overlap = 0.75
    minFrameLen = int(fs * frameSizeMs / 1000)
    frameLenPow = nextpow2(minFrameLen)
    N = 2 ** frameLenPow
    StepSize = int((1 - overlap) * minFrameLen)

    dataSet, data, datalen, iteration = PreProcessData(IRdata, minFrameLen, StepSize)

    B_EDR = _EnergyDecayRelief(dataSet, iteration, N)
    B_EDRdB = 10 * np.log10(B_EDR)
    minPlotDB = -80  # Set Threshold
    for i in range(0, np.size(B_EDRdB, 0)):
        for j in range(0, np.size(B_EDRdB, 1)):
            if (B_EDRdB[i, j] < minPlotDB):
                B_EDRdB[i, j] = minPlotDB

    k = np.linspace(0, int(N / 2), int(N / 2) + 1)
    fk = k / N * fs
    t = np.linspace(0, datalen / fs, iteration)

    X, Y = np.meshgrid(t * 1000, fk / 1000)
    Z = B_EDRdB
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