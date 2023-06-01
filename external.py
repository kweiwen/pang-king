import numpy as np
import sys
import wavio
from scipy.signal import butter, sosfilt, zpk2sos, freqz
from scipy.optimize import lsq_linear
from DecayFitNet.python.toolbox.DecayFitNetToolbox import DecayFitNetToolbox
import matplotlib.pyplot as plt

def mag2db(input_data):
    return 20*np.log10(input_data)

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
    normalization = np.array(normalization)
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
            b = np.multiply(np.array([1, 0, 0]), db2mag(gaindB[band]))
            a = np.array([1, 0, 0])
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

def shelvingFilter(omegaC, gain, type):
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

def probeSOS(SOS, controlFrequencies, fftLen, fs):
    numFreq = SOS.shape[0]

    H = np.zeros((fftLen, numFreq), dtype=np.complex_)
    W = np.zeros((fftLen, numFreq))
    G = np.zeros((len(controlFrequencies), numFreq))

    for band in range(numFreq):
        b = SOS[band, 0:3]
        a = SOS[band, 3:6]

        w, h = freqz(b, a, worN=fftLen, fs=fs)
        g = np.interp(controlFrequencies, w, 20 * np.log10(np.abs(h)))

        G[:, band] = g
        H[:, band] = h
        W[:, band] = w

    return G, H, W

def hertz2rad(freq, fs):
    return 2 * np.pi * np.array(freq) / fs

def designGEQ(targetG: np.array):
    fs = 48000
    fftLen = 2**16

    centerFrequencies = [63, 125, 250, 500, 1000, 2000, 4000, 8000]  # Hz
    ShelvingCrossover = [46, 11360]  # Hz
    numFreq = len(centerFrequencies) + len(ShelvingCrossover)
    shelvingOmega = hertz2rad(ShelvingCrossover, fs)
    centerOmega = hertz2rad(centerFrequencies, fs)
    R = 2.7

    # control frequencies are spaced logarithmically
    numControl = 100
    controlFrequencies = np.round(np.logspace(np.log10(1), np.log10(fs/2.1), numControl+1))

    # target magnitude response via command gains
    targetF = [1] + centerFrequencies + [fs]
    targetInterp = np.interp(controlFrequencies, targetF, targetG)

    # design prototype of the biquad sections
    prototypeGain = 10  # dB
    prototypeGainArray = prototypeGain * np.ones(numFreq+1)
    prototypeSOS = graphicEQ(centerOmega, shelvingOmega, R, prototypeGainArray)
    G, prototypeH, prototypeW = probeSOS(prototypeSOS, controlFrequencies, fftLen, fs)
    G = G / prototypeGain  # dB vs control frequencies

    # compute optimal parametric EQ gains
    # Either you can use a unconstrained linear solver or introduce gain bounds
    # at [-20dB,+20dB] with acceptable deviation from the self-similarity
    # property. The plot shows the deviation between design curve and actual
    # curve.
    upperBound = np.append(np.inf, 2 * prototypeGain * np.ones(numFreq))
    lowerBound = upperBound * -1

    # matlab code
    # x    =    lsqlin(C, d, A, b, Aeq, beq, lb, ub, x0, options)
    # optG =    lsqlin(G, targetInterp, [], [], [], [], lowerBound, upperBound, [], opts)

    optG = lsq_linear(G, targetInterp, bounds=(lowerBound, upperBound), method='bvls').x
    sos = graphicEQ(centerOmega, shelvingOmega, R, optG)
    return sos, targetF


def RIR2AbsCoefLvlCoef(data, delayLines, fs):
    n_slopes = 1
    filter_frequencies = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

    # Prepare the model
    net = DecayFitNetToolbox(n_slopes=n_slopes, sample_rate=fs, filter_frequencies=filter_frequencies)
    est_parameters_net, norm_vals_net = net.estimate_parameters(data, analyse_full_rir=True)
    estT = est_parameters_net[0].T
    estA = est_parameters_net[1].T
    estN = est_parameters_net[2].T
    estL, estA, estN = decayFitNet2InitialLevel(estT, estA, estN, norm_vals_net, fs, data.shape[0], filter_frequencies)

    targetT60 = np.hstack((estT[0, 0], estT[0], estT[0, -1]))
    targetT60 = np.multiply(targetT60, np.array([0.9, 1, 1, 1, 1, 1, 1, 1, 0.9, 0.5]))

    # Convert T60 to magnitude response
    targetG = RT602slope(targetT60, fs)

    output_data = np.zeros([len(delayLines)+1, len(filter_frequencies)+3, 6])

    for index, delayLine in enumerate(delayLines):
        sos, _ = designGEQ(delayLine * targetG)
        output_data[index] = sos

    estLevel = np.hstack((estL[0, 0], estL[0], estL[0, -1]))
    targetLevel = mag2db(estLevel)
    targetLevel = targetLevel - np.array([5, 0, 0, 0, 0, 0, 0, 0, 5, 30])
    sos, _ = designGEQ(targetLevel)
    output_data[-1] = sos

    return output_data

def RIR2FDN(fp, ch1, ch2, ch3, ch4):
    file = wavio.read(fp)
    data = file.data[:, 0] / (2 ** (file.sampwidth * 8 - 1) - 1)
    delayLines = np.array([ch1, ch2, ch3, ch4])
    output_data = RIR2AbsCoefLvlCoef(data, delayLines, file.rate)

    return output_data.tolist()

def demo_RIR2FDN():
    fp = "C:\Python37\Lib\DecayFitNet\data\exampleRIRs\singleslope_0001_1_sh_rirs.wav"
    output_data = RIR2FDN(fp, 1021, 2029, 3001, 4093)
    return np.array(output_data)

def demo_decayFitNet2InitialLevel():
    fs = 48000
    fBands = [0, 500, 1000, 2000, 4000, 8000, 16000, fs/2]
    rirLen = fs

    normalization = np.random.ranf(8)
    T = np.random.ranf(8)
    A = np.random.ranf(8)
    N = np.random.ranf(8)
    level, A_norm, N_norm = decayFitNet2InitialLevel(T, A, N, normalization, fs, rirLen, fBands)

    print(level, A_norm, N_norm)

if __name__ == "__main__":
    coefficients = demo_RIR2FDN()
    for index, coefficient in enumerate(coefficients[4]):
        b = coefficient[:3]
        a = coefficient[3:]
        w, h = freqz(b, a, worN=2**16, fs=48000)
        amplitude = mag2db(np.abs(h))
        plt.plot(amplitude, label="band" + str(index))

    # plt.ylim(-5, 5)
    plt.xscale('log')
    plt.legend(loc="lower right")
    plt.show()