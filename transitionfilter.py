import scipy.signal as signal
import numpy as np
import matplotlib.pyplot as plt
import wavio
import pyroomacoustics as pra


def nextpow2(n):
    return np.ceil(np.log2(np.abs(n))).astype('long')


def _Integral(array):
    array = np.array(array)
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

    B_energy = (np.abs(fftSet) / N * 2)**2
    B_EDR = np.zeros([int(N / 2) + 1, iteration])

    for i in range(np.size(B_EDR, 0)):
        B_EDR[i, :] = _Integral(B_energy[i, :])

    return B_EDR

if __name__ == "__main__":
    m1 = pra.make_materials(
        ceiling="reverb_chamber",
        floor="concrete_floor",
        east="concrete_floor",
        west="plywood_thin",
        north="hard_surface",
        south="hard_surface",
    )
    m2 = pra.make_materials(
        ceiling='hard_surface',
        floor='plywood_thin',
        east='brickwork',
        west='brickwork',
        north='concrete_floor',
        south='brickwork',

    )
    room_dim = [8, 6, 3]
    tmix = np.sqrt(np.prod(room_dim))
    fs = 48000
    room1 = pra.ShoeBox(
        room_dim, fs=fs, materials=m1, max_order=16, air_absorption=True, ray_tracing=False
    )
    room2 = pra.ShoeBox(
        room_dim, fs=fs, materials=m2, max_order=6, air_absorption=True, ray_tracing=False
    )
    room1.add_source(position=[1, 1.5, 0.2])
    room2.add_source(position=[1, 1.5, 0.2])
    room1.add_microphone(loc=[4, 2.5, 0.2])
    room2.add_microphone(loc=[4, 2.5, 0.2])
    room1.compute_rir()
    room2.compute_rir()

    h_n = room1.rir[0][0]
    h_fn = room2.rir[0][0]
    mixingNumber = int(np.floor(tmix * fs / 1000))
    h_cn = np.hstack((h_n[0:mixingNumber], h_fn[mixingNumber:]))   #ER & FDN combine

    # Generate Noise
    t = 3
    N = t * fs
    mu = 0.2
    sigma = 0.1
    s1 = np.random.normal(mu, sigma, N)

    conv_h_n = signal.convolve(s1, h_n)    # Output
    conv_h_cn = signal.convolve(s1, h_cn)

    # Generate Filter Coeff
    frameSizeMs = 20
    overlap = 0.75
    minFrameLen = int(fs * frameSizeMs / 1000)
    frameLenPow = nextpow2(minFrameLen)
    N = 2 ** frameLenPow
    StepSize = int((1 - overlap) * minFrameLen)

    dataSet, data, datalen, iteration = PreProcessData(h_n, minFrameLen, StepSize)

    B_EDR_hn = _EnergyDecayRelief(dataSet, iteration, N)
    B_EDRdb_hn = 10 * np.log10(B_EDR_hn)

    dataSet, data, datalen, iteration = PreProcessData(h_cn, minFrameLen, StepSize)

    B_EDR_hcn = _EnergyDecayRelief(dataSet, iteration, N)
    B_EDRdb_hcn = 10 * np.log10(B_EDR_hcn)

    k = np.linspace(0, int(N / 2), int(N / 2) + 1)
    fk = k / N * fs
    time = np.linspace(0, datalen / fs, iteration)

    for i, t in enumerate(time * 1000):
        if t > tmix:
            index = i
            break

    coeff = np.sqrt(B_EDR_hn[1:, index] / B_EDR_hcn[1:, index])
    taps = 255
    fir_coeff = signal.firls(taps, fk[1:], coeff, fs=fs)

    h_cn_f = np.hstack((h_cn[0:mixingNumber], np.zeros(len(h_cn[mixingNumber:]))))
    h_cn_b = np.hstack((np.zeros(len(h_cn[0:mixingNumber])), h_cn[mixingNumber:]))

    conv_h_cn_f = signal.convolve(s1, h_cn_f)
    conv_h_cn_b = signal.convolve(s1, h_cn_b)

    conv_h_cn_b_fir = signal.lfilter(fir_coeff, 1, conv_h_cn_b)

    conv_h_cn_cf = conv_h_cn_f + conv_h_cn_b_fir

    # MSE
    dataSet1, data, datalen, iteration = PreProcessData(conv_h_n, minFrameLen, StepSize)
    B_EDR_hn = _EnergyDecayRelief(dataSet1, iteration, N)
    dataSet1, data, datalen, iteration = PreProcessData(conv_h_cn, minFrameLen, StepSize)
    B_EDR_hcn = _EnergyDecayRelief(dataSet1, iteration, N)
    dataSet1, data, datalen, iteration = PreProcessData(conv_h_cn_cf, minFrameLen, StepSize)
    B_EDR_hcn_cf = _EnergyDecayRelief(dataSet1, iteration, N)

    Y1 = 10 * np.log10(B_EDR_hn[:, index])
    Y2 = 10 * np.log10(B_EDR_hcn[:, index])
    Y3 = 10 * np.log10(B_EDR_hcn_cf[:, index])
    mse_h_cn = np.sqrt(np.sum((Y1 - Y2) ** 2) / (len(Y1)))  # h(n)*x(n) & hc(n)*x(n)
    mse_h_cn_cf = np.sqrt(np.sum((Y1 - Y3) ** 2) / (len(Y1)))  # h(n)*x(n) & h(n)*x(n)+hf(n)*x(n)*c(f)
    print('MSE<h(n)*x(n)-hc(n)*x(n)>:      ', mse_h_cn)
    print('MSE<h(n)*x(n)-hc(n)*x(n)*c(f)>: ', mse_h_cn_cf)