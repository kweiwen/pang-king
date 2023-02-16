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
    # print(frameLen)
    F, T, B = signal.stft(data, fs, 'hann', frameLen, overlap * frameLen)
    # Calculating Energy
    B_energy = np.multiply(B, B.conjugate())
    B_EDR = np.zeros((np.size(B, 0), np.size(B, 1)))
    for i in range(0, np.size(B, 0)):
        B_EDR[i, :] = _Integral(B_energy[i, :])
    return T, F, B_EDR

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
        room_dim, fs=fs, materials=m1, max_order=6, air_absorption=True, ray_tracing=True
    )
    room2 = pra.ShoeBox(
        room_dim, fs=fs, materials=m2, max_order=6, air_absorption=True, ray_tracing=True
    )
    room1.add_source(position=[1, 1.5, 0.2])
    room2.add_source(position=[1, 1.5, 0.2])
    room1.add_microphone(loc=[4, 2.5, 0.2])
    room2.add_microphone(loc=[4, 2.5, 0.2])
    room1.compute_rir()
    room2.compute_rir()

    RIR_og=room1.rir[0][0]
    RIR_fd=room2.rir[0][0]


    if len(RIR_og) < len(RIR_fd):
        h_n = RIR_og
        h_fn = RIR_fd[0:len(RIR_og)]
    elif len(RIR_og) > len(RIR_fd):
        h_n = RIR_og[0:len(RIR_fd)]
        h_fn = RIR_fd
    else:
        h_n = RIR_og
        h_fn = RIR_fd

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
    T1, F1, B_EDR_hn = _EnergyDecayRelief(h_n)
    B_EDRdb_hn = 10 * np.log10(B_EDR_hn)  # Convert to Decibel Representation
    T2, F2, B_EDR_hcn = _EnergyDecayRelief(h_cn)
    B_EDRdb_hcn = 10 * np.log10(B_EDR_hcn)

    for i, t in enumerate(T1 * 1000):
        if t > tmix:
            index=i
            break

    coeff = np.sqrt(B_EDR_hn[1:, index] / B_EDR_hcn[1:, index])
    taps = 255
    fir_coeff = signal.firls(taps, F1[1:], coeff, fs=fs)

    h_cn_f = np.hstack((h_cn[0:mixingNumber], np.zeros(len(h_cn[mixingNumber:]))))
    h_cn_b = np.hstack((np.zeros(len(h_cn[0:mixingNumber])), h_cn[mixingNumber:]))

    conv_h_cn_f = signal.convolve(s1, h_cn_f)
    conv_h_cn_b = signal.convolve(s1, h_cn_b)

    conv_h_cn_b_fir = signal.lfilter(fir_coeff, 1, conv_h_cn_b)
    conv_h_cn_cf = conv_h_cn_f+conv_h_cn_b_fir

    # MSE
    T, F, B_h_n = _EnergyDecayRelief(conv_h_n)
    T, F, B_h_cn = _EnergyDecayRelief(conv_h_cn)
    T, F, B_h_cf = _EnergyDecayRelief(conv_h_cn_cf)
    Y1 = 10 * np.log10(B_h_n[:, index])
    Y2 = 10 * np.log10(B_h_cn[:, index])
    Y3 = 10 * np.log10(B_h_cf[:, index])
    mse_h_cn = np.sqrt(np.sum((Y1 - Y2) ** 2) / (len(Y1)))            #h(n)*x(n) & hc(n)*x(n)
    mse_h_cn_cf = np.sqrt(np.sum((Y1 - Y3) ** 2) / (len(Y1)))         #h(n)*x(n) & h(n)*x(n)+hf(n)*x(n)*c(f)
    print(mse_h_cn)
    print(mse_h_cn_cf)


    # Plot
    # fig1 : RIR of h(n)、 hf(n) 、hc(n)
    plt.plot(h_n, 'r')
    plt.plot(h_fn, 'b')
    plt.plot(h_cn, 'g')
    plt.legend(['h(n)', 'hf(n)','hc(n)'])
    plt.xlabel('Sample')
    plt.ylabel('magnitude')
    plt.title('RIR')
    plt.show()

    # fig2 : EDR of h(n) and hc(n) at mixing time
    plt.plot(F1/1000, B_EDRdb_hn[:, index], "r")
    plt.plot(F1/1000, B_EDRdb_hcn[:, index], "b")
    plt.legend(['h(n)', 'hc(n)'])
    plt.xlabel('frequency/kHz')
    plt.ylabel('magnitude/dB')
    plt.title('EDR at mixing time')
    plt.show()

    # fig3 : Fir Filter Magnitude Response
    freq, response = signal.freqz(fir_coeff)
    plt.plot(freq * fs / (2 * np.pi*1000), np.abs(response))
    plt.xlabel('frequency/Hz')
    plt.ylabel('magnitude')
    plt.title('Filter Magnitude Response')
    plt.show()

    # fig4 : EDR of h(n)、 hc(n) 、hcnf(n) at mixing time
    plt.plot(F/1000, Y1, "r")
    plt.plot(F/1000, Y2, "b")
    plt.plot(F/1000, Y3, "g")
    plt.legend(['h(n)', 'hc(n)', 'hcnf(n)'])
    plt.xlabel('frequency/KHz')
    plt.ylabel('magnitude/dB')
    plt.title('EDR at mixing time')
    plt.show()

