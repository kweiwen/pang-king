import pyroomacoustics as pra
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.io.wavfile import write, read

m = pra.make_materials(
    ceiling="reverb_chamber",
    floor="concrete_floor",
    east="concrete_floor",
    west="plywood_thin",
    north="hard_surface",
    south="hard_surface",
)

# Create the room
room = pra.ShoeBox(
    [5, 3, 2.2], fs=48000, materials=m, max_order=3, air_absorption=False, ray_tracing=False
)

# place the source in the room
room.add_source(position=[1, 1.5, 0.2])

# place the microphone in the room
room.add_microphone(loc=[4, 1.5, 0.2])

# plot room geometry
room.compute_rir()

room.plot()
plt.show()

# taps = np.append(room.rir[0][0], np.zeros(2048 - len(room.rir[0][0])))
# np.savetxt('impedance_1.dat', taps, delimiter=',\n', fmt='%.24f')
plt.plot(room.rir[0][0])
plt.show()
#
#
# taps = room.rir[0][0]
# # render
# rate, file = read("Protection.wav")
# fixed_data0 = file[:, 0]
# fixed_data1 = file[:, 1]
#
# float_data0 = fixed_data0.astype(np.float32, order='C') / 32767.0
# float_data1 = fixed_data1.astype(np.float32, order='C') / 32767.0
#
# outputData0 = signal.convolve(float_data0, taps, method='fft')
# outputData1 = signal.convolve(float_data1, taps, method='fft')
# temp = np.array([outputData0,outputData1])
#
# raw = np.array([outputData0 * 32767.0 / 8, outputData1 * 32767.0 / 8]).astype(np.int16).T
# write("fir_linear_conv_v2.wav", rate, raw)
