from ism import *
import matplotlib.pyplot as plt
from material import *

m = dict()
m["ceiling"] = Material(energy_absorption="reverb_chamber", scattering="rect_prism_boxes")
m["floor"]   = Material(energy_absorption="concrete_floor", scattering="rect_prism_boxes")
m["east"]    = Material(energy_absorption="concrete_floor", scattering="rect_prism_boxes")
m["west"]    = Material(energy_absorption="plywood_thin", scattering="rect_prism_boxes")
m["north"]   = Material(energy_absorption="hard_surface", scattering="rect_prism_boxes")
m["south"]   = Material(energy_absorption="hard_surface", scattering="rect_prism_boxes")

instance = ISM()
instance.defineSystem(48000, 343, 8192)
instance.createMultiBands()
instance.createRoom(3.2, 4, 2.7)

x1 = instance.resample(m['west'].energy_absorption['coeffs'], m['west'].energy_absorption['center_freqs'])
x2 = instance.resample(m['east'].energy_absorption['coeffs'], m['east'].energy_absorption['center_freqs'])
y1 = instance.resample(m['south'].energy_absorption['coeffs'], m['south'].energy_absorption['center_freqs'])
y2 = instance.resample(m['north'].energy_absorption['coeffs'], m['north'].energy_absorption['center_freqs'])
z1 = instance.resample(m['floor'].energy_absorption['coeffs'], m['floor'].energy_absorption['center_freqs'])
z2 = instance.resample(m['ceiling'].energy_absorption['coeffs'], m['ceiling'].energy_absorption['center_freqs'])

x1 = np.sqrt(1 - x1) * -1
x2 = np.sqrt(1 - x2) * -1
y1 = np.sqrt(1 - y1) * -1
y2 = np.sqrt(1 - y2) * -1
z1 = np.sqrt(1 - z1) * -1
z2 = np.sqrt(1 - z2) * -1

instance.createMaterialByCoefficient(x1, x2, y1, y2, z1, z2)
# instance.createMaterialByCoefficient(-0.92, -0.92, -0.92, -0.92, -0.92, -0.92)
instance.addMicrophone(2.2, 1, 1.2)
instance.addSource(2, 3, 2)
instance.computeISM()

taps = instance.computeRIR()

plt.plot(taps)
plt.show()

# import pyroomacoustics as pra
# room_dim=[10,10,10]
# room = pra.ShoeBox(room_dim, fs=16000, materials=m, max_order=17, ray_tracing=True)
#
# room.add_source([2,5,5])
# room.add_microphone([8,5,5])
# room.compute_rir()
# room.plot_rir()
# plt.show()

# # Sound velocity (m/s)
# c = 340
# # Sample frequency (samples/s)
# fs = 16000
# # Receiver position [ x y z ] (m)
# r = [2, 1.5, 2]
# # Source position [ x y z ] (m)
# s = [2, 3.5, 2]
# # Room dimensions [ x y z ] (m)
# L = [5, 4, 6]
# # Absorption
# beta = 0.4
# # Number of samples
# nsample = 4096
# # Type of microphone
# mtype = "hypercardioid"
# # −1 equals maximum reflection order!
# order = -1
# # Room dimension
# dim = 3
# # Microphone orientation [azimuth elevation] in radians
# orientation = [np.pi / 2]
# # Enable high-pass filter
# hp_filter = False
#
# taps = RG.rir_generator(c, fs, r, s, L, beta=beta, nsample=nsample, mtype=mtype, order=order, dim=dim, orientation=orientation, hp_filter=hp_filter)
# w, h = signal.freqz(taps[0])
# amplitude = 20 * np.log10(abs(h))
# angle = np.unwrap(np.angle(h))
#
# plt.plot(taps[0])
# plt.show()
#
#
# fig, ax1 = plt.subplots()
# ax2 = ax1.twinx()
#
# plt.title('frequency response of RIR')
# plt.xlabel(r'normalized frequency (x$\pi$rad/sample)')
#
# ax1.plot(w/max(w), amplitude, 'g')
# ax1.set_ylabel('amplitude (dB)', color='g')
# ax1.set_ylim(-120, 6)
# ax1.grid()
#
# ax2.plot(w/max(w), angle, 'b--')
# ax2.set_ylabel('phase (radians)', color='b')
#
# plt.xscale("log")
# plt.show()