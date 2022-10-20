from ism import *
import matplotlib.pyplot as plt
from material import *
from multibands import *

m = dict()
m["ceiling"] = Material(energy_absorption="reverb_chamber", scattering="rect_prism_boxes")
m["floor"]   = Material(energy_absorption="concrete_floor", scattering="rect_prism_boxes")
m["east"]    = Material(energy_absorption="concrete_floor", scattering="rect_prism_boxes")
m["west"]    = Material(energy_absorption="plywood_thin", scattering="rect_prism_boxes")
m["north"]   = Material(energy_absorption="hard_surface", scattering="rect_prism_boxes")
m["south"]   = Material(energy_absorption="hard_surface", scattering="rect_prism_boxes")

instance = ISM()
instance.defineSystem(48000, 343, 2048, 0.004)
instance.createMultiBands()
instance.createRoom(3.2, 4, 2.7)

x1 = instance.resample(m['west'].energy_absorption['coeffs'], m['west'].energy_absorption['center_freqs'])
x2 = instance.resample(m['east'].energy_absorption['coeffs'], m['east'].energy_absorption['center_freqs'])
y1 = instance.resample(m['south'].energy_absorption['coeffs'], m['south'].energy_absorption['center_freqs'])
y2 = instance.resample(m['north'].energy_absorption['coeffs'], m['north'].energy_absorption['center_freqs'])
z1 = instance.resample(m['floor'].energy_absorption['coeffs'], m['floor'].energy_absorption['center_freqs'])
z2 = instance.resample(m['ceiling'].energy_absorption['coeffs'], m['ceiling'].energy_absorption['center_freqs'])

instance.createMaterialByCoefficient(x1, x2, y1, y2, z1, z2, True)
instance.addMicrophone(2.2, 1, 1.2)
instance.addSource(2, 3, 2)
instance.computeISM()

taps = instance.computeRIR()

plt.plot(taps[:,0:4])
plt.show()

plt.plot(np.sum(taps[:,0:4], axis=1))
plt.show()

np.savetxt('impedance_1.dat', [np.sum(taps[:,:], axis=1)], delimiter=',\n', fmt='%.24f')
