from ism import *
import matplotlib.pyplot as plt
from material import *
from multibands import *

def example():
    room_size = [5, 3, 2.2]
    microphone_pos = [4, 1.5, 0.2]
    source_pos = [1, 1.5, 0.2]

    instance = ISM()
    instance.defineSystem(48000, 343, 2048-480, 0.001)
    instance.createMultiBands()
    instance.createRoom(room_size)

    # carpet_tufted_9.5mm
    # hard_surface

    m = dict()
    m["ceiling"] =  Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
    m["floor"] =    Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
    m["east"] =     Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
    m["west"] =     Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
    m["north"] =    Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
    m["south"] =    Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")

    x1 = instance.resample(m['west'].energy_absorption['coeffs'], m['west'].energy_absorption['center_freqs'])
    x2 = instance.resample(m['east'].energy_absorption['coeffs'], m['east'].energy_absorption['center_freqs'])
    y1 = instance.resample(m['south'].energy_absorption['coeffs'], m['south'].energy_absorption['center_freqs'])
    y2 = instance.resample(m['north'].energy_absorption['coeffs'], m['north'].energy_absorption['center_freqs'])
    z1 = instance.resample(m['floor'].energy_absorption['coeffs'], m['floor'].energy_absorption['center_freqs'])
    z2 = instance.resample(m['ceiling'].energy_absorption['coeffs'], m['ceiling'].energy_absorption['center_freqs'])

    instance.createMaterialByCoefficient(x1, x2, y1, y2, z1, z2, False)
    instance.addMicrophone(microphone_pos)
    instance.addSource(source_pos)
    instance.computeISM()

    taps = instance.computeRIR()

    plt.subplot(2, 1, 1)
    plt.plot(taps[:,:])
    plt.subplot(2, 1, 2)
    plt.plot(np.sum(taps[:, :], axis=1))
    plt.show()

    instance.render_room(space=2, alpha=0.2, x=0, y=0, z=0, dx=room_size[0], dy=room_size[1], dz=room_size[2], source=source_pos, mic=microphone_pos)

    # np.savetxt('impedance_1.dat', [np.sum(taps[:, :], axis=1)], delimiter=',\n', fmt='%.24f')

if __name__ == "__main__":
    example()