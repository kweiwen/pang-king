from ism import *
import matplotlib.pyplot as plt
from material import *
from multibands import *

def example(is_save):
    room_size = [5, 3, 2.2]
    microphone_pos = [4, 1.25, 1.2]
    source_pos = [1, 1.5, 0.2]

    instance = ISM()
    instance.define_system(48000, 343, 4096 - 512, 0.004)
    instance.create_multi_bands()
    instance.create_room(room_size)

    # carpet_tufted_9.5mm
    # hard_surface
    # reverb_chamber

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

    instance.create_material_by_coefficient(x1, x2, y1, y2, z1, z2, False)
    instance.add_receiver(microphone_pos)
    instance.add_transmitter(source_pos)
    instance.compute_ism()
    instance.compute_rir()

    # compensate the energy
    w, h = signal.freqz(instance.tap)
    scale = len(h) / np.sum(np.abs(h))
    h = h * scale

    plt.subplot(4, 1, 1)
    plt.plot(instance.tap)
    plt.subplot(4, 1, 2)
    plt.plot(instance.remove_direct_sound())
    plt.subplot(4, 1, 3)
    plt.plot(w, 20 * np.log10(abs(h)))
    plt.subplot(4, 1, 4)
    plt.plot(w, np.unwrap(np.angle(h)))
    plt.show()

    instance.render_room(space=2, alpha=0.2, x=0, y=0, z=0, dx=room_size[0], dy=room_size[1], dz=room_size[2], source=source_pos, mic=microphone_pos)

    if is_save:
        np.savetxt('impedance_1.dat', [instance.tap*scale], delimiter=',\n', fmt='%.24f')


def decodePlotRIR():
    f1 = open("impedance_1.dat", "r")
    arr1 = f1.readlines()

    for index, data in enumerate(arr1):
        data = float(data.replace(",", ""))
        arr1[index] = data

    f2 = open("impedance_2.dat", "r")
    arr2 = f2.readlines()

    for index, data in enumerate(arr2):
        data = float(data.replace(",", ""))
        arr2[index] = data


    plt.plot(arr1)
    plt.show()
    plt.plot(arr2)
    plt.show()


if __name__ == "__main__":
    example(True)