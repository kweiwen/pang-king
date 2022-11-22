from ism import *
import matplotlib.pyplot as plt
from material import *
from multibands import *

def example():
    room_size = [8, 3, 2.8]
    microphone_pos = [7, 1.5, 1.55]
    source_pos = [1.2, 1.5, 1.2]

    instance = ISM()
    instance.define_system(48000, 343, 2048 - 256, 0.008)
    instance.create_multi_bands()
    instance.create_room(room_size)

    # carpet_tufted_9.5mm
    # hard_surface
    # reverb_chamber

    m = dict()
    m["floor"] =    Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
    m["ceiling"] =  Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
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

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle('Original vs Compensated')

    # Original tap, time domain
    plt.subplot(2, 2, 1)
    plt.plot(instance.taps)
    # plt.ylim(-1, 1)

    # Original tap, frequency domain
    w, h = signal.freqz(instance.tap, worN=128)
    amplitude = 20 * np.log10(abs(h))
    angle = np.angle(h)
    ax1 = plt.subplot(2, 2, 2)
    ax2 = ax1.twinx()
    ax1.plot(w / max(w), amplitude, 'g')
    ax1.set_ylim(-72, 24)
    ax1.grid()
    ax2.plot(w / max(w), angle, 'y--')
    ax2.set_ylim(-np.pi, np.pi)

    # Compensated tap, time domain
    plt.subplot(2, 2, 3)
    plt.plot(instance.remove_direct_sound() * instance.compute_engery_scale())
    # plt.ylim(-1, 1)

    # Compensated tap, frequency domain
    w, h = signal.freqz(instance.remove_direct_sound() * instance.compute_engery_scale(), worN=128)
    amplitude = 20 * np.log10(abs(h))
    angle = np.angle(h)
    ax1 = plt.subplot(2, 2, 4)
    ax2 = ax1.twinx()
    ax1.plot(w / max(w), amplitude, 'g')
    ax1.set_ylim(-72, 24)
    ax1.grid()
    ax2.plot(w / max(w), angle, 'y--')
    ax2.set_ylim(-np.pi, np.pi)

    plt.show()

    np.savetxt('impedance_1.dat', [instance.remove_direct_sound() * instance.compute_engery_scale()], delimiter=',\n', fmt='%.24f')



if __name__ == "__main__":
    example()


