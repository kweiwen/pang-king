from ism import *
import matplotlib.pyplot as plt
from material import *
from multibands import *

def example():
    room_size = [8, 3, 2.8]
    microphone_pos = [7, 1.5, 1.55]
    source_pos = [1.2, 1.5, 1.2]

    instance = ISM()
    instance.defineSystem(48000, 343, 4096-512, 0.008)
    instance.createMultiBands()
    instance.createRoom(room_size)

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

    instance.createMaterialByCoefficient(x1, x2, y1, y2, z1, z2, False)
    instance.addMicrophone(microphone_pos)
    instance.addSource(source_pos)
    instance.computeISM()
    instance.computeRIR()

    fig, axs = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle('Original vs Compensated')

    # Original tap, time domain
    plt.subplot(2, 2, 1)
    plt.plot(instance.tap)
    plt.ylim(-1, 1)

    # Original tap, frequency domain
    w, h = signal.freqz(instance.tap, worN=64)
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
    plt.plot(instance.removeDirectSound() * instance.computeEngeryScale())
    plt.ylim(-1, 1)

    # Compensated tap, frequency domain
    w, h = signal.freqz(instance.removeDirectSound() * instance.computeEngeryScale(), worN=64)
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


if __name__ == "__main__":
    example()


