from ism import *
import matplotlib.pyplot as plt
from material import *
from multibands import *

import numpy as np
import struct

def float_to_byte_array_little(input_data):
    return list(struct.pack('<f', input_data))

def byte_to_string(input_data):
    output_data = ""
    for data in input_data:
        output_data += ('{:02x}'.format(data))
    return output_data

def convert_all(input_data):
    output_data = ""
    for data in input_data:
        output_data += byte_to_string(float_to_byte_array_little(data))
    # print(len(output_data))
    # print(output_data)
    return output_data

def example():
    room_size       = [3.2, 6, 2.7]
    microphone_pos  = [1.1, 3, 1.2]
    source_pos      = [2, 3, 2]
    sample_rate     = 48000
    velocity        = 340
    sample_length   = 2048
    block_size      = 240

    instance = ISM()
    instance.define_system(sample_rate, velocity, sample_length, 0.0025)
    instance.create_multi_bands()
    instance.create_room(room_size)

    # carpet_tufted_9.5mm
    # hard_surface
    # reverb_chamber

    m = dict()
    m["floor"] =    Material(energy_absorption="audience_orchestra_choir", scattering="rect_prism_boxes")
    m["ceiling"] =  Material(energy_absorption="audience_orchestra_choir", scattering="rect_prism_boxes")
    m["east"] =     Material(energy_absorption="hard_surface", scattering="rect_prism_boxes")
    m["west"] =     Material(energy_absorption="hard_surface", scattering="rect_prism_boxes")
    m["north"] =    Material(energy_absorption="hard_surface", scattering="rect_prism_boxes")
    m["south"] =    Material(energy_absorption="hard_surface", scattering="rect_prism_boxes")

    x1 = instance.resample(m['west'].energy_absorption['coeffs'], m['west'].energy_absorption['center_freqs'])
    x2 = instance.resample(m['east'].energy_absorption['coeffs'], m['east'].energy_absorption['center_freqs'])
    y1 = instance.resample(m['south'].energy_absorption['coeffs'], m['south'].energy_absorption['center_freqs'])
    y2 = instance.resample(m['north'].energy_absorption['coeffs'], m['north'].energy_absorption['center_freqs'])
    z1 = instance.resample(m['floor'].energy_absorption['coeffs'], m['floor'].energy_absorption['center_freqs'])
    z2 = instance.resample(m['ceiling'].energy_absorption['coeffs'], m['ceiling'].energy_absorption['center_freqs'])

    instance.create_material_by_coefficient(x1, x2, y1, y2, z1, z2, False)
    instance.add_receiver(microphone_pos)
    instance.add_transmitter(source_pos)
    instance.remove_direct_sound()
    instance.compute_ism()
    instance.compute_rir()


    # maximum_length(M + N - 1) to store rir coefficient for circular convolution
    maximum_length = len(instance.tap[:sample_length + 1 - block_size])
    padding_length = block_size - 1
    padding = np.zeros(padding_length)
    temp = np.concatenate((instance.tap[:maximum_length], padding)) * instance.compute_engery_scale()

    A = np.zeros(2048 * 2)

    H = np.fft.fft(temp)

    for index, data in enumerate(H):
        A[index*2    ] = data.real
        A[index*2 + 1] = data.imag

    result = convert_all(A)
    print(result)
    print(len(result))

    np.savetxt('impedance_1.dat', [temp], delimiter=',\n', fmt='%.24f')

def render_plot(instance):
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
    plt.plot(instance.tap * instance.compute_engery_scale())
    # plt.ylim(-1, 1)

    # Compensated tap, frequency domain
    w, h = signal.freqz(instance.tap * instance.compute_engery_scale(), worN=128)
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


