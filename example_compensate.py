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
    return output_data

def example(room_size, source_pos, microphone_pos, sample_rate, velocity, sample_length, block_size):
    # room_size =         [5, 5, 2.2]
    # source_pos =        [1, 1.5, 0.2]
    # microphone_pos =    [4, 1.25, 1.2]
    #
    # sample_rate     = 48000
    # velocity        = 340
    # sample_length   = 2048
    # block_size      = 240

    instance = ISM()
    instance.define_system(sample_rate, velocity, sample_length, 0.009)
    instance.create_multi_bands()
    instance.create_room(room_size)

    # carpet_tufted_9.5mm
    # hard_surface
    # reverb_chamber

    m = dict()
    m["floor"] =    Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
    m["ceiling"] =  Material(energy_absorption="carpet_tufted_9.5mm", scattering="rect_prism_boxes")
    m["east"] =     Material(energy_absorption="audience_orchestra_choir", scattering="rect_prism_boxes")
    m["west"] =     Material(energy_absorption="audience_orchestra_choir", scattering="rect_prism_boxes")
    m["north"] =    Material(energy_absorption="audience_orchestra_choir", scattering="rect_prism_boxes")
    m["south"] =    Material(energy_absorption="audience_orchestra_choir", scattering="rect_prism_boxes")

    x1 = instance.resample(m['west'].energy_absorption['coeffs'], m['west'].energy_absorption['center_freqs'])
    x2 = instance.resample(m['east'].energy_absorption['coeffs'], m['east'].energy_absorption['center_freqs'])
    y1 = instance.resample(m['south'].energy_absorption['coeffs'], m['south'].energy_absorption['center_freqs'])
    y2 = instance.resample(m['north'].energy_absorption['coeffs'], m['north'].energy_absorption['center_freqs'])
    z1 = instance.resample(m['floor'].energy_absorption['coeffs'], m['floor'].energy_absorption['center_freqs'])
    z2 = instance.resample(m['ceiling'].energy_absorption['coeffs'], m['ceiling'].energy_absorption['center_freqs'])

    instance.create_material_by_coefficient(x1, x2, y1, y2, z1, z2, True)
    instance.add_receiver(microphone_pos)
    instance.add_transmitter(source_pos)
    instance.remove_direct_sound()
    instance.compute_ism()
    instance.compute_rir()

    # maximum_length(M + N - 1) to store rir coefficient for circular convolution
    maximum_length = len(instance.tap[:sample_length + 1 - block_size])
    padding_length = block_size - 1
    padding = np.zeros(padding_length)

    refactor = np.sum(instance.taps[:, :], axis=1)

    temp = np.concatenate((refactor[:maximum_length], padding)) * instance.compute_engery_scale()
    A = []

    H = np.fft.fft(temp)

    for index, data in enumerate(H):
        A.append(data.real)
        A.append(data.imag)

    result = convert_all(A)
    print(result)
    print(len(result))

    return temp

    #
    # np.savetxt('impedance_1.dat', [temp], delimiter=',\n', fmt='%.24f')
    #
    # plt.plot(temp)
    # plt.show()
    #
    # w, h = signal.freqz(temp)
    # amplitude = 20 * np.log10(abs(h))
    # plt.plot(amplitude)
    # plt.show()

if __name__ == "__main__":
    L = example([15, 15, 2.2], [1.0, 2, 1.6], [4.5, 2.7, 1.2], 48000, 340, 2048, 256)
    R = example([15, 15, 2.2], [1.0, 3, 1.6], [4.5, 2.3, 1.2], 48000, 340, 2048, 256)

    plt.plot(R)
    plt.show()

    w, h = signal.freqz(R)
    amplitude = 20 * np.log10(abs(h))
    plt.plot(amplitude)
    plt.show()


# EROOMCONVOLUTION_PARAM_ID_EARLY_REFLECTION_TIME,    // 0x1667E304
# int, 1 - 1024
# EROOMCONVOLUTION_PARAM_ID_LATE_REFLECTION_TIME,     // 0x1667E305
# int, 1 - 1024
# EROOMCONVOLUTION_PARAM_ID_SPREAD,                   // 0x1667E306
# float, 0.0 - 1.0
# EROOMCONVOLUTION_PARAM_ID_COLOR,                    // 0x1667E307
# float, 0.0 - 1.0
# EROOMCONVOLUTION_PARAM_ID_DAMP,                     // 0x1667E308
# float, 0.0 - 1.0
# EROOMCONVOLUTION_PARAM_ID_DECAY,                    // 0x1667E309
# float, 0.0 - 1.0
# EROOMCONVOLUTION_PARAM_ID_DIRECT,                   // 0x1667E30A
# float, 0.0 - 1.0
# EROOMCONVOLUTION_PARAM_ID_EARLY_REFLECTION,         // 0x1667E30B
# float, 0.0 - 1.0
# EROOMCONVOLUTION_PARAM_ID_LATE_REFLECTION,          // 0x1667E30C
# float, 0.0 - 1.0
