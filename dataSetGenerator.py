import numpy
import pyroomacoustics as pra
import numpy as np
import itertools
import matplotlib.pyplot as plt

def total_images_size(n: int):
    return 1 + int(2 * n * (2 * n * n + 3 * n + 4) / 3)

def get_max_dist(images: numpy.ndarray, rx: numpy.ndarray, c: float, fs: float):
    dist = np.sqrt(np.sum((images - rx[:, None]) ** 2, axis=0))
    time = dist / c
    t_max = time.max()
    N = int(np.ceil(t_max * fs))
    return N

def get(nSample: int):
    power = np.ceil(np.log2(nSample))
    return np.power(2, power)

def generate(fs: int, temp: float, d: numpy.ndarray, tx: numpy.ndarray, rx: numpy.ndarray, order: int, materials: numpy.ndarray):

    ceiling = { "description": "", "coeffs": materials[0], "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    floor   = { "description": "", "coeffs": materials[1], "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    east    = { "description": "", "coeffs": materials[2], "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    west    = { "description": "", "coeffs": materials[3], "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    north   = { "description": "", "coeffs": materials[4], "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    south   = { "description": "", "coeffs": materials[5], "center_freqs": [125, 250, 500, 1000, 2000, 4000] }

    m = pra.make_materials(ceiling  =(ceiling,  None),
                           floor    =(floor,    None),
                           east     =(east,     None),
                           west     =(west,     None),
                           north    =(north,    None),
                           south    =(south,    None))

    room = pra.ShoeBox(d, fs=fs, materials=m, max_order=order, temperature=temp, air_absorption=False, ray_tracing=False)
    room.add_source(position=tx)
    room.add_microphone(loc=rx)
    room.compute_rir()

    image_size = total_images_size(order)

    images = room.sources[0].images
    max_dist = get_max_dist(images, rx, room.c, fs) + 81

    input_vector = [fs, d, tx, rx, order, image_size, max_dist]
    transform = np.fft.fft(room.rir[0][0])

    print(max_dist)

    print(len(transform.real))
    print(len(transform.imag))

    return input_vector, transform


# input_vector, transform = generate(48000, 25, np.array([10, 10, 10]), np.array([1, 5, 1]), np.array([4, 2, 3.7]), 3, np.array([0.2, 0.1, 0.21, 0.31, 0.21, 0.11]))
# data = {'input_vector':input_vector, 'output_vector':[transform.real, transform.imag]}
# np.save("d1.npy", data)

# data = np.load('d1.npy', allow_pickle=True)

# print(data)


# import itertools
# a = [[1,2,3],[4,5,6],[7,8,9,10]]
# list(itertools.product(*a))


size = [np.linspace(1, 10, 10)] * 3
tx = [np.linspace(0, 10, 11)] * 3
rx = [np.linspace(0, 10, 11)] * 3
temperature = [np.linspace(0, 40, 41)]
material = [np.linspace(0.01, 0.99, 99)] * 6
walls = material * 6
order = [np.linspace(1, 17, 17)]
fs = [[8000, 16000, 32000, 44100, 48000, 96000, 192000]]

args = size + tx + rx + walls + order + fs + temperature
combinations = itertools.product(*args)

for index, combination in enumerate(combinations):
    size = np.array(combination[0:3])
    tx = np.array(combination[3:6])
    rx = np.array(combination[6:9])

    m0 = np.array(combination[9:15])
    m1 = np.array(combination[15:21])
    m2 = np.array(combination[21:27])
    m3 = np.array(combination[27:33])
    m4 = np.array(combination[33:39])
    m5 = np.array(combination[39:45])
    m = np.array([m0,m1,m2,m3,m4,m5])

    order = combination[45]
    fs = combination[46]
    temperature = combination[47]
    print(index, size, tx, rx, order, fs, temperature)

    input_vector, transform = generate(fs, temperature, size, tx, rx, int(order), m)

    if index == 12:
        break


# class DSmanager:
#     def __init__(self):
#         print(123123)
#
#
# a = DSmanager()
