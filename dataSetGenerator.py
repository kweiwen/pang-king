import numpy
import pyroomacoustics as pra
import numpy as np
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

    ceiling = { "description": "", "coeffs": materials, "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    floor   = { "description": "", "coeffs": materials, "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    east    = { "description": "", "coeffs": materials, "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    west    = { "description": "", "coeffs": materials, "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    north   = { "description": "", "coeffs": materials, "center_freqs": [125, 250, 500, 1000, 2000, 4000] }
    south   = { "description": "", "coeffs": materials, "center_freqs": [125, 250, 500, 1000, 2000, 4000] }

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

    print("===Input Vector DATA==")
    print(fs)
    print(d)
    print(tx)
    print(rx)
    print(order)
    print(total_images_size(order))
    images = room.sources[0].images
    print(get_max_dist(images, rx, room.c, fs))

    plt.plot(room.rir[0][0])
    plt.show()

generate(48000, 25, np.array([10, 10, 10]), np.array([1, 5, 1]), np.array([4, 2, 3.7]), 3, np.array([0.2, 0.1, 0.21, 0.31, 0.21, 0.11]))