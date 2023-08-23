import numpy
import pyroomacoustics as pra
import numpy as np

DEBUG = True

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

    # input_vector = [temp, fs, d.flatten(), tx.flatten(), rx.flatten(), order, image_size, materials.flatten(), max_dist]
    input_vector = np.hstack([d.flatten(), tx.flatten(), rx.flatten(), order])
    materials = materials.flatten()
    transform = room.rir[0][0]

    return input_vector, materials, transform

results = []  # Create a list to store all results
save_dir = "D:/zengkuiwen/Desktop/dataset"
material_values = np.linspace(0.01, 0.99, 99)
order_size = 10

for index in range(4096):
    # m0 = [np.random.choice(material_values) for _ in range(6)]
    # m1 = [np.random.choice(material_values) for _ in range(6)]
    # m2 = [np.random.choice(material_values) for _ in range(6)]
    # m3 = [np.random.choice(material_values) for _ in range(6)]
    # m4 = [np.random.choice(material_values) for _ in range(6)]
    # m5 = [np.random.choice(material_values) for _ in range(6)]
    m = np.random.rand(6, 6)

    size = np.random.rand(3) * 10
    tx = np.random.rand(3) * size
    rx = np.random.rand(3) * size
    print(index)

    for order in range(order_size):
        input_vector, materials, transform = generate(48000, 25, size, tx, rx, int(order), m)
        transform_list = transform.tolist()

        result = {
            "index": index,
            "input_vector": input_vector,
            "materials_vector": materials,
            "transform_list": transform_list
        }
        results.append(result)

np.save('results.npy', results)
