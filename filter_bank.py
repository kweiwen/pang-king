import matplotlib.pyplot as plt
from multibands import *

object = MultiBands()

# filter bank
for instance in object.filters.T:
    w, h = signal.freqz(instance)
    amplitude = np.abs(h)

    plt.plot(amplitude)

plt.xscale("log")
plt.show()

# sum up taps in time domain
arr = sum(object.filters.T)
plt.plot(arr)
plt.show()

# transform in frequency domain
w, h = signal.freqz(arr)
amplitude = np.abs(h)
plt.plot(amplitude)
plt.show()