import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, fftfreq, ifft

# Domain Setup
# number of points
n = 1000

# Distance
Lx = 100

# Angular Frequency
omg = 2.0*np.pi/Lx

# Individual Signals
x = np.linspace(0, Lx, n)
y1 = 1.0*np.cos(10*omg*x)
y2 = 2.0*np.sin(10.0*omg*x)
y3 = 0.5*np.sin(20.0*omg*x)

# Full Signal
y = y1 + y2 + y3

# Preparatory steps
# Create all the necessary frequencies
freqs = fftfreq(n)

# Ignore half of the values (complex conjugates)
mask = freqs > 0

# FFT and power spectra calculations
# fft values
fft_vals = fft(y)

# true theoretical fft
fft_theo = 2.0*np.abs(fft_vals/n)

plt.figure(1)
plt.title('original signal)')
plt.plot(x, y, color='xkcd:salmon', label='original')
plt.legend()

plt.figure(2)
plt.title('Raw FFT values)')
plt.plot(freqs[mask], fft_theo[mask], label='True fft values')
plt.legend()

plt.show()
