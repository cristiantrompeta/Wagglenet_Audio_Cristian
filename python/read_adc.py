import sys
import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = "COM5"
BAUD = 921600
FS = 4000        # sample rate in Hz, must match the STM32 timer
SECONDS = 10      # how long to record

# Record raw bytes (2 bytes per sample, plus 2 spare for alignment)
ser = serial.Serial(PORT, BAUD, timeout=SECONDS + 2)
ser.reset_input_buffer()
raw = ser.read(FS * SECONDS * 2 + 2)
ser.close()

if len(raw) < 100:
    sys.exit("No data received. Is the board flashed and running?")

# Line up the bytes: real 12-bit samples are never above 4095
for offset in (0, 1):
    n = (len(raw) - offset) // 2
    samples = np.frombuffer(raw[offset:offset + n * 2], dtype="<u2")
    if samples.max() <= 4095:
        break

# Convert to volts and save
volts = samples * 3.3 / 4095
np.savetxt("samples.csv", volts, delimiter=",")

# Plot the signal and its frequency content
t = np.arange(len(volts)) / FS
freqs = np.fft.rfftfreq(len(volts), 1 / FS)
spectrum = np.abs(np.fft.rfft(volts - volts.mean()))

fig, (ax1, ax2) = plt.subplots(2, 1)
ax1.plot(t, volts)
ax1.set_xlabel("Time (s)")
ax1.set_ylabel("Voltage (V)")
ax2.plot(freqs, spectrum)
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Magnitude")
plt.tight_layout()
plt.show()