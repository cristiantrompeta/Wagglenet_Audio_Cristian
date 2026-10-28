import sys
import serial
import numpy as np
import matplotlib.pyplot as plt

PORT = "COM5"
BAUD = 921600
FS = 4000            # sample rate per channel in Hz, must match the STM32 timer
SECONDS = 10         # how long to record
CHANNELS = 3
NAMES = ["A0 (ADC1)", "A1 (ADC2)", "A4 (ADC3)"]

HALF_SAMPLES = 1536  # samples per half buffer, must match ADC_BUF_LEN / 2 on the STM32
SYNC = b"\xff\xff"   # marker the board sends before each half buffer
FRAME_BYTES = len(SYNC) + HALF_SAMPLES * 2

# How many frames make up SECONDS of data, plus one spare
# (the first partial frame gets thrown away)
sets_per_frame = HALF_SAMPLES // CHANNELS
n_frames = int(np.ceil(FS * SECONDS / sets_per_frame)) + 1

# Record raw bytes
ser = serial.Serial(PORT, BAUD, timeout=SECONDS + 5)
ser.reset_input_buffer()
raw = ser.read(n_frames * FRAME_BYTES)
ser.close()

if len(raw) < 100:
    sys.exit("No data received. Is the board flashed and running?")

# Find the first marker. Search from index 1 in case the read
# started halfway through a marker.
start = raw.find(SYNC, 1)
if start == -1:
    sys.exit("No sync marker found. Is the new firmware flashed?")

# Cut the stream into frames, checking each one starts with the marker
frames = []
pos = start
while pos + FRAME_BYTES <= len(raw):
    if raw[pos:pos + len(SYNC)] != SYNC:
        print(f"Warning: lost sync at byte {pos}, keeping data up to here")
        break
    frames.append(raw[pos + len(SYNC):pos + FRAME_BYTES])
    pos += FRAME_BYTES

# Every frame starts on A0, so the samples go A0, A1, A4, A0, A1, A4, ...
data = np.frombuffer(b"".join(frames), dtype="<u2")
volts = [data[i::CHANNELS] * 3.3 / 4095 for i in range(CHANNELS)]

# Save as three columns
np.savetxt("samples.csv", np.column_stack(volts), delimiter=",",
           header="A0,A1,A4", comments="")

# One column of plots per channel: signal, FFT, spectrogram
t = np.arange(len(volts[0])) / FS
freqs = np.fft.rfftfreq(len(volts[0]), 1 / FS)

fig, axes = plt.subplots(3, CHANNELS, figsize=(14, 9))
for i, v in enumerate(volts):
    centered = v - v.mean()
    ax_sig, ax_fft, ax_spec = axes[:, i]

    ax_sig.plot(t, v)
    ax_sig.set_title(NAMES[i])
    ax_sig.set_xlabel("Time (s)")
    ax_sig.set_ylabel("Voltage (V)")

    ax_fft.plot(freqs, np.abs(np.fft.rfft(centered)))
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude")

    # Spectrogram: frequency content over time (1024-sample windows, 50% overlap)
    _, _, _, im = ax_spec.specgram(centered, NFFT=1024, Fs=FS,
                                   noverlap=512, cmap="magma")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax_spec, label="Power (dB)")

plt.tight_layout()
plt.show()