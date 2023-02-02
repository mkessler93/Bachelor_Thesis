import numpy as np
from scipy import signal

def MA_filter(ecg, width=8):
    rec_win = 1/width * np.ones(width)
    avg_signal = signal.lfilter(rec_win, 1, ecg)
    return avg_signal

def high_low_pass_filter(ecg_signal):
    # as & bs
    b_lowpass = np.zeros([13])
    b_lowpass[0] = 1. / 32
    b_lowpass[6] = -2. / 32
    b_lowpass[12] = 1. / 32

    a_lowpass = np.zeros([13])
    a_lowpass[0] = 1.0
    a_lowpass[1] = -2.0
    a_lowpass[2] = 1.0

    a_highpass = np.zeros([33])
    a_highpass[0] = 1
    a_highpass[1] = -1

    b_highpass = np.zeros([33])
    b_highpass[0] = -1 / 32
    b_highpass[16] = 1
    b_highpass[17] = -1
    b_highpass[32] = 1 / 32

    # Filter signal
    ecg_signal = signal.lfilter(b_lowpass, a_lowpass, ecg_signal)
    ecg_signal = signal.lfilter(b_highpass, a_highpass, ecg_signal)


    return ecg_signal

def signal_power(signal):
    power = 1 / len(signal) * np.sum(signal**2)
    return power

def snr_1 (noise, signal):
    #power Signal/power Noise
    power_noise = signal_power(noise)
    power_signal = signal_power(signal)
    if (power_noise == 0):
        snr = 1
        return snr
    snr = power_signal / power_noise

    return snr

def snr_2(noise, signal):
    #(standard deviation signal)**2/(standard deviation noise)**2
    noise_std = np.std(noise)
    signal_std = np.std(signal)
    if (noise_std == 0):
        snr = 100
        return snr
    else:
        return signal_std**2 / noise_std**2
