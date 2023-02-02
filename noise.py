import numpy as np
import random
from datetime import datetime

def artifical_noise(T=10,fs = 250):
    '''

    :param T: duration [s] of noise signal
    :param fs: sampling frequency
    :return: x_ax, powerline, baseline, emg_noise, electrode_cable_noise
    '''
    x_ax = np.arange(0,T,1/fs)

    powerline = (1/3) * np.sin(2* np.pi* 50 * x_ax)

    baseline = _baseline(x_ax,fs,t=T)

    emg_noise = np.random.normal(0,0.05,len(x_ax))

    electrode_cable_noise = _cable_movement(x_ax)

    return x_ax, powerline, baseline, emg_noise, electrode_cable_noise

#private
def _cable_movement(x):
        freq = np.array([1.5,3.16, 6.3, 8.])
        ampl = np.array([0.1, 0.4 , 0.7, 1.0])
        y = np.zeros((4,len(x)))
        for i in range(len(freq)):
            y[i] = ampl[i] * np.sin(2 * np.pi * freq[i] * x)
        cable_noise = np.zeros((len(x)))
        for i in range(4):
            cable_noise += y[i]
        return cable_noise

#private
def _baseline(x,fs,t):
    random.seed(3)
    rnd_xax = random.sample(range(0,fs*9),3) #max start of last intervall at 9sec
    rnd_dur = random.sample(range(0,fs), 3) #duration of max 1s
    rnd_ampl = random.random()
    baseline = np.zeros(t*fs)
    for i in rnd_xax:
        for j in rnd_dur:
            baseline[i:i+j] = rnd_ampl * np.sin(2 * np.pi * (1 / 3) * x)[i:i+j]
    return baseline
