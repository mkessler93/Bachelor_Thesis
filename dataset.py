import random
from datetime import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from filter import *
from noise import *
from features import *

data = pd.read_csv('./data/ECG2.csv', index_col=0)


def slice_ecg(ecg_raw,t=10):
    '''
    :param ecg_raw: dataframe
    :return: ecg_list: list of 10s intervalls from ecg_raw
    '''
    ecg = clean_ecg(ecg_raw)
    ecg['ECG'] = MA_filter(ecg['ECG'])
    ecg['ECG'] = high_low_pass_filter(ecg['ECG'])
    ecg_rev = ecg.copy()
    ecg_rev = ecg_rev.assign(ECG = ecg['ECG'][::-1].to_numpy()) #reverse ECG

    len_ecg = int(ecg.index[-1]) -t
    ecg_list = list()

    #get 10sec intervalls from whole ecg and reversed version
    for i in range(0,len_ecg,2):
        ecg_list.append(ecg[i:i+t])
        ecg_list.append((ecg_rev[i:i+t]))

    return ecg_list

def clean_ecg(ecg_raw, fs_bool = False):

    '''
    removes unnecessary columns from dataframe and add index with time in s

    :param ecg_raw: raw dataframe
           fs: if True, return sampling rate of ECG as int; default: fs_bool = False
    :return: ecg: cleaned dataframe
             fs: sampling rate
    '''

    list_col = ecg_raw.columns

    for i in range(len(list_col)):
        if list_col[i] != 'ECG' and list_col[i] != 'RTC Time':
            ecg_raw = ecg_raw.drop([list_col[i]], axis=1)

    ecg = ecg_raw.set_index("RTC Time", drop=True)

    ecg['ECG'] *= -1
    s1 = ecg.index[0].split(":")[-1]
    s2 = ecg.index[1].split(":")[-1]
    step = float(s2) - float(s1)
    l = len(ecg.index)
    time = np.linspace(0, step * l, l)
    ecg['time [s]'] = time
    ecg = ecg.set_index('time [s]', drop=True)

    if fs_bool == True:
        fs = 1/step
        return ecg, int(fs)
    else:
        return ecg

def add_noise_col(ecg_list,t=10):
    #create random noise
    x_ax, power, base, emg, cable = artifical_noise(T=t)
    noise_sum = power+base+emg+cable
    seg = len(ecg_list)//4
    rest = len(ecg_list)%seg
    num_noise_types = 1
    it = 1
    num_power = 0
    num_base = 0
    num_emg = 0
    num_cable = 0

    for df in ecg_list:
        max_ampl = max(df['ECG']) * 0.24  # limit amplitutde of noise to 1/4 of the max amplitude of ecg signal
        if(it%seg == 0 and it <len(ecg_list)-rest):
            num_noise_types +=1
        index_noise = random.sample(range(0,4),num_noise_types)
        column_bool = {"power": False, "base": False, "emg": False, "cable": False}

        for ind in index_noise:
            if(ind == 0):
                df.insert(1, 'noise_powerline', max_ampl * power, True)
                column_bool["power"] = True
                num_power += 1
            if(ind == 1):
                df.insert(1, 'noise_baseline', max_ampl * base, True)
                column_bool["base"] = True
                num_base += 1
            if(ind == 2):
                df.insert(1, 'noise_emg', max_ampl * emg, True)
                column_bool["emg"] = True
                num_emg += 1
            if(ind == 3):
                df.insert(1, 'noise_cable_movement', max_ampl * cable, True)
                column_bool["cable"] = True
                num_cable += 1
        for b in column_bool.keys():
            if(column_bool[b]==False):
                if(b == "power"):
                    df.insert(1, 'noise_powerline', np.zeros(len(df['ECG'])), True)
                if(b == "base"):
                    df.insert(1, 'noise_baseline', np.zeros(len(df['ECG'])), True)
                if(b == "emg"):
                    df.insert(1, 'noise_emg', np.zeros(len(df['ECG'])), True)
                if(b == "cable"):
                    df.insert(1, 'noise_cable_movement', np.zeros(len(df['ECG'])), True)
        # insert column noise_ECG for plotting
        df.insert(1, "noise_ECG", df['ECG'].to_numpy() + df['noise_powerline'].to_numpy() + df['noise_baseline'].to_numpy() +  df['noise_emg'].to_numpy() +  df['noise_cable_movement'].to_numpy())


        it += 1
    #print("Num_power: {}, Num_base: {}, Num_emg: {}, Num_cable: {}".format(num_power,num_base,num_emg, num_cable))
    return ecg_list, num_power,num_base,num_emg, num_cable
