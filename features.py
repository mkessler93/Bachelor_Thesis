import numpy as np
import pandas as pd
from scipy.stats import kurtosis,skew
from scipy import signal
from noise import artifical_noise
from filter import snr_1,snr_2

def kSQI(ecg):

    return kurtosis(ecg)

def pSQI(ecg, fs=250):
    f, psd = signal.welch(ecg, fs)
    ecg_pow = sum(psd[5:40])
    qrs_pow = sum(psd[5:15])
    return qrs_pow / ecg_pow

def sSQI(ecg):
    return skew(ecg)

def fSQI():

    return 0

def basSQI(ecg, fs=250):
    f, psd = signal.welch(ecg, fs)
    ecg_pow = sum(psd[0:40])
    bas_pow = sum(psd[0:1])
    return 1-(bas_pow/ecg_pow)

def calculate_all_features(ecg_list,group,snrbool=True):
    if(snrbool==True):
        col = ('kSQI', 'pSQI', 'sSQI', 'basSQI','SNR', 'Quality', "group")
        features = pd.DataFrame(columns=col)
        for ecg in ecg_list:
            power = ecg['noise_powerline'].to_numpy()
            base = ecg['noise_baseline'].to_numpy()
            emg = ecg['noise_emg'].to_numpy()
            cable = ecg['noise_cable_movement'].to_numpy()
            noise = power + base + emg + cable


            k = kSQI(ecg['ECG'])
            p = pSQI(ecg['ECG'])
            s = sSQI(ecg['ECG'])
            bas = basSQI(ecg['ECG'])
            snr = snr_2(noise, ecg['ECG'])
            if (snr < 1):
                qual = 0
            else:
                qual = 1
            tmp_df = pd.DataFrame([[k,p,s,bas,snr,qual,group]],columns=col)
            features = features.append(tmp_df, ignore_index=True)
    if(snrbool==False):
        col = ('kSQI', 'pSQI', 'sSQI', 'basSQI', 'SNR', 'Quality', "group")
        features = pd.DataFrame(columns=col)
        for ecg in ecg_list:
            k = kSQI(ecg['ECG'])
            p = pSQI(ecg['ECG'])
            s = sSQI(ecg['ECG'])
            bas = basSQI(ecg['ECG'])
            qual = 0 #change according to dataset of qualitative analysis
            snr = 0.9 #change according to dataset of qualitative analysis
            tmp_df = pd.DataFrame([[k, p, s, bas, snr, qual, group]], columns=col)
            features = features.append(tmp_df, ignore_index=True)
    return features
