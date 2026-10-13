from math import nan
from pydoc import text
from re import S
import tkinter as tk
from tkinter import ttk
from turtle import end_fill
import numpy as np
from scripts.simulators import utils
import pandas as pd
import scipy
from scipy.interpolate import make_interp_spline
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian

#Spiketype Functions
def run_single(param, neu_param):  
    Fs = param["Fs"]
    t = param["t"]
    d = param["d"]
    num_neurons = param["num_neurons"]
    Intpol = param["Intpol"]
    lowpass_fil = param["lowpass_fill"]
    spiketype = param["spike_type"]

    speed = neu_param["speed"]
    rate = neu_param["rate"]
    on = neu_param["on"]
    off = neu_param["off"]
    amp = neu_param["amp"]
    mod_type = neu_param["mod_type"]

    spike_duration=0.002 
    spike_sigma= spike_duration/6
    spike_sample= int(np.round(spike_duration*Fs)) #How many spikes exist in the duration
    time_spike= np.arange(spike_sample)/Fs

    if spiketype== "Gaussian-modulated sine model":
        spike_waveform = utils.gaussian(time_spike, spike_sigma, spike_duration)
    elif spiketype== "Double exponential model":
        spike_waveform = utils.double_exp(spike_duration*(1/5), spike_duration*(4/5),time_spike)
    else:
        spike_waveform = utils.sum_gauss(time_spike, spike_sigma)

    spike_waveform = spike_waveform / np.max(np.abs(spike_waveform))

    #Noise
    time = np.linspace(0,t,num=int((t*Fs)+1))    #initialize time vector
    rndm = np.random.default_rng()
    noise1= rndm.standard_normal(time.size)  #Generate random timestamp for the noise
    noise2= rndm.standard_normal(time.size)
    rms_noise= (np.sqrt(np.mean(noise1**2))+np.sqrt(np.mean(noise2**2)))/2
    signal1= noise1
    signal2= noise2
    spike_trains = [None]*num_neurons           #Initialize list to store spike timestamp for each neurons

    for neuronid in range(num_neurons):
        min_gap = 0.002
        window  = off[neuronid] - on[neuronid]
        n_spikes=round(rate[neuronid]*(off[neuronid]-on[neuronid]))

        spikes_filt = []
        attempts = 0
        max_attempts = 100 * n_spikes
        while len(spikes_filt) < n_spikes and attempts < max_attempts:
            t = rndm.uniform(on[neuronid], off[neuronid])
            idx = np.searchsorted(spikes_filt, t)
            ok = True
            if idx > 0 and t - spikes_filt[idx-1] < min_gap: ok = False
            if idx < len(spikes_filt) and spikes_filt[idx] - t < min_gap: ok = False
            if ok:
                spikes_filt.insert(idx, t)
            attempts += 1

            spikes = np.array(spikes_filt)
    
        waveform= amp[neuronid]*rms_noise*spike_waveform                  #waveform
        for spike_time in spikes:
            electrode1 = int(np.round(spike_time*Fs))                     #Timestamp of spikes caught by electrode 1
            electrode2 = int(np.round((spike_time+d/speed[neuronid])*Fs))        #Timestamp of spikes caught by electrode 2
            if electrode1+spike_sample <= len(time):
                signal1[electrode1:electrode1+spike_sample] = signal1[electrode1:electrode1+spike_sample] + waveform
            if electrode2+spike_sample <= len(time):    
                signal2[electrode2:electrode2+spike_sample] = signal2[electrode2:electrode2+spike_sample] + waveform
        spike_trains[neuronid]=spikes
       
    #Interpolation
    if Intpol == 1:
        signal1_ups= np.nan
        signal2_ups=np.nan
    else:
        N=len(signal1)
        time_Ups= np.linspace(0,((N-1)/Fs),num= (N-1)*Intpol+1)
        Fs_Ups= Fs*Intpol
        #Spline interpolation
        interp1= make_interp_spline(time, signal1, k=3)
        interp2= make_interp_spline(time, signal2, k=3)
        signal1 = interp1(time_Ups)
        signal2 = interp2(time_Ups)

    #Filter signal Lowpass
    fc= np.array([5, lowpass_fil]) #Low and High pass cutoff frequency

    if Intpol==1:
        fc_n=fc/(Fs*2) #Cut off frewquency in terms of Nyquist rate
        b,a = butter(2,fc_n, btype='bandpass') #Generate the numerator and denominator for the bandpass filter
        signal1 = filtfilt(b,a,signal1)
        signal2 = filtfilt(b,a, signal2)

    else:
        fc_n=fc/(Fs_Ups*2)
        b,a = butter(2, fc_n , btype='bandpass')
        signal1_ups = filtfilt(b,a,signal1_ups)
        signal2_ups = filtfilt(b,a, signal2_ups)


    #Trace Modifications
    if mod_type=="None":
        signal1=signal1
        signal2=signal2
    elif mod_type=="Absolute":
        signal1=np.absolute(signal1)
        signal2=np.absolute(signal2)
        signal1_ups= np.absolute(signal1_ups)
        signal2_ups=np.absolute(signal2_ups)
    elif mod_type=="rectified":
        signal1=-np.min(signal1,0)
        signal2=-np.min(signal2,0)
        signal1_ups= -np.min(signal1_ups,0)
        signal2_ups=-np.min(signal2_ups,0)
    else:
        signal1=signal1**2
        signal2=signal2**2
        signal1_ups=signal1_ups**2
        signal2_ups=signal2_ups**2
    


    #Plot the  plot
    fig,(ax_signal, ax_raster)=plt.subplots(2,1, figsize=(12,6), sharex= True, gridspec_kw={"height_ratios":[3,1]})

    if Intpol ==1:
        ax_signal.plot(time, signal1, lw=0.5, label="Proximal electrode")
        ax_signal.plot(time, signal2, lw=0.5, label="Distal electrode")
        ax_signal.set_title("Nerce Signal Recorded from Two Electrodes")
    else:
        ax_signal.plot(time_Ups, signal1_ups, lw=0.5, label="Proximal electrode")
        ax_signal.plot(time_Ups, signal2_ups, lw=0.5, label="Distal electrode")
        ax_signal.set_title("Nerce Signal Recorded from Two Electrodes Interpolated")

    ax_signal.set_xlabel("Time(s)")
    ax_signal.set_ylabel("Amplitude")
    ax_signal.set_title("Nerve Signal Recorded from Two Electrodes")
    ax_signal.legend(loc="upper right")

    for x in range(num_neurons):
        ax_raster.plot(spike_trains[x], (x+1)*np.ones_like(spike_trains[x]),"k.", markersize=3)
        ax_raster.set_yticks(range(1, num_neurons + 1))
        ax_raster.set_yticklabels([f'n{i}'for i in range(1,num_neurons+1)]) 
    ax_raster.set_title("Neuron Spike Rasters Plots")
    ax_raster.set_xlabel("Time(s)")
    ax_raster.set_ylabel("Neuron #")
    ax_raster.set_ylim(0.5,num_neurons+0.5)

    plt.tight_layout()
    plt.show()
















