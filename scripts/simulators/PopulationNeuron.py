from math import nan
from pydoc import text
from re import S
import tkinter as tk
import numpy as np
from tkinter import ttk
from turtle import end_fill
from scripts.simulators import utils
import pandas as pd
import scipy
from scipy.interpolate import make_interp_spline
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal.windows import gaussian as gaussian_window

def run_pop (sim_param, fiber_param):
    n_fiber_types=6
    sampled_velocity=np.empty((n_fiber_types,1000000),dtype=float) #number of fiber types x 1000000 maximize until 1M
    
    #Simulation Parameter 
    Fs = sim_param["Fs"]
    t = sim_param["t"]
    d = sim_param["d"]
    num_neurons = sim_param["num_neurons"]
    Intpol=sim_param["Intpol"]
    lowpass_fil=sim_param["lowpass_fill"]
    
    
    spike_duration=0.002 
    spike_sigma= spike_duration/6
    spike_sample= int(np.round(spike_duration*Fs)) #How many spikes exist in the duration
    time_spike= np.arange(spike_sample)/Fs


    #Fiber names
    fiber_types = ['Aa m', 'Ay m', 'Aa s', 'Aβ s', 'Aδ s', 'C s']
    #Fiber Parameters
    fiber_attr =["fiber_n", "mean_speed", "SD_speed","amp_fact","rate","on","off"]


    #Noise
    time = np.linspace(0,t,num=int((t*Fs)+1))    #initialize time vector
    rndm = np.random.default_rng()
    noise1= rndm.standard_normal(time.size)  #Generate random timestamp for the noise
    noise2= rndm.standard_normal(time.size)
    rms_noise= (np.sqrt(np.mean(noise1**2))+np.sqrt(np.mean(noise2**2)))/2
    signal1= noise1
    signal2= noise2

    #Initialize list to store value
    num_groups, neurons_per_group=sampled_velocity.shape
    spike_trains = np.empty((num_groups, neurons_per_group), dtype=object)

    for i in range(n_fiber_types):
        fd=fiber_param.iloc[i]
        fiber_n = int(fd["fiber_n"])
        mean_speed=fd["mean_speed"]
        sd_speed=fd["SD_speed"]
        rate=fd["rate"]
        on= fd["on"]
        off=fd["off"]
        amp=fd["amp_fact"]
        
        if fiber_n > 0 and sd_speed >0:
            v_axis=np.arange(-120,120.01,0.01) #Simulate maximum signal velocity at 120m/s both ways at 0.01 m/s precision
            fiber_pdf = scipy.stats.norm.pdf(v_axis, loc=mean_speed, scale=sd_speed)            #Probability density function
            probs_weights= fiber_pdf/fiber_pdf.sum()
            sampled_velocity[i, :fiber_n]=np.random.choice(v_axis, size=fiber_n, replace=True, p=probs_weights)


            spike_waveform = utils.gaussian(time_spike, spike_sigma, spike_duration)

            for neuronid in range(fiber_n):
                v = sampled_velocity[i, neuronid]
                n_spikes = int(round(rate * (off - on)))     # ← FIRST

                min_gap = 0.002
                spikes_filt = []
                attempts = 0
                max_attempts = 100 * n_spikes
                while len(spikes_filt) < n_spikes and attempts < max_attempts:
                    t_sp = rndm.uniform(on, off)
                    idx = np.searchsorted(spikes_filt, t_sp)
                    ok = True
                    if idx > 0 and t_sp - spikes_filt[idx-1] < min_gap:
                        ok = False
                    if idx < len(spikes_filt) and spikes_filt[idx] - t_sp < min_gap:
                        ok = False
                    if ok:
                        spikes_filt.insert(idx, t_sp)
                    attempts += 1

                spikes = np.array(spikes_filt)
                
                waveform= amp*rms_noise*spike_waveform                  #waveform
                for spike_time in spikes:
                    electrode1 = int(np.round(spike_time*Fs))                     #Timestamp of spikes caught by electrode 1
                    electrode2 = int(np.round((spike_time+d/v)*Fs))        #Timestamp of spikes caught by electrode 2
                    if electrode1+spike_sample <= len(time):
                            signal1[electrode1:electrode1+spike_sample] = signal1[electrode1:electrode1+spike_sample] + waveform
                    if electrode2+spike_sample <= len(time):    
                        signal2[electrode2:electrode2+spike_sample] = signal2[electrode2:electrode2+spike_sample] + waveform
                spike_trains[i,neuronid]=spikes

    #Interpolation
    if Intpol == 1:
        signal1_ups = np.nan
        signal2_ups = np.nan
    else:
        N=len(signal1)
        time_Ups= np.linspace(0,((N-1)/Fs),num= (N-1)*Intpol+1)
        Fs_Ups= Fs*Intpol
        #Spline interpolation
        interp1= make_interp_spline(time, signal1, k=3)
        interp2= make_interp_spline(time, signal2, k=3)
        signal1_ups = interp1(time_Ups)
        signal2_ups = interp2(time_Ups)

    #Filter signal Lowpass
    fc = np.array([5, lowpass_fil]) #Low and High pass cutoff frequency

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

    #Plot the  plot
    fig, (ax_signal, ax_raster) = plt.subplots(
        2, 1, figsize=(12,6), sharex= True, gridspec_kw={"height_ratios":[3,1]}
    )

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
    
    row = 1

    for fi in range(n_fiber_types):
        n_in_group = int(fiber_param.iloc[fi]["fiber_n"])
        for ni in range(n_in_group):
            st = spike_trains[fi, ni]
            if st is not None and len(st) > 0:
                ax_raster.plot(st, row * np.ones_like(st), "k.", markersize=3)
            row += 1
                
        
    ax_raster.set_title("Neuron Spike Rasters Plots")
    ax_raster.set_xlabel("Time(s)")
    ax_raster.set_ylabel("Neuron #")

    plt.tight_layout()
    plt.show()


def plot_signal():
    return
    