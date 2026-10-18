import numpy as np
from scipy.signal.windows import gaussian as gaussian_window


def gaussian(x, sigma, spike_duration):
    freq = 2 / spike_duration
    center = spike_duration / 2
    window = np.exp(-0.5 * ((x - center) / sigma)**2)
    
    # Multiply by the sine wave
    return window * np.sin(2 * np.pi * freq * x)

def double_exp(t_rise, t_decay, x):
    t_shift = x - np.min(x)
    return -(np.exp(-t_shift / t_decay) - np.exp(-t_shift / t_rise))

def sum_gauss(x, sigma, amp1=-1.0, amp2=0.5, center=0.0007, delta_t=0.0005):
    sigma1 = sigma / 3
    sigma2 = sigma / 2
    center2 = center + delta_t
    gauss1 = amp1 * np.exp(-(x - center) ** 2 / (2 * sigma1 ** 2))
    gauss2 = amp2 * np.exp(-(x - center2) ** 2 / (2 * sigma2 ** 2))
    return gauss1 + gauss2

def get_spike_waveform(Fs,
                       spike_duration=0.002,
                       spike_sigma=None,
                       kind="sum_gauss",):
   
    if spike_sigma is None:
        spike_sigma = spike_duration / 6
        
    spike_sample = int(round(spike_duration * Fs))
    time_spike = np.arange(spike_sample) / Fs                 


    if kind in ("Gaussian-modulated sine model", "gaussian"):
        spike_waveform = gaussian(time_spike, spike_sigma, spike_duration)

    elif kind in ("Double exponential model", "double_exp"):
        spike_waveform = double_exp(spike_duration / 5, spike_duration * 4 / 5, time_spike)

    elif kind in ("sum_gauss",):
        spike_waveform = sum_gauss(time_spike, spike_sigma)
        
    else:
        raise ValueError(f"Unknown kind: {kind}")

    peak = np.max(np.abs(spike_waveform))
    if peak > 0:
        spike_waveform = spike_waveform / peak

    return time_spike, spike_waveform
