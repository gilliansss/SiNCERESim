import numpy as np

def gaussian (x, sigma,spike_duration):
    freq = 2/spike_duration
    window = gaussian(len(x),sigma)
    return window * np.sin(2*np.pi*freq*x)

def double_exp(t_rise, t_decay, x):
    t_shift = x - np.min(x) #Make sure the spike starts at 0s
    return -(np.exp(-t_shift/t_decay) - np.exp(-t_shift/t_rise))

def sum_gauss(x, sigma, amp1=-1, amp2=0.5,center=0.0007, delta_t=0.0005):
    sigma1 = sigma/3
    sigma2 = sigma/2
    
    center2 = center+delta_t

    gauss1 = amp1*np.exp(-(x-center)**2/(2*(sigma1**2)))
    gauss2 = amp2*np.exp(-(x-center2)**2/(2*(sigma2**2)))

    return gauss1+gauss2