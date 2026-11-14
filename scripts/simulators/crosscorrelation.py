import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.signal import correlate
from matplotlib.figure import Figure

def compute_and_plot_heatmap(sig1, sig2, Fs, distance_mm, window_width_s, step_size_s, range_lags_s):
    # Convert time parameters to sample points
    step_samples = int(round(step_size_s * Fs))
    window_samples = int(round(window_width_s * Fs))
    max_lag_samples = int(round(range_lags_s * Fs))
    
    total_samples = len(sig1)
    
    # Calculate how many windows we can fit
    valid_windows = 1 + (total_samples // step_samples) - (window_samples // step_samples)
    if valid_windows < 1:
        print("Window size is larger than the signal!")
        return
        
    # Calculate adjustment so the heatmap aligns with time
    total_cols = total_samples // step_samples
    ajustmnt = int(round((total_cols - valid_windows) / 2))
    
    # Initialize the matrix
    cross_correlations = np.full((2 * max_lag_samples + 1, total_cols), np.nan)
    
    # Sliding window loop
    for xx in range(valid_windows):
        start_idx = xx * step_samples
        end_idx = start_idx + window_samples
        
        win_p = sig1[start_idx:end_idx]
        win_d = sig2[start_idx:end_idx]
        
        # 1. Subtract the mean (Remove DC Offset)
        win_p = win_p - np.mean(win_p)
        win_d = win_d - np.mean(win_d)
        
        # 2. Normalize (Prevent noise from washing out the signal)
        norm_factor = (np.std(win_p) * np.std(win_d) * len(win_p))
        if norm_factor == 0:
            continue # Skip if signal is completely flat
            
        # 3. Compute normalized cross-correlation
        corr = correlate(win_p, win_d, mode='full') / norm_factor
        
        center = len(win_p) - 1
        lag_slice = corr[center - max_lag_samples : center + max_lag_samples + 1]
        
        cross_correlations[:, xx + ajustmnt] = lag_slice

    # --- Plotting Setup ---
    fig = Figure(figsize=(10, 5), layout="constrained")
    ax = fig.add_subplot(111)
    
    # X-axis: Time
    time_axis = np.arange(total_cols) * step_size_s
    
    # Y-axis: Lag or Velocity
    lags = np.linspace(-range_lags_s, range_lags_s, 2 * max_lag_samples + 1)
    
    # If we want velocity mode (based on your MATLAB code)
    velocity_mode = False # Set to False if you want Lag (ms)
    
    if velocity_mode:
        safe_lags = lags.copy()
        safe_lags[np.abs(safe_lags) < 1e-6] = 1e-6 #Prevent division by zero and extreme infinity values
        
        y_vals = (distance_mm / 1000.0) / safe_lags
        y_axis = np.arcsinh(y_vals)
        
        # FIX: pcolormesh requires the Y-axis to be monotonically increasing
        sort_idx = np.argsort(y_axis)
        y_axis = y_axis[sort_idx]
        cross_correlations = cross_correlations[sort_idx, :] # Reorder the data to match the sorted Y
        ylabel = 'Velocity (m/s)'
    else:
        y_axis = lags * 1000
        ylabel = 'Lead/Lag (ms)'
        
    # Plot the heatmap
    mesh = ax.pcolormesh(time_axis, y_axis, cross_correlations, shading='auto', cmap='bone')
    
    # Custom Colormap (Bone for background, Hot for high correlation)
    threshold_multiplier = 2.0
    corr_threshold = threshold_multiplier * np.nanstd(cross_correlations)
    
    # Get min/max
    c_min = np.nanmin(cross_correlations)
    c_max = np.nanmax(cross_correlations)
    
    if c_max > corr_threshold:
        # Create custom colormap
        n_bone = int(160 * (corr_threshold - c_min) / (c_max - c_min))
        n_hot = 160 - n_bone
        
        cmap_bone = cm.bone(np.linspace(0, 1, n_bone))
        cmap_hot = cm.hot(np.linspace(0.5, 1, n_hot)) # Start hot at 0.5 (red/yellow)
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', np.vstack((cmap_bone, cmap_hot)))
        
        mesh.set_cmap(custom_cmap)
        mesh.set_clim(c_min, c_max)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')
    ax.set_title('Cross-correlation Heatmap (Proximal vs Distal)')
    
    if velocity_mode:
        # Custom tick locations for asinh scale
        tick_vals = [-200, -50, -10, -2, 2, 10, 50, 200]
        ax.set_yticks(np.arcsinh(tick_vals))
        ax.set_yticklabels([str(v) for v in tick_vals])
    
    fig.colorbar(mesh, ax=ax, label="Correlation")
    return fig