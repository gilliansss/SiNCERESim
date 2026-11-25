import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.signal import correlate
from matplotlib.figure import Figure

def compute_and_plot_heatmap(sig1, sig2, Fs, distance_mm, window_width_s, step_size_s, range_lags_s, velocity_mode=True):
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
    
    
    if velocity_mode:
        safe_lags = lags.copy()
        safe_lags[np.abs(safe_lags) < 1e-6] = 1e-6 
        
        y_vals = -(distance_mm / 1000.0) / safe_lags

        valid_mask = np.abs(y_vals) <= 120 
        
        y_axis_masked = np.arcsinh(y_vals[valid_mask])
        cross_correlations_masked = cross_correlations[valid_mask, :]
        
        sort_idx = np.argsort(y_axis_masked)
        y_axis = y_axis_masked[sort_idx]
        cross_correlations = cross_correlations_masked[sort_idx, :]
        
        ylabel = 'Velocity (m/s)'
    else:
        # Standard Lag Mode (Milliseconds)
        y_axis = lags * 1000
        ylabel = 'Lead/Lag (ms)'
        
    # Plot the heatmap
    mesh = ax.pcolormesh(time_axis, y_axis, cross_correlations, shading='auto', cmap='bone')
    
    # Custom Colormap (Bone for background, Hot for high correlation)
    threshold_multiplier = 1.5 #increase or decrease threshold
    corr_threshold = threshold_multiplier * np.nanstd(cross_correlations)
    
    # Get min/max
        # Get min/max
    c_min = np.nanmin(cross_correlations)
    c_max = np.nanmax(cross_correlations)
    
    if c_max > corr_threshold:
        # Create custom colormap
        n_bone = max(1, int(160 * (corr_threshold - c_min) / (c_max - c_min)))
        n_hot = max(1, 160 - n_bone)
        
        # FIX: Only use the bottom half of bone (0 to 0.5) so it stays dark
        cmap_bone = cm.bone(np.linspace(0, 0.5, n_bone)) 
        
        # FIX: Only use the top half of hot (0.5 to 1) so it starts at orange, not black
        cmap_hot = cm.hot(np.linspace(0.5, 1, n_hot)) 
        
        # Combine them
        custom_cmap = mcolors.LinearSegmentedColormap.from_list('custom', np.vstack((cmap_bone, cmap_hot)))
        
        mesh.set_cmap(custom_cmap)
        mesh.set_clim(c_min, c_max)
    
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')
    ax.set_title('Cross-correlation Heatmap (Proximal vs Distal)')
    
    if velocity_mode:
        # Set ticks up to 120
        tick_vals = [-120, -100, -80, -50, -20, -10, 10, 20, 50, 80, 100, 120]
        ax.set_yticks(np.arcsinh(tick_vals))
        ax.set_yticklabels([str(v) for v in tick_vals])
        
        # FORCE the Y-axis limits
        ax.set_ylim(np.arcsinh(-120), np.arcsinh(120))
    
    fig.colorbar(mesh, ax=ax, label="Correlation")
    return fig