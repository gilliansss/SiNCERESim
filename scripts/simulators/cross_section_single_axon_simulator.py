import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# take the raw waveforms from singleneuron-2.py, doesnt include the filtering interpolation etc
from raw_waveform_generator import get_spike_waveform

spike_duration=0.002
grid_spacing_mm = 0.1
nerve_radius_mm = 0.4
axis_limit = 0.75
    
def simulate(positions, parameter, neu_parameter):
    Fs = parameter["Fs"]
    record_duration_s = parameter["t"]
    num_neurons = parameter["num_neurons"] 
    section_distance_mm = parameter["section_distance_mm"]
    noise_level = parameter["noise_level"]
    attenuation = parameter["attenuation"]
    
    amplitude = neu_parameter["amp"]
    velocity = neu_parameter["velocity"]
    rate = neu_parameter["rate"]
    
    spiketype = "Gaussian-modulated sine model"

    _, waveform = get_spike_waveform(Fs, spike_duration=spike_duration, kind=spiketype) 
    print("Waveform max:", np.max(waveform), "| Length:", len(waveform))

    time_axis = int(record_duration_s * Fs) + 1 
    time = np.arange(time_axis) / Fs 
    
    rng = np.random.default_rng() 
    signals = noise_level * rng.standard_normal((2, 4, time_axis))
    axon_positions = [] 
    all_spikes=[]
    final_delay_samples = 0

    # FIXED: Only ONE loop here
    for neuronid in range(num_neurons):
        r, c = positions[neuronid]
        axon_x = (c - 2) * grid_spacing_mm
        axon_y = (r - 2) * grid_spacing_mm
        axon_position = np.array([axon_x, axon_y])
        axon_positions.append(axon_position)
    
        angles = np.arange(4) * np.pi/2
        electrodes = nerve_radius_mm * np.column_stack((np.cos(angles), np.sin(angles)))
        
        distances = np.maximum(np.linalg.norm(electrodes - axon_position, axis=1), 0.01) 
        weights = 1.0 / (distances**attenuation) 
        weights /= np.max(weights)
        
        section_delay = int(round(((section_distance_mm / 1000) / velocity[neuronid]) * Fs))
        final_delay_samples = section_delay
        
        min_gap = 0.002
        window = record_duration_s
        n_spikes = round(rate[neuronid] * window) 
        
        spikes_filt = []
        attempts = 0
        max_attempts = 100 * n_spikes
        starts = []
        
        while len(spikes_filt) < n_spikes and attempts < max_attempts:
            spike_time = rng.uniform(0, record_duration_s)
            idx = np.searchsorted(spikes_filt, spike_time)
            ok = True
            if idx > 0 and spike_time - spikes_filt[idx-1] < min_gap: ok = False
            if idx < len(spikes_filt) and spikes_filt[idx] - spike_time < min_gap: ok = False
            if ok:
                spikes_filt.insert(idx, spike_time)
            attempts += 1
            
        spikes = np.array(spikes_filt)
        all_spikes.append(spikes)
        spike_samples = int(round(0.002 * Fs))
        
        for start_time in spikes:
            start_idx = int(round(start_time * Fs))
            starts.append([
                [start_idx, start_idx + spike_samples], 
                [start_idx + section_delay, start_idx + section_delay + spike_samples]
            ])
                # Put this right before the placement loop:
        print(f"Spikes generated: {len(spikes)}")
        print(f"Starts generated: {len(starts)}")
        print(f"First start index: {starts[0][0]}")    
        for time_pair in starts:
            for section in range(2):
                start, end = time_pair[section]
                if end > time_axis:
                    continue
                for electrode in range(4):
                    signals[section, electrode, start:end] += (amplitude[neuronid] * weights[electrode] * waveform)

    return time, signals, axon_positions, electrodes, final_delay_samples, all_spikes
class CrossSectionWidget:
    def __init__(self, root):
        self.root = root
        self.root.title("Single-Axon Cross-Section Simulator")
        self.root.geometry("1300x1050")

        # 1. Simulation Variables (Global)
        self.Fs_var = tk.DoubleVar(value=30000)
        self.duration_var = tk.DoubleVar(value=10.0)
        self.num_neurons_var = tk.IntVar(value=1)
        self.noise_level_var= tk.DoubleVar(value=0.05)
        self.section_distance_var= tk.IntVar(value=2) #distance between proximal and distal section 2mm
        self.attenuation_var=tk.IntVar(value= 1)

        # 2. Neuron Variables (Lists to hold values for multiple neurons)
        self.amplitude_vars = []
        self.velocity_vars = []
        self.rate_vars = []

        self.rng = np.random.default_rng()
        self.current_positions = None

        # 3. Build UI Frames
        # --- Simulation Frame ---
        sim_frame = tk.LabelFrame(root, text="Simulation Parameters", padx=5, pady=5)
        sim_frame.pack(padx=8, pady=5, fill="x")
        
        self.add_control(sim_frame, "Sampling Rate (Hz)", self.Fs_var, 1000, 50000, 1000, 0)
        self.add_control(sim_frame, "Record Duration (s)", self.duration_var, 1, 20, 1, 2)
        self.add_control(sim_frame, "Num Neurons", self.num_neurons_var, 1, 10, 1, 4)
        self.add_control(sim_frame, "Distance (mm)", self.section_distance_var,1,10, 1,6)
        self.add_control(sim_frame, "Attenuation Rate", self.attenuation_var, 1,5,1, 8)

        # --- Neuron Frame ---
        self.neu_frame = tk.LabelFrame(root, text="Neuron Parameters", padx=5, pady=5)
        self.neu_frame.pack(padx=8, pady=5, fill="x")
        
        # --- Button Frame ---
        button_frame = tk.Frame(root)
        button_frame.pack(padx=8, pady=5)
        
        tk.Button(button_frame, text="Run (new random axon)", command=self.run).pack(side="left", padx=8)
        tk.Button(button_frame, text="Run (same axon)", command=self.run_same_axon).pack(side="left", padx=8)
    
        # 4. Build Figure and Canvas
                # Create a 3x2 grid: cross-section on the left, 3 plots on the right
        self.figure, axes = plt.subplot_mosaic(
            [["cross", "sig1"], 
             ["cross", "sig2"],
             ["cross", "raster"]],
            figsize=(14, 10), 
            layout="constrained",
            gridspec_kw={'width_ratios': [1, 3], 'height_ratios': [2, 2, 1]}
        )
        
        self.ax_grid = axes["cross"]
        self.ax_section1 = axes["sig1"]
        self.ax_section2 = axes["sig2"]
        self.ax_raster = axes["raster"]

        # Sync the time axis for all 3 right-side plots
        self.ax_section2.sharex(self.ax_section1)
        self.ax_raster.sharex(self.ax_section1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        
        # --- ADD THESE 3 LINES FOR THE TOOLBAR ---
        self.toolbar = NavigationToolbar2Tk(self.canvas, root)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        # -----------------------------------------
        
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        # 5. Initialize the Neuron Controls for the first time
        self.rebuild_neuron_controls()

        # 6. Add the listener to rebuild controls when Num Neurons changes
        self.num_neurons_var.trace_add("write", lambda *args: self.rebuild_neuron_controls())

        # 7. Initial run
        self.run()

    def rebuild_neuron_controls(self):
        # Clear existing widgets in the frame
        for widget in self.neu_frame.winfo_children():
            widget.destroy()
            
        # Clear old variables
        self.amplitude_vars.clear()
        self.velocity_vars.clear()
        self.rate_vars.clear()


        num_neurons = self.num_neurons_var.get()

        # Create widgets for each neuron
        for i in range(num_neurons):
            self.amplitude_vars.append(tk.DoubleVar(value=1.0))
            self.velocity_vars.append(tk.DoubleVar(value=40.0))
            self.rate_vars.append(tk.DoubleVar(value=100))

            col = i * 3 # Offset columns for each neuron
            
            # Neuron Label
            tk.Label(self.neu_frame, text=f"Neuron {i+1}", font=("Arial", 10, "bold")).grid(
                row=0, column=col, columnspan=2, pady=(5, 0)
            )

            # Add controls for this neuron
            self.add_control(self.neu_frame, "Amplitude (µV)", self.amplitude_vars[i], 0.1, 5, 0.1, col, row=1)
            self.add_control(self.neu_frame, "Velocity (m/s)", self.velocity_vars[i], 1, 120, 1, col+1, row=1)
            self.add_control(self.neu_frame, "Firing Rate (Hz)", self.rate_vars[i], 1, 120, 1, col+2, row=1)

    def add_control(self, parent, label, variable, low, high, step, column, row=0):
        tk.Label(parent, text=label).grid(row=row, column=column, padx=4, sticky="w")
        tk.Spinbox(
            parent,
            textvariable=variable,
            from_=low,
            to=high,
            increment=step,
            width=9,
        ).grid(row=row + 1, column=column, padx=3, sticky="w", pady=(0, 5))

    # pick random grid
    def run(self, positions=None):
        num_neurons = self.num_neurons_var.get()
        
        # If no positions are passed in, generate new random ones
        if positions is None:
            positions = [(int(self.rng.integers(0, 5)), int(self.rng.integers(0, 5))) for _ in range(num_neurons)]
            
        # Save the positions so the "Same Axon" button can reuse them
        self.current_positions = positions

        param = {
        "Fs": self.Fs_var.get(),
        "t": self.duration_var.get(),
        "num_neurons": self.num_neurons_var.get(),
        "attenuation":self.attenuation_var.get(),
        "section_distance_mm":self.section_distance_var.get(),
        "noise_level": self.noise_level_var.get()}

        neu_param = {
        "amp": [var.get() for var in self.amplitude_vars],
        "velocity": [var.get() for var in self.velocity_vars],
        "rate": [var.get() for var in self.rate_vars],}

        # run simulation with current values
        time, signals, axons, electrodes, delay, all_spikes = simulate(
            positions=positions, 
            parameter=param, 
            neu_parameter=neu_param
            )

        self.ax_grid.clear()
        self.ax_section1.clear()
        self.ax_section2.clear()
        self.ax_raster.clear()

        # draw the 5x5 position grid.
        for i in range(5):
            p = (i - 2) * grid_spacing_mm
            self.ax_grid.axvline(p, color="lightgray", linewidth=0.7)
            self.ax_grid.axhline(p, color="lightgray", linewidth=0.7)

        # nerve outline
        self.ax_grid.add_patch(
            Circle((0.0, 0.0), nerve_radius_mm,
                facecolor="none", edgecolor="#a33",
                linewidth=1.2, zorder=1,))
            
        # electrode color
        for i, (px, py) in enumerate(electrodes):
            self.ax_grid.plot(
                px, py, "s", ms=6,
                mfc="tab:blue", mec="#1a365d", mew=0.8, zorder=3,)

            # electrode label
            r = np.hypot(px, py)
            lx = px + px / r * 0.18
            ly = py + py / r * 0.18
            self.ax_grid.text(
                lx, ly, f"E{i + 1}",
                ha="center", va="center",
                fontsize=5, color="#1a365d",)

        # axon plot
        for axon_pos in axons:
            self.ax_grid.plot(axon_pos[0], axon_pos[1], "o", ms=5, color="#a33", zorder=4)
        # plot
        self.ax_grid.set_aspect("equal")
        self.ax_grid.set_xlim(-axis_limit, axis_limit)
        self.ax_grid.set_ylim(-axis_limit, axis_limit)
        self.ax_grid.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        self.ax_grid.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        self.ax_grid.tick_params(axis="x", labelsize=3.5, pad=1)
        self.ax_grid.tick_params(axis="y", labelsize=4.2, pad=1)
        self.ax_grid.set_xlabel("x (mm)", size=6)
        self.ax_grid.set_ylabel("y (mm)", size=6)
        self.ax_grid.set_title("Nerve cross-section", size=7)

        
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        section_axes = [self.ax_section1, self.ax_section2]

        # electrode plot
        for section, ax in enumerate(section_axes):
            for electrode in range(4):
                ax.plot(
                    time,
                    signals[section, electrode],
                    color=colors[electrode],
                    linewidth=0.8,
                    label=f"E{electrode + 1}",)

            ax.set_ylabel("Signal (µV)", size=7)
            ax.set_title(f"Cross-section {section + 1}", size=7)
            ax.grid(True, color="0.92", linewidth=0.5)
            ax.axhline(0, color="0.75", linewidth=0.5)
            ax.legend(loc="upper right", ncol=4, framealpha=0.9, fontsize=4, handlelength=1.0, columnspacing=0.8,
                borderpad=0.3, labelspacing=0.3)

        self.ax_section2.set_xlabel("Time", size=6)

        #Raster Plot
        for i, spikes in enumerate(all_spikes):
            # Plot a horizontal row of dots for each neuron (i+1 is the y-value)
            self.ax_raster.plot(spikes, np.full_like(spikes, i + 1), 'k.', markersize=4)

        self.ax_raster.set_ylabel("Neuron #", size=7)
        self.ax_raster.set_yticks(range(1, len(all_spikes) + 1))
        self.ax_raster.set_ylim(0.5, len(all_spikes) + 0.5)
        self.ax_raster.set_xlabel("Time (s)", size=7)
        self.ax_raster.set_title("Spike Raster", size=7)
        self.ax_raster.grid(True, color="0.92", linewidth=0.5)

        # Get Fs to calculate the delay in milliseconds
        Fs = self.Fs_var.get()
        num_neurons = self.num_neurons_var.get()
        
        if num_neurons == 1:
            pos_str = f"row {positions[0][0] + 1}, col {positions[0][1] + 1}"
        else:
            pos_str = f"{num_neurons} axons"

        self.root.title(
            f"Single-Axon Cross-Section Simulator  |  "
            f"{pos_str}  |  "
            f"section delay {(delay / Fs) * 1000:.3f} ms")
        
        self.canvas.draw()

    def run_same_axon(self):          
        if self.current_positions is not None:
            self.run(positions=self.current_positions)
        else:
            self.run()


if __name__ == "__main__":
    root = tk.Tk()
    app = CrossSectionWidget(root)
    root.mainloop()
