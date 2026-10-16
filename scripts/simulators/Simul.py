import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle

# take the raw waveforms from singleneuron-2.py, doesnt include the filtering interpolation etc
from waveforms import get_spike_waveform

Fs = 30000 # Hz
record_duration_ms = 10
spike_duration = 0.002
firing_time_ms = 3
grid_spacing_mm = 0.1
nerve_radius_mm = 0.4
axis_limit = 0.75
    
def simulate(row, column, amplitude, velocity, section_distance_mm, noise_level):
    axon_x = (column - 2) * grid_spacing_mm
    axon_y = (row - 2) * grid_spacing_mm
    axon_position = np.array([axon_x, axon_y])

    angles = np.arange(4) * np.pi/2
    electrodes = nerve_radius_mm * np.column_stack((np.cos(angles), np.sin(angles)))

    distances = np.maximum(np.linalg.norm(electrodes - axon_position, axis = 1), 0.01) # minimum 0.01, avoid zero division
    weights = 1.0 /distances # closer electrode got larger weight
    weights /= np.max(weights) # normalize, closest electrode has weight of 1.0

    _, waveform = get_spike_waveform(Fs, spike_duration=spike_duration, kind="sum_gauss") # call waveforms.py

    n_samples = int(record_duration_ms * 1e-3 * Fs) + 1 # convert recording duration to samples
    time = np.arange(n_samples) / Fs # how long the duration for 1 sample

    section_delay = (section_distance_mm / 1000) / velocity
    starts = [
        int(round(firing_time_ms * 1e-3 * Fs)),
        int(round((firing_time_ms * 1e-3 + section_delay) * Fs)),]

    # noise
    rng = np.random.default_rng(row * 5 + column) # noise reproducible for (row, column), depends on axon position
    signals = noise_level * rng.standard_normal((2, 4, n_samples))

    for section in range(2):
        start, end = starts[section], starts[section] + len(waveform)

        if end > n_samples:
            continue
        for electrode in range(4):
             signals[section, electrode, start:end] += (amplitude * weights[electrode] * waveform)

    return time, signals, axon_position, electrodes, section_delay

class CrossSectionWidget:
    def __init__(self, root):
        self.root = root
        self.root.title("Single-Axon Cross-Section Simulator")

        self.root.geometry("950x850")
        
        self.amplitude_var = tk.DoubleVar(value=1.0)
        self.velocity_var = tk.DoubleVar(value=40.0)
        self.distance_var = tk.DoubleVar(value=10.0)
        self.noise_var = tk.DoubleVar(value=0.05)

        self.rng = np.random.default_rng()
        self.current_row = None
        self.current_column = None
         
        controls = tk.Frame(root)
        controls.pack(padx=8, pady=8)

        self.add_control(controls, "Amplitude (µV)", self.amplitude_var, 0.1, 5, 0.1, 0)
        self.add_control(controls, "Velocity (m/s)", self.velocity_var, 1, 120, 1, 2)
        self.add_control(controls, "Section spacing (mm)", self.distance_var, 1, 50, 1, 4)
        self.add_control(controls, "Noise level (µV RMS)", self.noise_var, 0, 1, 0.01, 6)

        tk.Button(controls, text="Run (new random axon)", command=self.run).grid(
            row=1, column=8, padx=8)
        tk.Button(controls, text="Run (same axon)", command=self.run_same_axon).grid(row=1, column=9, padx=8)

        self.figure, (self.ax_grid, self.ax_section1, self.ax_section2) = plt.subplots(3, 1, figsize=(9, 8), layout = "constrained", gridspec_kw = {"height_ratios":[1.3, 1.0, 1.0]},)
        
        
        self.canvas = FigureCanvasTkAgg(self.figure, master=root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True) # put figure to Tk window

        self.run()

    def add_control(self, parent, label, variable, low, high, step, column):
        tk.Label(parent, text=label).grid(row=0, column=column, padx=4, sticky="w")
        tk.Spinbox(
            parent,
            textvariable=variable,
            from_=low,
            to=high,
            increment=step,
            width=9,
        ).grid(row=1, column=column, padx=3, sticky="w")

    # pick random grid
    def run(self, row=None, column=None):
        if row is None:
            row = int(self.rng.integers(0, 5))
        if column is None:
            column = int(self.rng.integers(0, 5))

        self.current_row = row
        self.current_column = column

        # run simulation with current values
        time, signals, axon, electrodes, delay = simulate(
            row=row,
            column=column,
            amplitude=self.amplitude_var.get(),
            velocity=self.velocity_var.get(),
            section_distance_mm=self.distance_var.get(),
            noise_level=self.noise_var.get(),)

        self.ax_grid.clear()
        self.ax_section1.clear()
        self.ax_section2.clear()

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
        self.ax_grid.plot(axon[0], axon[1], "o", ms=5, color="#a33", zorder=4)

        # plot
        self.ax_grid.set_aspect("equal")
        self.ax_grid.set_xlim(-axis_limit, axis_limit)
        self.ax_grid.set_ylim(-axis_limit, axis_limit)
        self.ax_grid.set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        self.ax_grid.set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        self.ax_grid.tick_params(axis="x", labelsize=3.5, pad=1)
        self.ax_grid.tick_params(axis="y", labelsize=4.2, pad=1)
        self.ax_grid.set_xlabel("x (mm)", size=7)
        self.ax_grid.set_ylabel("y (mm)", size=7)
        self.ax_grid.set_title("Nerve cross-section", size=7)

        time_ms = time * 1000
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
        section_axes = [self.ax_section1, self.ax_section2]

        # electrode plot
        for section, ax in enumerate(section_axes):
            for electrode in range(4):
                ax.plot(
                    time_ms,
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

        self.ax_section2.set_xlabel("Time (ms)", size=7)

        self.root.title(
            f"Single-Axon Cross-Section Simulator  |  "
            f"row {row + 1}, col {column + 1}  |  "
            f"section delay {delay * 1000:.3f} ms")

        self.canvas.draw()

    def run_same_axon(self):          
        if self.current_row is None:
            self.run()
        else:
            self.run(row=self.current_row, column=self.current_column)


if __name__ == "__main__":
    root = tk.Tk()
    app = CrossSectionWidget(root)
    root.mainloop()
