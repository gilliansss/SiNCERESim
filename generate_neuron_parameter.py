import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import scipy.stats

class NeuronGeneratorWidget:
    def __init__(self, root):
        self.root = root
        self.root.title("Neuron Parameter Generator")
        self.root.geometry("400x450")
        
        # Variable
        self.num_neurons_var = tk.IntVar(value=100)
        self.mean_speed_var = tk.DoubleVar(value=40.0)
        self.sd_speed_var = tk.DoubleVar(value=5.0)
        self.rate_var = tk.DoubleVar(value=100.0)
        self.amp_var = tk.DoubleVar(value=1.0)
        
        self.correlate_amp_vel_var = tk.BooleanVar(value=False)
        
        
        self._updating_link = False
        self.amp_vel_ratio = 40.0  #Placeholder to be changed to the actual formula

        frame = tk.LabelFrame(root, text="Population Parameters", padx=10, pady=10)
        frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Create rows
        def add_row(row, label, var, min_val, max_val, step):
            tk.Label(frame, text=label).grid(row=row, column=0, sticky="w", pady=5)
            tk.Spinbox(
                frame, 
                textvariable=var, 
                from_=min_val, 
                to=max_val, 
                increment=step, 
                width=12
            ).grid(row=row, column=1, padx=10, pady=5)

        add_row(0, "Number of Neurons:", self.num_neurons_var, 1, 5000, 10)
        add_row(1, "Mean Velocity (m/s):", self.mean_speed_var, -120, 120, 1)
        add_row(2, "SD Velocity (m/s):", self.sd_speed_var, 0.1, 50, 1)
        add_row(3, "Firing Rate (Hz):", self.rate_var, 1, 500, 10)
        add_row(4, "Amplitude (µV):", self.amp_var, 0.1, 10, 0.1)
        
        #Correlate amplitude and velocity toggle
        tk.Checkbutton(
            frame, 
            text="Correlate Amplitude to Velocity", 
            variable=self.correlate_amp_vel_var, 
            font=("Arial", 9, "italic"),
            command=self._sync_all  
        ).grid(row=5, column=0, columnspan=2, pady=15, sticky="w")

        self.mean_speed_var.trace_add("write", self._on_vel_change)
        self.amp_var.trace_add("write", self._on_amp_change)

        # Run button
        tk.Button(
            root, 
            text="Generate & Save CSV", 
            command=self.generate_csv
        ).pack(pady=20, fill="x", padx=20)

    def _on_amp_change(self, *args):
        if not self.correlate_amp_vel_var.get() or self._updating_link:
            return
        self._updating_link = True
        try:
            amp = self.amp_var.get()
            self.mean_speed_var.set(amp * self.amp_vel_ratio)
        finally:
            self._updating_link = False
    def _on_vel_change(self, *args):
        if not self.correlate_amp_vel_var.get() or self._updating_link:
            return
        self._updating_link = True
        try:
            vel = self.mean_speed_var.get()
            self.amp_var.set(vel / self.amp_vel_ratio)
        finally:
            self._updating_link = False
    def _sync_all(self):
        # adjust values to the ratio when the checkbox is clicked
        if self.correlate_amp_vel_var.get():
            self._on_vel_change()

    def generate_csv(self):
        # Get values
        num_neurons = self.num_neurons_var.get()
        mean_speed = self.mean_speed_var.get()
        sd_speed = self.sd_speed_var.get()
        rate = self.rate_var.get()
        base_amp = self.amp_var.get()
        
        # Grid constraints
        grid_size = 81
        half_grid = 40
        grid_spacing_mm = 0.01
        nerve_radius_mm = 0.4
        
        # Randomized valid positions
        valid_positions = []
        for r in range(grid_size):
            for c in range(grid_size):
                x = (c - half_grid) * grid_spacing_mm
                y = (r - half_grid) * grid_spacing_mm
                if np.hypot(x, y) <= nerve_radius_mm:
                    valid_positions.append((r, c))
                    
        if num_neurons > len(valid_positions):
            messagebox.showerror("Error", f"Cannot fit {num_neurons} neurons.\nOnly {len(valid_positions)} valid grid spots exist inside the nerve.")
            return
            
        #Generate velocity distribution
        v_axis = np.arange(-120, 120.01, 0.01) 
        fiber_pdf = scipy.stats.norm.pdf(v_axis, loc=mean_speed, scale=sd_speed)
        probs_weights = fiber_pdf / fiber_pdf.sum()
        
        #Sample velocities & positions
        sampled_v = np.random.choice(v_axis, size=num_neurons, replace=True, p=probs_weights)
        chosen_positions = np.random.choice(len(valid_positions), size=num_neurons, replace=False)

        if self.correlate_amp_vel_var.get():
            # Scale amplitude by the ratio of the sampled velocity to the mean velocity
            safe_mean = max(abs(mean_speed), 1e-5)
            sampled_amp = base_amp * (np.abs(sampled_v) / safe_mean) #Placeholder for the relatinoship between velocity and amplitude
            sampled_amp = np.clip(sampled_amp, 0.1, base_amp * 3)
        else:
            # Independent: all amplitudes are just the base value
            sampled_amp = np.full(num_neurons, base_amp)
        
        #Build DataFrame
        all_neurons = []
        for i in range(num_neurons):
            r, c = valid_positions[chosen_positions[i]]
            neuron_data = {
                "row": r,
                "col": c,
                "amp": base_amp,
                "velocity": round(sampled_v[i], 2),
                "rate": rate
            }
            all_neurons.append(neuron_data)
            
        df_out = pd.DataFrame(all_neurons)
        
        #Save as csv
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            title="Save Neuron Parameters",
            initialfile=f"neurons_{num_neurons}.csv"
        )
        
        if filepath:
            df_out.to_csv(filepath, index=False)
            messagebox.showinfo("Success", f"Generated {num_neurons} neurons and saved to:\n{filepath}")


if __name__ == "__main__":
    root = tk.Tk()
    app = NeuronGeneratorWidget(root)
    root.mainloop()
    




