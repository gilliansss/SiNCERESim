import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import pandas as pd
import scipy.stats
from scipy.cluster.vq import kmeans2 # <-- NEW: For spatial clustering

class NeuronGeneratorWidget:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Population Neuron Generator")
        self.root.geometry("550x700")
        
        # --- Global Variables ---
        self.num_pops_var = tk.IntVar(value=1)
        self.rate_var = tk.DoubleVar(value=100.0)
        self.amp_var = tk.DoubleVar(value=1.0)
        
        self.correlate_var = tk.BooleanVar(value=False)
        self.cluster_var = tk.BooleanVar(value=False)
        
        self.populations = [] 
        
        # --- UI Setup ---
        top_frame = tk.LabelFrame(root, text="Global Settings", padx=10, pady=10)
        top_frame.pack(padx=20, pady=10, fill="x")
        
        tk.Label(top_frame, text="Number of Populations:").grid(row=0, column=0, sticky="w")
        tk.Spinbox(top_frame, textvariable=self.num_pops_var, from_=1, to=6, width=5, command=self._rebuild_population_ui).grid(row=0, column=1, padx=10, sticky="w")
        
        tk.Label(top_frame, text="Firing Rate (Hz):").grid(row=0, column=2, sticky="w", padx=(20, 0))
        tk.Spinbox(top_frame, textvariable=self.rate_var, from_=1, to=500, width=8).grid(row=0, column=3, padx=10, sticky="w")
        
        tk.Label(top_frame, text="Base Amplitude:").grid(row=0, column=4, sticky="w", padx=(20, 0))
        tk.Spinbox(top_frame, textvariable=self.amp_var, from_=0.1, to=10, increment=0.1, width=8).grid(row=0, column=5, padx=10, sticky="w")
        
        self.pop_frame = tk.LabelFrame(root, text="Population Parameters", padx=10, pady=10)
        self.pop_frame.pack(padx=20, pady=10, fill="x")
        
        bottom_frame = tk.Frame(root)
        bottom_frame.pack(padx=20, pady=10, fill="x")
        
        tk.Checkbutton(
            bottom_frame, text="Correlate Amplitude to Velocity", 
            variable=self.correlate_var, font=("Arial", 9)
        ).pack(anchor="w", pady=5)
        
        tk.Checkbutton(
            bottom_frame, text="Cluster spatially by Velocity", 
            variable=self.cluster_var, font=("Arial", 9)
        ).pack(anchor="w", pady=5)
        
        tk.Button(
            root, text="Generate & Save CSV", command=self.generate_csv
        ).pack(pady=20)
        
        self._rebuild_population_ui()

    def _rebuild_population_ui(self):
        for widget in self.pop_frame.winfo_children():
            widget.destroy()
            
        self.populations.clear()
        num_pops = self.num_pops_var.get()
        
        # Headers
        tk.Label(self.pop_frame, text="Pop", font=("Arial", 9, "bold")).grid(row=0, column=0)
        tk.Label(self.pop_frame, text="Count", font=("Arial", 9, "bold")).grid(row=0, column=1)
        tk.Label(self.pop_frame, text="Mean Vel (m/s)", font=("Arial", 9, "bold")).grid(row=0, column=2)
        tk.Label(self.pop_frame, text="SD Vel (m/s)", font=("Arial", 9, "bold")).grid(row=0, column=3)

        for i in range(num_pops):
            count_var = tk.IntVar(value=100)
            mean_var = tk.DoubleVar(value=40.0 - (i * 10)) # Default decreasing mean
            sd_var = tk.DoubleVar(value=5.0)
            
            self.populations.append({
                "count": count_var, "mean": mean_var, "sd": sd_var
            })
            
            tk.Label(self.pop_frame, text=f"#{i+1}").grid(row=i+1, column=0, pady=5)
            tk.Spinbox(self.pop_frame, textvariable=count_var, from_=0, to=5000, width=8).grid(row=i+1, column=1, padx=5)
            tk.Spinbox(self.pop_frame, textvariable=mean_var, from_=-120, to=120, width=8).grid(row=i+1, column=2, padx=5)
            tk.Spinbox(self.pop_frame, textvariable=sd_var, from_=0.1, to=50, width=8).grid(row=i+1, column=3, padx=5)

    def generate_csv(self):
        try:
            # 1. Grid constraints
            grid_size = 81
            half_grid = 40
            grid_spacing_mm = 0.01
            nerve_radius_mm = 0.4
            
            valid_positions = []
            for r in range(grid_size):
                for c in range(grid_size):
                    x = (c - half_grid) * grid_spacing_mm
                    y = (r - half_grid) * grid_spacing_mm
                    if np.hypot(x, y) <= nerve_radius_mm:
                        valid_positions.append((r, c))
            
            # 2. Generate Neurons from all populations
            all_neurons = []
            base_amp = self.amp_var.get()
            rate = self.rate_var.get()
            
            for pop in self.populations:
                n = pop["count"].get()
                if n <= 0: continue
                
                mean_v = pop["mean"].get()
                sd_v = pop["sd"].get()
                
                v_axis = np.arange(-120, 120.01, 0.01) 
                pdf = scipy.stats.norm.pdf(v_axis, loc=mean_v, scale=sd_v)
                probs = pdf / pdf.sum()
                sampled_v = np.random.choice(v_axis, size=n, replace=True, p=probs)
                
                for v in sampled_v:
                    if self.correlate_var.get():
                        safe_mean = max(abs(mean_v), 1e-5)
                        amp = base_amp * (np.abs(v) / safe_mean)
                        amp = np.clip(amp, 0.1, base_amp * 3)
                    else:
                        amp = base_amp
                        
                    all_neurons.append({"velocity": v, "amp": amp, "rate": rate})
                    
            total_neurons = len(all_neurons)
            if total_neurons == 0:
                messagebox.showwarning("Empty", "No neurons generated.")
                return
                
            if total_neurons > len(valid_positions):
                messagebox.showerror("Error", f"Cannot fit {total_neurons} neurons.\nOnly {len(valid_positions)} valid grid spots exist.")
                return
                
            # 3. Assign Positions (Clustered vs Random)
            if self.cluster_var.get():
                # --- NEW SPATIAL CLUSTERING LOGIC ---
                # Convert positions to float for K-Means
                pos_array = np.array(valid_positions, dtype=float)
                
                # Determine K clusters (based on number of active populations, min 2)
                active_pops = [p for p in self.populations if p["count"].get() > 0]
                num_clusters = max(2, len(active_pops))
                num_clusters = min(num_clusters, len(valid_positions))
                
                # Use K-Means to divide the nerve into K distinct spatial blobs
                centroids, labels = kmeans2(pos_array, num_clusters, minit='points')
                
                # Group positions by cluster label
                spatial_clusters = {i: [] for i in range(num_clusters)}
                for i, label in enumerate(labels):
                    spatial_clusters[label].append(valid_positions[i])
                    
                # Sort all neurons by velocity
                all_neurons.sort(key=lambda x: x["velocity"])
                
                # Distribute sorted neurons to the spatial clusters
                neuron_idx = 0
                for cluster_id in range(num_clusters):
                    positions = spatial_clusters[cluster_id]
                    np.random.shuffle(positions) # Shuffle within the blob
                    
                    capacity = len(positions)
                    count = min(capacity, total_neurons - neuron_idx)
                    
                    cluster_neurons = all_neurons[neuron_idx : neuron_idx + count]
                    neuron_idx += count
                    
                    for j, pos in enumerate(positions):
                        if j < count:
                            cluster_neurons[j]["row"], cluster_neurons[j]["col"] = pos
                            
                # Fallback (in case a few neurons got left out due to uneven cluster sizes)
                if neuron_idx < total_neurons:
                    remaining_pos = [pos for pos in valid_positions if pos not in [ (n["row"], n["col"]) for n in all_neurons if "row" in n ]]
                    np.random.shuffle(remaining_pos)
                    for k in range(neuron_idx, total_neurons):
                        all_neurons[k]["row"], all_neurons[k]["col"] = remaining_pos.pop()
            else:
                # Random assignment
                np.random.shuffle(valid_positions)
                for i in range(total_neurons):
                    all_neurons[i]["row"], all_neurons[i]["col"] = valid_positions[i]
                    
            # 4. Save
            df = pd.DataFrame(all_neurons)
            df["velocity"] = df["velocity"].round(2)
            df["amp"] = df["amp"].round(2)
            
            filepath = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv")],
                title="Save Neuron Parameters",
                initialfile=f"neurons_{total_neurons}.csv"
            )
            
            if filepath:
                df = df[["row", "col", "amp", "velocity", "rate"]]
                df.to_csv(filepath, index=False)
                messagebox.showinfo("Success", f"Generated {total_neurons} neurons and saved to:\n{filepath}")
                
        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = NeuronGeneratorWidget(root)
    root.mainloop()