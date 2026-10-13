from math import e, nan
from pydoc import text
import tkinter as tk
from tkinter import Button, ttk
from tkinter import font
from turtle import speed
from scripts.simulators.SingleNeuron import *
from scripts.simulators.PopulationNeuron import *

# SHARED PARAMETER
SPIKE_TYPES = [
    "Gaussian-modulated sine model",
    "Double exponential model",
    "Sum-of-gaussians model"
]
DEFAULT_SPIKE_TYPE = SPIKE_TYPES[0]

# ROOT WIDGET
root = tk.Tk()
root.title("Nerve Signal Simulator")
root.geometry("1000x450")

header_font = font.Font(family="Helvetica", size=10, weight="bold")

root_frame = ttk.Frame(root, padding = 10)

notebook = ttk.Notebook(root_frame)
notebook.pack(fill="both",expand=True, padx=10, pady=10)

# SHARED PARAMETERs
param = ttk.LabelFrame(root_frame, padding=10)\
    .grid(column=0, row=0, sticky="new", padx=(0,10))

ttk.Label(param, text="Simulation Parameter", font=header_font)\
    .grid(column=0, row=0,pady=0, sticky="w")

def entry(row, label, default):
    ttk.Label(param, text=label)\
        .grid(column=0, row=row, sticky="w", padx=(0,75))
    
    var = tk.StringVar(value=default)

    ttk.Entry(param, width=5,textvariable=var )\
        .grid(column=1, row=row, sticky="e")

    return var
 
 #Generate the entry fields for simulation parameter
Fs_var = entry( 1, "Fs (Hz)","30000")
d_var = entry(2, "d(m)","0.002")
t_var = entry(3, "t(s)","10")
lowpass_fill_var = entry(4,"Lowpass Filter (Hz)","2000")
num_neurons_var = entry(5,"Neurons #", "2")
Intpol_var = entry(6,"Interpolation","1")

#Save the entry into the variables
Fs=int(Fs_var.get())
d=float(d_var.get())
t=float(t_var.get())
lowpass_fill= int(lowpass_fill_var.get())
num_neurons=int(num_neurons_var.get())
Intpol=int(Intpol_var.get())

ttk.Label(param, text="Spike type")\
    .grid(column=0, row=7, sticky="w", padx=(0,75)) 

spike_def= tk.StringVar(value=DEFAULT_SPIKE_TYPE) #Create a default option for spike types
spike_type= ttk.Combobox(
    param, 
    textvariable=spike_def, 
    values=SPIKE_TYPES,
    state="readonly")
spike_type.grid(column=1, row=7, sticky="e")

#Store the value into a dictionary
param = {
    "Fs": Fs,
    "t":t,
    "d":d,
    "lowpass_fill":lowpass_fill,
    "num_neurons":num_neurons,
    "Intpol":Intpol,
    "spike_type":spike_type
}

#Single Neuron Simulation
sns = ttk.Frame(notebook, padding=10)
notebook.add(sns, text="Single Neuron Simulation")
sns.columnconfigure(0, weight=0, minsize=300)
sns.columnconfigure(1,weight=1)



#Parameters fro idividual neurons
neu_param= ttk.LabelFrame(sns, padding=10)
neu_param.grid(column=1, row=0,sticky="new")
ttk.Label(
    neu_param, 
    text="Neuron Parameter", 
    font=header_font)\
        .grid(column=0, row=0,pady=0, sticky="w")

#Initialize dictionaries for data storing
datas = {
    "speed":[10]*10,
    "rate":[20]*10,
    "on":[1]*10,
    "off":[9]*10,
    "amp":[2]*10,
    "mod_type":[None]
}
vars = {
    "speed":[10]*10,
    "rate":[20]*10,
    "on":[1]*10,
    "off":[9]*10,
    "amp":[2]*10
}

neu_labels=["speed","rate","on","off","amp"]
for x in range(len(neu_labels)):
    ttk.Label(neu_param, text=neu_labels[x])\
            .grid(column=0, row=x+1, sticky="w", padx=0)

def neuronp (num_neuron,datas):
    datas.pop("mod_type")

    for x in range(len(neu_labels)):
        ttk.Label(neu_param, text=neu_labels[x])\
            .grid(column=0, row=x+1, sticky="w", padx=0)
        
    for x in range(num_neuron):    #(x+1)-th Neuron
        ttk.Label(neu_param, text=f"n{x+1}")\
            .grid(column=x+1, row=0, sticky="")

        for y,var in enumerate(datas):  #y is index, var is the key
            default = tk.StringVar(value=datas[var][x]) #Set the default from datas dictionary, var (key), and x-th index of the value list

            ttk.Entry(neu_param, textvariable=default, width=5)\
                .grid(column= x+1, row=y+1, sticky="")
            
            vars[var][x] = default #Store the entry widget in the dictionary at key : variable, x-th object of the value list

    ttk.Label(neu_param, text="Trace mod")\
        .grid(column=0, row=7, sticky="w")

    mod_def = tk.StringVar(value='None')
    mod_type = ttk.Combobox(
        neu_param, 
        textvariable=mod_def,
        values=["None","Absolute","Rectified","Square"],
        state="readonly", 
        width=4
    )
    mod_type.grid(column=1, row=7, sticky="e")
    datas["mod_type"]=mod_def.get()

neuronp(num_neurons,datas)

def run1(*args):
    for var in vars:
        for x in range(num_neurons):
            datas[var][x]=int(vars[var][x].get())
    run_single(param, datas)

#Run Button
Run= ttk.Button(sns, text="Run Simulation", command=run1)
Run.grid(column=0, row= 8, pady=(5,0),sticky="w")

#Generate fields for neurons parameter for # Neurons
def rebuild(*args):
    for w in neu_param.winfo_children():
        w.destroy()

    ttk.Label(
        neu_param, text="Neuron Parameter", font=header_font)\
            .grid(column=0, row=0,pady=0, sticky="w")
    
    neuronp(int(num_neurons_var.get()),datas)

num_neurons_var.trace_add("write",rebuild) #Detect changes in the entry field

#Neuron Population Simulation
nps= ttk.Frame(notebook, padding=10)
notebook.add(nps, text="Neuron Population Simulation")
nps.columnconfigure(0, weight=0, minsize=300)
nps.columnconfigure(1,weight=1)

#Simulation Parameter (Population)
nps_param= ttk.LabelFrame(nps, padding=10)
nps_param.grid(column=0, row=0, sticky="new", padx=(0,10))

ttk.Label(
    nps_param, text="Simulation Parameter", font=header_font)\
        .grid(column=0, row=0,pady=0, sticky="w")

def nps_entry(row, label, default):
    ttk.Label(nps_param, text=label)\
        .grid(column=0, row=row, sticky="w", padx=(0,75))
    
    var = tk.StringVar(value=default)

    ttk.Entry(nps_param, width=5,textvariable=var )\
        .grid(column=1, row=row, sticky="e")

    return var

#Generate the entry fields for population simulation parameter
nps_Fs_var=nps_entry(1, "Fs (Hz)","30000")
nps_d_var=nps_entry(2, "d (m)","0.002")
nps_t_var=nps_entry(3, "t (s)","10")
nps_lowpass_var=nps_entry(4, "Lowpass (Hz)","2000")
nps_num_neurons_var=nps_entry(5, "Neuron #","2")
nps_Intpol_var=nps_entry(6, "Interpolation","1")

#Store initial values
nps_Fs=int(nps_Fs_var.get())
nps_d=float(nps_d_var.get())
nps_t=float(nps_t_var.get())
nps_lowpass=int(nps_lowpass_var.get())
nps_num_neurons=int(nps_num_neurons_var.get())
nps_Intpol=int(nps_Intpol_var.get())

#Spike type for population
ttk.Label(nps_param, text="Spike type")\
    .grid(column=0, row=7, sticky="w", padx=(0,75))

nps_spike_def= tk.StringVar(value=DEFAULT_SPIKE_TYPE)
nps_spike_type= ttk.Combobox(
    nps_param, 
    textvariable=nps_spike_def,                          
    values=SPIKE_TYPES,
    state="readonly"
)
nps_spike_type.grid(column=1, row=7, sticky="e")

#Store population parameters into a dictionary
nps_params={
    "Fs": nps_Fs,
    "t": nps_t,
    "d": nps_d,
    "lowpass_fill": nps_lowpass,
    "num_neurons": nps_num_neurons,
    "Intpol": nps_Intpol,
    "spike_type": nps_spike_type
}

# Fiber Type Table (right side)
fiber_frame = ttk.LabelFrame(nps, padding=10)
fiber_frame.grid(column=1, row=0, sticky="new")

ttk.Label(fiber_frame, text="Neuron Parameter", font=header_font).grid(
    column=0, row=0, columnspan=8, pady=0, sticky="w"
)

# Column headers — same order as the keys below
fiber_headers = ["Fiber #", "Mean v", "SD v", "Amp fact", "Rate", "t on", "t off"]

for i, h in enumerate(fiber_headers):
    ttk.Label(fiber_frame, text=h).grid(column=i+1, row=1, padx=2, sticky="w")

# Fiber type rows
fiber_types = ["Aa motor", "Ay motor", "Aa sensory", "Aβ sensory", "Aδ sensory", "C sensory"]

# Default data per fiber type
fiber_defaults = {
    "fiber_n":    [0, 0, 0, 0, 0, 0],
    "mean_speed": [50, 20, -58, -55, -17, -1],
    "SD_speed":   [8, 5, 3, 5, 5, 0.5],
    "amp_fact":   [3, 2, 3, 2, 1, 0.5],
    "rate":       [50, 20, 100, 100, 50, 10],
    "on":         [1, 1, 1, 1, 1, 1],
    "off":        [9, 9, 9, 9, 9, 9],
}

# Data storage dicts (same style as the single neuron tab)
fiber_datas = {k: list(v) for k, v in fiber_defaults.items()}
fiber_vars = {k: [None] * len(v) for k, v in fiber_defaults.items()}

# Build the table
for row_idx, fname in enumerate(fiber_types):
    # Fiber name label in column 0
    ttk.Label(fiber_frame, text=fname).grid(column=0, row=row_idx + 2, sticky="w", padx=2)
    # Entry fields for each attribute in columns 1..7
    for col_idx, key in enumerate(fiber_defaults.keys()):
        default = tk.StringVar(value=str(fiber_datas[key][row_idx]))
        ttk.Entry(fiber_frame, textvariable=default, width=5).grid(
            column=col_idx + 1, row=row_idx + 2, padx=2
        )
        fiber_vars[key][row_idx] = default


# Run Button
def run_population(*args):
    # Refresh simulation params from the entry widgets
    nps_params["Fs"]           = int(nps_Fs_var.get())
    nps_params["t"]            = float(nps_t_var.get())
    nps_params["d"]            = float(nps_d_var.get())
    nps_params["lowpass_fill"] = int(nps_lowpass_var.get())
    nps_params["num_neurons"]  = int(nps_num_neurons_var.get())
    nps_params["Intpol"]       = int(nps_Intpol_var.get())
    nps_params["spike_type"]   = nps_spike_def.get()

    # Read the fiber table back into fiber_datas
    for key in fiber_vars:
        for i in range(len(fiber_types)):
            try:
                fiber_datas[key][i] = float(fiber_vars[key][i].get())
            except (ValueError, TypeError):
                fiber_datas[key][i] = 0.0

    #Convert dicts of list to DataFrame
    fiber_df = pd.DataFrame(fiber_datas)
    fiber_df["label"] = fiber_types
    fiber_df["fiber_n"] = fiber_df["fiber_n"].astype(int)


    run_pop(nps_params, fiber_df)

Run = ttk.Button(nps, text="Run Simulation", command=run_population)
Run.grid(column=0, row=1, pady=(5,0), sticky="w")

root.mainloop()