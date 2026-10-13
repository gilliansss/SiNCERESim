import tkinter as tk

from tkinter import ttk
from scripts.simulator_panels.panel_constructor import Panel, ScrollableFrame

# ---  HELPERS  ---
HEADER_FONT = ("Helvetica", 12, "bold")

# SINGLE NEURON
MOD_TYPES = (
    "None",
    "Absolute",
    "Rectified",
    "Square"
)

DEFAULT_SINGLE_NEURON_PARAMS = {
    "sn_speed":("float", "Speed", 10),
    "sn_rate":("int", "Rate", 20),
    "sn_on":("int", "On", 1),
    "sn_off":("int", "Off", 9),
    "sn_amp":("float", "Amplitude", 2),
    "sn_mod_type":("str", "Trace Mod", MOD_TYPES, MOD_TYPES[0])
}

sn_params_panels : list[Panel] = []
sn_params : dict[str,dict] = {}

def make_sn_callback(neuron_key: str):
    """Closure so each panel writes into its own slot of `neuron_params`."""
    def callback(new_params):
        sn_params[neuron_key] = new_params
        print(neuron_key, new_params)
    return callback

# ENVIRONMENT PARAMETERS
SPIKE_TYPES = (
    "Gaussian-modulated sine model",
    "Double exponential model",
    "Sum-of-gaussians model"
)

DEFAULT_ENV_PARAMS = {
    "sample_freq": ("int", "Fs (Hz)", 30000),
    "electrode_dist": ("float", "d (mm)", 2),
    "recording_duration": ("float", "t (s)", 10),
    "lowpass_thresh": ("int", "Lowpass Filter (Hz)", 2000),
    "source_amt": ("int", "Neurons #", "02"),
    "do_interpolate": ("bool", "Interpolate", True),
    "spike_type": ("str", "Spike Type", SPIKE_TYPES, SPIKE_TYPES[0])
}

env_params = dict()
def on_env_params(new_params:dict):
    global env_params
    env_params = new_params
    # print(env_params) # for debugging

    n = env_params.get("source_amt")
    if isinstance(n, int) and n>=0:
        sync_neuron_panels(n)

# --- ROOT WIDGET ---
root = tk.Tk()
root.title("Nerve Signal Simulator")
root.geometry("1000x450")

root_frame = ttk.Frame(root, padding = 10)
root_frame.pack(side="top", fill="both")

root_frame.columnconfigure(0, weight=0, minsize=240)       # <-- keeps env from shrinking
root_frame.columnconfigure(1, weight=1)
root_frame.rowconfigure(0, weight=1)

notebook = ttk.Notebook(root_frame)
notebook.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

# ---  SHARED SIM PARAMETERS PANEL  ---
env_params_container = Panel(
    root_frame, DEFAULT_ENV_PARAMS, "Environment Parameters", on_env_params)
env_params_container.panel.grid(column=0, row=0, sticky="nw")

# ---  SINGLE NEURON PANEL  ---
single_neuron_panel = ttk.Frame(notebook, padding = 10)
notebook.add(single_neuron_panel, text = "Single Neuron Simulation")

sn_params_container = ScrollableFrame(single_neuron_panel, height=300)
sn_params_container.pack(fill="both", expand=True)

# management
def add_sn_panel():
    index = len(sn_params_panels)
    neuron_key = f"neuron_{index}"
    panel = Panel(
        sn_params_container.inner,
        DEFAULT_SINGLE_NEURON_PARAMS,
        f"Neuron {index + 1}",
        make_sn_callback(neuron_key),
    )
    panel.panel.pack(side="left",fill="y",padx=4)

    sn_params_panels.append(panel)
    sn_params[neuron_key] = panel.get()

def remove_sn_panel():
    sn_params_panels.pop().panel.destroy()
    sn_params.pop(f'neuron_{len(sn_params_panels)}')

def sync_neuron_panels(target_count: int):
    """
    Ensure exactly `target_count` neuron panels exist, adding/removing from the end.
    """
    current = len(sn_params_panels)
    if target_count > current:
        for _ in range(current, target_count):
            add_sn_panel()
    elif target_count < current:
        for _ in range(current - target_count):
            remove_sn_panel()

sync_neuron_panels(int(DEFAULT_ENV_PARAMS["source_amt"][-1]))

# ---  NEURON POPULATION PANEL  ---
population_neuron_panel = ttk.Frame(notebook, padding = 10)
notebook.add(population_neuron_panel, text = "Population Neuron Simulation")

# pn_params_container = ttk.Frame(population_neuron_panel)
# pn_params_container.grid(column=1, row=0, sticky="new")
# ttk.Label(pn_params_container, text="Neuron Parameters", font=HEADER_FONT)\
#     .grid(column=0, row=0, pady=0, sticky="w")


root.mainloop()