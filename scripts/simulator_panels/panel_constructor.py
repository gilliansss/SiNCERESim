import tkinter as tk
from tkinter import ttk, font

class Panel():
    """
    Panel for simulation parameters, shared between all types of simulation
    """

    def __init__(
            self,
            frame_parent:ttk.Frame,
            entries:dict[tuple[str,str,]], 
            panel_name:str="Panel",
            on_change = None,
        ):
        self.on_change = on_change  

        self.panel = ttk.LabelFrame(frame_parent, padding=6)

        self.entry_map = dict()
        self.entry_params = entries

        ttk.Label(self.panel, text=panel_name, 
                  font=("Helvetica", 12, "bold"))\
            .grid(column = 0, row = 0, pady = 0)

        self.__construct_parameters()

    # --- public methods --- #
    def get(self, key:str=None) -> dict | int | float | bool | str:
            """Gets Entry Mapping
            Args:
                key (str, optional): Entry key, if not given returns all entries. Defaults to None.
    
            Returns:
                if not given a key, returns a dictionary of all entries,
                if given a key, returns the value of that entry.
            """
            if key is None:
                return {k:self.__get_param_entry(*v) for k,v in self.entry_map.items()}
            if key in self.entry_map:
                return self.__get_param_entry(*self.entry_map[key])
            return None

    # --- private methods --- #
    def __notify(self, *_):
        """immediately push out new entry states to whatever on change is
        """
        if self.on_change is not None:
            self.on_change(self.get())

    def __construct_parameters(self):
        for (map_label, map_construction) in self.entry_params.items():
            self.__construct_entry((map_label, *map_construction))

    def __construct_entry(self,
            entry_info: tuple[str, int] | tuple[str, int, list],
        ):
        current_row = len(self.entry_map) + 1

        param_label_mapping = entry_info[0]
        param_type = entry_info[1]
        param_label_value = entry_info[2]
        param_default_value = entry_info[-1]

        param_label = ttk.Label(self.panel, text=param_label_value)
        
        if len(entry_info) > 4:
            var = tk.StringVar(value=param_default_value)
            param_field = ttk.Combobox(
                self.panel, 
                width = 10,
                textvariable = var, 
                values = entry_info[3], 
                state = "readonly"
            )
        elif isinstance(param_default_value, bool):
            var = tk.BooleanVar(value = param_default_value)
            param_field = ttk.Checkbutton(self.panel, variable = var)
        else:
            var = tk.StringVar(value = param_default_value)
            param_field = ttk.Entry(
                self.panel, width = 10, textvariable = var
            )

        for idx, element in enumerate((param_label, param_field)):
            element.grid(
                row = current_row, column = idx, 
                padx = 10, pady = 2, sticky = "w"
            )

        var.trace_add("write", self.__notify)

        self.entry_map[param_label_mapping] = (param_type, var)

    @staticmethod
    def __get_param_entry(param_type:str, param_obj):
        converters = {'int':int, 'str':str, 'bool':bool, 'float':float}
        return converters[param_type](param_obj.get() and param_obj.get() or 0)

class ScrollableFrame(ttk.Frame):
    """
    A frame with a horizontal scrollbar. Add children to `.inner`.

    Pack children with side="left", fill="y" so they line up left-to-right and
    stretch vertically. The inner frame's width grows with its children, which
    drives the horizontal scrollbar.
    """
    def __init__(self, parent, height: int = 300, **kw):
        super().__init__(parent, **kw)

        # Explicit height so the canvas never collapses during initial layout.
        self.canvas = tk.Canvas(self, highlightthickness=0, height=height)
        self.scrollbar = ttk.Scrollbar(
            self, orient="horizontal", command=self.canvas.xview
        )
        self.canvas.configure(xscrollcommand=self.scrollbar.set)

        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        # Reserve the scrollbar's strip first, then let the canvas take the rest.
        self.scrollbar.pack(side="bottom", fill="x")
        self.canvas.pack(side="top", fill="both", expand=True)

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # One global wheel binding; hit-test decides whether we act on it.
        toplevel = self.winfo_toplevel()
        toplevel.bind_all("<MouseWheel>", self._on_wheel, add="+")   # Win/mac
        toplevel.bind_all("<Button-4>",   self._on_wheel, add="+")   # Linux up
        toplevel.bind_all("<Button-5>",   self._on_wheel, add="+")   # Linux down

    # ---------- layout plumbing ----------

    def _on_inner_configure(self, _event):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        if event.height <= 1:
            return  # ignore initial junk before the widget is mapped
        # Pin inner's HEIGHT to canvas height; leave WIDTH free.
        self.canvas.itemconfigure(self._window_id, height=event.height)
        # Children may have re-flowed as a result.
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    # ---------- wheel ----------

    def _on_wheel(self, event):
        # Act only if the cursor is inside the canvas's visible bbox.
        px, py = self.canvas.winfo_pointerxy()
        cx = self.canvas.winfo_rootx()
        cy = self.canvas.winfo_rooty()
        if not (cx <= px < cx + self.canvas.winfo_width()
                and cy <= py < cy + self.canvas.winfo_height()):
            return

        if event.num == 4:                                          # Linux up
            delta = -1
        elif event.num == 5:                                        # Linux down
            delta = 1
        else:                                                       # Win/mac
            delta = -1 if event.delta > 0 else 1

        self.canvas.xview_scroll(delta, "units")
        return "break"   # stop other bindings from also reacting
