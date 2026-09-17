"""
CIELab a*–b* Plotter
====================
A desktop tool to plot experimental CIELab colour-measurement data on top of the
true sRGB colour gamut at a chosen lightness L*.

Run in PyCharm (green ▶) or:   python cielab.py

Requirements (install once in PyCharm's terminal):
    pip install matplotlib numpy

tkinter ships with standard Python on Windows/macOS.
On Linux:  sudo apt install python3-tk

Workflow
--------
1. Paste your measurement table into the CSV/TSV box (or use "From file").
   Accepted columns (tab- OR semicolon- OR comma-separated, header row optional):
       Composition, Phase, L*, a*, b*
   Example (copied straight from a spreadsheet):
       Composition (mass%)	Phase   L*      a*      b*	    Ref
        Au			        Au	    86	    4.7	    36.9	[1]
        Ag			        Ag	    92.65	-0.31	5.05	[6]
        Cu			        Cu	    87.2	13.4	14.9	[9]
2. Click Import.  Each unique Composition automatically gets its own
   colour + marker, and appears once in the legend.
3. Adjust the L* background slider and the a*/b* axis ranges as needed.
4. Save PNG  →  clean white publication-style figure for slides/papers.
   Save / Load Session  →  keep working later.
"""

"""CIELab a*–b* Plotter — plot experimental CIELab colour-measurement data
on the true sRGB colour gamut at a chosen lightness L*.
Copyright (C) 2026  Ulrich E. Klotz,
Hochschule München University of Applied Sciences
 
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
 
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
 
You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>."""

import tkinter as tk
from tkinter import ttk, colorchooser, filedialog, messagebox
import json
import itertools
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

# ════════════════════════════════════════════════════════════════════════════
# Theme (UI only — the plot itself is publication-styled separately)
# ════════════════════════════════════════════════════════════════════════════

BG       = "#0e0e18"
SURFACE  = "#16162a"
SURFACE2 = "#1e1e35"
BORDER   = "#2e2e50"
ACCENT   = "#e9c46a"
ACCENT2  = "#2a9d8f"
DANGER   = "#e63946"
TEXT     = "#e8e8f0"
MUTED    = "#7a7aa5"

# Marker pool — cycled per unique composition (all are 'filled' markers
# so they always show an edge cleanly)
MARKER_POOL = ["o", "s", "^", "D", "v", "p", "h", "<", ">", "*"]

# Colour palette — cycled per unique composition
PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
    "#dcbeff", "#9a6324", "#800000", "#aaffc3", "#808000",
    "#000075", "#a9a9a9", "#ffe119", "#000000", "#e6beff",
]

# ════════════════════════════════════════════════════════════════════════════
# CIELab → sRGB colour math   (D65 illuminant)
# ════════════════════════════════════════════════════════════════════════════

_M_XYZ2RGB = np.array([
    [ 3.2404542, -1.5371385, -0.4985314],
    [-0.9692660,  1.8760108,  0.0415560],
    [ 0.0556434, -0.2040259,  1.0572252],
])
_D65 = np.array([0.95047, 1.00000, 1.08883])


def _f_inv(t):
    delta = 6 / 29
    return np.where(t > delta, t**3, 3 * delta**2 * (t - 4 / 29))


def lab_to_xyz(L, a, b):
    L  = np.asarray(L, float) * np.ones_like(a)
    fy = (L + 16) / 116
    fx = np.asarray(a, float) / 500 + fy
    fz = fy - np.asarray(b, float) / 200
    return _D65[0] * _f_inv(fx), _D65[1] * _f_inv(fy), _D65[2] * _f_inv(fz)


def xyz_to_srgb(X, Y, Z):
    rgb = np.einsum("ij,...j->...i", _M_XYZ2RGB, np.stack([X, Y, Z], axis=-1))
    c   = np.clip(rgb, 0, None)
    return np.clip(np.where(c <= 0.0031308, 12.92 * c,
                            1.055 * c**(1 / 2.4) - 0.055), 0, 1)


def build_background(L, a0, a1, b0, b1, resolution=400, oog_grey=0.93):
    """
    Return an RGB image of the sRGB gamut in the a*-b* plane at lightness L.

    Row 0 corresponds to b = b0 (minimum).  Pair this with imshow(origin="lower")
    so that b* increases UPWARD — i.e. +b* (yellow) at top, -b* (blue) at bottom.
    Out-of-gamut pixels are filled with a neutral grey.
    """
    a = np.linspace(a0, a1, resolution)
    b = np.linspace(b0, b1, resolution)          # ascending
    aa, bb = np.meshgrid(a, b)                    # row index follows b ascending

    X, Y, Z  = lab_to_xyz(L, aa, bb)
    lin_rgb  = np.einsum("ij,...j->...i", _M_XYZ2RGB, np.stack([X, Y, Z], axis=-1))
    in_gamut = np.all((lin_rgb >= -0.001) & (lin_rgb <= 1.001), axis=-1)

    img = xyz_to_srgb(X, Y, Z)
    img[~in_gamut] = oog_grey
    return img                                    # NO flipud — origin="lower" handles it


def scatter_point(ax, x, y, colour, marker, size, edge_colour, edge_width):
    """Scatter one point; omit edgecolor for unfilled markers to avoid warnings."""
    if MarkerStyle(marker).is_filled():
        ax.scatter(x, y, c=colour, marker=marker, s=size,
                   edgecolors=edge_colour, linewidths=edge_width, zorder=5)
    else:
        ax.scatter(x, y, c=colour, marker=marker, s=size,
                   linewidths=edge_width + 0.5, zorder=5)


# ════════════════════════════════════════════════════════════════════════════
# Application
# ════════════════════════════════════════════════════════════════════════════

class CIELabPlotter:

    def __init__(self, root):
        self.root = root
        root.title("CIELab a*–b* Plotter")
        root.configure(bg=BG)
        root.minsize(1150, 720)

        # ── data model ──────────────────────────────────────────────────
        # points: list of dicts {composition, phase, Lpoint, a, b}
        # style:  composition → {"colour":..., "marker":...}
        self.points = []
        self.style  = {}
        self.selected_idx = None
        self.selected_idxs = []
        self._bg_cache = {}
        self._palette_cycle = itertools.cycle(PALETTE)
        self._marker_cycle  = itertools.cycle(MARKER_POOL)

        # Rendering / blitting state
        self._bg_region   = None    # cached static background (copy_from_bbox)
        self._dynamic_artists = []  # points + labels + guide lines drawn on top
        self._label_artists = {}    # point index → its label Text artist
        self._needs_full_draw = True

        # Drag state
        self._drag_label_idx = None
        self._drag_start = None     # (mouse_a, mouse_b, base_dx, base_dy)

        self._build_ui()
        self._load_demo()
        self._full_redraw()

    # ── demo data so the window isn't empty on first launch ─────────────
    def _load_demo(self):
        demo = [
            ("Pt 100",          "Pt",      89.12, -0.13, 0.54),
            ("Pt 87.8 Al 12.2", "PtAl",    82.93,  1.86, 2.28),
            ("Pt 83 Al 17",     "Pt2Al3",  78.34, -1.41, 1.93),
        ]
        for comp, phase, L, a, b in demo:
            self._ensure_style(comp)
            self.points.append({"composition": comp, "phase": phase,
                                 "Lpoint": L, "a": a, "b": b})

    # ════════════════════════════════════════════════════════════════════
    # UI
    # ════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        # ── toolbar ──────────────────────────────────────────────────────
        bar = tk.Frame(self.root, bg=SURFACE, pady=7, padx=12)
        bar.pack(fill="x")
        tk.Label(bar, text="CIELab  a*–b*  Plotter", bg=SURFACE, fg=TEXT,
                 font=("Helvetica", 14, "bold")).pack(side="left", padx=(0, 18))
        for label, cmd, c_bg, c_fg in [
            ("Save PNG",     self._save_png,     ACCENT2,  "white"),
            ("Save Session", self._save_session, ACCENT,   "#111"),
            ("Load Session", self._load_session, SURFACE2, TEXT),
        ]:
            tk.Button(bar, text=label, command=cmd, bg=c_bg, fg=c_fg,
                      relief="flat", padx=12, pady=4,
                      font=("Helvetica", 9, "bold"), cursor="hand2"
                      ).pack(side="left", padx=4)

        # ── split: plot | controls ───────────────────────────────────────
        pw = tk.PanedWindow(self.root, orient="horizontal", bg=BG,
                            sashwidth=5, sashrelief="flat")
        pw.pack(fill="both", expand=True)

        plot_side = tk.Frame(pw, bg=BG)
        pw.add(plot_side, minsize=480, width=680)
        self._build_plot_side(plot_side)

        ctrl_side = tk.Frame(pw, bg=SURFACE)
        pw.add(ctrl_side, minsize=380)
        self._build_control_side(ctrl_side)

    # ── plot side ──────────────────────────────────────────────────────
    def _build_plot_side(self, parent):
        # range + L* controls
        top = tk.Frame(parent, bg=BG, padx=8, pady=6)
        top.pack(fill="x")

        rng = tk.Frame(top, bg=BG)
        rng.pack(fill="x")
        self.v_amin = self._range_box(rng, "a* min", "-60",  0)
        self.v_amax = self._range_box(rng, "a* max",  "60",  1)
        self.v_bmin = self._range_box(rng, "b* min", "-60",  2)
        self.v_bmax = self._range_box(rng, "b* max",  "60",  3)
        tk.Button(rng, text="Apply", command=self._force_redraw,
                  bg=ACCENT, fg="#111", relief="flat", padx=10, pady=2,
                  font=("Helvetica", 9, "bold"), cursor="hand2"
                  ).grid(row=0, column=8, padx=(12, 0))

        lf = tk.Frame(top, bg=BG)
        lf.pack(fill="x", pady=(6, 0))
        tk.Label(lf, text="L* background", bg=BG, fg=ACCENT,
                 font=("Helvetica", 9, "bold")).pack(side="left", padx=(0, 8))
        self.v_L = tk.IntVar(value=80)
        self.lbl_L = tk.Label(lf, text="80", bg=BG, fg=ACCENT,
                              font=("Courier", 11, "bold"), width=3)
        self.lbl_L.pack(side="right")
        tk.Scale(lf, from_=1, to=99, orient="horizontal", variable=self.v_L,
                 showvalue=False, bg=BG, fg=ACCENT, troughcolor=SURFACE2,
                 highlightthickness=0, sliderrelief="flat",
                 command=self._on_L_change).pack(side="left", fill="x", expand=True)

        # Phase-label size (global) + show-all / hide-all
        sf = tk.Frame(top, bg=BG)
        sf.pack(fill="x", pady=(6, 0))
        tk.Label(sf, text="Phase label size", bg=BG, fg=MUTED,
                 font=("Helvetica", 9)).pack(side="left", padx=(0, 6))
        self.v_label_size = tk.IntVar(value=9)
        tk.Scale(sf, from_=5, to=24, orient="horizontal",
                 variable=self.v_label_size, showvalue=True,
                 bg=BG, fg=MUTED, troughcolor=SURFACE2,
                 highlightthickness=0, sliderrelief="flat",
                 length=120, command=lambda _=None: self._refresh_plot()
                 ).pack(side="left")
        tk.Button(sf, text="Show all", command=lambda: self._labels_all(True),
                  bg=SURFACE2, fg=TEXT, relief="flat", padx=8, pady=1,
                  font=("Helvetica", 8), cursor="hand2"
                  ).pack(side="left", padx=(10, 2))
        tk.Button(sf, text="Hide all", command=lambda: self._labels_all(False),
                  bg=SURFACE2, fg=TEXT, relief="flat", padx=8, pady=1,
                  font=("Helvetica", 8), cursor="hand2"
                  ).pack(side="left", padx=2)

        # ── matplotlib figure ─────────────────────────────────────────
        # Publication look: white axes face handled at draw-time.
        self.fig = Figure(figsize=(6, 6), facecolor="white")
        self.ax  = self.fig.add_subplot(111)
        # Reserve generous margins; box_aspect keeps the DATA area square.
        self.fig.subplots_adjust(left=0.14, right=0.97, top=0.92, bottom=0.12)

        canvas_holder = tk.Frame(parent, bg=BG)
        canvas_holder.pack(fill="both", expand=True, padx=8, pady=(0, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_holder)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        # Re-cache the static background whenever the canvas is (re)drawn
        self.canvas.mpl_connect("draw_event", self._on_draw_event)

        self.v_coord = tk.StringVar(value="")
        tk.Label(parent, textvariable=self.v_coord, bg=BG, fg=MUTED,
                 font=("Courier", 9)).pack(anchor="w", padx=10, pady=(0, 4))

    def _range_box(self, parent, label, default, col):
        tk.Label(parent, text=label, bg=BG, fg=MUTED, font=("Helvetica", 8)
                 ).grid(row=0, column=col * 2, padx=(8, 2))
        v = tk.StringVar(value=default)
        sb = tk.Spinbox(parent, from_=-128, to=128, textvariable=v, width=6,
                        bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE,
                        relief="flat", font=("Courier", 9), insertbackground=TEXT)
        sb.grid(row=0, column=col * 2 + 1, padx=(0, 4))
        sb.bind("<Return>",   lambda e: self._force_redraw())
        sb.bind("<FocusOut>", lambda e: self._force_redraw())
        return v

    # ── control side ───────────────────────────────────────────────────
    def _build_control_side(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=2)   # table grows
        parent.rowconfigure(4, weight=1)   # legend grows

        # 1) CSV import (top — primary workflow)
        self._build_csv(parent, row=0)

        # 2) table label
        tk.Label(parent, text="Data points", bg=SURFACE, fg=TEXT,
                 font=("Helvetica", 10, "bold")
                 ).grid(row=1, column=0, sticky="w", padx=10, pady=(8, 2))

        # 3) table
        self._build_table(parent, row=2)

        # 4) legend label
        tk.Label(parent, text="Legend  (one entry per composition)",
                 bg=SURFACE, fg=MUTED, font=("Helvetica", 9)
                 ).grid(row=3, column=0, sticky="w", padx=10, pady=(8, 2))

        # 5) legend
        self._build_legend(parent, row=4)

        # 6) editor
        self._build_editor(parent, row=5)

    def _build_csv(self, parent, row):
        f = tk.LabelFrame(parent, text=" Paste data → Import ",
                          bg=SURFACE, fg=ACCENT, font=("Helvetica", 9, "bold"),
                          bd=1, relief="groove")
        f.grid(row=row, column=0, sticky="ew", padx=8, pady=(8, 2))
        f.columnconfigure(0, weight=1)

        tk.Label(f, text="Columns:  Composition · Phase · L* · a* · b*"
                         "   (tab or comma, header optional)",
                 bg=SURFACE, fg=MUTED, font=("Courier", 7)
                 ).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(2, 0))

        self.csv_text = tk.Text(f, height=5, width=34, bg=SURFACE2, fg=TEXT,
                                insertbackground=TEXT, relief="flat",
                                font=("Courier", 8), wrap="none")
        self.csv_text.grid(row=1, column=0, sticky="ew", padx=5, pady=5)

        col = tk.Frame(f, bg=SURFACE)
        col.grid(row=1, column=1, padx=5, pady=5, sticky="n")
        for txt, cmd, c in [("Import",    self._csv_import, ACCENT2),
                            ("From file", self._csv_file,   SURFACE2),
                            ("Export",    self._csv_export, SURFACE2),
                            ("Clear all", self._clear_all,  DANGER)]:
            tk.Button(col, text=txt, command=cmd, bg=c,
                      fg="white" if c in (ACCENT2, DANGER) else TEXT,
                      relief="flat", padx=6, pady=3, width=9,
                      font=("Helvetica", 8, "bold"), cursor="hand2"
                      ).pack(pady=2)

    def _build_table(self, parent, row):
        frame = tk.Frame(parent, bg=SURFACE)
        frame.grid(row=row, column=0, sticky="nsew", padx=8, pady=2)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        cols = ("Composition", "Phase", "L*", "a*", "b*")
        self.tree = ttk.Treeview(frame, columns=cols, show="headings",
                                 selectmode="extended")
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", background=SURFACE2, foreground=TEXT,
                        fieldbackground=SURFACE2, rowheight=20,
                        font=("Courier", 8))
        style.configure("Treeview.Heading", background=BG, foreground=MUTED,
                        font=("Helvetica", 8, "bold"), relief="flat")
        style.map("Treeview", background=[("selected", ACCENT2)],
                  foreground=[("selected", "white")])

        widths = {"Composition": 130, "Phase": 70, "L*": 55, "a*": 55, "b*": 55}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c],
                             anchor="w" if c in ("Composition", "Phase") else "center")

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

    def _build_legend(self, parent, row):
        outer = tk.Frame(parent, bg=SURFACE2, bd=1, relief="groove")
        outer.grid(row=row, column=0, sticky="nsew", padx=8, pady=2)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        self.leg_canvas = tk.Canvas(outer, bg=SURFACE2, highlightthickness=0,
                                    height=90)
        vsb = ttk.Scrollbar(outer, orient="vertical",
                            command=self.leg_canvas.yview)
        self.leg_canvas.configure(yscrollcommand=vsb.set)
        self.leg_canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self.leg_inner = tk.Frame(self.leg_canvas, bg=SURFACE2)
        self.leg_win = self.leg_canvas.create_window((0, 0), window=self.leg_inner,
                                                     anchor="nw")
        self.leg_inner.bind("<Configure>",
            lambda e: self.leg_canvas.configure(scrollregion=self.leg_canvas.bbox("all")))
        self.leg_canvas.bind("<Configure>",
            lambda e: self.leg_canvas.itemconfig(self.leg_win, width=e.width))

    def _build_editor(self, parent, row):
        ed = tk.LabelFrame(parent, text=" Edit selected / add point ",
                           bg=SURFACE, fg=MUTED, font=("Helvetica", 8),
                           bd=1, relief="groove")
        ed.grid(row=row, column=0, sticky="ew", padx=8, pady=(2, 8))
        ed.columnconfigure(1, weight=1)
        ed.columnconfigure(3, weight=1)

        def lab(t, r, c):
            tk.Label(ed, text=t, bg=SURFACE, fg=MUTED, font=("Helvetica", 8)
                     ).grid(row=r, column=c, sticky="w", padx=4, pady=2)

        def ent(r, c, w=12):
            v = tk.StringVar()
            tk.Entry(ed, textvariable=v, width=w, bg=SURFACE2, fg=TEXT,
                     insertbackground=TEXT, relief="flat", font=("Courier", 9)
                     ).grid(row=r, column=c, sticky="ew", padx=4, pady=2)
            return v

        lab("Composition", 0, 0); self.e_comp  = ent(0, 1, 16)
        lab("Phase",        1, 0); self.e_phase = ent(1, 1, 10)
        lab("L*",           1, 2); self.e_L     = ent(1, 3, 7)
        lab("a*",           2, 0); self.e_a     = ent(2, 1, 7)
        lab("b*",           2, 2); self.e_b     = ent(2, 3, 7)

        # Show-phase-label checkbox for the focused point
        self.v_show_label = tk.BooleanVar(value=False)
        tk.Checkbutton(ed, text="Show phase label next to this point",
                       variable=self.v_show_label, command=self._toggle_label,
                       bg=SURFACE, fg=TEXT, selectcolor=SURFACE2,
                       activebackground=SURFACE, activeforeground=ACCENT,
                       font=("Helvetica", 8), cursor="hand2"
                       ).grid(row=3, column=0, columnspan=4, sticky="w", padx=4, pady=(2, 0))

        # Visible (plot on/off) checkbox for the focused point
        self.v_visible = tk.BooleanVar(value=True)
        tk.Checkbutton(ed, text="Show this point in the plot",
                       variable=self.v_visible, command=self._toggle_visible,
                       bg=SURFACE, fg=TEXT, selectcolor=SURFACE2,
                       activebackground=SURFACE, activeforeground=ACCENT,
                       font=("Helvetica", 8), cursor="hand2"
                       ).grid(row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(0, 2))

        brow = tk.Frame(ed, bg=SURFACE)
        brow.grid(row=5, column=0, columnspan=4, pady=6, padx=4, sticky="ew")
        brow.columnconfigure((0, 1, 2), weight=1)
        for i, (t, cmd, c_bg, c_fg) in enumerate([
            ("Add",    self._add,    ACCENT2, "white"),
            ("Update", self._update, ACCENT,  "#111"),
            ("Delete", self._delete, DANGER,  "white"),
        ]):
            tk.Button(brow, text=t, command=cmd, bg=c_bg, fg=c_fg, relief="flat",
                      pady=4, font=("Helvetica", 9, "bold"), cursor="hand2"
                      ).grid(row=0, column=i, sticky="ew", padx=2)

        # Reorder row — move the selected point up or down
        mrow = tk.Frame(ed, bg=SURFACE)
        mrow.grid(row=6, column=0, columnspan=4, pady=(0, 4), padx=4, sticky="ew")
        mrow.columnconfigure((0, 1), weight=1)
        tk.Button(mrow, text="↑ Move up", command=self._move_up,
                  bg=SURFACE2, fg=TEXT, relief="flat", pady=4,
                  font=("Helvetica", 9, "bold"), cursor="hand2"
                  ).grid(row=0, column=0, sticky="ew", padx=2)
        tk.Button(mrow, text="↓ Move down", command=self._move_down,
                  bg=SURFACE2, fg=TEXT, relief="flat", pady=4,
                  font=("Helvetica", 9, "bold"), cursor="hand2"
                  ).grid(row=0, column=1, sticky="ew", padx=2)

        # Per-composition style override (colour + marker)
        srow = tk.Frame(ed, bg=SURFACE)
        srow.grid(row=7, column=0, columnspan=4, pady=(0, 4), padx=4, sticky="w")
        tk.Label(srow, text="Style for this composition:", bg=SURFACE, fg=MUTED,
                 font=("Helvetica", 8)).pack(side="left", padx=(0, 6))
        self.style_swatch = tk.Label(srow, bg="#888", width=3, relief="groove",
                                     cursor="hand2")
        self.style_swatch.pack(side="left", padx=(0, 6))
        self.style_swatch.bind("<Button-1>", self._change_colour)
        tk.Label(srow, text="marker:", bg=SURFACE, fg=MUTED,
                 font=("Helvetica", 8)).pack(side="left")
        self.v_marker = tk.StringVar(value="o")
        om = tk.OptionMenu(srow, self.v_marker, *MARKER_POOL,
                           command=lambda _=None: self._change_marker())
        om.config(bg=SURFACE2, fg=TEXT, relief="flat", highlightthickness=0,
                  font=("Courier", 9), width=3, cursor="hand2")
        om["menu"].config(bg=SURFACE2, fg=TEXT)
        om.pack(side="left", padx=4)

        # ── Bulk edit for multi-selection (per-point overrides) ──────────
        bulk = tk.Frame(ed, bg=SURFACE2, bd=1, relief="groove")
        bulk.grid(row=8, column=0, columnspan=4, pady=(6, 2), padx=4, sticky="ew")
        self.lbl_multi = tk.Label(
            bulk, text="Select 2+ rows (Ctrl/Shift-click) to bulk-edit",
            bg=SURFACE2, fg=ACCENT, font=("Helvetica", 8, "bold"))
        self.lbl_multi.pack(anchor="w", padx=6, pady=(4, 2))

        brow2 = tk.Frame(bulk, bg=SURFACE2)
        brow2.pack(fill="x", padx=4, pady=(0, 5))
        tk.Button(brow2, text="Set colour…", command=self._bulk_colour,
                  bg=ACCENT2, fg="white", relief="flat", pady=3,
                  font=("Helvetica", 8, "bold"), cursor="hand2"
                  ).pack(side="left", expand=True, fill="x", padx=2)
        # Marker chooser for bulk
        self.v_bulk_marker = tk.StringVar(value="o")
        om2 = tk.OptionMenu(brow2, self.v_bulk_marker, *MARKER_POOL)
        om2.config(bg=SURFACE2, fg=TEXT, relief="flat", highlightthickness=1,
                   font=("Courier", 9), width=3, cursor="hand2")
        om2["menu"].config(bg=SURFACE2, fg=TEXT)
        om2.pack(side="left", padx=2)
        tk.Button(brow2, text="Set marker", command=self._bulk_marker,
                  bg=ACCENT2, fg="white", relief="flat", pady=3,
                  font=("Helvetica", 8, "bold"), cursor="hand2"
                  ).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(brow2, text="Clear override", command=self._bulk_clear,
                  bg=SURFACE, fg=MUTED, relief="flat", pady=3,
                  font=("Helvetica", 8, "bold"), cursor="hand2"
                  ).pack(side="left", expand=True, fill="x", padx=2)

        brow3 = tk.Frame(bulk, bg=SURFACE2)
        brow3.pack(fill="x", padx=4, pady=(0, 5))
        tk.Button(brow3, text="Labels ON", command=lambda: self._bulk_label(True),
                  bg=SURFACE2, fg=TEXT, relief="flat", pady=3,
                  font=("Helvetica", 8, "bold"), cursor="hand2"
                  ).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(brow3, text="Labels OFF", command=lambda: self._bulk_label(False),
                  bg=SURFACE2, fg=TEXT, relief="flat", pady=3,
                  font=("Helvetica", 8, "bold"), cursor="hand2"
                  ).pack(side="left", expand=True, fill="x", padx=2)

        brow4 = tk.Frame(bulk, bg=SURFACE2)
        brow4.pack(fill="x", padx=4, pady=(0, 6))
        tk.Button(brow4, text="Show in plot", command=lambda: self._bulk_visible(True),
                  bg=SURFACE2, fg=TEXT, relief="flat", pady=3,
                  font=("Helvetica", 8, "bold"), cursor="hand2"
                  ).pack(side="left", expand=True, fill="x", padx=2)
        tk.Button(brow4, text="Hide from plot", command=lambda: self._bulk_visible(False),
                  bg=SURFACE2, fg=TEXT, relief="flat", pady=3,
                  font=("Helvetica", 8, "bold"), cursor="hand2"
                  ).pack(side="left", expand=True, fill="x", padx=2)

    # ════════════════════════════════════════════════════════════════════
    # Style management
    # ════════════════════════════════════════════════════════════════════

    def _ensure_style(self, composition):
        if composition not in self.style:
            self.style[composition] = {
                "colour": next(self._palette_cycle),
                "marker": next(self._marker_cycle),
            }
        return self.style[composition]

    def _point_style(self, p):
        """Resolve a point's effective colour & marker.

        A point may carry its own 'colour'/'marker' override keys.
        Otherwise it inherits its composition's style.
        """
        base = self._ensure_style(p["composition"])
        return {
            "colour": p.get("colour", base["colour"]),
            "marker": p.get("marker", base["marker"]),
        }

    # ════════════════════════════════════════════════════════════════════
    # Plot
    # ════════════════════════════════════════════════════════════════════

    def _ranges(self):
        def p(v, d):
            try: return float(v.get())
            except Exception: return d
        a0, a1 = p(self.v_amin, -60), p(self.v_amax, 60)
        b0, b1 = p(self.v_bmin, -60), p(self.v_bmax, 60)
        if a0 >= a1: a1 = a0 + 1
        if b0 >= b1: b1 = b0 + 1
        return a0, a1, b0, b1

    def _force_redraw(self, *_):
        self._bg_cache.clear()
        self._full_redraw()

    def _on_L_change(self, val):
        self.lbl_L.config(text=str(val))
        self._full_redraw()

    # ── render in two layers for speed ──────────────────────────────────
    #
    # _full_redraw      draws the expensive static background (gamut image +
    #                   axes + grid), then the dynamic artists, then caches the
    #                   background pixels.  Call only when L*, ranges, or the
    #                   window size change.
    # _redraw_dynamic   restores the cached background and blits ONLY the
    #                   points / labels / guide lines.  Fast — use for
    #                   selection, label drags, show/hide, styling, etc.

    def _draw_background(self):
        """Draw the static layer (image, axes, grid, titles). No data points."""
        L = self.v_L.get()
        a0, a1, b0, b1 = self._ranges()

        self.ax.clear()

        key = (L, a0, a1, b0, b1)
        if key not in self._bg_cache:
            self._bg_cache[key] = build_background(L, a0, a1, b0, b1)
        self.ax.imshow(self._bg_cache[key], extent=[a0, a1, b0, b1],
                       origin="lower", aspect="auto",
                       interpolation="bilinear", zorder=0)

        if a0 <= 0 <= a1:
            self.ax.axvline(0, color="#444", lw=0.7, ls="--", alpha=0.6, zorder=1)
        if b0 <= 0 <= b1:
            self.ax.axhline(0, color="#444", lw=0.7, ls="--", alpha=0.6, zorder=1)
        self.ax.grid(color="#000", lw=0.3, alpha=0.12, zorder=1)

        self.ax.set_box_aspect(1)
        self.ax.set_xlim(a0, a1)
        self.ax.set_ylim(b0, b1)
        self.ax.set_xlabel("a*   (green − / + red)", fontsize=11)
        self.ax.set_ylabel("b*   (blue − / + yellow)", fontsize=11)
        self.ax.set_title(f"CIELab a*–b*  at  L* = {L}", fontsize=12, pad=8)
        self.ax.tick_params(labelsize=9)

    def _build_dynamic_artists(self):
        """Create (but rely on blitting to show) the points, labels, guides."""
        # Remove any previous dynamic artists
        for art in self._dynamic_artists:
            try: art.remove()
            except Exception: pass
        self._dynamic_artists = []
        self._label_artists = {}

        size = self.v_label_size.get()
        for i, p in enumerate(self.points):
            if not p.get("visible", True):
                continue
            st  = self._point_style(p)
            sel = (i in self.selected_idxs)
            # marker
            if MarkerStyle(st["marker"]).is_filled():
                sc = self.ax.scatter(p["a"], p["b"], c=st["colour"],
                                     marker=st["marker"], s=90,
                                     edgecolors="#ffff00" if sel else "#222",
                                     linewidths=2.0 if sel else 0.8, zorder=5)
            else:
                sc = self.ax.scatter(p["a"], p["b"], c=st["colour"],
                                     marker=st["marker"], s=90,
                                     linewidths=2.5 if sel else 1.3, zorder=5)
            self._dynamic_artists.append(sc)

            # label (+ guide line if moved)
            if p.get("show_label") and p.get("phase"):
                dx, dy = p.get("label_offset", (None, None))
                if dx is None:
                    # default offset in points (not data units)
                    txt = self.ax.annotate(
                        p["phase"], (p["a"], p["b"]),
                        textcoords="offset points", xytext=(6, 4),
                        fontsize=size, color="#111", ha="left", va="bottom",
                        zorder=7,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
                else:
                    lx, ly = p["a"] + dx, p["b"] + dy
                    # guide line from point to label
                    line, = self.ax.plot([p["a"], lx], [p["b"], ly],
                                         color="#444", lw=0.7, zorder=6)
                    self._dynamic_artists.append(line)
                    txt = self.ax.annotate(
                        p["phase"], (lx, ly),
                        fontsize=size, color="#111", ha="left", va="bottom",
                        zorder=7,
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")])
                self._dynamic_artists.append(txt)
                self._label_artists[i] = txt

    def _full_redraw(self):
        self._draw_background()
        self._build_dynamic_artists()
        self.canvas.draw()          # triggers _on_draw_event → caches bg

    def _redraw_dynamic(self):
        """Fast path: restore cached bg, blit dynamic artists only."""
        if self._bg_region is None:
            self._full_redraw(); return
        self.canvas.restore_region(self._bg_region)
        for art in self._dynamic_artists:
            self.ax.draw_artist(art)
        self.canvas.blit(self.ax.bbox)

    def _on_draw_event(self, event):
        """After a full canvas draw, cache the clean background for blitting."""
        if getattr(self, "_capturing_bg", False):
            return
        self._capturing_bg = True
        try:
            # Hide dynamic artists, capture the static background, then restore.
            for art in self._dynamic_artists:
                art.set_visible(False)
            self.canvas.draw()
            self._bg_region = self.canvas.copy_from_bbox(self.ax.bbox)
            for art in self._dynamic_artists:
                art.set_visible(True)
                self.ax.draw_artist(art)
            self.canvas.blit(self.ax.bbox)
        finally:
            self._capturing_bg = False

    # Backwards-compatible alias — many call sites use _redraw()
    def _redraw(self):
        self._redraw_dynamic()

    def _refresh_plot(self):
        """Rebuild dynamic artists (appearance changed) then fast-blit."""
        self._build_dynamic_artists()
        self._redraw_dynamic()

    def _on_motion(self, event):
        # Live a*/b* readout
        if event.inaxes == self.ax and event.xdata is not None:
            self.v_coord.set(f"a* = {event.xdata:6.2f}    b* = {event.ydata:6.2f}")
        else:
            self.v_coord.set("")
        # Label dragging
        if self._drag_label_idx is not None and event.xdata is not None:
            ma0, mb0, base_dx, base_dy = self._drag_start
            p = self.points[self._drag_label_idx]
            new_dx = base_dx + (event.xdata - ma0)
            new_dy = base_dy + (event.ydata - mb0)
            p["label_offset"] = (new_dx, new_dy)
            self._build_dynamic_artists()
            self._redraw_dynamic()

    def _on_press(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        hit = self._label_at(event)
        if hit is None:
            return
        # Right-click resets the label to its default position
        if event.button == 3:
            self.points[hit].pop("label_offset", None)
            self._refresh_plot()
            return
        # Left-click starts a drag
        p = self.points[hit]
        dx, dy = p.get("label_offset", (None, None))
        if dx is None:
            dx, dy = self._default_offset_in_data()
        self._drag_label_idx = hit
        self._drag_start = (event.xdata, event.ydata, dx, dy)

    def _on_release(self, event):
        self._drag_label_idx = None
        self._drag_start = None

    def _label_at(self, event):
        """Return point index whose label contains the click, else None."""
        for i, txt in self._label_artists.items():
            contains, _ = txt.contains(event)
            if contains:
                return i
        return None

    def _default_offset_in_data(self):
        """Approximate the 6,4 points default offset in data coordinates."""
        a0, a1, b0, b1 = self._ranges()
        # 6 pts ≈ small fraction of axis; use 2% of range as a sensible seed
        return 0.02 * (a1 - a0), 0.02 * (b1 - b0)

    # ════════════════════════════════════════════════════════════════════
    # Table + legend refresh
    # ════════════════════════════════════════════════════════════════════

    def _refresh_all(self):
        self._refresh_table()
        self._refresh_legend()
        self._build_dynamic_artists()
        self._redraw_dynamic()

    def _refresh_table(self):
        for it in self.tree.get_children():
            self.tree.delete(it)
        for i, p in enumerate(self.points):
            hidden = not p.get("visible", True)
            if i in self.selected_idxs:
                tag = "sel"
            elif hidden:
                tag = "hidden"
            else:
                tag = "e" if i % 2 == 0 else "o"
            comp = p["composition"] + ("  (hidden)" if hidden else "")
            self.tree.insert("", "end", iid=str(i), tags=(tag,),
                             values=(comp, p.get("phase", ""),
                                     f"{p.get('Lpoint', ''):.2f}" if p.get("Lpoint") not in (None, "") else "",
                                     f"{p['a']:.2f}", f"{p['b']:.2f}"))
        self.tree.tag_configure("e", background=SURFACE2)
        self.tree.tag_configure("o", background=SURFACE)
        self.tree.tag_configure("sel", background=ACCENT2)
        self.tree.tag_configure("hidden", background=SURFACE, foreground=MUTED)
        # Restore selection highlight in the widget itself
        valid = [str(i) for i in self.selected_idxs if i < len(self.points)]
        if valid:
            self.tree.selection_set(valid)
            self.tree.see(valid[-1])

    def _refresh_legend(self):
        for w in self.leg_inner.winfo_children():
            w.destroy()
        # Group by the EFFECTIVE appearance so per-point overrides show up.
        # Key = (composition, colour, marker); first-seen order preserved.
        seen = []          # list of (comp, colour, marker)
        for p in self.points:
            if not p.get("visible", True):
                continue
            st = self._point_style(p)
            key = (p["composition"], st["colour"], st["marker"])
            if key not in seen:
                seen.append(key)
        for comp, colour, marker in seen:
            rowf = tk.Frame(self.leg_inner, bg=SURFACE2)
            rowf.pack(fill="x", padx=4, pady=1)
            tk.Label(rowf, bg=colour, width=2, relief="groove"
                     ).pack(side="left", padx=(2, 4))
            tk.Label(rowf, text=f"[{marker}]", bg=SURFACE2, fg=MUTED,
                     font=("Courier", 8), width=3).pack(side="left")
            tk.Label(rowf, text=comp, bg=SURFACE2, fg=TEXT,
                     font=("Helvetica", 8), anchor="w"
                     ).pack(side="left", fill="x", expand=True)
        self.leg_inner.update_idletasks()
        self.leg_canvas.configure(scrollregion=self.leg_canvas.bbox("all"))

    # ════════════════════════════════════════════════════════════════════
    # Selection + editor
    # ════════════════════════════════════════════════════════════════════

    def _on_select(self, _evt):
        sel = self.tree.selection()
        if not sel:
            self.selected_idxs = []
            self.selected_idx = None
            self._update_multi_label()
            self._refresh_plot()
            return
        self.selected_idxs = sorted(int(s) for s in sel)
        # Primary = the focused row (for the single-point editor fields)
        focus = self.tree.focus()
        self.selected_idx = int(focus) if focus else self.selected_idxs[-1]

        p = self.points[self.selected_idx]
        self.e_comp.set(p["composition"])
        self.e_phase.set(p.get("phase", ""))
        self.e_L.set(str(p.get("Lpoint", "")))
        self.e_a.set(str(p["a"]))
        self.e_b.set(str(p["b"]))
        st = self._point_style(p)
        self.style_swatch.config(bg=st["colour"])
        self.v_marker.set(st["marker"])
        self.v_show_label.set(bool(p.get("show_label", False)))
        self.v_visible.set(bool(p.get("visible", True)))
        self._update_multi_label()
        self._refresh_plot()

    def _update_multi_label(self):
        """Refresh the count shown on the bulk-edit panel."""
        n = len(self.selected_idxs)
        if hasattr(self, "lbl_multi"):
            if n <= 1:
                self.lbl_multi.config(
                    text="Select 2+ rows (Ctrl/Shift-click) to bulk-edit")
            else:
                self.lbl_multi.config(text=f"{n} points selected")

    def _read_editor(self):
        def f(v, d=0.0):
            try: return float(v.get())
            except Exception: return d
        comp = self.e_comp.get().strip() or "Sample"
        self._ensure_style(comp)
        Lp = self.e_L.get().strip()
        return {"composition": comp,
                "phase": self.e_phase.get().strip(),
                "Lpoint": float(Lp) if Lp else None,
                "a": f(self.e_a), "b": f(self.e_b),
                "show_label": self.v_show_label.get(),
                "visible": self.v_visible.get()}

    def _add(self):
        self.points.append(self._read_editor())
        self.selected_idx = len(self.points) - 1
        self.selected_idxs = [self.selected_idx]
        self._refresh_all()

    def _update(self):
        if self.selected_idx is None:
            messagebox.showwarning("No selection", "Select a row first."); return
        new = self._read_editor()
        # Preserve any per-point colour/marker override on the edited point
        old = self.points[self.selected_idx]
        if "colour" in old: new["colour"] = old["colour"]
        if "marker" in old: new["marker"] = old["marker"]
        if "label_offset" in old: new["label_offset"] = old["label_offset"]
        self.points[self.selected_idx] = new
        self.selected_idxs = [self.selected_idx]
        self._refresh_all()

    def _delete(self):
        if not self.selected_idxs:
            messagebox.showwarning("No selection", "Select a row first."); return
        # Delete all selected points (descending so indices stay valid)
        for i in sorted(self.selected_idxs, reverse=True):
            self.points.pop(i)
        first = min(self.selected_idxs)
        self.selected_idx = min(first, len(self.points) - 1) if self.points else None
        self.selected_idxs = [self.selected_idx] if self.selected_idx is not None else []
        self._refresh_all()

    def _move_up(self):
        i = self.selected_idx
        if i is None or i <= 0:
            return
        self.points[i - 1], self.points[i] = self.points[i], self.points[i - 1]
        self.selected_idx = i - 1
        self.selected_idxs = [i - 1]
        self._refresh_all()

    def _move_down(self):
        i = self.selected_idx
        if i is None or i >= len(self.points) - 1:
            return
        self.points[i + 1], self.points[i] = self.points[i], self.points[i + 1]
        self.selected_idx = i + 1
        self.selected_idxs = [i + 1]
        self._refresh_all()

    def _change_colour(self, _evt=None):
        comp = self.e_comp.get().strip()
        if not comp or comp not in self.style:
            return
        res = colorchooser.askcolor(color=self.style[comp]["colour"],
                                    title=f"Colour for {comp}")
        if res and res[1]:
            self.style[comp]["colour"] = res[1]
            self.style_swatch.config(bg=res[1])
            self._refresh_legend(); self._refresh_plot()

    def _change_marker(self):
        comp = self.e_comp.get().strip()
        if comp and comp in self.style:
            self.style[comp]["marker"] = self.v_marker.get()
            self._refresh_legend(); self._refresh_plot()

    # ── bulk edits on the multi-selection (per-point overrides) ──────────

    def _bulk_targets(self):
        if not self.selected_idxs:
            messagebox.showinfo("No selection",
                                "Select one or more rows in the table first.")
            return []
        return self.selected_idxs

    def _bulk_colour(self):
        idxs = self._bulk_targets()
        if not idxs:
            return
        # Seed the picker with the first selected point's effective colour
        seed = self._point_style(self.points[idxs[0]])["colour"]
        res = colorchooser.askcolor(
            color=seed, title=f"Colour for {len(idxs)} selected point(s)")
        if res and res[1]:
            for i in idxs:
                self.points[i]["colour"] = res[1]
            self._refresh_all()

    def _bulk_marker(self):
        idxs = self._bulk_targets()
        if not idxs:
            return
        m = self.v_bulk_marker.get()
        for i in idxs:
            self.points[i]["marker"] = m
        self._refresh_all()

    def _bulk_clear(self):
        """Remove per-point overrides → revert to composition style."""
        idxs = self._bulk_targets()
        if not idxs:
            return
        for i in idxs:
            self.points[i].pop("colour", None)
            self.points[i].pop("marker", None)
        self._refresh_all()

    # ── phase labels ─────────────────────────────────────────────────────

    def _toggle_label(self):
        """Editor checkbox: toggle the focused point's phase label."""
        if self.selected_idx is None:
            return
        self.points[self.selected_idx]["show_label"] = self.v_show_label.get()
        self._refresh_plot()

    def _bulk_label(self, on):
        """Turn phase labels on/off for all selected points."""
        idxs = self._bulk_targets()
        if not idxs:
            return
        for i in idxs:
            self.points[i]["show_label"] = on
        if self.selected_idx in idxs:
            self.v_show_label.set(on)
        self._refresh_plot()

    def _labels_all(self, on):
        """Global Show all / Hide all phase labels."""
        for p in self.points:
            p["show_label"] = on
        if self.selected_idx is not None:
            self.v_show_label.set(on)
        self._refresh_plot()

    # ── point visibility (hide without deleting) ─────────────────────────

    def _toggle_visible(self):
        if self.selected_idx is None:
            return
        self.points[self.selected_idx]["visible"] = self.v_visible.get()
        self._refresh_all()      # table greying also updates

    def _bulk_visible(self, on):
        idxs = self._bulk_targets()
        if not idxs:
            return
        for i in idxs:
            self.points[i]["visible"] = on
        if self.selected_idx in idxs:
            self.v_visible.set(on)
        self._refresh_all()

    # ════════════════════════════════════════════════════════════════════
    # CSV / TSV
    # ════════════════════════════════════════════════════════════════════

    def _csv_import(self):
        text = self.csv_text.get("1.0", "end").strip()
        if not text:
            messagebox.showinfo("Empty", "Paste data into the box first."); return
        added = 0
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" in line:
                sep = "\t"
            elif ";" in line:
                sep = ";"
            else:
                sep = ","
            parts = [c.strip() for c in line.split(sep)]
            if len(parts) < 5:
                continue
            # skip header rows
            head = parts[0].lower()
            if head.startswith("composition") or head in ("label", "comp"):
                continue
            try:
                comp  = parts[0] or "Sample"
                phase = parts[1]
                Lp    = float(parts[2])
                a     = float(parts[3])
                b     = float(parts[4])
            except ValueError:
                continue
            self._ensure_style(comp)
            self.points.append({"composition": comp, "phase": phase,
                                "Lpoint": Lp, "a": a, "b": b})
            added += 1
        if added:
            self.selected_idx = len(self.points) - 1
            self.selected_idxs = [self.selected_idx]
            self.csv_text.delete("1.0", "end")
            self._refresh_all()
            messagebox.showinfo("Import", f"{added} point(s) imported.")
        else:
            messagebox.showwarning("Import",
                "No valid rows found.\n\nExpected 5 columns:\n"
                "  Composition · Phase · L* · a* · b*\n\n"
                "Example:\n  Pt 100\tPt\t89.12\t-0.13\t0.54")

    def _csv_file(self):
        path = filedialog.askopenfilename(
            title="Open data file",
            filetypes=[("Data", "*.csv *.tsv *.txt"), ("All", "*.*")])
        if not path:
            return
        with open(path, encoding="utf-8-sig") as fh:
            self.csv_text.delete("1.0", "end")
            self.csv_text.insert("1.0", fh.read())
        self._csv_import()

    def _csv_export(self):
        lines = ["Composition\tPhase\tL*\ta*\tb*"]
        for p in self.points:
            Lp = f"{p['Lpoint']:.2f}" if p.get("Lpoint") is not None else ""
            lines.append(f"{p['composition']}\t{p.get('phase','')}\t"
                         f"{Lp}\t{p['a']:.2f}\t{p['b']:.2f}")
        self.csv_text.delete("1.0", "end")
        self.csv_text.insert("1.0", "\n".join(lines))

    def _clear_all(self):
        if self.points and messagebox.askyesno(
                "Clear all", "Remove all data points?"):
            self.points.clear()
            self.selected_idx = None
            self.selected_idxs = []
            self._refresh_all()

    # ════════════════════════════════════════════════════════════════════
    # Publication PNG  (white background)
    # ════════════════════════════════════════════════════════════════════

    def _save_png(self):
        L = self.v_L.get()
        a0, a1, b0, b1 = self._ranges()

        fig = Figure(figsize=(8, 8), facecolor="white")
        ax  = fig.add_subplot(111)
        fig.subplots_adjust(left=0.12, right=0.74, top=0.93, bottom=0.10)

        img = build_background(L, a0, a1, b0, b1, resolution=600)
        ax.imshow(img, extent=[a0, a1, b0, b1], origin="lower",
                  aspect="auto", interpolation="bilinear", zorder=0)
        if a0 <= 0 <= a1:
            ax.axvline(0, color="#333", lw=0.7, ls="--", alpha=0.6)
        if b0 <= 0 <= b1:
            ax.axhline(0, color="#333", lw=0.7, ls="--", alpha=0.6)

        # Draw every VISIBLE point with its effective (possibly overridden) style
        for p in self.points:
            if not p.get("visible", True):
                continue
            st = self._point_style(p)
            scatter_point(ax, p["a"], p["b"], st["colour"], st["marker"],
                          120, "#222", 0.8)
            if p.get("show_label") and p.get("phase"):
                dx, dy = p.get("label_offset", (None, None))
                if dx is None:
                    ax.annotate(
                        p["phase"], (p["a"], p["b"]),
                        textcoords="offset points", xytext=(7, 5),
                        fontsize=self.v_label_size.get() + 1, color="#111",
                        ha="left", va="bottom", zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])
                else:
                    lx, ly = p["a"] + dx, p["b"] + dy
                    ax.plot([p["a"], lx], [p["b"], ly],
                            color="#444", lw=0.8, zorder=6)
                    ax.annotate(
                        p["phase"], (lx, ly),
                        fontsize=self.v_label_size.get() + 1, color="#111",
                        ha="left", va="bottom", zorder=7,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

        # Legend: one handle per unique (composition, colour, marker), visible only
        seen = []
        for p in self.points:
            if not p.get("visible", True):
                continue
            st = self._point_style(p)
            key = (p["composition"], st["colour"], st["marker"])
            if key not in seen:
                seen.append(key)
        handles = []
        for comp, colour, marker in seen:
            handles.append(Line2D([0], [0], marker=marker, color="none",
                                  markerfacecolor=colour,
                                  markeredgecolor="#222", markersize=9,
                                  label=comp))
        if handles:
            ncol = 1 if len(handles) <= 16 else 2
            ax.legend(handles=handles, loc="upper left",
                      bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
                      ncol=ncol, fontsize=8, framealpha=1.0,
                      edgecolor="#ccc")

        ax.set_box_aspect(1)
        ax.set_xlim(a0, a1); ax.set_ylim(b0, b1)
        ax.set_xlabel("a*   (green − / + red)", fontsize=13)
        ax.set_ylabel("b*   (blue − / + yellow)", fontsize=13)
        ax.set_title(f"CIELab a*–b*  at  L* = {L}", fontsize=14, pad=10)
        ax.tick_params(labelsize=11)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

        name = f"cielab_L{L}_{datetime.now():%Y%m%d}.png"
        path = filedialog.asksaveasfilename(
            title="Save PNG", defaultextension=".png", initialfile=name,
            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            messagebox.showinfo("Saved", f"Figure saved:\n{path}")
        plt.close(fig)

    # ════════════════════════════════════════════════════════════════════
    # Session save / load
    # ════════════════════════════════════════════════════════════════════

    def _save_session(self):
        name = f"cielab_session_{datetime.now():%Y%m%d_%H%M}.json"
        path = filedialog.asksaveasfilename(
            title="Save session", defaultextension=".json", initialfile=name,
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        state = {"points": self.points, "style": self.style,
                 "L": self.v_L.get(),
                 "label_size": self.v_label_size.get(),
                 "a_min": self.v_amin.get(), "a_max": self.v_amax.get(),
                 "b_min": self.v_bmin.get(), "b_max": self.v_bmax.get()}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, ensure_ascii=False)
        messagebox.showinfo("Saved", f"Session saved:\n{path}")

    def _load_session(self):
        path = filedialog.askopenfilename(
            title="Load session", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                s = json.load(fh)
            self.points = s.get("points", [])
            self.style  = s.get("style", {})
            self.v_L.set(int(s.get("L", 80)))
            self.v_label_size.set(int(s.get("label_size", 9)))
            self.lbl_L.config(text=str(s.get("L", 80)))
            self.v_amin.set(s.get("a_min", "-60"))
            self.v_amax.set(s.get("a_max", "60"))
            self.v_bmin.set(s.get("b_min", "-60"))
            self.v_bmax.set(s.get("b_max", "60"))
            self.selected_idx = None
            self.selected_idxs = []
            self._bg_cache.clear()
            self._refresh_table()
            self._refresh_legend()
            self._full_redraw()
            messagebox.showinfo("Loaded", f"Session loaded:\n{path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load:\n{e}")


# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    CIELabPlotter(root)
    root.mainloop()
