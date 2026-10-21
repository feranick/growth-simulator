#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*****************************************************************************
                    Growth simulation - interactive GUI

                     v. 2026.10.21.1

               (C) Nicola Ferralis - 2006-2026

  This program is free software; you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation; either version 2 of the License, or any later
  version. See http://www.gnu.org/licenses/gpl.txt
*****************************************************************************

Tkinter front-end for gr.py: the film grows live on screen and every
parameter can be changed while it runs.

  python3 gr_gui.py

Keys: Space = start/pause, S = step 1 ML, R = reset.
"""

import math
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import gr

__version__ = gr.__version__

MODE_LABELS = {"RD": "Random deposition",
               "BD1": "Ballistic deposition (1st NN)",
               "BD2": "Ballistic deposition (2nd NN)"}
CMAPS = ("viridis", "plasma", "inferno", "magma", "cividis", "turbo",
         "coolwarm", "Greys")


def take(buf, idx):
    """Copy buf[idx] out of an array.array without keeping a buffer export
    alive (array.array cannot grow while numpy views it)."""
    v = np.frombuffer(buf, dtype=np.dtype(buf.typecode))
    out = v[idx]
    del v
    return out


# ********************** Simulation session ********************************

class Session:
    """One or more independent replicas grown in lockstep."""

    def __init__(self, L, mode, relax, constant, seed, replicas):
        self.L = L
        self.reps = [gr.Growth(L, mode, relax, constant,
                               None if seed is None else seed + k)
                     for k in range(max(1, replicas))]
        self.events = []        # (t particles, label) when the model changes

    @property
    def g(self):
        return self.reps[0]

    @property
    def t(self):
        return self.reps[0].t

    def set_model(self, mode, relax):
        for g in self.reps:
            g.mode, g.relax = mode, relax
        if self.t > 0:
            self.events.append((self.t, mode + (" + relax" if relax else "")))

    def set_constant(self, c):
        for g in self.reps:
            g.constant = c

    def advance(self, n, budget_s=None):
        """Deposit up to n particles per replica; stop early after budget_s."""
        done, t0 = 0, time.perf_counter()
        chunk = max(200, 4000 // len(self.reps))
        while done < n:
            k = min(chunk, n - done)
            for g in self.reps:
                g.step(k)
            done += k
            if budget_s is not None and time.perf_counter() - t0 > budget_s:
                break
        return done

    def series(self, idx):
        """<h> and w = sqrt(<w^2>) over replicas at time indices idx."""
        if len(self.reps) == 1:
            return take(self.g.h, idx), take(self.g.w, idx)
        h = np.mean([take(g.h, idx) for g in self.reps], axis=0)
        w = np.sqrt(np.mean([take(g.w, idx) ** 2 for g in self.reps], axis=0))
        return h, w

    def result(self):
        res = self.g.result()
        if len(self.reps) > 1:
            res["h"] = np.mean([np.array(g.h) for g in self.reps], axis=0)
            res["w"] = np.sqrt(np.mean([np.array(g.w) ** 2 for g in self.reps],
                                       axis=0))
        res["runs"] = len(self.reps)
        return res


class DepositGrid:
    """Image of the deposit (arrival time per site, NaN = void), filled
    incrementally. Keeps at most max_cells sites: for very thick films the
    bottom of the film is dropped and only the top is kept."""

    def __init__(self, L, max_cells=20_000_000):
        self.L = L
        self.cap = max(256, max_cells // L)
        self.z0, self.rows = 0, min(256, self.cap)
        self.a = np.full((self.rows, L), np.nan, np.float32)
        self.drawn = 0

    def _rewindow(self, z0, rows):
        new = np.full((rows, self.L), np.nan, np.float32)
        lo, hi = max(self.z0, z0), min(self.z0 + self.rows, z0 + rows)
        if hi > lo:
            new[lo - z0:hi - z0] = self.a[lo - self.z0:hi - self.z0]
        self.a, self.z0, self.rows = new, z0, rows

    def add(self, g):
        t = g.t
        if t <= self.drawn:
            return
        c = np.array(g.cols[self.drawn:t])
        z = np.array(g.zp[self.drawn:t]) - 1
        tt = np.arange(self.drawn + 1, t + 1, dtype=np.float32) / self.L
        self.drawn = t
        zmax = int(z.max())
        need = zmax - self.z0 + 1
        if need > self.rows:
            if need <= self.cap:
                self._rewindow(self.z0, min(self.cap, max(need, 2 * self.rows)))
            else:
                self._rewindow(zmax + 1 - (3 * self.cap) // 4, self.cap)
        m = z >= self.z0
        self.a[z[m] - self.z0, c[m]] = tt[m]

    def view(self, lo, hi):
        lo, hi = max(lo, self.z0), min(hi, self.z0 + self.rows)
        return self.a[lo - self.z0:hi - self.z0], lo, hi


# ********************** Figure ********************************************

class LiveView:
    """Matplotlib figure updated in place (no Tk dependency)."""

    def __init__(self, fig):
        self.fig = fig
        # fixed margins: a layout engine would re-run on every frame
        gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1], left=0.07,
                              right=0.95, bottom=0.07, top=0.93, hspace=0.3,
                              wspace=0.2)
        self.ax_dep = fig.add_subplot(gs[0, :])
        self.ax_h = fig.add_subplot(gs[1, 0])
        self.ax_w = fig.add_subplot(gs[1, 1])

        a = self.ax_dep
        self.im = a.imshow(np.full((1, 1), np.nan), origin="lower", aspect="auto",
                           interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
        self.cb = fig.colorbar(self.im, ax=a, pad=0.01, fraction=0.03)
        self.cb.set_label("deposition time [ML]")
        self.prof, = a.step([], [], where="post", color="crimson", lw=1,
                            label="interface")
        a.set_xlabel("X [arb. units]")
        a.set_ylabel("Z [arb. units]")
        self.void_txt = a.text(0.01, 0.98, "", transform=a.transAxes, ha="left",
                               va="top", fontsize=8, color="0.3")

        a = self.ax_h
        self.h_line, = a.plot([], [], color="tab:blue", lw=1.2,
                              label=r"$\langle h\rangle$")
        self.c_line, = a.plot([], [], "k--", lw=0.8, label="compact film")
        self.dens_txt = a.text(0.97, 0.05, "", transform=a.transAxes, ha="right",
                               va="bottom", fontsize=9)
        a.set_xlabel("time [ML]")
        a.set_ylabel(r"average Z  $\langle h\rangle$")
        a.grid(alpha=0.3)

        a = self.ax_w
        self.w_line, = a.plot([], [], color="tab:green", lw=1.2, label="w(t)")
        self.fit_line, = a.plot([], [], "r--", lw=1.2, label="fit")
        self.th_line, = a.plot([], [], ":", color="0.25", lw=1.4, label="theory")
        a.set_xlabel("time [ML]")
        a.set_ylabel("interface width w")
        a.grid(alpha=0.3, which="both")

        self._span = None
        self._evlines = []
        self.grid = None

    def reset(self, session):
        self.grid = DepositGrid(session.L)

    def update(self, s, o):
        """s: Session; o: dict of display options (see App.options)."""
        g, L, c = s.g, s.L, o["constant"]
        t = s.t
        self.grid.add(g)

        # ---- titles ----
        lbl = MODE_LABELS[g.mode] + (" + relaxation" if g.relax else "")
        extra = f",  averaged over {len(s.reps)} runs" if len(s.reps) > 1 else ""
        self.fig.suptitle(f"{lbl}  -  L = {L},  N = {t} ({t / L:.1f} ML){extra}",
                          fontsize=12)

        # ---- deposit ----
        top = max(g.H) if t else 1
        if o["follow"]:
            lo = max(0, top - o["view_h"])
            hi = top + max(2, o["view_h"] // 20)
        else:
            lo, hi = 0, int(top * 1.05) + 2
        data, lo, hi = self.grid.view(lo, hi)
        self.im.set_data(data if data.size else np.full((1, 1), np.nan))
        self.im.set_extent((0, L * c, lo, hi))
        self.im.set_cmap(o["cmap"])
        self.im.set_clim(0, max(o["target_ml"], 1e-9))
        self.ax_dep.set_xlim(0, L * c)
        self.ax_dep.set_ylim(lo, hi)
        self.void_txt.set_text(f"film below z = {self.grid.z0} not stored"
                               if self.grid.z0 > 0 and lo <= self.grid.z0 else "")
        if o["interface"] and t:
            self.prof.set_data(np.arange(L + 1) * c, np.append(g.H, g.H[0]))
        else:
            self.prof.set_data([], [])

        for ln in self._evlines:
            ln.remove()
        self._evlines = []
        if self._span is not None:
            self._span.remove()
            self._span = None

        if t < 2:
            for ln in (self.h_line, self.c_line, self.w_line, self.fit_line,
                       self.th_line):
                ln.set_data([], [])
            self.dens_txt.set_text("")
            self.ax_w.set_title("")
            return {"t": t, "ml": t / L, "h": 0.0, "w": 0.0, "rho": float("nan"),
                    "beta": None}

        # ---- time series (decimated) ----
        il = np.unique(np.linspace(0, t - 1, 1500).astype(np.int64))
        hl, wl = s.series(il)
        tl = (il + 1) / L

        self.h_line.set_data(tl, hl)
        self.c_line.set_data([0, tl[-1]], [0, tl[-1]])
        rho = tl[-1] / hl[-1] if hl[-1] > 0 else float("nan")
        self.dens_txt.set_text(f"film density = {rho:.3f}")
        self.ax_h.relim()
        self.ax_h.autoscale_view()
        self.ax_h.legend(loc="upper left", fontsize=8)

        a = self.ax_w
        if o["loglog"]:
            ig = np.unique(np.geomspace(1, t, 1500).astype(np.int64) - 1)
            _, wg = s.series(ig)
            tg = (ig + 1) / L
            a.set_xscale("log")
            a.set_yscale("log")
        else:
            tg, wg = tl, wl
            a.set_xscale("linear")
            a.set_yscale("linear")
        pos = wg > 0
        self.w_line.set_data(tg[pos], wg[pos])

        if o["auto_fit"]:
            tmin, tmax = gr.auto_fit_window(tl, wl)
        else:
            tmin, tmax = o["fit_min"], o["fit_max"]
        beta, cc = gr.fit_beta(tg[pos], wg[pos], tmin, tmax)
        bt, cls = gr.BETA_THEORY[(g.mode, g.relax)]
        if beta is not None:
            tf = np.geomspace(max(tmin, tg[0]), min(tmax, tg[-1]), 50)
            self.fit_line.set_data(tf, np.exp(cc) * tf ** beta)
            self.fit_line.set_label(rf"fit: $\beta$ = {beta:.3f}")
            self._span = a.axvspan(tf[0], tf[-1], color="0.85", alpha=0.5, lw=0,
                                   zorder=0)
            if o["theory"]:
                w0 = np.interp(tf[0], tg[pos], wg[pos])
                self.th_line.set_data(tf, w0 * (tf / tf[0]) ** bt)
                self.th_line.set_label(rf"theory: $\beta$ = {bt:.3g}")
            else:
                self.th_line.set_data([], [])
        else:
            self.fit_line.set_data([], [])
            self.th_line.set_data([], [])
        a.set_title(rf"expected $\beta$ = {bt:.3g}  ({cls})", fontsize=10)

        for te, lab in s.events:
            for ax in (self.ax_h, a):
                self._evlines.append(ax.axvline(te / L, color="0.4", lw=0.8,
                                                ls="-."))
            self._evlines.append(a.text(te / L, 0.02, " " + lab, rotation=90,
                                        transform=a.get_xaxis_transform(),
                                        fontsize=7, color="0.3", va="bottom"))
        a.relim()
        a.autoscale_view()
        handles = [h for h in (self.w_line, self.fit_line, self.th_line)
                   if len(h.get_xdata())]
        a.legend(handles=handles, loc="upper left", fontsize=8)

        return {"t": t, "ml": t / L, "h": hl[-1], "w": wl[-1], "rho": rho,
                "beta": beta}


# ********************** Tk application ************************************

class App:
    def __init__(self, root):
        self.root = root
        root.title(f"Growth simulation  v{__version__}")
        root.geometry("1450x900")
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass

        cfg = gr.load_cfg()

        # ---- parameters ----
        self.mode = tk.StringVar(value="BD1")
        self.relax = tk.BooleanVar(value=False)
        self.restart_on_change = tk.BooleanVar(value=True)
        self.L = tk.IntVar(value=200)
        self.constant = tk.DoubleVar(value=cfg["constant"])
        self.target_ml = tk.DoubleVar(value=125)
        self.speed = tk.DoubleVar(value=2000)
        self.replicas = tk.IntVar(value=1)
        self.seed = tk.StringVar(value="")
        self.follow = tk.BooleanVar(value=False)
        self.view_h = tk.IntVar(value=300)
        self.cmap = tk.StringVar(value="viridis")
        self.interface = tk.BooleanVar(value=True)
        self.loglog = tk.BooleanVar(value=True)
        self.theory = tk.BooleanVar(value=True)
        self.auto_fit = tk.BooleanVar(value=True)
        self.fit_min = tk.DoubleVar(value=1.0)
        self.fit_max = tk.DoubleVar(value=100.0)
        self.status = tk.StringVar(value="Ready")

        self.running = False
        self._after = None
        self._restart_job = None
        self._rate = (time.perf_counter(), 0)
        self.pps = 0.0

        # ---- layout ----
        main = ttk.Frame(root)
        main.pack(fill="both", expand=True)
        panel = ttk.Frame(main, padding=8)
        panel.pack(side="left", fill="y")
        plot = ttk.Frame(main)
        plot.pack(side="left", fill="both", expand=True)
        ttk.Label(root, textvariable=self.status, anchor="w", relief="sunken",
                  padding=(8, 2)).pack(side="bottom", fill="x")

        self._build_controls(panel)

        self.fig = Figure(figsize=(10, 7.5))
        self.view = LiveView(self.fig)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot)
        tb = NavigationToolbar2Tk(self.canvas, plot, pack_toolbar=False)
        tb.update()
        tb.pack(side="bottom", fill="x")
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

        root.bind("<space>", lambda e: self._key(e, self.toggle))
        root.bind("<Key-s>", lambda e: self._key(e, self.step_ml))
        root.bind("<Key-r>", lambda e: self._key(e, self.reset))
        root.protocol("WM_DELETE_WINDOW", self.quit)

        self.reset()

    # ---------- widgets ----------
    def _slider(self, parent, text, var, lo, hi, res=1.0, log=False,
                fmt="{:g}", on_change=None):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        row.columnconfigure(0, weight=1)
        ttk.Label(row, text=text).grid(row=0, column=0, sticky="w")
        val = ttk.Label(row, width=10, anchor="e")
        val.grid(row=0, column=1, sticky="e")
        pos = tk.DoubleVar(value=math.log10(var.get()) if log else var.get())
        is_int = isinstance(var, tk.IntVar)

        def changed(*_):
            p = pos.get()
            if log:
                v = float(f"{10 ** p:.3g}")
            else:
                v = round(p / res) * res
            if is_int:
                v = int(round(v))
            val.config(text=fmt.format(v))
            if v != var.get():
                var.set(v)
                if on_change:
                    on_change()

        sc = ttk.Scale(row, from_=math.log10(lo) if log else lo,
                       to=math.log10(hi) if log else hi, variable=pos,
                       command=changed)
        sc.grid(row=1, column=0, columnspan=2, sticky="ew")
        changed()
        return sc

    def _check(self, parent, text, var, cmd):
        ttk.Checkbutton(parent, text=text, variable=var, command=cmd).pack(
            anchor="w", pady=1)

    def _build_controls(self, p):
        box = ttk.LabelFrame(p, text="Growth model", padding=6)
        box.pack(fill="x", pady=3)
        for m in gr.MODES:
            ttk.Radiobutton(box, text=MODE_LABELS[m], value=m, variable=self.mode,
                            command=self.model_changed).pack(anchor="w")
        self._check(box, "Relaxation", self.relax, self.model_changed)
        self._check(box, "Restart when the model changes", self.restart_on_change,
                    lambda: None)

        box = ttk.LabelFrame(p, text="Substrate", padding=6)
        box.pack(fill="x", pady=3)
        self._slider(box, "Sites L", self.L, 20, 1000, res=10,
                     on_change=self.schedule_restart)
        self._slider(box, "Lattice constant", self.constant, 0.1, 5.0, res=0.1,
                     fmt="{:.1f}", on_change=self.constant_changed)

        box = ttk.LabelFrame(p, text="Run", padding=6)
        box.pack(fill="x", pady=3)
        self._slider(box, "Stop at [ML]", self.target_ml, 1, 5000, log=True,
                     on_change=self.redraw)
        self._slider(box, "Speed [particles/frame]", self.speed, 1, 100000,
                     log=True)
        self._slider(box, "Replicas (averaged)", self.replicas, 1, 20,
                     on_change=self.schedule_restart)
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Seed (blank = random)").pack(side="left")
        e = ttk.Entry(row, textvariable=self.seed, width=8)
        e.pack(side="right")
        e.bind("<Return>", lambda ev: self.schedule_restart())
        e.bind("<FocusOut>", lambda ev: self.schedule_restart())

        row = ttk.Frame(box)
        row.pack(fill="x", pady=(6, 2))
        self.btn_run = ttk.Button(row, text="Start", command=self.toggle)
        self.btn_run.pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Step 1 ML", command=self.step_ml).pack(
            side="left", expand=True, fill="x")
        ttk.Button(row, text="Reset", command=self.reset).pack(
            side="left", expand=True, fill="x")
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Button(row, text="Save figure...", command=self.save_figure).pack(
            side="left", expand=True, fill="x")
        ttk.Button(row, text="Save data...", command=self.save_data).pack(
            side="left", expand=True, fill="x")

        box = ttk.LabelFrame(p, text="Display", padding=6)
        box.pack(fill="x", pady=3)
        self._check(box, "Follow the surface", self.follow, self.redraw)
        self._slider(box, "View height", self.view_h, 20, 5000, log=True,
                     on_change=self.redraw)
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="Colormap").pack(side="left")
        cb = ttk.Combobox(row, textvariable=self.cmap, values=CMAPS, width=10,
                          state="readonly")
        cb.pack(side="right")
        cb.bind("<<ComboboxSelected>>", lambda e: self.redraw())
        self._check(box, "Show interface", self.interface, self.redraw)
        self._check(box, "Log-log width plot", self.loglog, self.redraw)
        self._check(box, "Show theory slope", self.theory, self.redraw)

        box = ttk.LabelFrame(p, text="Beta fit", padding=6)
        box.pack(fill="x", pady=3)
        self._check(box, "Automatic window", self.auto_fit, self.redraw)
        self._slider(box, "From [ML]", self.fit_min, 0.01, 5000, log=True,
                     on_change=self.redraw)
        self._slider(box, "To [ML]", self.fit_max, 0.01, 5000, log=True,
                     on_change=self.redraw)

        ttk.Label(p, text="Space: start/pause   S: step   R: reset",
                  foreground="gray").pack(anchor="w", pady=(8, 0))

    def _key(self, event, fn):
        if not isinstance(event.widget, (tk.Entry, ttk.Entry, ttk.Combobox)):
            fn()

    # ---------- state ----------
    def options(self):
        return {"constant": self.constant.get(), "target_ml": self.target_ml.get(),
                "follow": self.follow.get(), "view_h": self.view_h.get(),
                "cmap": self.cmap.get(), "interface": self.interface.get(),
                "loglog": self.loglog.get(), "theory": self.theory.get(),
                "auto_fit": self.auto_fit.get(), "fit_min": self.fit_min.get(),
                "fit_max": self.fit_max.get()}

    def _seed(self):
        s = self.seed.get().strip()
        try:
            return int(s) if s else None
        except ValueError:
            messagebox.showwarning("Seed", "The seed must be an integer.")
            self.seed.set("")
            return None

    def reset(self):
        self._restart_job = None
        self.session = Session(self.L.get(), self.mode.get(), self.relax.get(),
                               self.constant.get(), self._seed(),
                               self.replicas.get())
        self.view.reset(self.session)
        self._rate = (time.perf_counter(), 0)
        self.redraw()

    def schedule_restart(self):
        """Debounced restart (sliders fire many events while dragged)."""
        if self._restart_job is not None:
            self.root.after_cancel(self._restart_job)
        self._restart_job = self.root.after(300, self.reset)

    def model_changed(self):
        if self.restart_on_change.get() or self.session.t == 0:
            self.schedule_restart()
        else:
            self.session.set_model(self.mode.get(), self.relax.get())
            self.redraw()

    def constant_changed(self):
        self.session.set_constant(self.constant.get())
        self.redraw()

    def target(self):
        return int(round(self.target_ml.get() * self.session.L))

    # ---------- run loop ----------
    def toggle(self):
        if self.running:
            self.pause()
        else:
            if self.session.t >= self.target():
                self.status.set("Target reached: raise 'Stop at' or press Reset.")
                return
            self.running = True
            self.btn_run.config(text="Pause")
            self._rate = (time.perf_counter(), self.session.t)
            self._tick()

    def pause(self):
        self.running = False
        self.btn_run.config(text="Start")
        if self._after is not None:
            self.root.after_cancel(self._after)
            self._after = None
        self.redraw()

    def _tick(self):
        self._after = None
        if not self.running:
            return
        n = min(int(self.speed.get()), self.target() - self.session.t)
        if n > 0:
            self.session.advance(n, budget_s=0.06)
        t0, n0 = self._rate
        dt = time.perf_counter() - t0
        if dt > 0.5:
            self.pps = (self.session.t - n0) * len(self.session.reps) / dt
            self._rate = (time.perf_counter(), self.session.t)
        self.redraw()
        if self.session.t >= self.target():
            self.pause()
            return
        self._after = self.root.after(10, self._tick)

    def step_ml(self):
        if self.running:
            self.pause()
        self.session.advance(self.session.L)
        self.redraw()

    def redraw(self):
        if not hasattr(self, "view") or not hasattr(self, "session"):
            return
        st = self.view.update(self.session, self.options())
        self.canvas.draw_idle()
        beta = f"{st['beta']:.3f}" if st["beta"] is not None else "-"
        state = "running" if self.running else "paused"
        self.status.set(
            f"{state}   N = {st['t']:,}   t = {st['ml']:.1f} ML   "
            f"<h> = {st['h']:.2f}   w = {st['w']:.3f}   density = {st['rho']:.3f}"
            f"   beta = {beta}   speed = {self.pps:,.0f} particles/s")

    # ---------- files ----------
    def save_figure(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png", initialfile=self._name() + ".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("SVG", "*.svg"),
                       ("EPS", "*.eps")])
        if path:
            self.fig.savefig(path, dpi=150)
            self.status.set(f"Saved {path}")

    def save_data(self):
        if self.session.t == 0:
            return
        folder = filedialog.askdirectory(title="Folder for the .dat files")
        if folder:
            name = f"{folder}/{self._name()}"
            gr.save_data(self.session.result(), name)
            self.status.set(f"Saved {name}_z.dat, _h.dat, _w.dat")

    def _name(self):
        g = self.session.g
        return gr.base_name(g.L, g.mode, g.relax, f"_{self.session.t / g.L:.0f}ML")

    def quit(self):
        self.running = False
        if self._after is not None:
            self.root.after_cancel(self._after)
        self.root.quit()
        self.root.destroy()


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
