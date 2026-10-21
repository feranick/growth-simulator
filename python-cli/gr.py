#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
*****************************************************************************
                        Growth simulation

                     v. 2026.10.21.1

               (C) Nicola Ferralis - 2006-2026

  This program (source code and binaries) is free software;
  you can redistribute it and/or modify it under the terms of the
  GNU General Public License as published by the Free Software
  Foundation; either version 2 of the License, or any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You can find a complete copy of the GNU General Public License at:
  http://www.gnu.org/licenses/gpl.txt
*****************************************************************************

1D surface growth on a periodic substrate of L columns.

Deposition modes:
  RD   Random deposition: the particle lands on top of its own column.
  BD1  Ballistic deposition, first NN: the particle sticks to the first
       neighbour it touches -> h_i = max(h_{i-1}, h_i + 1, h_{i+1}).
  BD2  Ballistic deposition, "second NN" (as in gr.cpp v0.2.2):
       h_i = max(h_{i-1}, h_i, h_{i+1}) + 1.
Optional relaxation: after deposition, the column hands one particle to
a neighbour that is lower by more than one unit (left first, then right).

Outputs (same naming as the C++ version):
  <name>_z.dat   x, z of every deposited particle
  <name>_h.dat   t, average height <h>(t)
  <name>_w.dat   t, interface width w(t) = sqrt(<h^2> - <h>^2)
  <name>.png     summary figure (deposit, profile, <h>(t), w(t) with beta fit)

Usage examples:
  python3 gr.py                          # interactive menu (like gr.cpp)
  python3 gr.py run -L 200 -n 25 -m BD1  # direct run, 25 x 1000 particles
  python3 gr.py run -L 200 -n 25 -m RD --relax --show
  python3 gr.py run -L 200 -n 25 -m BD1 --runs 20   # ensemble-averaged w(t)
"""

import argparse
import configparser
import math
import os
import random
import sys
import time
from array import array

import numpy as np

__version__ = "2026.10.21.1"

CFG_FILE = "gr.cfg"
DEFAULTS = {
    "constant": 1.0,        # lattice constant
    "plot": True,           # make the figure
    "multfacttime": 1000,   # multiplication factor for the number of particles
}

# Expected growth exponents (w ~ t^beta) in the growth regime
BETA_THEORY = {
    ("RD", False): (0.5, "random deposition"),
    ("RD", True): (0.25, "Edwards-Wilkinson / Family"),
    ("BD1", False): (1 / 3, "KPZ"),
    ("BD2", False): (1 / 3, "KPZ"),
    ("BD1", True): (1 / 3, "KPZ"),
    ("BD2", True): (1 / 3, "KPZ"),
}


# ********************** Parameters ****************************************

def load_cfg(path=CFG_FILE):
    cfg = dict(DEFAULTS)
    cp = configparser.ConfigParser()
    if os.path.exists(path):
        try:
            cp.read(path)
            s = cp["growth"]
            cfg["constant"] = s.getfloat("constant", cfg["constant"])
            cfg["plot"] = s.getboolean("plot", cfg["plot"])
            cfg["multfacttime"] = s.getint("multfacttime", cfg["multfacttime"])
        except (configparser.Error, KeyError, ValueError):
            print(f" Could not parse {path}: using defaults.")
    else:
        save_cfg(cfg, path)
    return cfg


def save_cfg(cfg, path=CFG_FILE):
    cp = configparser.ConfigParser()
    cp["growth"] = {k: str(v) for k, v in cfg.items()}
    with open(path, "w") as f:
        cp.write(f)


# ********************** Simulation ****************************************

MODES = ("RD", "BD1", "BD2")


class Growth:
    """Incremental growth simulation (used by the CLI and by gr_gui.py).

    Call step(n) to deposit n more particles. 'mode' and 'relax' are read
    at every call, so they can be changed between calls.
    Per-particle history is kept in compact array.array buffers:
      cols  column where each particle ended up
      zp    its height
      h     average height after each deposition
      w     interface width after each deposition
    """

    def __init__(self, L, mode="BD1", relax=False, constant=1.0, seed=None):
        if mode not in MODES:
            raise ValueError("mode must be RD, BD1 or BD2")
        self.L = int(L)
        self.mode = mode
        self.relax = bool(relax)
        self.constant = constant
        self.rng = random.Random(seed)
        self.H = [0] * self.L
        self.S1 = 0         # sum of heights        (exact integers: no drift)
        self.S2 = 0         # sum of squared heights
        self.cols = array("l")
        self.zp = array("q")
        self.h = array("d")
        self.w = array("d")

    @property
    def t(self):
        return len(self.h)

    def step(self, n):
        L, H, mode, relax = self.L, self.H, self.mode, self.relax
        S1, S2 = self.S1, self.S2
        randrange = self.rng.randrange
        sqrt = math.sqrt
        cols_app, zp_app = self.cols.append, self.zp.append
        h_app, w_app = self.h.append, self.w.append

        for _ in range(n):
            i = randrange(L)
            left = (i - 1) % L          # periodic substrate
            right = (i + 1) % L

            # ---- deposition ----
            old = H[i]
            if mode == "RD":
                new = old + 1
            elif mode == "BD1":
                new = max(H[left], old + 1, H[right])
            else:  # BD2
                new = max(H[left], old, H[right]) + 1
            H[i] = new
            S1 += new - old
            S2 += new * new - old * old
            px, pz = i, new

            # ---- relaxation ----
            if relax:
                for k in (left, right):
                    if H[k] < H[i] - 1:
                        a, b = H[i], H[k]
                        H[i] = a - 1
                        H[k] = b + 1            # S1 unchanged
                        S2 += (a - 1) ** 2 - a * a + (b + 1) ** 2 - b * b
                        px, pz = k, b + 1

            cols_app(px)
            zp_app(pz)
            av = S1 / L
            h_app(av)
            w_app(sqrt(max(S2 / L - av * av, 0.0)))

        self.S1, self.S2 = S1, S2

    def result(self):
        return {"H": np.array(self.H), "xp": np.array(self.cols) * self.constant,
                "zp": np.array(self.zp), "h": np.array(self.h),
                "w": np.array(self.w), "L": self.L, "mode": self.mode,
                "relax": self.relax, "constant": self.constant}


def simulate(L, nsteps, mode="BD1", relax=False, constant=1.0, seed=None,
             progress=True):
    """Run the growth simulation.

    Returns a dict with:
      H    final column heights (int array, length L)
      xp   x of each deposited particle (float array, length nsteps)
      zp   z of each deposited particle (int array, length nsteps)
      h    average height after each deposition (length nsteps)
      w    interface width after each deposition (length nsteps)
    """
    g = Growth(L, mode, relax, constant, seed)
    t0 = time.time()
    chunk = max(nsteps // 20, 1)
    while g.t < nsteps:
        g.step(min(chunk, nsteps - g.t))
        if progress:
            sys.stdout.write(f"\r Depositing... {100 * g.t // nsteps:3d}%")
            sys.stdout.flush()
    if progress:
        print(f"\r Deposited {nsteps} particles in {time.time() - t0:.2f} s")
    return g.result()


def auto_fit_window(t, w):
    """Growth-regime window: skip the early transient, stop before saturation.

    tmin = 1 ML (or the first point for very short runs).
    tmax = end of run, unless w has saturated (the width at half the run is
    already > 90% of the late-time width): then the first time w reaches
    80% of the late-time width.
    """
    n = len(w)
    tmin = 1.0 if t[-1] > 3.0 else t[0]
    w_late = np.median(w[int(0.8 * n):])
    tmax = t[-1]
    if w_late > 0 and np.median(w[int(0.45 * n):int(0.55 * n)]) > 0.9 * w_late:
        reached = np.nonzero(w >= 0.8 * w_late)[0]
        if len(reached):
            tmax = t[reached[0]]
    if tmax <= 3 * tmin:            # window too short: fall back to whole run
        tmin, tmax = t[0], t[-1]
    return tmin, tmax


def fit_beta(t, w, tmin=None, tmax=None):
    """Least-squares fit of log w = beta log t + c over [tmin, tmax]."""
    if tmin is None or tmax is None:
        tmin, tmax = auto_fit_window(t, w)
    m = (t >= tmin) & (t <= tmax) & (w > 0)
    if m.sum() < 3:
        return None, None
    # log-spaced subsample so every decade weighs the same
    idx = np.unique(np.geomspace(1, m.sum(), 200).astype(int) - 1)
    tt, ww = t[m][idx], w[m][idx]
    beta, c = np.polyfit(np.log(tt), np.log(ww), 1)
    return beta, c


# ********************** Output ********************************************

def base_name(L, mode, relax, maxtime):
    tag = {"RD": "_RD", "BD1": "_BD", "BD2": "_BD2nn"}[mode]
    return f"{L}{tag}{'r' if relax else ''}{maxtime}"


def save_data(res, name):
    n = len(res["h"])
    t = np.arange(1, n + 1)
    np.savetxt(f"{name}_z.dat", np.column_stack([res["xp"], res["zp"]]),
               fmt=["%g", "%d"], delimiter="\t")
    np.savetxt(f"{name}_h.dat", np.column_stack([t, res["h"]]),
               fmt=["%d", "%.6g"], delimiter="\t")
    np.savetxt(f"{name}_w.dat", np.column_stack([t, res["w"]]),
               fmt=["%d", "%.6g"], delimiter="\t")
    print(f" Saved: {name}_z.dat, {name}_h.dat, {name}_w.dat")


def plot_results(res, name, show=False, fit_range=None, max_cells=40_000_000,
                 fmt="png"):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L, c = res["L"], res["constant"]
    n = len(res["h"])
    t_ml = np.arange(1, n + 1) / L                  # time in monolayers
    mode_lbl = {"RD": "Random deposition", "BD1": "Ballistic deposition (1st NN)",
                "BD2": "Ballistic deposition (2nd NN)"}[res["mode"]]
    if res["relax"]:
        mode_lbl += " + relaxation"

    fig = plt.figure(figsize=(13, 8.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax_dep = fig.add_subplot(gs[0, :])
    ax_h = fig.add_subplot(gs[1, 0])
    ax_w = fig.add_subplot(gs[1, 1])
    runs = res.get("runs", 1)
    fig.suptitle(f"{mode_lbl}  -  L = {L},  N = {n} particles "
                 f"({n / L:.0f} ML)"
                 + (f",  <h>, w averaged over {runs} runs" if runs > 1 else ""),
                 fontsize=13)

    # ---- deposit: every particle coloured by arrival time, voids white ----
    zmax = int(res["zp"].max())
    if L * zmax <= max_cells:
        grid = np.full((zmax, L), np.nan)
        cols = np.rint(res["xp"] / c).astype(int) % L
        grid[res["zp"] - 1, cols] = t_ml        # later particles overwrite
        sc = ax_dep.imshow(grid, origin="lower", aspect="auto", cmap="viridis",
                           interpolation="nearest", extent=(0, L * c, 0, zmax))
    else:                                       # very large deposits
        step = max(1, n // 200_000)
        sc = ax_dep.scatter(res["xp"][::step] + 0.5 * c, res["zp"][::step] - 0.5,
                            c=t_ml[::step], s=2, marker="s", cmap="viridis",
                            linewidths=0, rasterized=True)
    x_prof = np.arange(L + 1) * c
    ax_dep.step(x_prof, np.append(res["H"], res["H"][0]), where="post",
                color="crimson", lw=1, label="final interface")
    ax_dep.set_xlim(0, L * c)
    ax_dep.set_ylim(0, res["H"].max() * 1.05 + 1)
    ax_dep.set_xlabel("X [arb. units]")
    ax_dep.set_ylabel("Z [arb. units]")
    ax_dep.legend(loc="upper right", fontsize=9)
    cb = fig.colorbar(sc, ax=ax_dep, pad=0.01)
    cb.set_label("deposition time [ML]")

    # ---- average height ----
    ax_h.plot(t_ml, res["h"], color="tab:blue", lw=1.2, label=r"$\langle h\rangle$")
    ax_h.plot(t_ml, t_ml, "k--", lw=0.8, label="compact film (1 ML / ML)")
    ax_h.set_xlabel("time [ML]")
    ax_h.set_ylabel(r"average Z  $\langle h \rangle$")
    ax_h.grid(alpha=0.3)
    ax_h.legend(fontsize=9)
    rho = t_ml[-1] / res["h"][-1] if res["h"][-1] > 0 else float("nan")
    ax_h.text(0.97, 0.05, f"film density = {rho:.3f}", transform=ax_h.transAxes,
              ha="right", va="bottom", fontsize=9)

    # ---- interface width, log-log with beta fit ----
    ax_w.loglog(t_ml, res["w"], color="tab:green", lw=1.2, label="w(t)")
    tmin, tmax = fit_range if fit_range else auto_fit_window(t_ml, res["w"])
    beta, cc = fit_beta(t_ml, res["w"], tmin, tmax)
    if beta is not None:
        tf = np.geomspace(max(tmin, t_ml[0]), min(tmax, t_ml[-1]), 50)
        ax_w.axvspan(tf[0], tf[-1], color="0.85", alpha=0.5, lw=0,
                     label="fit window")
        ax_w.loglog(tf, np.exp(cc) * tf ** beta, "r--", lw=1.2,
                    label=rf"fit: $\beta$ = {beta:.3f}")
    bt, cls = BETA_THEORY[(res["mode"], res["relax"])]
    ax_w.set_title(rf"expected $\beta$ = {bt:.3g}  ({cls})", fontsize=10)
    ax_w.set_xlabel("time [ML]")
    ax_w.set_ylabel("interface width w")
    ax_w.grid(alpha=0.3, which="both")
    ax_w.legend(fontsize=9)

    out = f"{name}.{fmt}"
    fig.savefig(out, dpi=150)
    print(f" Saved: {out}")
    if beta is not None:
        print(f" Growth exponent beta = {beta:.3f}  (expected {bt:.3g}, {cls})")
    if show:
        plt.show()
    plt.close(fig)
    return beta


def run(L, maxtime, mode, relax, cfg, seed=None, show=False, fit_range=None,
        fmt="png", outdir=".", runs=1):
    """Run 'runs' independent realizations. Deposit and profile come from the
    first one; <h>(t) and w(t) = sqrt(<w^2>) are averaged over all of them."""
    nsteps = maxtime * cfg["multfacttime"]
    res = simulate(L, nsteps, mode, relax, cfg["constant"], seed)
    if runs > 1:
        h_sum, w2_sum = res["h"].copy(), res["w"] ** 2
        for k in range(1, runs):
            print(f" Realization {k + 1}/{runs}")
            s = None if seed is None else seed + k
            r = simulate(L, nsteps, mode, relax, cfg["constant"], s)
            h_sum += r["h"]
            w2_sum += r["w"] ** 2
        res["h"], res["w"] = h_sum / runs, np.sqrt(w2_sum / runs)
    res["runs"] = runs
    os.makedirs(outdir, exist_ok=True)
    name = os.path.join(outdir, base_name(L, mode, relax, maxtime))
    save_data(res, name)
    if cfg["plot"] or show:
        plot_results(res, name, show=show, fit_range=fit_range, fmt=fmt)
    return res


# ********************** Interactive menu (as in gr.cpp) *******************

def ask(prompt, cast=int, default=None):
    s = input(prompt).strip()
    if not s and default is not None:
        return default
    try:
        v = cast(s)
    except ValueError:
        return default
    return v


def manual():
    print(f"\nGrowth simulation (v. {__version__})\n")
    print(" 1. Start the growth. The program asks for:")
    print("    a. number of atoms in a 1D periodic substrate;")
    print("    b. the number of incoming atoms (or the time for the deposition);")
    print("    c. growth mode:")
    print("       i.  Ballistic growth: new atoms stick to the first or second NN;")
    print("       ii. Random growth: new atoms sit on the column they land on.")
    print(" 3. Relaxation: after deposition, atoms can move to a nearby column")
    print("    with lower height.")
    print("\nCommand line: python3 gr.py run -h")
    print("Comments or suggestions: Nicola Ferralis <ferralis@mit.edu>")


def interactive():
    cfg = load_cfg()
    while True:
        print("\n_______________________________________________________")
        print("Growth simulation\n")
        print("0) Exit\t\t\t\t2) View parameters")
        print("1) Start growth\t\t\t3) Change parameters")
        print(" \t\t\t\t4) About & Manual")
        choice = ask("", int, 0)

        if choice == 1:
            L = ask("\n Number of substrate atoms (suggested 200): ", int, 200)
            dep = ask(" Ballistic deposition (1) or Random deposition (2): ", int, 1)
            mode = "RD"
            if dep == 1:
                nn = ask(" First NN (1) or Second NN (2): ", int, 1)
                mode = "BD2" if nn == 2 else "BD1"
            maxtime = ask(f" Number of atoms to be deposited (x{cfg['multfacttime']}, "
                          "suggested '25'): ", int, 25)
            relax = ask(" Relaxation? (1: no;  2: yes): ", int, 1) == 2
            show = cfg["plot"]
            run(L, maxtime, mode, relax, cfg, show=show)
        elif choice == 2:
            print(f"\n Lattice constant: {cfg['constant']}")
            print(f" Plot: {'yes' if cfg['plot'] else 'no'}")
            print(f" Multiplication factor for the growth time: {cfg['multfacttime']}")
        elif choice == 3:
            p = ask("\n1: Restore default,  2: custom parameters,  3: no change ", int, 3)
            if p == 1:
                cfg = dict(DEFAULTS)
                print("\n Default parameters successfully restored!")
            elif p == 2:
                cfg["constant"] = ask(f" Lattice constant [{cfg['constant']}]: ",
                                      float, cfg["constant"])
                pl = ask(f" Plot (1: yes, 2: no) [{1 if cfg['plot'] else 2}]: ", int, 0)
                if pl in (1, 2):
                    cfg["plot"] = pl == 1
                cfg["multfacttime"] = ask(f" Multiplication factor "
                                          f"[{cfg['multfacttime']}]: ",
                                          int, cfg["multfacttime"])
            else:
                print("\n Parameters not updated!")
                continue
            save_cfg(cfg)
            print(" Parameters successfully changed")
        elif choice == 4:
            manual()
        else:
            print("Bye Bye")
            return


# ********************** Main **********************************************

def main():
    if len(sys.argv) == 1:
        interactive()
        return

    ap = argparse.ArgumentParser(description=f"1D growth simulation v{__version__}")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run a simulation")
    r.add_argument("-L", type=int, default=200, help="substrate size (default 200)")
    r.add_argument("-n", type=int, default=25,
                   help="particles to deposit, in units of multfacttime (default 25)")
    r.add_argument("-m", "--mode", choices=["RD", "BD1", "BD2"], default="BD1")
    r.add_argument("-r", "--relax", action="store_true", help="enable relaxation")
    r.add_argument("--mult", type=int, help="override multfacttime (default from gr.cfg)")
    r.add_argument("--constant", type=float, help="lattice constant")
    r.add_argument("--seed", type=int, help="random seed (reproducible runs)")
    r.add_argument("--runs", type=int, default=1,
                   help="independent realizations to average <h> and w over "
                        "(cleaner beta; default 1)")
    r.add_argument("--fit", type=float, nargs=2, metavar=("TMIN", "TMAX"),
                   help="beta fit window in monolayers "
                        "(default: automatic, growth regime before saturation)")
    r.add_argument("--show", action="store_true", help="open an interactive window")
    r.add_argument("--no-plot", action="store_true")
    r.add_argument("--format", default="png", choices=["png", "pdf", "svg", "eps"])
    r.add_argument("-o", "--outdir", default=".")
    a = ap.parse_args()

    cfg = load_cfg()
    if a.mult:
        cfg["multfacttime"] = a.mult
    if a.constant:
        cfg["constant"] = a.constant
    if a.no_plot:
        cfg["plot"] = False
    run(a.L, a.n, a.mode, a.relax, cfg, seed=a.seed, show=a.show,
        fit_range=a.fit, fmt=a.format, outdir=a.outdir, runs=max(1, a.runs))


if __name__ == "__main__":
    main()
