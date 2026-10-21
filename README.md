# Growth simulation

Monte Carlo simulation of 1D surface growth on a periodic substrate: random
deposition, ballistic deposition and surface relaxation, with automatic
analysis of average height, interface width and growth exponent β.

Version: **2026.10.21.1**  
(C) Nicola Ferralis, 2006–2026 — GPL v2 or later

![Ballistic deposition, L = 200, 125 ML](example_BD.png)

## Files

| File         | Description                                              |
|--------------|----------------------------------------------------------|
| `gr.py`      | Simulation engine, command line and text menu            |
| `gr_gui.py`  | Tkinter GUI with live visualization (uses `gr.py`)       |
| `gr.cpp`     | Original C++ version (v0.2.2, 2014)                      |

## Requirements

- Python 3.8+
- numpy
- matplotlib
- tkinter (GUI only)

```bash
pip install numpy matplotlib
```

tkinter ships with the python.org installers. Otherwise:
`brew install python-tk` (Homebrew on macOS) or
`sudo apt install python3-tk` (Debian/Ubuntu).

## GUI

```bash
python3 gr_gui.py
```

The film grows live in the window, and the deposit, ⟨h⟩(t), w(t) and the β
fit update on every frame.

| Panel          | Controls                                                          |
|----------------|-------------------------------------------------------------------|
| Growth model   | RD / BD1 / BD2, relaxation, restart-on-change                     |
| Substrate      | Sites L (20–1000), lattice constant                               |
| Run            | Stop at [ML], speed [particles/frame], replicas, seed             |
|                | Start/Pause, Step 1 ML, Reset, Save figure, Save data             |
| Display        | Follow the surface, view height, colormap, interface, log-log, theory slope |
| Beta fit       | Automatic window, or manual From/To [ML]                          |

- Changing **L, replicas or seed** restarts the growth (it keeps running if
  it was running).
- Changing the **model or relaxation** restarts by default. Untick *Restart
  when the model changes* to switch the model mid-growth instead; the switch
  is marked by a vertical line on the time plots, so crossovers can be
  studied.
- **Speed**, **Stop at**, the lattice constant, and all display and fit
  options apply immediately without restarting.
- **Replicas** grow N independent films in lockstep; ⟨h⟩ and w are averaged
  over them, and the deposit panel shows the first one.
- **Follow the surface** shows only the top *view height* rows, useful for
  thick films. Above about 20 million sites the bottom of the film is no
  longer stored for display (the statistics are unaffected).
- The colour scale of the deposit runs from 0 to *Stop at*.
- **Save data** writes the same `_z`, `_h`, `_w` files as the command line.
- Keys: `Space` start/pause, `S` step 1 ML, `R` reset. Pause before using
  the matplotlib toolbar to zoom, since a running simulation rescales the axes.

## Quick start (command line)

Interactive menu (same flow as the original C++ program):

```bash
python3 gr.py
```

Direct run from the command line:

```bash
# Ballistic deposition, 200 sites, 25 x 1000 particles
python3 gr.py run -L 200 -n 25 -m BD1

# Random deposition with relaxation, open an interactive plot window
python3 gr.py run -L 200 -n 25 -m RD --relax --show

# Average w(t) over 20 independent runs for a cleaner beta
python3 gr.py run -L 200 -n 25 -m BD1 --runs 20 --seed 1
```

## Growth models

At each time step a particle lands on a random column `i` of a substrate with
`L` sites and periodic boundaries.

| Mode  | Rule                                         | Expected β          |
|-------|----------------------------------------------|---------------------|
| `RD`  | `h_i = h_i + 1`                              | 1/2 (random)        |
| `BD1` | `h_i = max(h_{i-1}, h_i + 1, h_{i+1})`       | 1/3 (KPZ)           |
| `BD2` | `h_i = max(h_{i-1}, h_i, h_{i+1}) + 1`       | 1/3 (KPZ)           |

`BD1` is standard ballistic deposition, where the particle sticks to the first
neighbour it touches. `BD2` is the "second NN" rule from the original program.

**Relaxation** (`-r`, `--relax`): after deposition, column `i` hands one
particle to each neighbour (left first, then right) that is lower by more than
one unit. `RD` with relaxation is in the Edwards–Wilkinson class (β = 1/4).

## Command-line options

`python3 gr.py run [options]`

| Option                  | Default | Description                                           |
|-------------------------|---------|-------------------------------------------------------|
| `-L`                    | 200     | Number of substrate sites                             |
| `-n`                    | 25      | Particles to deposit, in units of `multfacttime`      |
| `-m`, `--mode`          | `BD1`   | `RD`, `BD1` or `BD2`                                  |
| `-r`, `--relax`         | off     | Enable relaxation                                     |
| `--runs N`              | 1       | Independent runs to average `<h>` and `w` over        |
| `--seed S`              | random  | Random seed (run `k` uses `S + k`)                    |
| `--fit TMIN TMAX`       | auto    | β fit window, in monolayers                           |
| `--mult M`              | cfg     | Override `multfacttime`                               |
| `--constant A`          | cfg     | Lattice constant                                      |
| `--show`                | off     | Open an interactive matplotlib window                 |
| `--no-plot`             | off     | Write data files only                                 |
| `--format`              | `png`   | Figure format: `png`, `pdf`, `svg`, `eps`             |
| `-o`, `--outdir`        | `.`     | Output folder                                         |

The total number of particles is `n × multfacttime` (default 25 × 1000 = 25,000).

## Configuration

`gr.cfg` is created on first run and can be edited from menu option 3 or by
hand:

```ini
[growth]
constant = 1.0
plot = True
multfacttime = 1000
```

## Output

Files are named `<L>_<mode>[r]<n>`, for example `200_BD25`, `200_RDr25` or
`200_BD2nn25`.

| File           | Columns                     | Content                                |
|----------------|-----------------------------|----------------------------------------|
| `*_z.dat`      | x, z                        | Final position of every particle       |
| `*_h.dat`      | t, ⟨h⟩                      | Average height after each deposition   |
| `*_w.dat`      | t, w                        | Interface width after each deposition  |
| `*.png`        | —                           | Summary figure                         |

Time `t` in the data files is the number of deposited particles. The figure
uses monolayers (ML = t / L).

The interface width is `w = sqrt(<h²> − <h>²)`. With `--runs N`, `<h>` is
averaged over runs and `w` is `sqrt(<w²>)`. The deposit panel always shows
the first run.

### Figure panels

1. **Deposit**: each particle coloured by arrival time, voids in white, final
   interface in red.
2. **Average height** compared with a compact film (1 ML per ML), with the
   film density printed (about 0.48 for ballistic deposition).
3. **Interface width** on log–log axes with a power-law fit `w ~ t^β` and the
   expected β for the chosen model.

The automatic fit window starts at 1 ML and runs to the end, or stops where
`w` reaches 80% of its late-time value if the width has saturated. Use `--fit`
to set it by hand.

## Notes on accuracy

- A single run is noisy; use `--runs 10` or more for a reliable β.
- Ballistic deposition approaches β = 1/3 slowly; short runs on small
  substrates typically give 0.37–0.39.
- Saturation sets in at about `L^z` monolayers (z = 3/2 for KPZ, 2 for EW), so
  larger `L` gives a longer growth regime.

## Changes from the C++ version (v0.2.2, 2014)

- The x position now uses the column where the particle lands (the old code
  used an uninitialised index).
- Neighbour lookups wrap around at the edges (periodic substrate) instead of
  reading outside the array.
- The time loop runs sequentially; the OpenMP parallel loop had a race
  condition because each step depends on the previous one.
- `*_z.dat` now has one line per particle instead of repeating the last one.
- ⟨h⟩ and w are updated incrementally with exact integer sums: O(1) per step.
- Plots use matplotlib instead of gnuplot; the unused precision parameter was
  removed; `BD2` files are named `_BD2nn` to avoid ambiguous names.
- New: β fit, ensemble averaging (`--runs`), reproducible seeds, CLI options,
  and an interactive GUI (`gr_gui.py`).

## Using the engine from Python

`gr.Growth` is an incremental simulator that both front-ends use:

```python
import numpy as np
import gr
g = gr.Growth(L=200, mode="BD1", relax=False, seed=1)
g.step(25_000)            # deposit more particles; mode/relax can change between calls
res = g.result()          # dict: H, xp, zp, h, w, ...
beta, _ = gr.fit_beta(np.arange(1, g.t + 1) / g.L, res["w"])
```

## License

GNU General Public License v2 or later: <http://www.gnu.org/licenses/gpl.txt>
