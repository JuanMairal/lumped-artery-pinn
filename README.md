# lumped-artery-pinn

A Physics-Informed Neural Network (PINN) framework for calibrating 0-dimensional (lumped-parameter) arterial network models. The neural network simultaneously fits measured hemodynamic data and enforces cardiovascular physical laws, enabling parameter identification in branching arterial trees.

## Overview

Arterial networks can be modeled as electrical circuits where vessels have resistance, compliance, and inertance. Calibrating these parameters from partial measurements is an inverse problem. lumped-artery-pinn solves this by training a neural network whose outputs represent pressure and flow throughout the network over one cardiac cycle, subject to:

- Physics residuals (vessel momentum, junction continuity, bed RCR dynamics)
- Periodicity constraints (cardiac cycle)
- Fit to available pressure and flow measurements

The result is a calibrated model that predicts hemodynamics at any vessel, including unmeasured locations.

## Repository Structure

```
lumped-artery-pinn/
├── main.py              # Main entry point
├── config.py            # All hyperparameters and case settings
├── src/
│   ├── pinn.py          # Neural network architecture, loss functions, training loop
│   ├── tree.py          # Arterial network topology and parameter computation
│   ├── load.py          # Input file parsers
│   ├── plot.py          # Visualization utilities
│   └── report.py        # LaTeX report generation
├── input/
│   └── invivo/          # Example case: in-vivo thoracic aorta data (default)
└── output/              # Generated outputs (timestamped per run)
```

## Physics

### Governing Equations

The 0D model enforces four sets of equations:

**Vessel momentum** (pressure-flow relationship along each vessel):
```
ΔP = R·Q + L·dQ/dt
```

**Junction continuity** (mass conservation at bifurcations):
```
Q_upstream = Σ Q_downstream
```

**Inlet compliance**:
```
C_inlet · dP/dt = Q_inlet - Σ Q_outlet
```

**Terminal bed (RCR model)**:
```
P_bed - P_out = R_bed · (Q_bed - C_bed · dP_bed/dt)
```

**Periodicity**:
```
P(0) = P(T),   Q(0) = Q(T)
```

### Vessel Parameters

Lumped parameters are computed from geometry and wave speed, then corrected for pressure-dependent deformation (nonlinear compliance):

| Parameter | Symbol | Formula |
|-----------|--------|---------|
| Compliance | C₀ | V_d / (ρ · c²) |
| Resistance | R₀ | π · β · ν · l³ / V_d² |
| Inertance  | L₀ | ρ · l² / V_d |

where V_d is vessel volume, c is pulse wave speed, l is length, ρ is blood density, ν is viscosity.

### Notation

**Tree size**: `Nv` vessels, `Nj` junctions, `Nb` terminal beds, with `Nv = Nj + Nb` (every vessel ends either at a junction or a terminal bed).

**Pressure/flow arrays**: `P_all`/`Q_all` hold one value per vessel end. Column 0 is the network inlet; column `v+1` is the downstream (distal) pressure/flow of vessel `v`.

**Incidence matrices** (built in `src/tree.py`, used to assemble the loss residuals in `src/pinn.py`):

| Matrix | Shape | Meaning |
|---|---|---|
| `mJ` | `(Nj, Nv+1)` | Per-junction flow balance: `+1` at the upstream vessel's column, `-1` at each downstream vessel's column |
| `mJD` | `(Nj, Nv)` | 0/1 mask selecting a junction's downstream vessels |
| `mJid` | `(Nj, Nv+1)` | Maps each junction to its pressure column in `P_all` |
| `mV` | `(Nv, Nv+1)` | Per-vessel pressure drop: `+1` at the upstream pressure column, `-1` at the downstream (`v+1`) column |
| `mVid` | `(Nv, Nv+1)` | Maps each vessel to its downstream pressure column |
| `mB` / `mBid` | `(Nb, Nv+1)` | Maps each terminal bed to its vessel's `Q_all`/`P_all` column |

**Terminal bed (RCR) parameters**, from `beds.input`: `Zb` (characteristic impedance), `Cb` (peripheral compliance), `Rb` (peripheral resistance), `Pout` (venous outflow pressure), `Vfrac` (fraction of total flow/resistance assigned to that bed). `RT` is the total peripheral resistance across all beds, used when `trainable_params` includes `"Rb"` (all `Rb`/`Cb` are then re-derived from a single trained `RT` rather than trained individually).

`c0` (in `trainable_params`) is the reference/diastolic pulse wave velocity — see [Vessel Parameters](#vessel-parameters) above.

## Installation

**Dependencies**:
```
tensorflow >= 2.x
numpy
matplotlib
networkx
pandas
scipy
```

Install with pip:
```bash
pip install tensorflow numpy matplotlib networkx pandas scipy
```

## Usage

### 1. Configure the Case

Edit [config.py](config.py) to select the case and set training parameters:

```python
# Case selection
origin   = "input"        # Input data root folder
casename = "invivo"       # Subfolder with input files

# Known signals (what measurements are available)
known_signals = [("Q", 0), ("P", 4), ("P", 5), ("P", 6), ("P", 7)]
#                type  vessel_index

# Whether bed (RCR) parameters are provided in beds.input or estimated from data
known_beds = False

# Training
max_epochs = 2_000_000
ncoloc     = 256          # Collocation points in time
```

### 2. Prepare Input Files

Place four input files in `input/<casename>/`:

#### `tree.input` — Vessel geometry

```
number_vessels 7
#   cm      mm      mm      cm/s
n   length  r_in    r_out   c_avg    tag
1   6.58    12.3    12.3    456      AoI
2   0.57    12.3    11.8    456      AoII
...
```

#### `junctions.input` — Network topology

```
number_junctions 3
1; 2 5       # vessel 1 upstream → vessels 2 and 5 downstream
2; 3 6
3; 4 7
```

#### `beds.input` — Terminal RCR beds

```
number_beds 4
#   mmHg/mls  ml/mmHg  mmHg/mls  mmHg
n   vessel    Zb       Cb        Rb    Pout  Vfrac
1   5         0.618    0.193     0.788 65.55 19.3
...
```

If `known_beds = False`, bed parameters are estimated from diastolic decay of the measured signals.

#### `cardiac.input` — Pulse parameters

```
ps(mmHg)  117.74
pd(mmHg)   73.83
qa(ml/s)   95.40
tm(s)       0.992
```

#### `data.input` — Measured time series

```
# T  s
# P  mmHg
# Q  ml/s
type  vessel  time    value
P     0       0.000   72.33
P     0       0.005   72.19
Q     0       0.000    0.451
...
```

### 3. Run

```bash
python main.py
```

Outputs are written to `output/<casename>_<timestamp>/`.

Common settings can also be overridden from the command line without editing `config.py`:

```bash
python main.py --seed 3 --hidden_layers 5 --neurons_per_layer 64 --max_epochs 500000 --name my_run
```

These flags work by patching the already-imported `config` module's values in place, at the very top of `main.py`, before any other module is imported. Everything downstream (`src/tree.py`, `src/pinn.py`, ...) imports its settings from `config` afterwards, so it sees the overridden values transparently — this is why the CLI-parsing block sits above the rest of `main.py`'s imports rather than below them.

## Output

Each run generates a timestamped directory:

```
output/<casename>_<timestamp>/
├── tree_graph.svg              # Network topology diagram
├── loss_evolution.svg          # Training loss curves
├── predictions/
│   ├── predP*.svg              # Predicted vs. measured pressure
│   └── predQ*.svg              # Predicted vs. measured flow
└── latex/
    ├── report.tex              # Auto-generated LaTeX report
    ├── report.pdf              # Compiled PDF
    └── figures/                # All figures embedded in the report
```

The report includes:
- Vessel geometry and computed lumped parameters (C₀, R₀, L₀)
- Bed RCR parameters (estimated or prescribed)
- Input signal plots
- Prediction overlays for all vessels

## Configuration Reference

All settings live in [config.py](config.py).

### Neural Network

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_layers` | 5 | Number of fully connected layers |
| `neurons_per_layer` | 24 | Neurons per hidden layer |
| `activation_f` | `"tanh"` | Activation function |
| `FF` | `True` | Use Fourier feature embedding |
| `nff_harmon` | 2 | Number of harmonic Fourier features |

### Learning Rate Schedule (exponential decay)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `net_learning["initial"]` | 1e-3 | Initial LR for network weights |
| `param_learning["initial"]` | 1e-3 | Initial LR for physical parameters |
| `steps` | 1e5 | Decay step count |
| `decay` | 0.667 | Decay factor per step |

### Physical Properties

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fluid["dens"]` | 1050 kg/m³ | Blood density |
| `fluid["visc"]` | 0.0025 Pa·s | Blood viscosity |
| `fluid["beta"]` | 9 | Viscosity profile coefficient |
| `INER` | 1 | Inertance scaling factor (0 to disable) |

## Module Descriptions

### [src/pinn.py](src/pinn.py)
Defines the neural network, all residual loss functions, training step, and utilities for converting predictions back to physical units. Uses TensorFlow's `GradientTape` for both training gradients and physics derivative computation (`batch_jacobian`).

### [src/tree.py](src/tree.py)
Builds the arterial network data structure from input files: computes lumped parameters, nondimensionalizes variables, constructs incidence matrices for junctions and beds, estimates bed parameters when not prescribed, and generates collocation points.

### [src/load.py](src/load.py)
Parses all five input files and performs unit conversions (mmHg → Pa, cm → m, ml/s → m³/s).

### [src/plot.py](src/plot.py)
Produces all figures: network graph, loss history, signal overlays, diastolic decay fits, and normalized data.

### [src/report.py](src/report.py)
Writes a LaTeX document with structured tables and figures summarizing inputs and results, then compiles it to PDF.

## Example Case

`input/invivo/` is a 7-vessel model of the thoracic aorta (aortic arch plus the brachiocephalic, left common carotid, and left subclavian branches), built from non-invasive *in-vivo* pressure and flow measurements. It's included as a worked example of the four input files described above — copy it as a template for a new case, or point `casename` at a different subfolder under `input/` with the same file layout.
