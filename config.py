from pathlib import Path
from datetime import datetime
import os



### === Settings === ###
# Case
origin = "input"
casename = "invivo"

# Known stuff
known_signals = [("Q", 0)]  # list of (type, vessel_index) tuples; type is "P" or "Q"
known_beds = False
compile_latex_report = True   # set False on machines without pdflatex
known_bed_params = ["Vfrac"]  # Subset of: "Zb", "Vfrac", "Pout", "Rb" (Rb implies Cb)
trainable_params = ["Rb"]           # Subset of: "Rb", "c0"

# Random seed
random_seed = 1

# Physical model
INER = 1

# Training
ncoloc = 256

NN_params = {"FF" : True,
             "nff_harmon": 2,
             "nff_random" : 0,
             "hidden_layers" : 5,
             "neurons_per_layer" : 64,
             "activation_f" : "tanh"}

max_epochs = 1000000
dump_every = 5000
plot_every = 20000

early_stopping = {
    "min_epochs":  100000,  # never stop before this many epochs
    "patience":    10,     # consecutive dump windows without sufficient improvement
    "threshold":   5e-3,   # minimum relative drop in data loss to reset patience counter
}


net_learning = {"initial": 1e-3,
                  "steps": int(1e5),
                  "decay": 0.5}
param_learning = {"initial": 1e-4,   # physical parameters (Rb, c0, ...); smaller than
                  "steps": int(1e5),  # network LR since initial estimates are already close
                  "decay": 0.667}



### === Fluid properties === ###
fluid = {"beta" : 8,
        "visc" : 0.0025,
        "dens" : 1050}



### === File Paths === ###
outname = f"{casename}_{datetime.now().strftime('%Y%m%d%H%M')}"
BASE_DIR = Path(__file__).resolve().parent
directories = { "in" : os.path.join(BASE_DIR, origin, casename),
                "out" : os.path.join(BASE_DIR, "output", outname),
                "param" : os.path.join(BASE_DIR, "output", outname, "param"),
                "predictions": os.path.join(BASE_DIR, "output", outname, "predictions"),
                "latex": os.path.join(BASE_DIR, "output", outname, "latex"),
                "figures": os.path.join(BASE_DIR, "output", outname, "latex", "figures"),
                "solutions": os.path.join(BASE_DIR, "output", outname, "latex", "figures", "solutions")
}


### === Unit Conversion Constants === ###
Pa2mmHg = 0.00750062            # Pascal to mmHg
mmHg2Pa = 133.32238740           # mmHg to Pascal


unit_conv = {"-": 1.0,
            "s": 1.0,
             "m": 1.0,
             "m3/s":1.0,
             "Pa": 1.0,
             "ms": 1e-3,
             "mm": 1e-3,
             "cm": 1e-2,
             "mm2": 1e-6,
             "cm2": 1e-4,
             "ml/s": 1e-6,
             "cm3/s": 1e-6,
             "mmHg": mmHg2Pa,
             "kPa": 1e3
             }
