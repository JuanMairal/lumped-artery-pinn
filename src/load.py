import os
import numpy as np
import re
import pandas as pd
from config import Pa2mmHg, unit_conv
from src.report import report_vessels, report_beds, report_cardiac, report_signals, report_junctions


## The goal of these functions is to read the case files and fill up the vessel, bed, ground_truth and cardiac dictionaries

def get_file(keyword, file_list, extension=".input"):
    return [f for f in file_list if keyword in f and f.endswith(extension)]


def full_path(input_dir, filename):
    return os.path.join(input_dir, filename)


def load_vessels(filepath):

    print(f"Opening file {filepath}\n.")

    with open(filepath, "r") as f:
        Nvessel = int(re.findall(r'\d+', f.readline())[0])
        f.readline() #Ignore units
        header = f.readline().split()
        vessels_dict = {key: [] for key in header}
        for line in f:
            tokens = line.split()
            if tokens:
                for i, key in enumerate(header):
                    vessels_dict[key].append(tokens[i])
                    

    vessels_dict["Nv"] = Nvessel
    vessels_dict["n"] = [int(x) for x in vessels_dict["n"]]
    for key in ["length", "c_avg"]:
        vessels_dict[key] = [1e-2*float(x) for x in vessels_dict[key]]  # cm to m
    for key in ["r_in", "r_out"]:
        vessels_dict[key] = [1e-3* float(x) for x in vessels_dict[key]]  # mm to m
    r_avg = [0.5 * (r1 + r0) for r1, r0 in zip(vessels_dict["r_out"], vessels_dict["r_in"])]
    vessels_dict["r_avg"] = r_avg
    vessels_dict["Ad"] = [np.pi * r**2 for r in r_avg]
    vessels_dict["Vd"] = [a * l for a, l in zip(vessels_dict["Ad"], vessels_dict["length"])]
    
    return vessels_dict



def load_beds(filepath):

    print(f"Opening file {filepath}\n.")

    with open(filepath, "r") as f:
        Nbeds = int(f.readline().split()[-1])
        f.readline()  # units
        header = f.readline().split()
        beds_dict = {key: [] for key in header}
        for line in f:
            tokens = line.split()
            if tokens:
                for i, key in enumerate(header):
                    beds_dict[key].append(tokens[i])
    beds_dict["Nb"] = Nbeds
    beds_dict["n"] = [int(x) for x in beds_dict["n"]]
    beds_dict["vessel"] = [int(x) for x in beds_dict["vessel"]]
    beds_dict["Zb"] = [1e6 / Pa2mmHg * float(x) for x in beds_dict["Zb"]]
    beds_dict["Cb"] = [Pa2mmHg / 1e6 * float(x) for x in beds_dict["Cb"]]
    beds_dict["Rb"] = [1e6 / Pa2mmHg * float(x) for x in beds_dict["Rb"]]
    beds_dict["Pout"] = [1 / Pa2mmHg * float(x) for x in beds_dict["Pout"]]
    beds_dict["Vfrac"] = [0.01 * float(x) for x in beds_dict["Vfrac"]]


    return beds_dict


def load_cardiac(filepath):

    print(f"Opening file {filepath}\n.")
    cardiac = {}

    with open(filepath, "r") as f:
        cardiac["Ps"] = 1.0/Pa2mmHg*float(f.readline().split()[-1])
        cardiac["Pd"] = 1.0/Pa2mmHg*float(f.readline().split()[-1])
        cardiac["Qavg"]  = 1e-6*float(f.readline().split()[-1])
        cardiac["T"]  = float(f.readline().split()[-1])

    return cardiac



def load_signals(filepath):

    """ Loads the singals from data.input into the ground_truth dictionary. 
    This dictionary has tuples (VARIABLE, VESSEL) as keys.
    Also stores units in input_units."""

    ground_truth = {}
    input_units = {}

    print(f"Opening file {filepath}\n.")

    with open(filepath, "r") as f:
        for line in f:
            if not line.startswith("#"):
                break
            units = line.lstrip("#").strip().split()
            input_units[units[0]] = units[1]

    df = pd.read_csv(filepath, sep=r'\s+', comment="#")

    for _, row in df.iterrows():
        key = (row["type"], int(row["vessel"]))
        if key not in ground_truth:
            ground_truth[key] = {"time": [], "value": [], "unit": None}
        ground_truth[key]["time"].append(row["time"])
        ground_truth[key]["value"].append(row["value"])

    for key, series in ground_truth.items():
        t = np.array(series["time"], dtype=np.float32)
        v = np.array(series["value"], dtype=np.float32)
        sort_idx = np.argsort(t)

        var = key[0]
        unit = input_units[var]
        ground_truth[key]["t_phys"] = t[sort_idx]
        ground_truth[key]["val_phys"] = unit_conv[unit]*v[sort_idx]
        ground_truth[key]["unit"] = unit

    return ground_truth



def load_junctions(filepath):

    print(f"Opening file {filepath}\n.")

    with open(filepath, "r") as f:
        Njunctions = int(f.readline().split()[-1])
        junctions = {"Nj": Njunctions,
                    "nvup" : [[] for _ in range(Njunctions)],
                    "nvdw" : [[] for _ in range(Njunctions)]
                    }
        nj = 0
        for line in f:
            group = line.split(";")
            up = [int(nv) for nv in group[0].split()]
            dw = [int(nv) for nv in group[-1].split()]
            junctions["nvup"][nj] = up
            junctions["nvdw"][nj] = dw
            nj+=1
    

    return junctions


def load_all(input_dir, files_list):

    f_vessels = get_file("tree", files_list)[0]
    path_vessels = full_path(input_dir, f_vessels)
    vessels = load_vessels(path_vessels) 
    report_vessels(vessels)  

    f_beds = get_file("beds", files_list)[0]
    path_beds = full_path(input_dir, f_beds)
    beds = load_beds(path_beds) 
    report_beds(beds, vessels)  

    f_cardiac = get_file("cardiac", files_list)[0]
    path_cardiac = full_path(input_dir, f_cardiac)
    cardiac = load_cardiac(path_cardiac) 
    report_cardiac(cardiac)  
    
    f_signals = get_file("data", files_list)[0]
    path_signals = full_path(input_dir, f_signals)
    signals = load_signals(path_signals) 
    report_signals(signals)      

    f_junctions = get_file("junctions", files_list)[0]
    path_junctions = full_path(input_dir, f_junctions)
    junctions = load_junctions(path_junctions) 
    report_junctions(junctions)  

    return vessels, beds, cardiac, signals, junctions
    

