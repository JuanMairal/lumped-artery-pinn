import numpy as np
from config import fluid, INER, known_signals, known_beds, known_bed_params, ncoloc
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
import math
from src.plot import plot_diastolic_decay, plot_ready_data
from src.report import report_lumped_model, report_computed_beds


def generate_tree(vessels, beds, cardiac, signals, junctions):
    """
    Builds the full 0D arterial tree data structure consumed by src.pinn.

    Non-dimensionalizes the physical quantities, builds the junction/vessel/
    bed incidence matrices, computes lumped vessel parameters (Com0, Res0,
    Ine0) and bed (RCR) parameters (from beds.input or estimated from the
    known signals), and prepares the time collocation grid and normalized
    training data.

    Returns a single `tree` dict holding all of the above.
    """

    adim = nondimensionalization(vessels, cardiac)

    input_data = extract_known_signals(signals, adim)

    Nv = vessels["Nv"]
    Nb = beds["Nb"]
    Nj = junctions["Nj"]

    maps_dict = build_maps(Nv, Nb, Nj, junctions, beds)

    mJ, mJD, mJid = build_junction_matrices(Nj, Nv, junctions, maps_dict)
    mV, mVid = build_vessel_matrices(Nv, maps_dict)
    mB = build_bed_matrices(Nv, Nb, maps_dict)

    validate_tree(Nv, Nj, Nb, maps_dict, mJ, mJD, mJid, mV, mB)

    rho = fluid["dens"]
    nu = fluid["visc"]
    beta = fluid["beta"]

    l = np.array(vessels["length"])
    Ad = np.array(vessels["Ad"])
    Aout = (np.pi*np.array(vessels["r_out"])**2)
    vd = np.array(vessels["Vd"])
    cavg = np.array(vessels["c_avg"])
    k = 2.0 * rho * cavg**2  

    cardT = cardiac["T"]
    ps = cardiac["Ps"]
    pd = cardiac["Pd"]
    qavg = cardiac["Qavg"]


    # Lumped vessel parameters
    Com0 = vd/(rho*cavg**2)
    Res0 = np.pi*beta*nu*l**3/(vd**2)
    Ine0 = INER*rho*l**2/vd

    report_lumped_model(vessels, Com0, Res0, Ine0)  

    # Bed parameters
    RT = None   # set in the estimated-Rb path; derived below for prescribed paths

    if known_beds:
        Zb, Cb, Rb, Pout, Vfrac = handle_bed_parameters_known(beds)
    else:
        # ── Per-parameter logic driven by known_bed_params ────────────────────
        # Zb
        if "Zb" in known_bed_params:
            Zb = _require_known(beds["Zb"], "Zb")
        else:
            Zb = (cavg*rho/Aout)[maps_dict["vessB"]]

        # Vfrac
        if "Vfrac" in known_bed_params:
            Vfrac = _require_known(beds["Vfrac"], "Vfrac")
        else:
            Vfrac = compute_volume_fractions(beds, Ad, maps_dict["vessB"])

        # Pout: prescribed or estimated from signal analysis
        if "Pout" in known_bed_params:
            Pout_arr = _require_known(beds["Pout"], "Pout")
            pout = float(np.mean(Pout_arr))   # scalar used for ResT
        else:
            Pout_arr = None
            pout     = None                   # set by signal analysis below

        # Rb / Cb (coupled): prescribed together or distributed from analysis
        if "Rb" in known_bed_params:
            Rb = _require_known(beds["Rb"], "Rb")
            Cb = _require_known(beds["Cb"], "Cb")
            if pout is None:
                pout = 0.5 * pd               # fallback heuristic (Mariscal-Harana)
        else:
            # Signal analysis to obtain tau (and pout/ResT when not prescribed)
            tau   = None
            p_avg = 0.4*ps + 0.6*pd

            for data in input_data:
                if data["var"] == "P" and data["nv"] in maps_dict["vessB"]:
                    t_sig = data["t_phys"]
                    p_sig = data["val_phys"]
                    p_avg = np.trapezoid(p_sig, x=t_sig) / np.max(t_sig)
                    pout_analysis, tau = diastolic_decay(t_sig, p_sig, cardT, ps, pd)
                    if Pout_arr is None:
                        pout = pout_analysis
                    break

            if tau is None:
                for data in input_data:
                    if data["var"] == "Q" and data["nv"] == 0:
                        t_sig = data["t_phys"]
                        q_sig = data["val_phys"]
                        pout_analysis, _, tau = Qin_analysis(t_sig, q_sig, ps, pd, qavg)
                        if Pout_arr is None:
                            pout = pout_analysis
                        break

            if tau is None:
                raise ValueError(
                    "Failed to estimate tau. Provide a peripheral P or inlet Q signal."
                )

            ResT = (p_avg - pout) / qavg
            ComT = tau / ResT
            Cb, Rb = distribute_bed_parameters(Com0, Vfrac, Zb, ResT, ComT)
            RT = ResT   # total peripheral resistance (used as trainable anchor)

        # Final per-bed Pout array
        Pout = Pout_arr if Pout_arr is not None else np.full(Nb, pout)

    # Ensure RT is always defined regardless of which bed-param path was taken.
    # In the estimated-Rb path RT = ResT was set explicitly.
    # In the prescribed-Rb paths derive it from the parallel combination of Rb+Zb.
    if RT is None:
        RT = 1.0 / np.sum(1.0 / (Rb + Zb))

    report_computed_beds(beds, vessels, Zb, Cb, Rb, Pout, Vfrac)

    # Collocation points: T will be smallest of max(t(known_signal)) and cardT
    true_data, t_coloc = colocation_points(cardT, input_data, adim)

    norm = compute_means(input_data, ps/adim["P"], pd/adim["P"], qavg/adim["Q"])
    data_ready = normalize_data(true_data, norm)

    
    tree = {"Nv" : Nv,
            "Nb" : Nb,
            "Nj" : junctions["Nj"],
            "maps": maps_dict,
            "mJ" : mJ,
            "mJD": mJD,
            "mJid": mJid,
            "mV": mV,
            "mVid": mVid,
            "mB": mB,
            "K" : k/(adim["P"]*norm["stdP"]),
            "Com0": Com0/adim["Com"],
            "Res0": Res0/adim["Res"],
            "Ine0": Ine0/adim["Res"],
            "Zb": Zb/adim["Res"],
            "Cb": Cb/adim["Com"],
            "Rb": Rb/adim["Res"],
            "Pout": Pout/adim["P"]/norm["stdP"],
            "Vfrac": Vfrac,
            "Ps": (ps/adim["P"])/norm["stdP"],
            "Pd": (pd/adim["P"])/norm["stdP"],
            "tau": tau/adim["T"],
            "RT":  RT/adim["Res"],
            # Prescribed Windkessel scalars in normalised units, used by loss_RT
            # to replace NN-predicted means when the corresponding signal is unknown.
            # RT * qavg = p_avg - pout  by construction (ResT definition).
            "Qavg_norm":  qavg / (adim["Q"] * norm["stdQ"]),
            "dP_RT_norm": RT * qavg / (adim["P"] * norm["stdP"]),
            "t_coloc": t_coloc,      
            "data" : data_ready,      
            "adim" : adim,
            "norm": norm
            }
    
    return tree


def build_maps(Nv, Nb, Nj, junctions, beds):
    """Builds the index maps (vessel<->pressure/flow, bed<->pressure/flow)
    used to assemble the incidence matrices below."""
    vessel_of_bed = [v-1 for v in beds["vessel"]]
    maps_dict = {"vessP": build_vessel_to_pressure_map(Nv, junctions),
                 "vessQ": build_vessel_to_outlet_flow_map(Nv),
                 "bedP":  build_bed_to_pressure_map(vessel_of_bed),
                 "bedQ":  build_bed_to_flow_map(vessel_of_bed),
                 "vessB": vessel_of_bed
    }
    return maps_dict

def build_vessel_to_pressure_map(Nv, junctions):
    """
    For each vessel, the P_all column its upstream pressure depends on.
    Size (Nv,). Vessel 0 -> inlet (P_all[:,0]); vessel v>0 -> pressure of
    its upstream junction.
    """

    map_vessP = np.zeros(Nv, dtype=int)
    map_vessP[0] = 0

    for j, nvdw in enumerate(junctions["nvdw"]):
        for v in nvdw:
            map_vessP[v-1] = 1 + j

    assert np.all(map_vessP[1:] > 0)
    return map_vessP

def build_vessel_to_outlet_flow_map(Nv):
    """For each vessel v, the Q_all column of its outlet flow: Q_all[:,v+1]."""

    map_vessQ = np.arange(1, Nv+1)
    return map_vessQ

def build_bed_to_pressure_map(vessel_of_bed):
    """
    For each bed, its vessel-end pressure column in P_all (equals bedQ by
    construction). Convention: P_all[:,v+1] is the downstream (vessel-end)
    pressure of vessel v; for a terminal vessel this is P_n before the Zb
    drop, and the true bed pressure Pb is recovered from it via Ohm's law.
    """

    map_bedP = np.array(vessel_of_bed, dtype=int) + 1
    return map_bedP

def build_bed_to_flow_map(vessel_of_bed):
    """For each bed, its vessel's Q_all column (needed to recover true Pb)."""

    map_bedQ = np.array(vessel_of_bed, dtype=int)+1
    return map_bedQ



def build_junction_matrices(Nj, Nv, junctions, maps):
    """
    Builds the per-junction incidence matrices used in the mass-conservation
    residual: mJ (flow balance, +1 upstream/-1 downstream per junction),
    mJD (0/1 mask of a junction's downstream vessels), and mJid (maps each
    junction to its pressure column in P_all). See Notation in the README.
    """

    vessP = maps["vessP"]

    mJ = np.zeros((Nj, Nv+1), dtype=float)
    mJD = np.zeros((Nj, Nv), dtype=float)
    mJid = np.zeros((Nj, Nv+1), dtype=float)

    for j, nvup in enumerate(junctions["nvup"]):
        v = [nv for nv in nvup]
        mJ[j, v] = 1
    for j, nvdw in enumerate(junctions["nvdw"]):
        v = [nv for nv in nvdw]
        vD = [nv-1 for nv in nvdw]
        vP = vessP[vD[0]]
        mJ[j, v] = -1
        mJD[j, vD] = 1
        mJid[j, vP] = 1

    return mJ, mJD, mJid

def build_vessel_matrices(Nv, maps):
    """
    Builds mV (per-vessel pressure-drop incidence matrix: +1 at the upstream
    pressure column, -1 at the downstream column v+1) and mVid (maps each
    vessel to its downstream pressure column), used in the vessel-momentum
    residual.
    """

    vessP = maps["vessP"]
    mV = np.zeros((Nv, Nv+1), dtype=float)
    mVid = np.zeros((Nv, Nv+1), dtype=float)

    for v in range(Nv):
        pidx_up = vessP[v]
        pidx_dw = v + 1           # downstream pressure always at v+1 (new convention)
        mV[v, pidx_up] = 1
        mV[v, pidx_dw] = -1
        mVid[v, v+1] = 1

    return mV, mVid


def build_bed_matrices(Nv, Nb, maps):
    """Builds mBid, mapping each terminal bed to its vessel's column in
    Q_all/P_all, used in the terminal-bed (RCR) residual."""

    mBid = np.zeros((Nb, Nv+1), dtype=float)
    vessB = maps["vessB"]

    for b in range(Nb):
        v = vessB[b] + 1
        mBid[b, v] = 1

    return mBid


def compute_means(data, ps, pd, qavg):
    """
    Computes the global mean/std of the known P and Q signals, used to
    normalize both the training data and the network output to a common
    scale. Falls back to cardiac-cycle statistics (pulse-pressure heuristic,
    mean flow) for whichever signal type has no known measurement.
    """

    mu_P = std_P = None
    mu_Q = std_Q = None
    for signal in data:
        values = signal["val_adim"]  
        var = signal["var"]

        if var == "P" and mu_P == None:
            mu_P  = np.mean(values)
            std_P = np.std(values)
        
        if var == "Q" and mu_Q == None:
            mu_Q  = np.mean(values)
            std_Q = np.std(values)

    
    if mu_P == None:
        mu_P = 0.4*ps + 0.6*pd
        std_P = ps - pd
    
    if mu_Q == None:
        mu_Q = qavg
        std_Q = qavg
        

    norm = {"meanP": mu_P,
            "meanQ": mu_Q,
            "stdP" : std_P,
            "stdQ" : std_Q
            }

    return norm

    
def normalize_data(data, norm):
    """
    Normalizes each signal for the NN using the global type std (stdP / stdQ).

    Only divides by std — does NOT subtract the mean.  Mean-centering would
    require updating all physics constants (Pout, Po, bed equations) that use
    standalone absolute pressure/flow values, creating consistency risk.
    The global std keeps the NN output scale consistent with the physics
    residuals (which use stdP/stdQ as unit conversion factors).
    """

    for signal in data:
        values = signal["value_interp"]
        var = signal["var"]

        if var == "P":
            new_values = values / norm["stdP"]
        if var == "Q":
            new_values = values / norm["stdQ"]

        signal["t_ready"] = signal["t_interp"]
        signal["val_ready"] = new_values

        plot_ready_data(signal)

    return data

        

def colocation_points(cardT, input_data, adim):
    """
    Builds the time collocation grid over one cardiac cycle and interpolates
    each known signal onto it. Also flags, per signal, which grid points
    actually fall within that signal's own measured time range (valid_mask) —
    signals shorter than the full cycle are not trimmed, they just stop
    contributing to the data loss past their last measured point.
    """

    # Always use the full cardiac cycle for collocation so that periodicity
    # P(0)=P(cardT), Q(0)=Q(cardT) is physically meaningful.
    # Signals shorter than cardT are used as partial data constraints and are
    # NOT trimmed here; the data loss only evaluates over the signal's own time axis.
    T = cardT / adim["T"]

    t_coloc = np.linspace(0.0, T, num=ncoloc)
        
    for signal in input_data:
        t = signal["t_adim"]
        v = signal["val_adim"]
        interp_v = np.interp(t_coloc, t, v)
        signal["t_interp"] = t_coloc
        signal["value_interp"] = interp_v
        # Boolean mask: True only where this signal actually has data.
        # Prevents the constant-extrapolated tail from entering the data loss.
        signal["valid_mask"] = t_coloc <= t[-1]

    return input_data, t_coloc


def _require_known(values, name):
    """Return np.array of bed parameter values.
    Raises if any value is negative (parameter declared known but not provided)."""
    arr = np.array(values, dtype=float)
    if np.any(arr < 0):
        raise ValueError(
            f"Bed parameter '{name}' is listed in known_bed_params but has "
            f"negative values in beds.input: {arr}"
        )
    return arr


def handle_bed_parameters_known(beds):
    """
    Reads all five bed (RCR) parameters directly from beds.input, validating
    that none are negative (a negative value there means the parameter was
    declared known but never actually provided).
    """

    for key, serie in beds.items():
        if any(serie < 0.0):
            raise ValueError(f"Beds set as known but negative value found in bed.input file: {key}: {serie}")

    Zb = np.array(beds["Zb"])
    Cb = np.array(beds["Cb"])
    Rb = np.array(beds["Rb"])
    Pout = np.array(beds["Pout"])
    Vfrac = np.array(beds["Vfrac"]) 

    return Zb, Cb, Rb, Pout, Vfrac


def compute_volume_fractions(beds, Ad, vessel_of_bed):
    """Check if any of the volume fractions are negative and then estimates them from diastolic area
    """
    if any(x < 0 for x in beds["Vfrac"]):
        Aout_total = 0.0
        Aout = []
        for nb in beds["n"]:
            nv = vessel_of_bed[nb]
            Aout.append(Ad[nv])
            Aout_total += Ad[nv]

        Vfrac= np.array([a/Aout_total for a in Aout])
    else:   # This is preferred
        Vfrac = np.array(beds["Vfrac"]) 

    return Vfrac



def diastolic_decay(t, p, cardT, ps, pd):
    """
    Fits an exponential decay to a peripheral pressure signal's diastolic
    phase (from end-systole to the end of the cycle) to estimate the
    terminal outflow pressure Pout and the Windkessel time constant tau.
    """

    def pexp(t, t0, tau, Pout, P0):
            return Pout + (P0-Pout)*np.exp(-(t-t0)/tau)
    
    lvet, i_lvet, i_min = find_lvet_fromP(t,p,cardT)

    t_fit = np.linspace(lvet,t[-1],1000)
    t_diastole = t[i_min:-1]
    p_diastole = p[i_min:-1]

    opt_param, cov = curve_fit(lambda t, tau, Pout: pexp(t, lvet, tau, Pout, p[i_lvet]), 
                        t_diastole, p_diastole, 
                        p0=[0.3, 9200],
                        maxfev=2000)
    
    tau = opt_param[0]
    Pout = opt_param[1]
    P_exp = pexp(t_fit, lvet, tau, Pout, p[i_lvet])

    if tau < 0.0 or Pout < 0.0:
        Pout = 0.0
    if Pout >= pd:
        Pout = pd

    plot_diastolic_decay(Pout,lvet,tau,t,p,t_diastole,p_diastole,t_fit,P_exp,ps,pd)

    return Pout, tau



def Qin_analysis(t, q, ps, pd, qavg):
    """Estimate Windkessel time constant tau from inlet flow Q.

    ResT = (p_avg - pout) / qavg  (total resistance)
    ComT = SV / ΔP                (total compliance, Remington formula)
    tau  = ResT * ComT
    """
    max_pos = np.argmax(q)

    pp    = ps - pd
    p_avg = 0.4 * ps + 0.6 * pd
    pout  = 0.5 * pd
    ResT  = (p_avg - pout) / qavg

    # Forward stroke volume: integrate Q over the ejection window
    idx_Qn2p = np.sort(np.where((q[:-1] < 0) & (q[1:] >= 0))[0] + 1)
    idx_Qp2n = np.sort(np.where((q[:-1] > 0) & (q[1:] <= 0))[0] + 1)

    before_peak = idx_Qn2p[idx_Qn2p < max_pos]
    idx_lwr = before_peak[-1] if len(before_peak) > 0 else np.argmin(q[:max_pos + 1])

    after_peak = idx_Qp2n[idx_Qp2n > max_pos]
    idx_hgr = after_peak[0] if len(after_peak) > 0 else max_pos + np.argmin(q[max_pos:])

    sv   = np.trapezoid(q[idx_lwr:idx_hgr], x=t[idx_lwr:idx_hgr])
    ComT = sv / pp

    return pout, ResT, ComT * ResT


def find_lvet_fromQ(t,q):
    """
    Estimates left-ventricular ejection time (LVET) from a flow signal.
    NOTE: currently unused elsewhere in the codebase — find_lvet_fromP
    (used inside diastolic_decay) is the one actually called.
    """

    ## LVET Q analysis (LV4)
    n_Q = len(q)
    max_pos = np.argmax(q)
    min_pos = np.argmin(q)

    Q_after = q[min_pos:2*n_Q//3]
    if np.all(np.abs(Q_after)<=0.01*np.abs(q[max_pos])):
        lvet = t[min_pos]
    else:
        Q_frag = q[min_pos:-1]
        sign_change = np.where((Q_frag[:-1] < 0) & (Q_frag[1:] >= 0))[0] + 1
        local_maxima = np.where((Q_frag[1:-1] > Q_frag[:-2]) & (Q_frag[1:-1] > Q_frag[2:]))[0] + 1
        zeroes = np.where(Q_frag == 0)[0]
        filtered = [arr for arr in [sign_change, local_maxima, zeroes] if arr.size > 0]
        if filtered:
            lvet_idx = np.concatenate(filtered).min()
            lvet = t[min_pos + lvet_idx]
        else:
            lvet = 0.37*np.sqrt(t[-1])
    return lvet


def find_lvet_fromP(t, p, cardT):
    """
    Estimates LVET from the pressure signal: locates the dicrotic-notch-like
    inflection marking end-systole, used as the start of the diastolic-decay
    fit window in diastolic_decay.
    """

    maxima = argrelextrema(p, np.greater)[0]
    n_maxima = np.size(maxima)
    first_peak = maxima[0]
    real_maxima = []
    for maxidx in maxima:
        if t[maxidx]<=0.6:
            real_maxima.append(maxidx)
    
    n_maxima = len(real_maxima)

    if(n_maxima == 1):
        # Only one maximum (LV2)
        dpdt = np.gradient(p,t)
        hr = 60/cardT
        aux = np.square(0.5 - np.abs(0.5-hr*t/60.0))
        i_min = np.argmin(aux*dpdt)
        lvet = t[i_min]
        i_lvet = i_min+1


    if(n_maxima >1):
        # More than one maximum (LV1) 
        second_peak = real_maxima[-1]+math.floor(0.04*np.size(t))
        lvet = t[second_peak]
        i_min = second_peak+1
        i_lvet = second_peak

    return lvet, i_lvet, i_min 



def distribute_bed_parameters(Com0, Vfrac, Zb, ResT, ComT):
    """
    Distributes total peripheral resistance/compliance (ResT/ComT, estimated
    from signal analysis) across the individual terminal beds in proportion
    to each bed's volume fraction (Vfrac), producing per-bed Rb/Cb.
    """

    ## First distribute resistance:
    RT_beds = ResT/Vfrac
    R_beds = RT_beds - Zb

    ## Distribute Compliance
    C_vessel = Com0
    C_central = np.sum(C_vessel)
    C_periph = ComT - C_central

    Cw_beds = C_periph*Vfrac
    C_beds = Cw_beds*RT_beds/R_beds

    return C_beds, R_beds


def extract_known_signals(signals, adim):
    """Filters the full parsed signal set down to just the ones listed in
    config.known_signals, converting each to non-dimensional units."""

    input_data = []
    for key, series in signals.items():
        if key in known_signals:
            var, nv = key
            t = series["t_phys"]
            val = series["val_phys"]

            if var == "P":
                dimfactor = adim["P"]
            elif var == "Q":
                dimfactor = adim["Q"]
    
            input_data.append({
            "var": var,
            "nv": nv,
            "t_phys": t,
            "val_phys": val,
            "t_adim": t / adim["T"],
            "val_adim": val / dimfactor
            })
    return input_data



def nondimensionalization(vessels, cardiac):
    """
    Defines the characteristic scales (pressure, time, length, and their
    derived combinations) used to non-dimensionalize every physical
    quantity in the tree: pressure by systolic pressure, time by the
    cardiac period, length by the cube root of the largest vessel volume.
    """

    adim={"P": cardiac["Ps"],
          "T" : cardiac["T"],
          "L" : np.cbrt(max(vessels["Vd"])),
    }
    
    adim["A"]=adim["L"]**2
    adim["U"]=adim["L"]/adim["T"]
    adim["Q"]=adim["L"]**3/adim["T"]
    adim["rho"]=adim["P"]*adim["T"]**2/adim["L"]**2
    adim["nu"]=adim["P"]*adim["T"]

    adim["Com"] = adim["L"]**3/adim["P"]
    adim["Res"] = adim["P"]*adim["T"]/adim["L"]**3
    adim["Ine"] = adim["P"]*adim["T"]**2/adim["L"]**3

    return adim


def validate_tree(Nv, Nj, Nb, maps, mJ, mJD, mJid, mV, mB):
    """
    Validates the topology of the 0-D vascular tree.

    Checks:
      - Map sizes, value ranges, and internal consistency (bedP == bedQ == vessB+1)
      - Matrix shapes and non-zero patterns for mJ, mJD, mJid, mV, mB
      - New P_all convention: downstream pressure of vessel v is at column v+1

    Raises AssertionError with a descriptive message if anything is wrong.
    """

    vessP = maps["vessP"]
    vessQ = maps["vessQ"]
    bedP  = maps["bedP"]
    bedQ  = maps["bedQ"]
    vessB = maps["vessB"]

    # --- Map sizes ---
    assert len(vessP) == Nv, f"vessP length {len(vessP)} != Nv={Nv}"
    assert len(vessQ) == Nv, f"vessQ length {len(vessQ)} != Nv={Nv}"
    assert len(bedP)  == Nb, f"bedP length {len(bedP)} != Nb={Nb}"
    assert len(bedQ)  == Nb, f"bedQ length {len(bedQ)} != Nb={Nb}"
    assert len(vessB) == Nb, f"vessB length {len(vessB)} != Nb={Nb}"

    # --- Map value ranges ---
    assert np.all((vessP >= 0) & (vessP <= Nj)), \
        f"vessP must be in [0, Nj={Nj}]; got {vessP}"
    assert np.all(np.sort(vessQ) == np.arange(1, Nv+1)), \
        f"vessQ must be a permutation of [1..Nv]; got {vessQ}"
    assert np.all((bedP >= 1) & (bedP <= Nv)), \
        f"bedP must be in [1, Nv={Nv}]; got {bedP}"
    assert np.all((bedQ >= 1) & (bedQ <= Nv)), \
        f"bedQ must be in [1, Nv={Nv}]; got {bedQ}"
    assert np.all((np.array(vessB) >= 0) & (np.array(vessB) < Nv)), \
        f"vessB must be in [0, Nv-1={Nv-1}]; got {vessB}"

    # --- Convention: bedP == bedQ == vessel_of_bed + 1 ---
    assert np.all(bedP == bedQ), \
        f"bedP must equal bedQ (P_all convention); bedP={bedP}, bedQ={bedQ}"
    assert np.all(bedP == np.array(vessB) + 1), \
        f"bedP[b] must equal vessB[b]+1; bedP={bedP}, vessB+1={np.array(vessB)+1}"

    # --- Uniqueness: no two beds share a P_all or Q_all slot ---
    assert len(np.unique(bedP)) == Nb, \
        f"bedP values must be unique; got {bedP}"

    # --- Tree topology ---
    assert Nv == Nj + Nb, \
        f"Tree condition: Nv={Nv} must equal Nj+Nb={Nj}+{Nb}={Nj+Nb}"

    # --- Matrix shapes ---
    assert mJ.shape   == (Nj, Nv+1), f"mJ shape {mJ.shape} != ({Nj},{Nv+1})"
    assert mJD.shape  == (Nj, Nv),   f"mJD shape {mJD.shape} != ({Nj},{Nv})"
    assert mJid.shape == (Nj, Nv+1), f"mJid shape {mJid.shape} != ({Nj},{Nv+1})"
    assert mV.shape   == (Nv, Nv+1), f"mV shape {mV.shape} != ({Nv},{Nv+1})"
    assert mB.shape   == (Nb, Nv+1), f"mB shape {mB.shape} != ({Nb},{Nv+1})"

    # --- mJ: one +1 (upstream Q), at least one -1 (downstream Q) per junction ---
    for j in range(Nj):
        row = mJ[j]
        assert np.sum(row == 1) == 1, \
            f"Junction {j}: mJ must have exactly one +1 (upstream Q)"
        assert np.sum(row == -1) >= 1, \
            f"Junction {j}: mJ must have at least one -1 (downstream Q)"

    # --- mJD: only 0/1, at least one downstream vessel selected per junction ---
    for j in range(Nj):
        row = mJD[j]
        assert np.all((row == 0) | (row == 1)), \
            f"Junction {j}: mJD must contain only 0 and 1"
        assert np.sum(row) >= 1, \
            f"Junction {j}: mJD must select at least one downstream vessel"

    # --- mJid: exactly one +1 per junction, pointing to a junction pressure [1..Nj] ---
    for j in range(Nj):
        row = mJid[j]
        nz = np.where(row != 0)[0]
        assert len(nz) == 1, \
            f"Junction {j}: mJid must have exactly one non-zero entry; got {len(nz)}"
        assert row[nz[0]] == 1.0, \
            f"Junction {j}: mJid non-zero entry must be +1"
        assert 1 <= nz[0] <= Nj, \
            f"Junction {j}: mJid +1 must point to a junction pressure col in [1,{Nj}], got col {nz[0]}"

    # --- mV: new convention — downstream pressure at column v+1 for every vessel ---
    for v in range(Nv):
        row = mV[v]
        assert np.sum(row) == 0, \
            f"Vessel {v}: mV row must sum to zero"
        assert row[v+1] == -1, \
            f"Vessel {v}: mV downstream pressure must be at col v+1={v+1} (P_all convention)"
        assert row[vessP[v]] == 1, \
            f"Vessel {v}: mV upstream pressure must be at col vessP[v]={vessP[v]}"

    # --- mB: exactly one +1 per bed, at the correct Q slot ---
    for b in range(Nb):
        row = mB[b]
        nz = np.where(row != 0)[0]
        assert len(nz) == 1, \
            f"Bed {b}: mB must have exactly one non-zero entry; got {len(nz)}"
        assert row[nz[0]] == 1.0, \
            f"Bed {b}: mB non-zero entry must be +1"
        assert nz[0] == bedQ[b], \
            f"Bed {b}: mB +1 must be at col bedQ[b]={bedQ[b]}, got col {nz[0]}"

    print("Tree validation passed")
