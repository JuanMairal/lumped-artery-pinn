import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp
from src.metrics import save_loss_csv
from config import NN_params, dump_every, net_learning, param_learning, trainable_params, known_signals



def log_step(logger, epoch, loss, mean_residuals, residuals):
    """Appends the current epoch's loss and per-residual values (both raw
    and loss-weighted) to `logger`, skipping residuals whose weight is
    currently zero (i.e. not yet ramped in — see compute_weights)."""
    logger["loss"].append((epoch, loss))

    for key, value in mean_residuals.items():
        if mean_residuals.get(key, 0.0) > 0.0:
            if key not in logger["weighted_residuals"]:
                logger["weighted_residuals"][key] = []
            logger["weighted_residuals"][key].append((epoch, value))

    for key, value in residuals.items():
        if mean_residuals.get(key, 0.0) > 0.0:
            if key not in logger["residuals"]:
                logger["residuals"][key] = []
            logger["residuals"][key].append((epoch, value))


def construct_DNN(Nv):
    """Builds the PINN model and its two Adam optimizers: one for the
    network weights, one for the trainable physical parameters."""
    model = define_network_architecture(Nv)
    optimizers = {"model": build_model_optimizer(),
                  "param": build_param_optimizer()}
    return model, optimizers



@tf.function
def train_step(model, optimizers, weights, pinn_dict):
    """
    One Adam training step: computes the loss and residuals, backpropagates,
    and applies gradients through two separate optimizers — network weights
    and trainable physical parameters (RT/c0) each get their own learning
    rate. Wrapped in @tf.function for graph-mode speed.
    """

    with tf.GradientTape(persistent=True) as tape:
        loss_tf, mean_res_tf, res_tf = compute_loss(model, weights, pinn_dict)

    # Collect variables — split Rb (slow LR) from other physical params
    model_vars = model.trainable_variables
    param_vars = [v for k, v in pinn_dict["trainable"].items()]

    all_vars = model_vars + param_vars
    grads    = tape.gradient(loss_tf, all_vars)

    n_model = len(model_vars)
    n_param = len(param_vars)

    model_grads = grads[:n_model]
    param_grads = grads[n_model : n_model + n_param]

    # Apply updates
    optimizers["model"].apply_gradients(zip(model_grads, model_vars))
    if param_vars:
        optimizers["param"].apply_gradients(zip(param_grads, param_vars))

    return loss_tf, mean_res_tf, res_tf 


def data_loss(pall, qall, signals):
    """
    pall       : (Nc, Nv+1) — P_all[:,v+1] is vessel-end pressure of vessel v
    qall       : (Nc, Nv+1)

    Normalization chain:
      val_ready = val_adim / stdX_global   (see normalize_data in tree.py)
         reference signal (e.g. Q0) has std ≈ 1 in val_ready space
         terminal flows (e.g. Q6 ≈ 4 ml/s vs Q0 ≈ 100 ml/s) have std ≈ 0.03

    Dividing by signals["signal_std"] (per-signal std in val_ready space)
    gives each signal equal weight in the loss, equivalent to computing
    the NRMSE (normalized root-mean-square error) per signal.
    """

    var_type   = signals["var_type"]    # (Ns,)
    idx        = signals["idx"]         # (Ns,)
    values     = signals["values"]      # (Ns, Nc)
    signal_std = signals["signal_std"]  # (Ns,)
    mask       = signals["mask"]        # (Ns, Nc) bool — False beyond signal extent

    P_pred = tf.transpose(tf.gather(pall, idx, axis=1))
    Q_pred = tf.transpose(tf.gather(qall, idx, axis=1))

    pred = tf.where(
        var_type[:, None] == 0,
        P_pred,
        Q_pred
    )

    # Normalise by per-signal std → each vessel contributes equally.
    # Average only over valid time points so that partial signals (shorter than
    # cardT) do not contribute a spurious flat-tail constraint.
    sq_res = tf.square((pred - values) / signal_std[:, None])  # (Ns, Nc)
    n_valid = tf.reduce_sum(tf.cast(mask, tf.float64))
    return tf.reduce_sum(tf.where(mask, sq_res, tf.zeros_like(sq_res))) / n_valid

def loss_periodicity(p, q):
    """Periodicity residual: penalizes P(0) != P(T) and Q(0) != Q(T) so the
    predicted solution repeats consistently over one cardiac cycle."""
    q0 = q[0, :]
    qT = q[-1, :]
    p0 = p[0, :]
    pT = p[-1, :]

    r_cycl_p = tf.square(p0 - pT)  # (Nv+1,)
    r_cycl_q = tf.square(q0 - qT)  # (Nv+1,)

    tf.debugging.check_numerics(r_cycl_q, "r_cycl_q contains NaN or Inf!")
    tf.debugging.check_numerics(r_cycl_p, "r_cycl_p contains NaN or Inf!")

    return tf.reduce_mean(r_cycl_p + r_cycl_q)


def loss_inlet(dpdt, qall, Com, normalization):
    """Inlet compliance residual: enforces C_inlet * dP/dt = Q_inlet -
    Q_downstream at the network root (vessel 0's upstream end)."""
    dpdt = dpdt[:, 0]
    C_in = Com[:, 0]
    q_in = qall[:, 0]
    q_down = qall[:, 1]

    stdQ = normalization["stdQ"]
    stdP = normalization["stdP"]

    r_inlet = tf.square( C_in * dpdt - stdQ/stdP*(q_in - q_down))

    tf.debugging.check_numerics(r_inlet, "r_inlet contains NaN or Inf!")

    return tf.reduce_mean(r_inlet)

def loss_junctions(dpdt, qall, Com, normalization, matrices, Nv, Nj):
    """
    Mass-conservation residual at every internal junction: enforces the
    junction's compliance-weighted pressure derivative against its net
    inflow, using the mJ/mJD/mJid incidence matrices from src.tree.
    """

    mJ = matrices["mJ"]     # (Nj, Nv+1)
    mJD = matrices["mJD"]   # (Nj, Nv)
    mID = matrices["mJid"]

    stdQ = normalization["stdQ"]
    stdP = normalization["stdP"]

    Com_down = tf.matmul(Com, mJD)
    dQ = tf.matmul(qall, mJ)
    dpdt_j = tf.matmul(dpdt, mID)

    r_junctions = tf.square( Com_down * dpdt_j - stdQ/stdP*dQ)

    return  tf.reduce_mean(r_junctions)


def loss_vessels(dqdt, pall, qall, Res, Ine, normalization, matrices):
    """
    Vessel-momentum residual: enforces the pressure-drop/resistance-inertance
    balance (ΔP = R*Q + L*dQ/dt) along every vessel, using the mV incidence
    matrix.
    """

    mV = matrices["mV"]     # (Nv+1, Nv)

    stdQ = normalization["stdQ"]
    stdP = normalization["stdP"]


    dp_v = tf.matmul(pall, mV)
    q_v = qall[:,1:]
    dqdt_v = dqdt[:,1:]

    r_vessels= tf.square(dp_v - stdQ/stdP*(Res*q_v + Ine*dqdt_v))

    return  tf.reduce_mean(r_vessels)


def loss_beds(dpdt, dqdt, pall, qall, Rb, Cb, pout, normalization, matrices, Zb):
    """
    Bed (RCR Windkessel) residual.
    P_all stores vessel-end pressure P_n (before Zb). Pb is recovered via Ohm's law:
      Pb     = P_n    - Zb*(stdQ/stdP)*Q_n
      dPb/dt = dP_n/dt - Zb*(stdQ/stdP)*dQ_n/dt
    Bed equation: Pb - Pout = Rb*(Q_n - Cb*dPb/dt)  [in normalised units]
    """
    mB = matrices["mB"]     # (Nv+1, Nb)

    stdQ = normalization["stdQ"]
    stdP = normalization["stdP"]

    p_n    = tf.matmul(pall, mB)   # vessel-end P (Nc, Nb)
    q_n    = tf.matmul(qall, mB)   # vessel outlet Q (Nc, Nb)
    dpn_dt = tf.matmul(dpdt, mB)   # d/dt of vessel-end P (Nc, Nb)
    dqn_dt = tf.matmul(dqdt, mB)   # d/dt of vessel outlet Q (Nc, Nb)

    pb     = p_n    - Zb * (stdQ/stdP) * q_n
    dpb_dt = dpn_dt - Zb * (stdQ/stdP) * dqn_dt

    r_beds = tf.square(pb - pout - Rb * (stdQ/stdP * q_n - Cb * dpb_dt))

    return tf.reduce_mean(r_beds)



def loss_RT(qall, pall, pout, RT_eff, normalization, constants, Q0_known, P_known):
    """Windkessel balance residual: RT_eff * (stdQ/stdP) * Q_avg = P_avg - Pout.

    When the inlet flow is unknown (Q0_known=False), uses the prescribed cardiac
    mean flow instead of the NN-predicted mean — eliminating one degree of freedom.
    Similarly, when no pressure signal is known (P_known=False), uses the prescribed
    mean pressure estimate 0.4*Ps + 0.6*Pd.
    """
    stdQ = normalization["stdQ"]
    stdP = normalization["stdP"]

    q0avg = tf.reduce_mean(qall[:, 0]) if Q0_known else constants["Qavg_norm"]

    # dP = (P_avg - Pout) in normalised units.
    # P_known: compute from NN-predicted inlet pressure minus mean outlet pressure.
    # P_unknown: use prescribed dP_RT_norm = (p_avg - pout) / (adim_P * stdP),
    #            which already has pout subtracted — do NOT subtract again.
    if P_known:
        dP = tf.reduce_mean(pall[:, 0]) - tf.reduce_mean(pout)
    else:
        dP = constants["dP_RT_norm"]

    return tf.square(RT_eff * (stdQ/stdP) * q0avg - dP)




def total_loss(residual_vec, weights):
    """Combines the per-residual vector into a single scalar loss via a
    weighted sum; returns both the total and the per-residual weighted
    contributions (the latter used for logging)."""
    tf.debugging.assert_rank(residual_vec, 1)
    tf.debugging.assert_rank(weights, 1)
    tf.debugging.assert_equal(
        tf.shape(residual_vec)[0],
        tf.shape(weights)[0],
        message="Residuals and weights must have same length",
    )
    weighted_res = weights * residual_vec
    return tf.reduce_sum(weighted_res), weighted_res


def compute_loss(model, weights, pinn_dict):
    """
    Full forward pass and loss computation for one training step: evaluates
    the network, computes time derivatives via automatic differentiation,
    applies the trainable-parameter substitutions (c0, RT) to get effective
    physical constants, and returns the total weighted loss plus every
    individual residual (data, periodicity, inlet, junctions, vessels,
    beds, RT).
    """

    # Unpacking
    t = pinn_dict["t"]
    topology = pinn_dict["topology"]
    maps = topology["maps"]
    matrices = topology["matrices"]
    constants = pinn_dict["constants"]
    trainable = pinn_dict["trainable"]
    signals = pinn_dict["signals"]
    normalization = pinn_dict["normalization"]
    Q0_known = pinn_dict["Q0_known"]
    P_known  = pinn_dict["P_known"]


    Nv = topology["Nv"]
    Nj = topology["Nj"]

    # The foward pass
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        pall, qall = model(t)

    tf.debugging.assert_rank(pall, 2)
    tf.debugging.assert_rank(qall, 2)

    #Derivatives
    dpdt = tape.batch_jacobian(pall, t) # So apparently this has rank 3. (Nc, Nv+1, 1)
    dpdt = tf.squeeze(dpdt, axis=-1)  # Get rid of last dimension.
    tf.debugging.assert_rank(dpdt, 2)
    tf.debugging.assert_equal(tf.shape(dpdt)[1], Nv + 1)
    tf.debugging.check_numerics(dpdt, "dpdt contains NaN or Inf!")

    dqdt = tape.batch_jacobian(qall, t)
    dqdt = tf.squeeze(dqdt, axis=-1)  
    tf.debugging.assert_rank(dqdt, 2)
    tf.debugging.assert_equal(tf.shape(dqdt)[1], Nv + 1)
    tf.debugging.check_numerics(dqdt, "dqdt contains NaN or Inf!")
    del tape


    # Effective wave-speed constants (scaled by trainable c0 when active).
    # log_c0_scale is in log-space; recover c0 = exp(log_c0_scale) > 0 always.
    if "log_c0_scale" in trainable:
        c0 = tf.exp(trainable["log_c0_scale"])
        K_eff    = constants["K"]    * tf.square(c0)   # K ∝ c²
        Com0_eff = constants["Com0"] / tf.square(c0)   # Com0 ∝ 1/c²
        Zb_eff   = constants["Zb"]   * c0              # Zb ∝ c
    else:
        K_eff    = constants["K"]
        Com0_eff = constants["Com0"]
        Zb_eff   = constants["Zb"]

    # Effective bed parameters.
    # RT is trained as a single scalar; individual Rb_i are derived hard as
    # Rb_i = RT_eff / Vfrac_i - Zb_i, preserving the anatomical distribution
    # while reducing degeneracy from Nb DOF to 1.
    # Cb_i = tau / Rb_i  (hard Windkessel time-constant constraint).
    if "log_RT_scale" in trainable:
        RT_eff = constants["RT"] * tf.exp(trainable["log_RT_scale"])
        Rb     = RT_eff / constants["Vfrac"] - constants["Zb"]
        Cb     = constants["tau"] / Rb
    else:
        Rb     = constants["Rb"]
        Cb     = constants["Cb"]

    pout  = constants["Pout"]

    # Lumped parameters (use effective K and Com0 so c0 gradient flows through)
    alpha = compute_alphas(pall, K_eff, constants["Po"], maps)
    tf.debugging.check_numerics(alpha, "alpha contains NaN or Inf!")

    Com = Com0_eff * tf.pow(alpha, 0.5)
    Res = constants["Res0"] * tf.pow(alpha, -2.0)
    Ine = constants["Ine0"] * tf.pow(alpha, -1.0)

    ## Residual computations
    res_data = data_loss(pall, qall, signals)
    tf.debugging.check_numerics(res_data, "res_data contains NaN or Inf!")

    res_period = loss_periodicity(pall, qall)
    tf.debugging.check_numerics(res_period, "r_period contains NaN or Inf!")

    res_inlet = loss_inlet(dpdt, qall, Com, normalization)
    tf.debugging.check_numerics(res_inlet, "res_inlet contains NaN or Inf!")

    res_junctions = loss_junctions(dpdt, qall, Com, normalization, matrices, Nv, Nj)
    tf.debugging.check_numerics(res_junctions, "res_junctions contains NaN or Inf!")

    res_vessels = loss_vessels(dqdt, pall, qall, Res, Ine, normalization, matrices)
    tf.debugging.check_numerics(res_vessels, "res_vessels contains NaN or Inf!")

    res_beds = loss_beds(dpdt, dqdt, pall, qall, Rb, Cb, pout, normalization, matrices, Zb_eff)
    tf.debugging.check_numerics(res_beds, "res_beds contains NaN or Inf!")

    res_RT = (loss_RT(qall, pall, pout, RT_eff, normalization, constants, Q0_known, P_known)
              if "log_RT_scale" in trainable
              else tf.constant(0.0, dtype=tf.float64))
    tf.debugging.check_numerics(res_RT, "res_RT contains NaN or Inf!")

    res_tf = tf.stack([res_data,
                       res_period,
                       res_inlet,
                       res_junctions,
                       res_vessels,
                       res_beds,
                       res_RT
                        ],
                        axis=0)

    loss_tf, mean_res_tf = total_loss(res_tf, weights)


    return loss_tf, mean_res_tf, res_tf


def compute_weights(epoch):
    """
    Builds the per-residual loss weight vector for the given epoch: physics
    residuals ramp in linearly over the first 5000 epochs (see `ramp`) so
    the network first learns to fit the data before physics constraints are
    enforced; the RT residual is only active when RT is trainable.
    """
    epoch = tf.cast(epoch, tf.float64)
    # RT residual is only active when RT is trainable (otherwise res_RT = 0 anyway).
    rt_w = 1.0 if "Rb" in trainable_params else 0.0
    return tf.stack([
        1.0*tf.constant(1.0, dtype=tf.float64),                                    # DATA
        1.0*tf.cast(ramp(epoch, start=0.0, end=5000.0), tf.float64),               # PERIODICITY
        1.0*tf.cast(ramp(epoch, start=0.0, end=5000.0), tf.float64),               # INLET
        1.0*tf.cast(ramp(epoch, start=0.0, end=5000.0), tf.float64),               # JUNCTIONS
        1.0*tf.cast(ramp(epoch, start=0.0, end=5000.0), tf.float64),               # VESSELS
        1.0*tf.cast(ramp(epoch, start=0.0, end=5000.0), tf.float64),               # BEDS
        rt_w*tf.cast(ramp(epoch, start=0.0, end=5000.0), tf.float64),              # RT
        ],axis=0)


def convert_tree_to_tensors(tree):
    """
    Converts the `tree` dict (numpy-based, from src.tree.generate_tree) into
    the nested dict of TensorFlow tensors used throughout training: time
    collocation points, topology (incidence matrices, transposed for the
    forward pass), physical constants, normalization, and the trainable-
    parameter variables (log_RT_scale and/or log_c0_scale, depending on
    config.trainable_params).
    """

    ## Known data
    signals_tf = convert_signals_to_tensor(tree["data"], tree["maps"])

    for i, s in enumerate(tree["data"]):
        print(
            f"Signal {i}: var={s['var']}, nv={s['nv']}, "
            f"resolved idx={signals_tf['idx'][i].numpy()}, "
        )

    ## Time
    t = tf.constant(tree["t_coloc"], dtype=tf.float64)
    t = tf.reshape(t, (-1, 1)) 
    
    ## Network constants
    # Tensors of structure, never trainable
    topology = {        
        "Nv" : tf.constant(tree["Nv"], tf.int32),
        "Nb" : tf.constant(tree["Nb"], tf.int32),
        "Nj" : tf.constant(tree["Nj"], tf.int32),
        "maps": {
            "vessP": tf.constant(tree["maps"]["vessP"], dtype=tf.int32),
            "vessQ": tf.constant(tree["maps"]["vessQ"], dtype=tf.int32),
            "bedP":  tf.constant(tree["maps"]["bedP"],  dtype=tf.int32),
            "bedQ":  tf.constant(tree["maps"]["bedQ"],  dtype=tf.int32),
            "vessB": tf.constant(tree["maps"]["vessB"], dtype=tf.int32),
        },
        "matrices": {
            "mJ"  : tf.constant(np.transpose(tree["mJ"]), dtype=tf.float64),
            "mJD" : tf.constant(np.transpose(tree["mJD"]), dtype=tf.float64),
            "mJid" :tf.constant(np.transpose(tree["mJid"]), dtype=tf.float64),
            "mV" : tf.constant(np.transpose(tree["mV"]), dtype=tf.float64),
            "mB": tf.constant(np.transpose(tree["mB"]), dtype=tf.float64)
        }
    }

    normalization = {
        "stdP" : tf.constant(tree["norm"]["stdP"], tf.float64),
        "stdQ" : tf.constant(tree["norm"]["stdQ"], tf.float64),
        "meanP" : tf.constant(tree["norm"]["meanP"], tf.float64),
        "meanQ" : tf.constant(tree["norm"]["meanQ"], tf.float64),
    }

    ## Physical constants
    # Tensors of physical parameters, never trainable
    constants = {
        # global pressures (1,)
        "Po": tf.constant(tree["Pd"], dtype=tf.float64),
        "Pmax": tf.constant(tree["Ps"], dtype=tf.float64),

        # vessel constants (Nv,)
        "K": tf.constant(tree["K"], dtype=tf.float64),
        "Res0": tf.constant(tree["Res0"], dtype=tf.float64),
        "Ine0": tf.constant(tree["Ine0"], dtype=tf.float64),
        "Com0": tf.constant(tree["Com0"], dtype=tf.float64),

        # bed constants (Nb,)
        "Zb":   tf.constant(tree["Zb"],    dtype=tf.float64),
        "Rb":   tf.constant(tree["Rb"],    dtype=tf.float64),
        "Cb":   tf.constant(tree["Cb"],    dtype=tf.float64),
        "tau":  tf.constant(tree["tau"],   dtype=tf.float64),
        "Pout": tf.constant(tree["Pout"],  dtype=tf.float64),
        "Vfrac":tf.constant(tree["Vfrac"], dtype=tf.float64),
        # total peripheral resistance (scalar) used as anchor when RT is trainable
        "RT":          tf.constant(tree["RT"],          dtype=tf.float64),
        # Prescribed Windkessel scalars (normalised) for loss_RT
        "Qavg_norm":   tf.constant(tree["Qavg_norm"],   dtype=tf.float64),
        "dP_RT_norm":  tf.constant(tree["dP_RT_norm"],  dtype=tf.float64),
    }

    # Static flags: which signal types are known — resolved once at graph compile time.
    Q0_known = any(s[0] == "Q" and s[1] == 0 for s in known_signals)
    P_known  = any(s[0] == "P" for s in known_signals)


    ## Physical trainable parameters
    # Stored in log-space so physical values remain strictly positive for any
    # gradient step size.  Recover via exp() in compute_loss.
    # "log_RT_scale" : scalar log-scale for total peripheral resistance RT.
    #                  RT_eff = RT_0 * exp(s).  Individual bed resistances are
    #                  derived hard: Rb_i = RT_eff/Vfrac_i - Zb_i.
    #                  Cb is derived from the tau constraint: Cb_i = tau/Rb_i.
    # "log_c0_scale" : global log-wave-speed scale, scalar, init 0 → scale=1.
    #                  Affects K (∝c²), Com0 (∝1/c²), Zb (∝c).

    trainable = {}
    if "Rb" in trainable_params:
        trainable["log_RT_scale"] = tf.Variable(
            tf.random.normal([], mean=0.0, stddev=0.3, dtype=tf.float64),
            name="log_RT_scale", trainable=True)

    if "c0" in trainable_params:
        trainable["log_c0_scale"] = tf.Variable(
            tf.random.normal([], mean=0.0, stddev=0.3, dtype=tf.float64),
            name="log_c0_scale", trainable=True)

    return {
        "t": t,
        "topology": topology,
        "normalization": normalization,
        "constants": constants,
        "trainable": trainable,
        "signals": signals_tf,
        "Q0_known": Q0_known,
        "P_known":  P_known,
    }



def resolve_signal_indices(signals, maps):
    """
    Resolves each signal to an index in P_all or Q_all
    """

    var_type   = []   # 0 = P, 1 = Q
    idx        = []   # index into P_all or Q_all

    for s in signals:
        v = s["nv"]

        if s["var"] == "Q":
            var_type.append(1)
        elif s["var"] == "P":
            var_type.append(0)
        else:
            raise ValueError(f"Character {s['var']} not allowed for known signal type.")
        idx.append(v)   # Direct relation from human-readable nv to vessQ!

    return var_type, idx


def convert_signals_to_tensor(signals, maps):
    """Converts the known signals (from the case's data.input) into the
    tensors used by data_loss: predicted-array indices, values, per-signal
    normalization std, and validity mask."""

    var_type, idx = resolve_signal_indices(signals, maps)
    values = np.stack([s["val_ready"] for s in signals], axis=0)  # (Ns, Nc)

    # Per-signal std in val_ready space.
    # normalize_data divides all P by stdP_global and all Q by stdQ_global, so
    # the reference signal (e.g. Q0) has std ≈ 1, while terminal vessels (e.g.
    # Q6 ≈ 4 ml/s vs Q0 ≈ 100 ml/s) have std ≈ 0.03.  Using these as divisors
    # in data_loss gives each signal equal fractional-error weight.
    # Clip at 1e-6 to guard against degenerate constant signals.
    signal_std = np.maximum(np.std(values, axis=1), 1e-6)  # (Ns,)

    # Boolean mask (Ns, Nc): False for colocation points beyond a signal's actual
    # time extent (only relevant when a signal is shorter than cardT).
    masks = np.stack([s["valid_mask"] for s in signals], axis=0)  # (Ns, Nc)

    return {
        "var_type":   tf.constant(var_type,    tf.int32),    # (Ns,)
        "idx":        tf.constant(idx,         tf.int32),    # (Ns,)
        "values":     tf.constant(values,      tf.float64),  # (Ns, Nc)
        "signal_std": tf.constant(signal_std,  tf.float64),  # (Ns,)
        "mask":       tf.constant(masks,       tf.bool),     # (Ns, Nc)
    }


def compute_alphas(pall, K, Po, maps):
    """
    Computes vessel deformation alpha for each vessel.
    K  : (Nv,) stiffness constant, may be scaled by trainable c0
    Po : scalar diastolic reference pressure
    pall : (Nc, Nv+1)
    returns alpha : (Nc, Nv)
    """
    p_up = tf.gather(pall, maps["vessP"], axis=1)  # (Nc, Nv)
    alpha = tf.square(1.0 + (p_up - Po) / K)
    return alpha


def recover_bed_pressure(maps, pall, qall, Zb, normalization):
    """
    Recovers Pb (pressure after Zb, before Rb) from P_all via Ohm's law.
    Convention: P_all[:,v+1] = P_n (vessel-end pressure, before Zb).
    Ohm's law: P_n - Pb = Zb * Q  →  Pb = P_n - Zb * (stdQ/stdP) * Q
    """
    stdQ = normalization["stdQ"]
    stdP = normalization["stdP"]
    p_sel = tf.gather(pall, maps["bedP"], axis=1)  # (Nc, Nb) — vessel-end pressures
    q_sel = tf.gather(qall, maps["bedQ"], axis=1)  # (Nc, Nb) — vessel outlet flows
    pb = p_sel - Zb * (stdQ / stdP) * q_sel
    return pb

def recover_values_for_loss(loss_tf, mean_res_tf, res_tf):
    """Converts the loss/residual tensors back to plain Python floats,
    labeled by residual name, for logging and CSV export."""
    residual_names = [  ## MUST MATCH STACKING OF RESIDUALS
    "data",
    "period",
    "inlet",
    "junctions",
    "vessels",
    "beds",
    "RT",
    ]

    loss = float(loss_tf.numpy())

    mean_res = {
        name: float(mean_res_tf[i].numpy())
        for i, name in enumerate(residual_names)
    }

    res = {
        name: float(res_tf[i].numpy())
        for i, name in enumerate(residual_names)
    }

    return loss, mean_res, res



def recover_prediction_as_arrays(model, pinn_dict, adim, norm):
    """Runs the trained model once and converts its normalized, non-
    dimensional output back to physical units (P in the input's pressure
    unit, Q in its flow unit) for every vessel end, ready for
    plotting/export."""

    t = pinn_dict["t"]
    Nv = pinn_dict["topology"]["Nv"]

    Pall, Qall = model(t)

    # Convert to numpy and de-normalize (undo std scaling)
    # Pall[:,v+1] = vessel-end pressure of vessel v (before Zb for terminal vessels)
    t_pred = t.numpy().squeeze()                 # (Nc,)
    q_pred = norm["stdQ"] * Qall.numpy()         # (Nc, Nv+1)
    p_pred = norm["stdP"] * Pall.numpy()         # (Nc, Nv+1)

    # Build output list
    pred_arrays = []

    for nv in range(Nv + 1):
        pred_arrays.append({
            "t": adim["T"]*t_pred,
            "Q": adim["Q"]*q_pred[:, nv],
            "P": adim["P"]*p_pred[:, nv],
        })

    return pred_arrays


def run_lbfgs(model, pinn_dict, weights, maxiter=50000, ftol=1e-10, gtol=1e-5,
              print_fn=print, output_dir=None, epoch_offset=0):
    """Run L-BFGS via tensorflow_probability, staying entirely on GPU.

    Returns a SimpleNamespace with .fun, .nit, .message to match the old
    scipy OptimizeResult interface so main.py needs no changes.
    """
    from types import SimpleNamespace

    all_vars = model.trainable_variables + list(pinn_dict["trainable"].values())
    sizes = [int(np.prod(v.shape)) for v in all_vars]

    _count = [0]

    @tf.function
    def _compute(x):
        offset = 0
        for v, s in zip(all_vars, sizes):
            v.assign(tf.reshape(x[offset:offset + s], v.shape))
            offset += s
        with tf.GradientTape() as tape:
            loss_tf, mean_res_tf, res_tf = compute_loss(model, weights, pinn_dict)
        grads = tape.gradient(loss_tf, all_vars)
        grad_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)
        return loss_tf, mean_res_tf, res_tf, grad_flat

    def value_and_gradients(x):
        loss_tf, mean_res_tf, res_tf, grad_flat = _compute(x)

        _count[0] += 1
        if _count[0] % dump_every == 0:
            loss_val = float(loss_tf)
            print_fn(f"L-BFGS eval {_count[0]}, loss = {loss_val:.4e}", flush=True)
            for name, var in pinn_dict["trainable"].items():
                val = np.exp(var.numpy()) if name.startswith("log_") else var.numpy()
                label = name[4:] if name.startswith("log_") else name
                print_fn(f"  {label} = {val}", flush=True)
            if output_dir is not None:
                _, _, res = recover_values_for_loss(loss_tf, mean_res_tf, res_tf)
                save_loss_csv(epoch_offset + _count[0], loss_val, res, output_dir)

        return loss_tf, grad_flat

    x0 = tf.cast(
        tf.concat([tf.reshape(v, [-1]) for v in all_vars], axis=0),
        tf.float64,
    )

    result = tfp.optimizer.lbfgs_minimize(
        value_and_gradients,
        initial_position=x0,
        max_iterations=maxiter,
        f_relative_tolerance=ftol,
        tolerance=gtol,
    )

    offset = 0
    for v, s in zip(all_vars, sizes):
        v.assign(tf.reshape(result.position[offset:offset + s], v.shape))
        offset += s

    msg = ("converged" if result.converged.numpy()
           else "failed" if result.failed.numpy()
           else "max iterations reached")
    return SimpleNamespace(
        fun=float(result.objective_value),
        nit=int(result.num_iterations),
        message=msg,
    )


def define_network_architecture(Nv):
    """
    Builds the feed-forward network: an optional random/harmonic Fourier-
    feature embedding of the (scalar) time input, followed by
    NN_params["hidden_layers"] dense tanh layers, ending in two linear
    output heads (P and Q) of size Nv+1.
    """
    hidden_layers = NN_params["hidden_layers"]
    neurons_per_layer = NN_params["neurons_per_layer"]

    tf.keras.backend.set_floatx('float64')
    initializer = tf.keras.initializers.GlorotNormal()
    input_layer = tf.keras.layers.Input(shape=(1,), dtype='float64')

    if NN_params["FF"]:
        Bff = random_fourier_features(NN_params["nff_random"], NN_params["nff_harmon"])
        fourier = tf.keras.layers.Lambda(embed_fourier_features(Bff))(input_layer)
        x = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', kernel_initializer=initializer)(fourier)
    else:
        x = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', kernel_initializer=initializer)(input_layer)

    for _ in range(hidden_layers):
        x = tf.keras.layers.Dense(neurons_per_layer, activation='tanh', kernel_initializer=initializer)(x)

    q   = tf.keras.layers.Dense(Nv+1, name='output_q')(x)
    p   = tf.keras.layers.Dense(Nv+1, name='output_p')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=[p, q])
    return model


def build_model_optimizer():
    """Adam optimizer with exponential learning-rate decay for the network
    weights (see config.net_learning)."""
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=net_learning["initial"],
        decay_steps=net_learning["steps"],
        decay_rate=net_learning["decay"])
    return tf.keras.optimizers.Adam(learning_rate=schedule, clipnorm=1.0)


def build_param_optimizer():
    """Adam optimizer with exponential learning-rate decay for the
    trainable physical parameters (see config.param_learning)."""
    schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=param_learning["initial"],
        decay_steps=param_learning["steps"],
        decay_rate=param_learning["decay"])
    return tf.keras.optimizers.Adam(learning_rate=schedule, clipnorm=1.0)


def ramp(epoch, start, end, max_val=1.0):
    """Linear ramp from 0 to max_val between epochs `start` and `end`,
    clamped outside that range. Used to gradually turn on the physics
    residuals early in training (see compute_weights)."""
    if epoch <= start:
        return 0.0
    if epoch >= end:
        return float(max_val)

    x = (epoch - start) / (end - start)
    x = max(0.0, min(1.0, x))
    return x * max_val


def random_fourier_features(nff_random, nff_harmon):
    """Builds the frequency vector for the Fourier-feature embedding:
    nff_harmon fixed harmonics of the cardiac frequency plus nff_random
    random frequencies (used when config.NN_params["FF"] is True)."""
    fbeat = 1.0
    w0 = 2.0*np.pi*fbeat
    sigma = w0*0.5
    w_max = 10*w0

    k = np.arange(1, nff_harmon)
    B_harmon = k*w0
    B_random = np.random.normal(0.0, sigma, size=nff_random)
    B_random = np.clip(B_random, -w_max, w_max)

    b = np.concatenate([B_harmon, B_random])
    B_tensor = tf.constant(b.reshape(1, -1), dtype=tf.float64)  # shape (1, m)

    return B_tensor

def embed_fourier_features(B_tensor):
    """Returns a function mapping the scalar time input to its Fourier-
    feature embedding [cos(Bt), sin(Bt)], for use in a Keras Lambda
    layer."""
    def fourier_features(t):
        arg = tf.matmul(t, B_tensor)
        return  tf.concat([tf.cos(arg), tf.sin(arg)], axis=-1)  # (batch, 2m)
    return fourier_features
