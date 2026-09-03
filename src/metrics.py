import numpy as np
import os
import csv
import time
from config import Pa2mmHg


def save_param_csv(epoch, trainable, param_dir):
    """Append one row per dump step to <param>_evolution.csv in param_dir.

    Columns: epoch, log_scale, scale  (scale = exp(log_scale), dimensionless
    relative to the initial estimate).  For vector params (Rb) one pair of
    columns per bed: log_scale_b0, scale_b0, log_scale_b1, scale_b1, ...
    """
    import numpy as np
    for name, var in trainable.items():
        label = name[4:] if name.startswith("log_") else name
        log_val = var.numpy()
        scale   = np.exp(log_val)

        fpath = os.path.join(param_dir, f"{label}_evolution.csv")
        write_header = not os.path.exists(fpath)

        with open(fpath, "a", newline="") as f:
            writer = csv.writer(f)
            if np.isscalar(log_val):
                if write_header:
                    writer.writerow(["epoch", "log_scale", "scale"])
                writer.writerow([epoch, f"{log_val:.6f}", f"{float(scale):.6f}"])
            else:
                Nb = len(log_val)
                if write_header:
                    header = ["epoch"]
                    for b in range(Nb):
                        header += [f"log_scale_b{b}", f"scale_b{b}"]
                    writer.writerow(header)
                row = [epoch]
                for b in range(Nb):
                    row += [f"{log_val[b]:.6f}", f"{scale[b]:.6f}"]
                writer.writerow(row)


def save_Rb_from_RT_csv(epoch, Rb_scales, param_dir):
    """Append per-bed Rb scale factors (derived from trainable RT) to Rb_scale_evolution.csv.
    Matches the column layout of the old per-bed log_Rb_scale CSV so postprocessing
    can read it without changes."""
    Nb = len(Rb_scales)
    fpath = os.path.join(param_dir, "Rb_scale_evolution.csv")
    write_header = not os.path.exists(fpath)
    with open(fpath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["epoch"]
            for b in range(Nb):
                header += [f"log_scale_b{b}", f"scale_b{b}"]
            writer.writerow(header)
        row = [epoch]
        for s in Rb_scales:
            row += [f"{float(np.log(s)):.6f}", f"{float(s):.6f}"]
        writer.writerow(row)


def save_loss_csv(epoch, total_loss, residuals, output_dir):
    """Appends one row per dump step to loss_history.csv."""
    fpath = os.path.join(output_dir, "loss_history.csv")
    write_header = not os.path.exists(fpath)
    names = ["data", "period", "inlet", "junctions", "vessels", "beds", "RT"]
    with open(fpath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "total"] + names)
        writer.writerow(
            [epoch, f"{total_loss:.6e}"] +
            [f"{residuals.get(n, 0.0):.6e}" for n in names]
        )


def check_early_stopping(logger, cfg, current_epoch):
    """
    Returns True if training should stop.

    Scans the full data-loss history from the logger and counts consecutive
    dump windows in which the data loss did not drop by more than `threshold`
    relative to the running best. Stops when that count reaches `patience`.

    Uses data loss (not total loss) because total loss rises as curriculum
    weights ramp physics residuals from 0→1, masking genuine convergence.
    """
    if current_epoch < cfg["min_epochs"]:
        return False

    history = logger["residuals"].get("data", [])
    if len(history) < cfg["patience"] + 1:
        return False

    patience_count = 0
    best = history[0][1]
    for _, v in history[1:]:
        if v < best * (1.0 - cfg["threshold"]):
            best = v
            patience_count = 0
        else:
            best = min(best, v)
            patience_count += 1

    return patience_count >= cfg["patience"]


def measure_time_now(t_start, epoch, output_dir):
    """Appends one row per (epoch, total_time, avg_time) to time.out in output_dir."""
    t_now = time.time()
    total_time = t_now - t_start
    avg_time = total_time / epoch

    fpath = os.path.join(output_dir, "time.out")
    write_header = not os.path.exists(fpath)

    with open(fpath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "time [s]", "time/epoch [ms]"])
        writer.writerow([epoch, f"{total_time:.1f}", f"{avg_time*1e3:.2f}"])


def save_timing_summary(adam_time, epochs_run, lbfgs_time, lbfgs_iters, output_dir):
    """Appends a final two-phase timing summary to time.out."""
    avg_adam_ms  = adam_time / epochs_run * 1e3
    avg_lbfgs_ms = lbfgs_time / lbfgs_iters * 1e3 if lbfgs_iters > 0 else 0.0
    total_time   = adam_time + lbfgs_time

    fpath = os.path.join(output_dir, "time.out")
    with open(fpath, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["phase", "time [s]", "time/unit [ms]", "units"])
        writer.writerow(["Adam",   f"{adam_time:.1f}",  f"{avg_adam_ms:.2f}",  f"{epochs_run} epochs"])
        writer.writerow(["L-BFGS", f"{lbfgs_time:.1f}", f"{avg_lbfgs_ms:.2f}", f"{lbfgs_iters} iters"])
        writer.writerow(["Total",  f"{total_time:.1f}", "",                     ""])



def compute_error_metrics(pred_signals, ground_truth):
    """
    Computes per-vessel error metrics (MAE, SYS, DIAS) for pressure and flow.
    Parameters
    ----------
    pred_signals : list of dict, length Nv+1
        Each entry has keys "t" (s), "P" (Pa), "Q" (m3/s).
    ground_truth : dict with keys (var, nv) -> {"t_phys": array, "val_phys": array}
        All available reference signals (not just the known subset).

    Returns
    -------
    metrics : dict with keys (var, nv) -> {"avg", "sys", "dias"}
    """
    metrics = {}

    for (var, nv), ref in ground_truth.items():
        t_ref = ref["t_phys"]
        val_ref = ref["val_phys"]

        t_pred        = pred_signals[nv]["t"]
        val_pred_full = pred_signals[nv][var]  # "P" or "Q"

        # Interpolate prediction onto reference time grid
        val_pred = np.interp(t_ref, t_pred, val_pred_full)

        max_ref = np.max(np.abs(val_ref))

        if max_ref < 1e-12:
            continue  # skip degenerate signals

        if var == "P":
            eps_avg = float(np.mean(np.abs(val_pred - val_ref) / np.abs(val_ref)))
        else:  # Q
            # Eq. 6.78: normalized by max|Q| (avoid division by near-zero diastolic flow)
            eps_avg = float(np.mean(np.abs(val_pred - val_ref)) / max_ref)


        eps_sys = float((np.max(val_pred) - np.max(val_ref)) / max_ref)

        eps_dias = float((np.min(val_pred) - np.min(val_ref)) / max_ref)

        metrics[(var, nv)] = {
            "avg": eps_avg,
            "sys": eps_sys,
            "dias": eps_dias,
        }

    return metrics

def compute_only_msq(pred_signal, ref_signal, var):
    """
    Computes only one error metric MAE given signal.
    Parameters
    ----------
    pred_signal : tuple with t, P or t, Q
    ref_signal :tuple with t, P or t, Q
    var: char, type of variable

    Returns
    -------
    mse : float
    """
        
    t_ref = ref_signal[0]
    val_ref = ref_signal[1]

    t_pred = pred_signal[0]
    val_pred_full = pred_signal[1]  # "P" or "Q"

    # Interpolate prediction onto reference time grid
    val_pred = np.interp(t_ref, t_pred, val_pred_full)

    max_ref = np.max(np.abs(val_ref))

    if max_ref < 1e-12:
        return None

    if var == "P":
        eps_avg = float(np.mean(np.abs(val_pred - val_ref) / np.abs(val_ref)))
    else:  # Q
        eps_avg = float(np.mean(np.abs(val_pred - val_ref)) / max_ref)

    return eps_avg
    

def find_optimal_time_shift(pred_signals, ground_truth):
    """
    Computes per vessel shift in time axis in order to minimize mean error.
    Calls compute_error_metrics.

    Parameters
    ----------
    pred_signals : list of dict, length Nv+1
        Each entry has keys "t" (s), "P" (Pa), "Q" (m3/s).
    ground_truth : dict with keys (var, nv) -> {"t_phys": array, "val_phys": array}
        All available reference signals (not just the known subset).

    Returns
    -------
    dt : dict with keys (var, nv) -> dt
    """

    max_dt = 0.20
    dt_array = np.linspace(-max_dt, max_dt, 100)
    dt_dict = {}

    for (var, nv), ref in ground_truth.items():
        t_pred  = pred_signals[nv]["t"]
        val_pred = pred_signals[nv][var]
        t_ref   = ref["t_phys"]
        val_ref = ref["val_phys"]

        best_dt  = 0.0
        best_msq = np.inf

        for dt in dt_array:
            msq = compute_only_msq(
                (t_pred + dt, val_pred),
                (t_ref,       val_ref),
                var,
            )
            if msq is not None and msq < best_msq:
                best_msq = msq
                best_dt  = dt

        dt_dict[(var, nv)] = best_dt

    return dt_dict



def print_metrics_table(metrics, vessels_tag):
    """Prints a formatted table of error metrics to stdout."""
    header = f"{'Var':>4}  {'Vessel':>8}  {'eps_avg':>10}  {'eps_sys':>10}  {'eps_dias':>10}"
    print("\n" + "=" * len(header))
    print("Error metrics")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for (var, nv), m in sorted(metrics.items()):
        tag = vessels_tag[nv - 1] if 0 < nv <= len(vessels_tag) else "inlet"
        print(
            f"{var:>4}  {tag:>8}  "
            f"{m['avg']:>10.4f}  {m['sys']:>+10.4f}  {m['dias']:>+10.4f}"
        )
    print("=" * len(header) + "\n")


def save_prediction_csv(pred_signals, output_dir):
    """Save predicted P and Q waveforms to semicolon-separated CSV files.

    Units: pressure in mmHg, flow in ml/s, time in s.
    One file per variable; all vessels as columns.
    """
    Nv = len(pred_signals) - 1
    t = pred_signals[0]["t"]

    for var, unit_label, conv in [("P", "mmHg", Pa2mmHg), ("Q", "ml/s", 1e6)]:
        fname = os.path.join(output_dir, f"{'p' if var == 'P' else 'q'}_pred.csv")
        header_units = "s;" + ";".join([unit_label] * (Nv + 1))
        header_vars  = "t;" + ";".join([f"{var.lower()}_n{n}" for n in range(Nv + 1)])
        data = np.column_stack([t] + [pred_signals[n][var] * conv for n in range(Nv + 1)])
        np.savetxt(fname, data, fmt="%12.8f", delimiter=";",
                   header=header_units + "\n# " + header_vars, comments="# ")


def save_metrics_csv(metrics, vessels_tag, epoch, output_dir):
    """Appends one row per (var, vessel) to metrics.csv in output_dir."""
    fpath = os.path.join(output_dir, "metrics.csv")
    write_header = not os.path.exists(fpath)

    with open(fpath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["epoch", "var", "nv", "vessel", "eps_avg", "eps_sys", "eps_dias"])
        for (var, nv), m in sorted(metrics.items()):
            tag = vessels_tag[nv - 1] if 0 < nv <= len(vessels_tag) else "inlet"
            writer.writerow([epoch, var, nv, tag,
                             f"{m['avg']:.6f}", f"{m['sys']:+.6f}", f"{m['dias']:+.6f}"])
