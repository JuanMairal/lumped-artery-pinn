import subprocess
import os
import json
from config import (known_beds, known_bed_params, Pa2mmHg, directories,
                    casename, origin, known_signals, trainable_params,
                    INER, ncoloc, NN_params, max_epochs, dump_every, plot_every,
                    net_learning, param_learning, fluid, early_stopping, random_seed)
from src.plot import plot_one_signal


# ── Helpers ───────────────────────────────────────────────────────────────────

def escape_latex(s):
    """Escape characters that have special meaning in LaTeX."""
    return str(s).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def write_tex(filepath, content):
    with open(filepath, "a") as f:
        f.write(content)


def generate_latex_table(header, rows, caption=" "):
    col_fmt = "c" * len(header)
    lines = [
        "\\begin{table}[htb]",
        "\\centering",
        f"\\begin{{tabular}}{{|{col_fmt}|}}",
        "\\hline",
        " & ".join(header) + " \\\\ \\hline"
    ]
    for row in rows:
        lines.append(" & ".join(map(str, row)) + " \\\\ \\hline")
    lines += [
        "\\end{tabular}",
        f"\\caption{{{caption}}}",
        "\\end{table}\n"
    ]
    return "\n".join(lines)


# ── Document lifecycle ────────────────────────────────────────────────────────

def create_tex_document(filepath, title, author, date):
    preamble = r"""\documentclass{article}
        \usepackage{amsmath}
        \usepackage{geometry}
        \usepackage{booktabs}
        \usepackage{graphicx}
        \usepackage{subcaption}
        \usepackage[table]{xcolor}
        \usepackage{colortbl}
        \title{%s}
        \author{%s}
        \date{%s}
        \begin{document}
        \maketitle
        """ % (title, author, date)
    with open(filepath, "w") as f:
        f.write(preamble)


def start_report(outname):
    outpath = directories["latex"]
    file_path = os.path.join(outpath, "report.tex")
    title = "Report of " + escape_latex(outname)
    author = "Automatically generated"
    date = r"\today"
    create_tex_document(file_path, title, author, date)


def close_tex_document():
    tex_path = os.path.join(directories["latex"], "report.tex")
    with open(tex_path, "a") as f:
        f.write("\n\\end{document}")


def compile_latex(logfile):
    outpath = directories["latex"]
    tex_path = os.path.join(outpath, "report.tex")
    cmd = [
        "pdflatex",
        "-interaction=nonstopmode",
        f"-output-directory={outpath}",
        tex_path
    ]
    log_path = os.path.join(outpath, logfile)
    with open(log_path, "w") as log_file:
        try:
            subprocess.run(cmd, check=True, stdout=log_file, stderr=subprocess.STDOUT)
            print("PDF compilation successful! Log written to", log_path)
        except subprocess.CalledProcessError as e:
            print("PDF compilation failed. See log:", log_path)
            print(e)


def reopen_for_results(latex_dir):
    """Remove the closing \\end{document} and open the Results section."""
    tex_path = os.path.join(latex_dir, "report.tex")
    with open(tex_path, "r") as f:
        content = f.read()
    content = content.rstrip()
    if content.endswith(r"\end{document}"):
        content = content[: -len(r"\end{document}")].rstrip()
    with open(tex_path, "w") as f:
        f.write(content)
    with open(tex_path, "a") as f:
        f.write("\n\n\\section{Results}\n")


def close_and_compile_results(latex_dir, logfile="report_results.log", compile=True):
    tex_path = os.path.join(latex_dir, "report.tex")
    with open(tex_path, "a") as f:
        f.write("\n\\end{document}\n")
    if compile:
        compile_latex(logfile)


# ── Config snapshot ───────────────────────────────────────────────────────────

def save_config_snapshot(output_dir):
    snapshot = {
        "casename":         casename,
        "origin":           origin,
        "known_signals":    [[v, i] for v, i in known_signals],
        "known_beds":       known_beds,
        "known_bed_params": known_bed_params,
        "trainable_params": trainable_params,
        "INER":             INER,
        "ncoloc":           ncoloc,
        "NN_params":        NN_params,
        "max_epochs":       max_epochs,
        "dump_every":       dump_every,
        "plot_every":       plot_every,
        "net_learning":     net_learning,
        "param_learning":   param_learning,
        "fluid":            fluid,
        "early_stopping":   early_stopping,
        "random_seed":      random_seed,
    }
    fpath = os.path.join(output_dir, "config_snapshot.json")
    with open(fpath, "w") as f:
        json.dump(snapshot, f, indent=2)
    return snapshot


# ── Input section ─────────────────────────────────────────────────────────────

def report_config():
    """Appends case and NN/training config tables (LaTeX-safe) to report.tex."""
    sig_str = ", ".join(escape_latex(f"{v}{i}") for v, i in known_signals) or "none"
    tp_str  = ", ".join(escape_latex(p) for p in trainable_params) or "none"

    rows_case = [
        ["Case",                 escape_latex(casename)],
        ["Known signals",        sig_str],
        ["Known beds",           str(known_beds)],
        ["Known bed params",     ", ".join(known_bed_params) or "none"],
        ["Trainable params",     tp_str],
        ["Inertance (INER)",     str(INER)],
        ["Collocation pts",      str(ncoloc)],
        ["Max epochs",           str(max_epochs)],
        ["Early stop patience",  str(early_stopping["patience"])],
        ["Early stop threshold", f"{early_stopping['threshold']:.0e}"],
    ]

    rows_nn = [
        ["Hidden layers",    str(NN_params["hidden_layers"])],
        ["Neurons/layer",    str(NN_params["neurons_per_layer"])],
        ["Activation",       NN_params["activation_f"]],
        ["Fourier features", str(NN_params["FF"])],
        ["FF harmonics",     str(NN_params["nff_harmon"])],
        ["LR network",       f"{net_learning['initial']:.0e}"],
        ["LR params",        f"{param_learning['initial']:.0e}"],
        ["LR decay / steps", f"{net_learning['decay']} / {net_learning['steps']:.0e}"],
    ]

    file_path = os.path.join(directories["latex"], "report.tex")
    write_tex(file_path, generate_latex_table(["Setting", "Value"], rows_case,
                                               "Case configuration."))
    write_tex(file_path, generate_latex_table(["Setting", "Value"], rows_nn,
                                               "Neural network and training configuration."))


def report_vessels(vessels):
    file_path = os.path.join(directories["latex"], "report.tex")
    header = ["Vessel", r"$l$ [cm]", r"$\bar{r}_d$ [mm]",
              r"$\bar{V}_d$ [cm$^3$]", r"$c_o$ [cm/s]"]
    rows = []
    for i in range(vessels["Nv"]):
        rows.append([
            vessels["tag"][i],
            f"{1e2*vessels['length'][i]:.2f}",
            f"{1e3*vessels['r_avg'][i]:.2f}",
            f"{1e6*vessels['Vd'][i]:.2f}",
            f"{1e2*vessels['c_avg'][i]:.2f}"
        ])
    write_tex(file_path, generate_latex_table(header, rows,
                                              "Loaded properties of the vessels."))


def report_lumped_model(vessels, Com0, Res0, Ine0):
    file_path = os.path.join(directories["latex"], "report.tex")
    header = ["Vessel", r"$C_o$ [ml/mmHg]", r"$R_o$ [mmHg/(ml s)]",
              r"$L_{o}$ [mmHg]"]
    rows = []
    for i in range(vessels["Nv"]):
        rows.append([
            vessels["tag"][i],
            f"{1e6/Pa2mmHg*Com0[i]:.6f}",
            f"{Pa2mmHg/1e6*Res0[i]:.6f}",
            f"{Pa2mmHg/1e6*Ine0[i]:.6f}",
        ])
    write_tex(file_path, generate_latex_table(header, rows,
                                              "Estimated lumped parameters of vessels."))


def report_beds(beds, vessels):
    file_path = os.path.join(directories["latex"], "report.tex")
    header = ["End of vessel",
              r"$Z_b$ [mmHg/(ml s)]", r"$C_b$ [ml/mmHg]",
              r"$R_b$ [mmHg/(ml s)]", r"$P_{out}$ [mmHg]", r"$V_{frac}$ [\%]"]
    rows = []
    for i in range(beds["Nb"]):
        nv = beds["vessel"][i] - 1
        rows.append([
            vessels["tag"][nv],
            "Unknown" if beds['Zb'][i] < 0 else f"{Pa2mmHg/1e6*beds['Zb'][i]:.4f}",
            "Unknown" if beds['Cb'][i] < 0 else f"{1e6/Pa2mmHg*beds['Cb'][i]:.4f}",
            "Unknown" if beds['Rb'][i] < 0 else f"{Pa2mmHg/1e6*beds['Rb'][i]:.4f}",
            "Unknown" if beds['Pout'][i] < 0 else f"{Pa2mmHg*beds['Pout'][i]:.4f}",
            "Unknown" if beds['Vfrac'][i] < 0 else f"{100*beds['Vfrac'][i]:.2f}"
        ])
    write_tex(file_path, generate_latex_table(header, rows,
                                              "Loaded properties of the beds."))


def report_computed_beds(beds, vessels, Zb, Cb, Rb, Pout, Vfrac):
    file_path = os.path.join(directories["latex"], "report.tex")
    header = [r"$n_b$",
              r"$Z^b$ [mmHg/(ml s)]", r"$C^b$ [ml/mmHg]",
              r"$R^b$ [mmHg/(ml s)]", r"$P_{out}$ [mmHg]",
              r"$V_{frac}$ [\%]", "End of vessel"]
    rows = []
    for i in range(beds["Nb"]):
        nv = beds["vessel"][i] - 1
        rows.append([
            f"{i+1}",
            f"{Pa2mmHg/1e6*Zb[i]:.4f}",
            f"{1e6/Pa2mmHg*Cb[i]:.4f}",
            f"{Pa2mmHg/1e6*Rb[i]:.4f}",
            f"{Pa2mmHg*Pout[i]:.4f}",
            f"{100*Vfrac[i]:.2f}" + r"\%",
            vessels["tag"][nv],
        ])
    write_tex(file_path, generate_latex_table(header, rows,
                                              "Estimated parameters of the RCR beds."))


def report_cardiac(cardiac):
    file_path = os.path.join(directories["latex"], "report.tex")
    header = [r"$P_s$ [mmHg]", r"$P_d$ [mmHg]",
              r"$\bar{Q}$ [ml/s]", r"$T$ [s]"]
    rows = [[
        f"{Pa2mmHg*cardiac['Ps']:.2f}",
        f"{Pa2mmHg*cardiac['Pd']:.2f}",
        f"{1e6*cardiac['Qavg']:.2f}",
        f"{cardiac['T']:.2f}"
    ]]
    write_tex(file_path, generate_latex_table(header, rows,
                                              "Loaded properties of the pulse."))


def report_junctions(junctions):
    """Include the tree topology graph instead of a text table."""
    file_path = os.path.join(directories["latex"], "report.tex")
    img_path = "figures/tree_graph.pdf"
    content = (
        "\n\\begin{figure}[htb]\n"
        "\\centering\n"
        f"\\includegraphics[width=0.6\\textwidth]{{{img_path}}}\n"
        "\\caption{Arterial network topology.}\n"
        "\\end{figure}\n\n"
    )
    write_tex(file_path, content)


def report_signals(signals):
    """Plot and include only the known signals in the report."""
    figpath = os.path.join(directories["latex"], "figures", "solutions")
    _plot_known_signals(figpath, signals)

    fig_files = sorted(os.listdir(figpath))
    if not fig_files:
        return

    file_path = os.path.join(directories["latex"], "report.tex")
    with open(file_path, "a") as f:
        f.write("\n\\begin{figure}[htb]\n\\centering\n")
        for k, fig in enumerate(fig_files):
            img_path = f"figures/solutions/{fig}"
            f.write(
                f"\\begin{{subfigure}}{{0.48\\textwidth}}\n"
                f"    \\centering\n"
                f"    \\includegraphics[width=\\textwidth]{{{img_path}}}\n"
                f"\\end{{subfigure}}\n"
            )
            if (k + 1) % 6 == 0 and k + 1 < len(fig_files):
                f.write("\\caption{Known input signals.}\n\\end{figure}\n\n"
                        "\\begin{figure}[htb]\n\\centering\n")
        f.write("\\caption{Known input signals.}\n\\end{figure}\n\n")


def _plot_known_signals(figpath, signals):
    for key, series in signals.items():
        if key not in known_signals:
            continue
        var, vessel = key[0], key[1]
        save_path = os.path.join(figpath, f"v{vessel}_{var}.pdf")
        if var == "P":
            ylabel, y = f"{var} [mmHg]", series["val_phys"] * Pa2mmHg
        elif var == "Q":
            ylabel, y = f"{var} [ml/s]",  series["val_phys"] * 1e6
        else:
            continue
        plot_one_signal(series["t_phys"], y, ylabel,
                        f"{var} at vessel {vessel}", save_path)


# ── Results section ───────────────────────────────────────────────────────────

def report_timing(adam_time, epochs_run, lbfgs_time, lbfgs_iters, latex_dir):
    avg_adam_ms  = adam_time / epochs_run * 1e3
    avg_lbfgs_ms = lbfgs_time / lbfgs_iters * 1e3 if lbfgs_iters > 0 else 0.0
    total_time   = adam_time + lbfgs_time
    rows = [
        ["Adam — total time",       f"{adam_time:.1f} s  ({adam_time/60:.1f} min)"],
        ["Adam — epochs",           str(epochs_run)],
        ["Adam — time per epoch",   f"{avg_adam_ms:.2f} ms"],
        ["L-BFGS — total time",     f"{lbfgs_time:.1f} s  ({lbfgs_time/60:.1f} min)"],
        ["L-BFGS — iterations",     str(lbfgs_iters)],
        ["L-BFGS — time per iter",  f"{avg_lbfgs_ms:.2f} ms"],
        ["Total time",              f"{total_time:.1f} s  ({total_time/60:.1f} min)"],
    ]
    file_path = os.path.join(latex_dir, "report.tex")
    write_tex(file_path, generate_latex_table(["Metric", "Value"], rows,
                                              "Training time summary."))


def report_loss_figure(latex_dir):
    file_path = os.path.join(latex_dir, "report.tex")
    content = (
        "\n\\begin{figure}[htb]\n"
        "\\centering\n"
        "\\includegraphics[width=0.7\\textwidth]{figures/loss_evolution.pdf}\n"
        "\\caption{Loss evolution during training.}\n"
        "\\end{figure}\n\n"
    )
    write_tex(file_path, content)


def report_error_table(metrics, vessels_tag, latex_dir):
    """
    Error metrics table: one row per location, P and Q side-by-side.
    Known-signal cells: gray background with white text.
    Locations without reference data show --.
    """
    file_path = os.path.join(latex_dir, "report.tex")
    known_set = set(known_signals)

    def _cell(var, nv, field):
        key = (var, nv)
        if key not in metrics:
            return "--"
        val = metrics[key][field] * 100.0
        fmt = (f"{val:+.1f}\\%" if field in ("sys", "dias")
               else f"{abs(val):.1f}\\%")
        if key in known_set:
            return f"\\cellcolor{{gray!60}}\\textcolor{{white}}{{{fmt}}}"
        return fmt

    header = ["Vessel",
              r"P MAE", r"P SYS", r"P DIAS",
              r"Q MAE", r"Q SYS", r"Q DIAS"]

    Nv = len(vessels_tag)
    rows = []
    for nv in range(Nv + 1):
        tag = "inlet" if nv == 0 else vessels_tag[nv - 1]
        rows.append([
            tag,
            _cell("P", nv, "avg"),  _cell("P", nv, "sys"),  _cell("P", nv, "dias"),
            _cell("Q", nv, "avg"),  _cell("Q", nv, "sys"),  _cell("Q", nv, "dias"),
        ])

    caption = (
        "Error metrics at final epoch (\\%). "
        "\\colorbox{gray!60}{\\textcolor{white}{Gray}} cells were known during training."
    )
    write_tex(file_path, generate_latex_table(header, rows, caption))


def report_param_figures(latex_dir):
    """Include parameter evolution plots saved in param/."""
    param_dir = directories["param"]
    if not os.path.isdir(param_dir):
        return
    fig_dir = os.path.join(latex_dir, "figures")
    plots = sorted(f for f in os.listdir(fig_dir) if f.startswith("param_") and f.endswith(".pdf"))
    if not plots:
        return
    file_path = os.path.join(latex_dir, "report.tex")
    with open(file_path, "a") as f:
        f.write("\n\\begin{figure}[htb]\n\\centering\n")
        for fig in plots:
            img_path = os.path.join("figures", fig)
            f.write(
                f"\\begin{{subfigure}}{{0.48\\textwidth}}\n"
                f"    \\centering\n"
                f"    \\includegraphics[width=\\textwidth]{{{img_path}}}\n"
                f"\\end{{subfigure}}\n"
            )
        f.write("\\caption{Calibrated parameter evolution during training.}\n"
                "\\end{figure}\n\n")


def report_prediction_figures(Nv, latex_dir):
    """
    2-column (P | Q) layout, one row per vessel (0 = inlet, 1..Nv).
    Figures must already be saved to ../predictions/ by plot_reference_and_prediction.
    """
    file_path = os.path.join(latex_dir, "report.tex")
    with open(file_path, "a") as f:
        f.write("\n\\begin{figure}[htb]\n\\centering\n")
        for nv in range(Nv + 1):
            for var in ("P", "Q"):
                img_path = f"figures/pred{var}{nv}.pdf"
                f.write(
                    f"\\begin{{subfigure}}{{0.48\\textwidth}}\n"
                    f"    \\centering\n"
                    f"    \\includegraphics[width=\\textwidth]{{{img_path}}}\n"
                    f"\\end{{subfigure}}\n"
                )
            if nv > 0 and nv % 4 == 0:
                f.write(
                    "\\caption{Predicted (dashed red) vs.\\ reference (black) signals.}\n"
                    "\\end{figure}\n\n"
                    "\\begin{figure}[htb]\\ContinuedFloat\n\\centering\n"
                )
        f.write(
            "\\caption{Predicted (dashed red) vs.\\ reference (black) signals.}\n"
            "\\end{figure}\n\n"
        )
