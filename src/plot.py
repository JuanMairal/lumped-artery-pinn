import matplotlib.pyplot as plt
import numpy as np
import os
import networkx as nx
from config import Pa2mmHg, directories


def remove_box(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


def plot_one_signal(t, y, ylabel, title, save_path, lc='k'):

    fig, ax = plt.subplots(1, 1, figsize=(6,4))

    format_axis(ax)

    ax.plot(t, y, '-', linewidth=3, color=lc)
    ax.grid(color='k', linestyle=(0, (4, 8)), linewidth=1, alpha=0.55)
    ax.locator_params(tight=True, nbins=4)
    ax.tick_params(axis='both', which='major', labelsize=24)

    plt.xlabel("t (s)", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_loss_history(logger, output_dir, scale_tolerance=1e-3, pdf_dir=None):
    """
    Plots the training loss history.

    Residuals whose median value is more than 1/scale_tolerance below the
    largest residual median are omitted — they would compress the y-axis
    without adding useful information (e.g. the periodicity term, which
    operates on a single collocation point and is typically many orders of
    magnitude smaller than the physics residuals).
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    epochs, values = zip(*logger["loss"])
    ax.plot(epochs, values, label="total")
    ax.scatter(epochs[0], values[0], marker="o")

    # Compute median of each residual to decide which ones to show
    medians = {
        key: np.median([v for _, v in logger["residuals"][key]])
        for key in logger["residuals"]
    }
    if medians:
        max_median = max(medians.values())
        visible = {key for key, med in medians.items() if med >= max_median * scale_tolerance}
    else:
        visible = set()

    for key in logger["residuals"].keys():
        if key not in visible:
            continue

        # Unweighted
        epochs_u, values_u = zip(*logger["residuals"][key])
        line, = ax.plot(epochs_u, values_u, linestyle="--", alpha=0.6, label="_nolegend_")
        color = line.get_color()
        ax.scatter(epochs_u[0], values_u[0], color=color, marker="o", alpha=0.6)

        # Weighted
        epochs_w, values_w = zip(*logger["weighted_residuals"][key])
        ax.plot(epochs_w, values_w, linestyle="-", color=color, label=key)
        ax.scatter(epochs_w[0], values_w[0], color=color, marker="o")

    ax.set_xlabel("Epoch", fontsize=20)
    ax.set_ylabel("Loss", fontsize=20)
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.title("Loss Evolution")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_evolution.svg"))
    if pdf_dir is not None:
        plt.savefig(os.path.join(pdf_dir, "loss_evolution.pdf"))
    plt.close()



def plot_diastolic_decay(Pout,lvet,tau,t,p,t_diastole,p_diastole,t_fit,p_exp, ps,pd):
    lgndP = "Est. $P_{out}$ = " + f"{Pa2mmHg*Pout:.3f}" + " mmHg"
    lgndt = "LVET = " + f"{lvet:.3f}" + " s"
    lgndtau = "tau = " + f"{tau:.3f}" + " s"

    figPout, axPout = plt.subplots(1,1, figsize=(5,5))

    axPout.plot(t, Pa2mmHg*p, label="Peripheral pressure", color="k")
    axPout.plot(t_diastole, Pa2mmHg*p_diastole, label="Diastolic decay", color="g")
    axPout.plot(t_fit, Pa2mmHg*p_exp, label="Diastole fit", color="red", linestyle='--')


    axPout.axvline(lvet, linestyle="--", linewidth=1, color = 'k')
    axPout.axhline(y=Pa2mmHg*Pout, linestyle="--", linewidth=1, label=lgndt, color = 'k')
    axPout.text(1.05*lvet,Pa2mmHg*pd, lgndt, ha="left", va="bottom")
    axPout.text(0.85*t_fit[-1], 0.9*Pa2mmHg*(ps), lgndtau, ha="right", va="top")
    axPout.text(0.60, Pa2mmHg*Pout, lgndP, ha="left", va="bottom")
    axPout.set_xlabel("Time [s]")
    axPout.set_ylabel("Pressure [mmHg]")
    format_axis(axPout)
    figPout.savefig(os.path.join(directories["figures"], f"diastole_fit.svg"), format='svg')
    plt.close()


def plot_ready_data(signal):
     
    fig, ax = plt.subplots(1,1, figsize=(5,5))
    
    varname = signal["var"]
    locat = signal["nv"]
    name = f"signal{varname}at{locat}.svg"
    ax.plot(signal["t_ready"], signal["val_ready"])

    ax.set_xlabel("Time [-]")
    ax.set_ylabel(f"{varname} [-]")
    format_axis(ax)

    fig.savefig(os.path.join(directories["figures"], name), format='svg')
    plt.close()


def format_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(color='k', linestyle=(0, (4, 8)), linewidth=1, alpha=0.55)
    ax.locator_params(tight=True, nbins=3)
    ax.tick_params(axis='both', which='major',labelsize=16)



def plot_residuals(ne, residuals, output_dir):
    fig, ax = plt.subplots()

    resnames = residuals.keys()
    values = residuals.values()
    ax.bar(resnames, values)

    ax.set_title(f"Distribution of residuals at epoch {ne}")
    fig.savefig(os.path.join(output_dir, f"residuals.svg"), format='svg')
    plt.close(fig)


def plot_reference_and_prediction(pred_signals, ground_truth, save_path, pdf_dir=None):
    """Save one P and one Q figure per vessel (0..Nv).
    Reference line is drawn only when ground-truth data exists for that signal."""
    Nv = len(pred_signals) - 1   # pred_signals has Nv+1 entries (0=inlet, 1..Nv)

    for nv in range(Nv + 1):
        t_pred = pred_signals[nv]["t"]
        for var in ("P", "Q"):
            if var == "P":
                serie_pred = Pa2mmHg * pred_signals[nv]["P"]
                ylabel = "P [mmHg]"
            else:
                serie_pred = 1e6 * pred_signals[nv]["Q"]
                ylabel = "Q [ml/s]"

            fig, ax = plt.subplots(1, 1, figsize=(6, 4))

            key = (var, nv)
            if key in ground_truth:
                ref = ground_truth[key]
                t_ref = ref["t_phys"]
                serie_ref = (Pa2mmHg * ref["val_phys"] if var == "P"
                             else 1e6 * ref["val_phys"])
                ax.plot(t_ref, serie_ref, '-', linewidth=3,
                        label="Reference", color="k")

            ax.plot(t_pred, serie_pred, linestyle="--", linewidth=3,
                    label="Predicted", color='r')
            ax.grid(color='k', linestyle=(0, (4, 8)), linewidth=1, alpha=0.55)
            ax.locator_params(tight=True, nbins=4)
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_xlabel("t [s]", fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
            ax.legend(fontsize=14)

            fig.savefig(os.path.join(save_path, f"pred{var}{nv}.svg"),
                        bbox_inches="tight")
            if pdf_dir is not None:
                fig.savefig(os.path.join(pdf_dir, f"pred{var}{nv}.pdf"),
                            bbox_inches="tight")
            plt.close()


def plot_param_evolution(param_history, param_dir, pdf_dir=None):
    """Plot calibrated parameter scale vs epoch.

    param_history : dict  label -> [(epoch, value)]
        Scalar params (e.g. c0): value is a float (the dimensionless scale).
        Vector params (e.g. Rb): value is a list of floats, one per bed.
    """
    for label, history in param_history.items():
        if not history:
            continue
        epochs = [e for e, _ in history]
        values = [v for _, v in history]

        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        format_axis(ax)

        if np.isscalar(values[0]):
            ax.plot(epochs, values, '-', linewidth=2, color='k')
            ax.axhline(y=1.0, linestyle='--', linewidth=1, color='gray',
                       label="Initial estimate")
            ax.set_ylabel(f"{label} scale [-]", fontsize=16)
        else:
            arr = np.array(values)   # (n_epochs, Nb)
            for b in range(arr.shape[1]):
                ax.plot(epochs, arr[:, b], linewidth=2, label=f"bed {b+1}")
            ax.axhline(y=1.0, linestyle='--', linewidth=1, color='gray',
                       label="Initial estimate")
            ax.set_ylabel(f"{label} scale [-]", fontsize=16)
            ax.legend(fontsize=12)

        ax.set_xlabel("Epoch", fontsize=16)
        ax.set_title(f"Calibration of {label}", fontsize=16)
        ax.set_xscale("log")
        plt.tight_layout()
        plt.savefig(os.path.join(param_dir, f"{label}_evolution.svg"))
        if pdf_dir is not None:
            plt.savefig(os.path.join(pdf_dir, f"param_{label}_evolution.pdf"))
        plt.close()

def extract_tree_edges(mJ):
    """
    Returns a list of (parent_vessel, child_vessel) pairs.
    """
    edges = []

    Nj, Nv = mJ.shape

    for j in range(Nj):
        upstream = np.where(mJ[j] == 1)[0][0]
        downstream = np.where(mJ[j] == -1)[0]

        for v in downstream:
            edges.append((upstream, v))

    return edges


def hierarchy_pos(G, root=0, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    pos = {}

    def _hierarchy_pos(v, x, y, dx):
        pos[v] = (x, y)
        children = list(G.successors(v))
        if children:
            dx_child = dx / len(children)
            for i, child in enumerate(children):
                _hierarchy_pos(child,
                               x - dx/2 + dx_child/2 + i*dx_child,
                               y - vert_gap,
                               dx_child)

    _hierarchy_pos(root, xcenter, vert_loc, width)
    return pos

def draw_tree(mJ, tags):

    matrix = mJ[:,1:]
    
    G = nx.DiGraph()

    edges = extract_tree_edges(matrix)
    G.add_edges_from(edges)

    pos = hierarchy_pos(G)
    labels = {v: f"v{v+1}" for v in G.nodes()}
    nx.draw(G, pos, labels=labels, with_labels=True, node_size=1200, arrows=True)

    legend_text = "\n".join(
        f"v{i+1}: {tags[i]}" for i in range(len(tags))
    )

    plt.gcf().text(
        0.85, 0.85,
        legend_text,
        fontsize=9,
        verticalalignment="center",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    isave = os.path.join(directories["out"], f"tree_graph.svg")
    plt.savefig(isave, bbox_inches="tight")
    plt.savefig(os.path.join(directories["figures"], "tree_graph.pdf"), bbox_inches="tight")
    plt.close()
