# ── CLI / config patch ────────────────────────────────────────────────────────
import argparse, os, config
from datetime import datetime


_p = argparse.ArgumentParser()
_p.add_argument("--seed",              type=int, default=None)
_p.add_argument("--hidden_layers",     type=int, default=None)
_p.add_argument("--neurons_per_layer", type=int, default=None)
_p.add_argument("--ncoloc",            type=int, default=None)
_p.add_argument("--max_epochs",        type=int, default=None)
_p.add_argument("--name",              type=str, default=None)
_cli = _p.parse_args()

if _cli.seed              is not None: config.random_seed                    = _cli.seed
if _cli.hidden_layers     is not None: config.NN_params["hidden_layers"]     = _cli.hidden_layers
if _cli.neurons_per_layer is not None: config.NN_params["neurons_per_layer"] = _cli.neurons_per_layer
if _cli.ncoloc            is not None: config.ncoloc                         = _cli.ncoloc
if _cli.max_epochs        is not None: config.max_epochs                     = _cli.max_epochs
if _cli.name is not None:
    _old = config.outname
    config.outname = f"{config.casename}_{_cli.name}_{datetime.now().strftime('%Y%m%d%H%M')}"
    for k in config.directories:           # update in-place so all importers see the change
        config.directories[k] = config.directories[k].replace(_old, config.outname)


# ── Imports ───────────────────────────────────────────────────────────────────
from src.load import load_all
from src.report import (start_report, close_tex_document, compile_latex,
                        save_config_snapshot, report_config,
                        reopen_for_results, report_timing, report_loss_figure,
                        report_error_table, report_param_figures,
                        report_prediction_figures, close_and_compile_results)
from src.tree import generate_tree
from src.plot import draw_tree, plot_reference_and_prediction, plot_loss_history, plot_param_evolution
from src.pinn import construct_DNN, convert_tree_to_tensors, recover_values_for_loss, train_step, compute_weights, recover_prediction_as_arrays, log_step, run_lbfgs
from src.metrics import (compute_error_metrics, save_metrics_csv,
                         measure_time_now, save_loss_csv, check_early_stopping, save_param_csv,
                         save_timing_summary, save_prediction_csv, save_Rb_from_RT_csv)
from config import directories, outname, max_epochs, dump_every, plot_every, early_stopping, trainable_params, compile_latex_report
import time, random, numpy as np, tensorflow as tf



def init_logger():
    return {
        "loss": [],               # [(epoch, loss)]
        "residuals": {},          # name -> [(epoch, value)]
        "weighted_residuals": {}, # name -> [(epoch, value)]
        "params": {},             # label -> [(epoch, scale)]  (dimensionless)
    }

if __name__ == "__main__":

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    for key, dir in directories.items():
        if "in" not in key:
            os.makedirs(dir, exist_ok=True)

    log_file = open(os.path.join(directories["out"], "run.log"), "w")

    # Print to both terminal and run.log
    def log_print(*args, **kwargs):
        print(*args, **kwargs)
        kwargs.pop("file", None)
        print(*args, file=log_file, **kwargs)

    tf.random.set_seed(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    save_config_snapshot(directories["out"])



    start_report(outname)

    in_files_list = os.listdir(directories["in"])
    vessels, beds, cardiac, signals, junctions = load_all(directories["in"], in_files_list)

    tree = generate_tree(vessels, beds, cardiac, signals, junctions)
    draw_tree(tree["mJ"], vessels["tag"])
    report_config()
    close_tex_document()
    if compile_latex_report:
        compile_latex("report.log")

    model, optimizers = construct_DNN(tree["Nv"])
    pinn_dict = convert_tree_to_tensors(tree)


    ## train 
    logger = init_logger()

    t_start = time.time()
    for epoch in range(max_epochs):
        weights = compute_weights(epoch)
        loss_tf, mean_res_tf, res_tf = train_step(model, optimizers, weights, pinn_dict)

        if epoch % dump_every == 0:
            loss_value, mean_res, res = recover_values_for_loss(loss_tf, mean_res_tf, res_tf)

            log_print(f"Epoch {epoch}, loss = {loss_value:.4e}", flush=True)
            for name, var in pinn_dict["trainable"].items():
                val = np.exp(var.numpy()) if name.startswith("log_") else var.numpy()
                label = name[4:] if name.startswith("log_") else name
                log_print(f"  {label} = {val}", flush=True)
            log_step(logger, epoch, loss_value, mean_res, res)
            save_loss_csv(epoch, loss_value, res, directories["out"])

            # Log and save trainable parameter evolution
            for name, var in pinn_dict["trainable"].items():
                label = name[4:] if name.startswith("log_") else name
                scale = np.exp(var.numpy())
                if label not in logger["params"]:
                    logger["params"][label] = []
                val = float(scale) if np.isscalar(scale) else scale.tolist()
                logger["params"][label].append((epoch, val))
            if pinn_dict["trainable"]:
                save_param_csv(epoch, pinn_dict["trainable"], directories["param"])

            # Derive and log per-bed Rb scales from trainable RT
            if "log_RT_scale" in pinn_dict["trainable"]:
                RT_scale   = float(np.exp(pinn_dict["trainable"]["log_RT_scale"].numpy()))
                Rb_eff     = (tree["RT"] * RT_scale) / tree["Vfrac"] - tree["Zb"]
                Rb_scales  = Rb_eff / tree["Rb"]
                if "Rb_scale" not in logger["params"]:
                    logger["params"]["Rb_scale"] = []
                logger["params"]["Rb_scale"].append((epoch, Rb_scales.tolist()))
                save_Rb_from_RT_csv(epoch, Rb_scales, directories["param"])

            if check_early_stopping(logger, early_stopping, epoch):
                log_print(f"Early stopping triggered at epoch {epoch} "
                      f"(patience={early_stopping['patience']} windows, "
                      f"threshold={early_stopping['threshold']:.0e})", flush=True)
                break


        if epoch > 0 and epoch % plot_every == 0:
            pred_signals = recover_prediction_as_arrays(model, pinn_dict, tree["adim"], tree["norm"])
            plot_reference_and_prediction(pred_signals, signals, directories["predictions"])
            plot_loss_history(logger, directories["out"])
            if logger["params"]:
                plot_param_evolution(logger["params"], directories["param"])
            metrics = compute_error_metrics(pred_signals, signals)
            save_metrics_csv(metrics, vessels["tag"], epoch, directories["out"])
            save_prediction_csv(pred_signals, directories["out"])
            measure_time_now(t_start, epoch,  directories["out"])

    t_adam_end = time.time()
    adam_time = t_adam_end - t_start
    epochs_run = epoch + 1
    avg_time = adam_time / epochs_run
    log_print(f"\n--- Adam complete ---")
    log_print(f"Total time:        {adam_time:.1f} s  ({adam_time/60:.2f} min)")
    log_print(f"Average per epoch: {avg_time*1e3:.2f} ms  ({avg_time:.4f} s)")
    log_print(f"Epochs:            {epochs_run}")

    # ── L-BFGS refinement phase ───────────────────────────────────────────────
    log_print("\n--- L-BFGS phase ---", flush=True)
    weights_final = compute_weights(epoch)
    lbfgs_result = run_lbfgs(model, pinn_dict, weights_final,
                             print_fn=log_print,
                             output_dir=directories["out"],
                             epoch_offset=epochs_run)
    t_lbfgs_end = time.time()
    lbfgs_time = t_lbfgs_end - t_adam_end
    total_time = t_lbfgs_end - t_start
    log_print(f"L-BFGS: {lbfgs_result.message}")
    log_print(f"Iterations:  {lbfgs_result.nit}")
    log_print(f"Final loss:  {lbfgs_result.fun:.4e}")
    log_print(f"L-BFGS time: {lbfgs_time:.1f} s  ({lbfgs_time/60:.2f} min)")
    log_print(f"Total time:  {total_time:.1f} s  ({total_time/60:.2f} min)", flush=True)
    save_timing_summary(adam_time, epochs_run, lbfgs_time, lbfgs_result.nit,
                        directories["out"])

    # ── Results section of the report ────────────────────────────────────────
    pred_signals = recover_prediction_as_arrays(model, pinn_dict, tree["adim"], tree["norm"])
    plot_reference_and_prediction(pred_signals, signals, directories["predictions"],
                                  pdf_dir=directories["figures"])
    plot_loss_history(logger, directories["out"], pdf_dir=directories["figures"])
    if logger["params"]:
        plot_param_evolution(logger["params"], directories["param"],
                             pdf_dir=directories["figures"])
    metrics = compute_error_metrics(pred_signals, signals)
    save_metrics_csv(metrics, vessels["tag"], epochs_run, directories["out"])
    save_prediction_csv(pred_signals, directories["out"])

    reopen_for_results(directories["latex"])
    report_timing(adam_time, epochs_run, lbfgs_time, lbfgs_result.nit, directories["latex"])
    report_loss_figure(directories["latex"])
    if trainable_params:
        report_param_figures(directories["latex"])
    report_error_table(metrics, vessels["tag"], directories["latex"])
    report_prediction_figures(tree["Nv"], directories["latex"])
    close_and_compile_results(directories["latex"], compile=compile_latex_report)

