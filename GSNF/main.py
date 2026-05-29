from pathlib import Path
import argparse
import traceback

from experiments.exp_biclass import Exp_BiClass


parser = argparse.ArgumentParser(
    description="Run GSNF binary classification experiments.")

# Training process
parser.add_argument("--random-state", type=int, default=1, help="Random seed")
parser.add_argument("--proj-path", type=str, default=str(Path(__file__).parents[0]))
parser.add_argument("--test-info", default="testing")
parser.add_argument("--num-dl-workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--epochs-max", type=int, default=1000)
parser.add_argument("--patience", type=int, default=10)
parser.add_argument("--early-stop-metric", type=str, default="val_loss",
                    choices=["val_loss", "ce_loss", "auprc", "auroc"])
parser.add_argument("--weight-decay", type=float, default=1e-4)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-scheduler-step", type=int, default=20)
parser.add_argument("--lr-decay", type=float, default=0.5)
parser.add_argument("--lr-scheduler", type=str, default="step",
                    choices=["none", "step", "plateau"])
parser.add_argument("--plateau-monitor", type=str, default="auprc",
                    choices=["val_loss", "ce_loss", "auprc", "auroc"])
parser.add_argument("--plateau-mode", type=str, default="auto",
                    choices=["auto", "min", "max"])
parser.add_argument("--plateau-factor", type=float, default=0.1)
parser.add_argument("--plateau-patience", type=int, default=1)
parser.add_argument("--plateau-threshold", type=float, default=1e-4)
parser.add_argument("--plateau-min-lr", type=float, default=1e-8)
parser.add_argument("--ema-decay", type=float, default=0.0)
parser.add_argument("--swa-start", type=int, default=0)
parser.add_argument("--clip-gradient", action="store_false")
parser.add_argument("--clip", type=float, default=1.0)

# Datasets
parser.add_argument("--data", default="physionet12",
                    choices=["physionet12", "p12", "eICU", "P19", "MIMIC4"])
parser.add_argument("--data-dir", type=str, default=None,
                    help="Optional dataset root. Defaults to <proj-path>/data.")
parser.add_argument("--num-samples", type=int, default=-1)
parser.add_argument("--ts-full", action="store_true")
parser.add_argument("--mask-type", default="default", choices=["default", "cumsum"])
parser.add_argument("--time-scale", default="time_max",
                    choices=["time_max", "self_max", "constant", "none", "max"])
parser.add_argument("--time-constant", type=float, default=2880)
parser.add_argument("--first-dim", default="batch", choices=["batch", "time_series"])
parser.add_argument("--batch-size", type=int, default=50)
parser.add_argument("--t-offset", type=float, default=0.1)
parser.add_argument("--down-times", type=int, default=1)
parser.add_argument("--time-max", type=int, default=2880)
parser.add_argument("--next-start", type=float, default=1440)
parser.add_argument("--mask-drop-rate", type=float, default=0.0)
parser.add_argument("--recon-mask-drop-rate", type=float, default=0.0)
parser.add_argument("--scale-train-only", action="store_true")
parser.add_argument("--norm", action="store_false")

# GSNF solver
parser.add_argument("--hidden-layers", type=int, default=3)
parser.add_argument("--hidden-dim", type=int, default=128)
parser.add_argument("--flow-layers", type=int, default=2)
parser.add_argument("--time-net", type=str, default="TimeTanh",
                    choices=["TimeFourier", "TimeFourierBounded", "TimeLinear", "TimeTanh"])
parser.add_argument("--time-hidden-dim", type=int, default=8)
parser.add_argument("--graph-segments", type=int, default=4)
parser.add_argument("--graph-hidden-dim", type=int, default=16)
parser.add_argument("--graph-pool", type=str, default="learned",
                    choices=["learned", "kl_weighted", "uniform"])
parser.add_argument("--gnn-flow-conv", type=str, default="basic",
                    choices=["basic", "residual", "gated"])
parser.add_argument("--gcn-hidden-dim", type=int, default=64)

# Latent model / classifier
parser.add_argument("--k-iwae", type=int, default=3)
parser.add_argument("--kl-coef", type=float, default=1.0)
parser.add_argument("--latent-dim", type=int, default=20)
parser.add_argument("--classifier-input", default="z0", choices=["z0", "zn"])
parser.add_argument("--classifier-pool", default="none",
                    choices=["none", "mean", "last", "mean_last"])
parser.add_argument("--classifier-head", default="gsnf",
                    choices=["gsnf", "deep", "norm", "small", "linear", "bn_mlp"])
parser.add_argument("--classifier-hidden-dim", type=int, default=0)
parser.add_argument("--classifier-dropout", type=float, default=0.0)
parser.add_argument("--classifier-grad-scale", type=float, default=-1.0)
parser.add_argument("--init-classifier-bias", action="store_true")
parser.add_argument("--use-static-input", action="store_true")
parser.add_argument("--static-source", type=str, default="auto",
                    choices=["auto", "batch", "summary"])
parser.add_argument("--static-dim", type=int, default=0)
parser.add_argument("--static-summary-vars", type=int, default=5)
parser.add_argument("--train-w-reconstr", action="store_false")
parser.add_argument("--ratio-ce", type=float, default=1000)
parser.add_argument("--ratio-itg", type=float, default=0.1)
parser.add_argument("--ratio-rtg", type=float, default=0.1)
parser.add_argument("--ratio-graph-kl", type=float, default=1.0)
parser.add_argument("--itg-margin", type=float, default=1e-6)
parser.add_argument("--itg-margin-mode", type=str, default="fixed",
                    choices=["fixed", "lb"])
parser.add_argument("--itg-reinit-frac", type=float, default=0.5)
parser.add_argument("--balanced-ce", action="store_true")
parser.add_argument("--balanced-batch-sampler", action="store_true")
parser.add_argument("--balanced-batch-pos-repeat", type=int, default=3)
parser.add_argument("--use-delta-input", action="store_true")
parser.add_argument("--ce-pos-weight", type=float, default=0.0)
parser.add_argument("--ce-label-smoothing", type=float, default=0.0)
parser.add_argument("--focal-gamma", type=float, default=0.0)
parser.add_argument("--ratio-zz", type=float, default=0.0)
parser.add_argument("--prior-mu", type=float, default=0.0)
parser.add_argument("--prior-std", type=float, default=1.0)
parser.add_argument("--obsrv-std", type=float, default=0.01)
parser.add_argument("--combine-methods", default="average",
                    choices=["average", "kl_weighted"])
if __name__ == "__main__":
    args = parser.parse_args()
    experiment = Exp_BiClass(args)
    try:
        experiment.run()
        experiment.finish()
    except Exception:
        err_path = experiment.proj_path / "log" / f"err_{experiment.args.exp_name}.log"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with open(err_path, "w") as fout:
            print(traceback.format_exc(), file=fout)
        raise
