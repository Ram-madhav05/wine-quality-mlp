"""Reproducible MLP architecture study on UCI red Wine Quality.

Run: python train_mlp.py
All results are computed, never hard-coded. The test set is used only after
architecture selection. Exact duplicate records are removed before splitting.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import platform
import time
import urllib.request
import zipfile
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
    log_loss, classification_report, confusion_matrix, mean_absolute_error)
from threadpoolctl import threadpool_limits
import joblib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "results"
FIG = ROOT / "figures"
ARCHITECTURES = [(16,), (32,), (64,), (128,), (32, 32), (64, 64),
                 (32, 32, 32), (64, 64, 64)]
SEEDS = [11, 22, 33, 44, 55]
SPLIT_SEED = 42
MAX_EPOCHS = 500
PATIENCE = 30
MIN_DELTA = 1e-4
LABELS = np.arange(3, 9)
SOURCE = "https://archive.ics.uci.edu/static/public/186/wine+quality.zip"


def dump_json(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def metrics(y, pred):
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_f1=float(f1_score(y, pred, labels=LABELS,
                                       average="macro", zero_division=0)),
                weighted_f1=float(f1_score(y, pred, labels=LABELS,
                                          average="weighted", zero_division=0)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)),
                mae=float(mean_absolute_error(y, pred)),
                within_one=float(np.mean(np.abs(y - pred) <= 1)))


def fit_run(arch, seed, x_train, y_train, x_val, y_val):
    """One partial_fit call is one epoch; restore lowest validation loss."""
    model = MLPClassifier(hidden_layer_sizes=arch, activation="relu",
        solver="adam", alpha=0.0001, batch_size=64,
        learning_rate_init=0.001, random_state=seed,
        early_stopping=False, shuffle=True, max_iter=1)
    best_loss = float("inf")
    significant_best = float("inf")
    best_model = None
    stale = 0
    trace = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.partial_fit(x_train, y_train, classes=LABELS)
        tr_prob = model.predict_proba(x_train)
        va_prob = model.predict_proba(x_val)
        tr_loss = log_loss(y_train, tr_prob, labels=LABELS)
        va_loss = log_loss(y_val, va_prob, labels=LABELS)
        trace.append(dict(epoch=epoch, train_log_loss=float(tr_loss),
            validation_log_loss=float(va_loss),
            train_accuracy=float(accuracy_score(y_train, model.classes_[tr_prob.argmax(1)])),
            validation_accuracy=float(accuracy_score(y_val, model.classes_[va_prob.argmax(1)]))))
        if va_loss < best_loss:
            best_loss, best_epoch = va_loss, epoch
            best_model = deepcopy(model)
        if va_loss < significant_best - MIN_DELTA:
            significant_best, stale = va_loss, 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    elapsed = time.perf_counter() - started
    tr = metrics(y_train, best_model.predict(x_train))
    va = metrics(y_val, best_model.predict(x_val))
    row = dict(architecture="x".join(map(str, arch)), hidden_layers=len(arch),
        seed=seed, parameters=sum(w.size for w in best_model.coefs_) +
        sum(b.size for b in best_model.intercepts_), best_epoch=best_epoch,
        epochs_run=epoch, seconds=elapsed, validation_log_loss=float(best_loss))
    row.update({"train_" + k: v for k, v in tr.items()})
    row.update({"validation_" + k: v for k, v in va.items()})
    return best_model, row, trace


def make_plots(frame, counts, traces, selected, cm, report):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "savefig.dpi": 200})
    fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
    bars = ax.bar(counts.index.astype(str), counts.values, color="#305e83", width=.65)
    ax.bar_label(bars, padding=3)
    ax.set(xlabel="Original quality grade", ylabel="Unique records", ylim=(0, 650),
           title="Class distribution after exact duplicate removal")
    fig.savefig(FIG / "class_distribution.png"); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    pos = np.arange(len(frame))
    for ax, metric, title in zip(axes, ["validation_accuracy", "validation_macro_f1"],
                               ["Validation accuracy", "Validation macro F1"]):
        ax.bar(pos, frame[metric + "_mean"], yerr=frame[metric + "_std"],
               color=["#1b755a" if a == selected else "#4578a0" for a in frame.architecture],
               capsize=3)
        ax.set_xticks(pos, frame.architecture, rotation=40, ha="right")
        ax.set(title=title, ylabel="Score", ylim=(0, .8))
        ax.grid(axis="y", alpha=.15)
    fig.suptitle("Eight architectures on the same validation set\nMean ± sample SD across five initialization seeds", fontsize=12)
    fig.savefig(FIG / "architecture_comparison.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.6), layout="constrained")
    ax.plot(frame.architecture, frame.train_accuracy_mean, "o-", label="Training")
    ax.plot(frame.architecture, frame.validation_accuracy_mean, "s-", label="Validation")
    ax.set(ylabel="Mean accuracy", ylim=(.3, 1), title="Generalization gap at restored checkpoints")
    ax.tick_params(axis="x", rotation=25); ax.legend(); ax.grid(alpha=.15)
    fig.savefig(FIG / "generalization_gap.png"); plt.close(fig)
    trace = pd.DataFrame(traces[selected + "_11"])
    best_epoch = int(report["selected_run"]["best_epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    axes[0].plot(trace.epoch, trace.train_log_loss, label="Training")
    axes[0].plot(trace.epoch, trace.validation_log_loss, label="Validation")
    axes[0].set(title="Cross entropy excluding L2 penalty", ylabel="Log loss", xlabel="Epoch")
    axes[1].plot(trace.epoch, trace.train_accuracy, label="Training")
    axes[1].plot(trace.epoch, trace.validation_accuracy, label="Validation")
    axes[1].set(title="Classification accuracy", ylabel="Accuracy", xlabel="Epoch")
    for ax in axes:
        ax.axvline(best_epoch, color="#7c4040", linestyle="--", label=f"Restored epoch {best_epoch}")
        ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.suptitle(f"Selected architecture {selected}, predeclared seed 11", fontsize=12)
    fig.savefig(FIG / "learning_curves.png"); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.2, 5), layout="constrained")
    im = ax.imshow(cm, cmap="Blues")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black")
    ax.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=LABELS,
           yticklabels=LABELS, xlabel="Predicted quality", ylabel="Actual quality",
           title="Confusion matrix on untouched test data")
    fig.colorbar(im, ax=ax, label="Records")
    fig.savefig(FIG / "confusion_matrix.png"); plt.close(fig)


def main():
    for folder in [DATA, OUT, FIG]: folder.mkdir(exist_ok=True, parents=True)
    csv = DATA / "winequality-red.csv"
    if not csv.exists():
        with urllib.request.urlopen(SOURCE, timeout=60) as response:
            z = zipfile.ZipFile(io.BytesIO(response.read()))
        csv.write_bytes(z.read("winequality-red.csv"))
    raw = pd.read_csv(csv, sep=";")
    assert raw.shape == (1599, 12), "Unexpected source data shape"
    assert not raw.isna().any().any(), "Missing values need explicit handling"
    assert all(pd.api.types.is_numeric_dtype(raw[c]) for c in raw.columns)
    df = raw.drop_duplicates().copy()
    df["source_row"] = df.index
    x = df.drop(columns=["quality", "source_row"])
    y = df.quality.astype(int)
    assert not x.duplicated().any(), "Repeated feature vectors need grouped splitting"
    train_val, test = train_test_split(np.arange(len(df)), test_size=.2,
                                      stratify=y, random_state=SPLIT_SEED)
    train, val = train_test_split(train_val, test_size=.25,
                                stratify=y.iloc[train_val], random_state=SPLIT_SEED)
    scaler = StandardScaler().fit(x.iloc[train])
    xt, xv, xe = [scaler.transform(x.iloc[idx]) for idx in [train, val, test]]
    yt, yv, ye = [y.iloc[idx].to_numpy() for idx in [train, val, test]]
    split_rows = []
    for label, idx in [("train", train), ("validation", val), ("test", test)]:
        split_rows.extend(dict(source_row=int(df.iloc[i].source_row), split=label,
                               quality=int(y.iloc[i])) for i in idx)
    pd.DataFrame(split_rows).to_csv(OUT / "split_manifest.csv", index=False)
    counts = y.value_counts().sort_index()
    split_counts = pd.DataFrame({"original": raw.quality.value_counts(), "unique": counts,
        "train": pd.Series(yt).value_counts(), "validation": pd.Series(yv).value_counts(),
        "test": pd.Series(ye).value_counts()}).sort_index().astype(int)
    split_counts.index.name = "quality"
    split_counts.to_csv(OUT / "class_counts.csv")
    audit = dict(source=SOURCE, sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
        original_shape=list(raw.shape), duplicates_removed=int(raw.duplicated().sum()),
        unique_records=len(df), missing_values=int(raw.isna().sum().sum()),
        split_sizes={"train": len(train), "validation": len(val), "test": len(test)},
        features=x.columns.tolist(), labels=LABELS.tolist(), split_seed=SPLIT_SEED,
        initialization_seeds=SEEDS, python=platform.python_version(),
        numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        matplotlib=matplotlib.__version__, joblib=joblib.__version__, platform=platform.platform())
    dump_json("data_audit.json", audit)
    audit_text = ("UCI RED WINE QUALITY DATA AUDIT\n" +
        f"Raw shape: {raw.shape}; missing values: {audit['missing_values']}\n" +
        f"Removed exact duplicates: {audit['duplicates_removed']}\n" +
        f"Unique records: {len(df)}; input features: {x.shape[1]}\n" +
        f"Train / validation / test: {len(train)} / {len(val)} / {len(test)}\n" +
        "Scaler fitted on training partition only.\n\n" + split_counts.to_string() + "\n")
    (OUT / "data_audit.txt").write_text(audit_text)
    print(audit_text, flush=True)
    rows, traces, seed11_models = [], {}, {}
    with threadpool_limits(limits=1):
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                model, row, trace = fit_run(arch, seed, xt, yt, xv, yv)
                rows.append(row); traces[row["architecture"] + "_" + str(seed)] = trace
                if seed == 11: seed11_models[row["architecture"]] = model
                print(f"{row['architecture']:9} seed={seed} best_epoch={row['best_epoch']:3} "
                      f"val_acc={row['validation_accuracy']:.4f} val_F1={row['validation_macro_f1']:.4f}", flush=True)
        runs = pd.DataFrame(rows)
        runs.to_csv(OUT / "all_runs.csv", index=False)
        means = ["train_accuracy", "validation_accuracy", "validation_macro_f1",
                 "validation_weighted_f1", "validation_balanced_accuracy", "best_epoch", "seconds"]
        summary = runs.groupby("architecture", sort=False)[means].agg(["mean", "std"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary = summary.reset_index()
        summary["parameters"] = summary.architecture.map(runs.groupby("architecture").parameters.first())
        summary.to_csv(OUT / "architecture_summary.csv", index=False)
        # Selection rule is fixed before any test predictions are produced.
        selected = summary.sort_values(["validation_macro_f1_mean", "parameters"],
                                       ascending=[False, True]).iloc[0].architecture
        final_model = seed11_models[selected]
        selected_row = runs[(runs.architecture == selected) & (runs.seed == 11)].iloc[0].to_dict()
        pred = final_model.predict(xe)
        baseline = DummyClassifier(strategy="most_frequent").fit(xt, yt)
        dummy_pred = baseline.predict(xe)
        cm = confusion_matrix(ye, pred, labels=LABELS)
        report = dict(selected_architecture=selected, selected_seed=11,
            selected_run=selected_row, test=metrics(ye, pred),
            baseline_test=metrics(ye, dummy_pred), test_count=len(ye),
            correct_test=int((ye == pred).sum()),
            classification_report=classification_report(ye, pred, labels=LABELS,
                                                       output_dict=True, zero_division=0))
        dump_json("final_metrics.json", report)
        dump_json("training_histories.json", traces)
        pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(OUT / "confusion_matrix.csv")
        pd.DataFrame(dict(source_row=df.iloc[test].source_row.to_numpy(), actual=ye,
                          predicted=pred)).to_csv(OUT / "test_predictions.csv", index=False)
        joblib.dump(dict(scaler=scaler, model=final_model, features=x.columns.tolist(),
                         audit=audit), OUT / "selected_model.joblib")
    summary_text = summary[["architecture", "parameters", "validation_accuracy_mean",
                           "validation_macro_f1_mean", "validation_macro_f1_std"]].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    (OUT / "architecture_output.txt").write_text(summary_text + "\n")
    final_text = (f"Selected architecture: {selected}\nFixed reporting seed: 11\n"
        f"Restored epoch: {selected_row['best_epoch']}\n"
        f"Test correct: {report['correct_test']} / {len(ye)}\n" +
        "\n".join(f"{k}: {v:.6f}" for k,v in report['test'].items()) +
        "\n\n" + classification_report(ye, pred, labels=LABELS, digits=4, zero_division=0) +
        f"\nMajority baseline accuracy: {report['baseline_test']['accuracy']:.6f}\n")
    (OUT / "evaluation_output.txt").write_text(final_text)
    print("\n" + summary_text + "\n\n" + final_text, flush=True)
    make_plots(summary, counts, traces, selected, cm, report)
    # A simple quality prediction for a saved test row, not an invented example.
    example = dict(source_row=int(df.iloc[test[0]].source_row),
                   features=x.iloc[test[0]].to_dict(), actual=int(ye[0]), predicted=int(pred[0]),
                   probabilities=final_model.predict_proba(xe[:1])[0].tolist())
    dump_json("example_prediction.json", example)


if __name__ == "__main__":
    main()
