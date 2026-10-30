"""
52_inference_demo.py

WHAT THIS DOES
    Runs a packed sample folder the way a device would: read sixty seconds,
    predict, report, move on. And for each window it also says how much each
    context column changed the answer.

    Point it at a folder made by btp/51_make_sample.py. Nothing else from
    the project is needed, because that folder carries the model, the
    encoder and the scaler with its paths already rewritten.

HOW THE CONTRIBUTION IS MEASURED
    Not by reading weights off the gate. The gate gives one weight per
    embedding dimension, not per context column, so there is no weight to
    read for "posture".

    Instead, each column is turned off and the change is measured:

        1. score the window with the real context
        2. replace ONE column with its ordinary value for that day -
           the mean for a number, the commonest value for a word -
           leaving everything else alone
        3. score it again
        4. the difference is what that column was worth

    This is an ablation, and it answers the question actually being asked:
    if this column had said nothing unusual, how different would the answer
    have been?

    A column that never varies will show zero, correctly. On the sample day
    IS_WEEKEND and INDOOR_OUTDOOR are constant, so expect nothing from them.

WHAT THE SCORE IS
        stress_score = 1 x P(Medium) + 2 x P(High)        0 to 2

    Ours, not the paper's. AdaStress gives three probabilities and stops.

BEFORE SHOWING THIS TO ANYONE
    The model packed into the sample was trained with a pooled random split
    and scores 0.890 balanced accuracy. That number is inflated and the
    reason is known: windows overlap by 30 seconds and each survey answer
    labels about 60 of them, so all 100 answers had windows on both sides of
    the split. The model recognises moments it was trained on.

    The honest figures, same data, same code:

        survey answers kept whole      0.380
        a person never seen before     0.286
        chance                         0.333

    So the contributions below describe what THIS model does, which is a
    fair thing to demonstrate. They are not evidence that context predicts
    stress. Measured properly, it does not: with these eight columns the
    honest score was 0.398 against 0.428 with no context at all.

HOW TO RUN
    python3 btp/52_inference_demo.py --sample sample_data/user25_2026-09-16

    slowed down, with the context breakdown printed each time:

    python3 btp/52_inference_demo.py --sample sample_data/user25_2026-09-16 \\
        --sleep 2 --limit 20 --explain-every 1
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import torch
import joblib

# psutil gives processor use and memory. It is not essential: without it
# the script still runs and simply says so rather than reporting nothing.
try:
    import psutil
    HAVE_PSUTIL = True
except ImportError:
    HAVE_PSUTIL = False

VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "vendor", "adastress")
sys.path.insert(0, VENDOR)

from adastress.models import (ContextGateFusion, MLPClassifier,     # noqa: E402
                              StressModel, load_encoder)

LABEL_NAMES = ["Low", "Medium", "High"]
TIME_COL = "datetime_ist"
DEVICE = torch.device("cpu")


def load_bundle(folder):
    """Rebuild the model from the folder, which carries everything."""
    path = os.path.join(folder, "model.pth")
    if not os.path.exists(path):
        raise SystemExit("no model.pth in %s" % folder)

    b = torch.load(path, map_location="cpu", weights_only=False)

    # the packer rewrote these to bare filenames inside the folder
    enc = os.path.join(folder, os.path.basename(b["encoder_ckpt"]))
    scl = os.path.join(folder, os.path.basename(b["scaler_ckpt"]))
    for p in (enc, scl):
        if not os.path.exists(p):
            raise SystemExit("%s is missing from the sample folder" % p)

    encoder = load_encoder(enc, b["input_channels"], b["lstm_hidden"], DEVICE)

    ctx_dim = int(b.get("context_dim", 0))
    gate = None
    if ctx_dim > 0 and b.get("gate"):
        gate = ContextGateFusion(encoder.output_dim, ctx_dim).to(DEVICE)
        gate.load_state_dict(b["gate"])

    clf = MLPClassifier(encoder.output_dim, len(b["classes"])).to(DEVICE)
    clf.load_state_dict(b["classifier"])

    model = StressModel(encoder, clf, gate).to(DEVICE)
    model.eval()
    return model, joblib.load(scl), b


def rss_mb():
    """How much memory this process is holding, in megabytes."""
    if not HAVE_PSUTIL:
        return float("nan")
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def model_footprint(model, folder):
    """What the model costs to carry and to hold in memory.

    Three different numbers, and they are not the same thing:

      on disk      what has to be copied onto the device
      parameters   the weights themselves, counted and sized
      in memory    how much the process grew when the model was loaded,
                   which includes everything torch allocates around it
    """
    disk = 0
    for f in ("model.pth", "encoder.pth", "scaler.pkl"):
        p = os.path.join(folder, f)
        if os.path.exists(p):
            disk += os.path.getsize(p)

    n_params = sum(p.numel() for p in model.parameters())
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_buf = sum(b.numel() * b.element_size() for b in model.buffers())

    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    return {"disk_mb": disk / 1e6,
            "params": n_params,
            "params_frozen": frozen,
            "weights_mb": (n_bytes + n_buf) / 1e6}


def encode_context(df, fitted):
    """Turn context columns into the numbers the gate was trained on.

    Uses the transformers saved at training time. Fitting new ones here
    would produce different numbers for the same words, and the gate would
    be reading something it has never seen.
    """
    num = fitted.get("numeric_cols") or []
    cat = fitted.get("categorical_cols") or []
    pieces = []
    if num:
        pieces.append(fitted["scaler"].transform(
            df[num].apply(pd.to_numeric, errors="coerce").fillna(0.0)))
    if cat:
        pieces.append(fitted["onehot"].transform(df[cat].astype(str)))
    return np.concatenate(pieces, axis=1).astype(np.float32)


def ordinary_values(df, cols):
    """What each column usually says, for turning it off."""
    out = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        if v.notna().all():
            out[c] = v.mean()
        else:
            out[c] = df[c].astype(str).mode().iloc[0]
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Run a packed sample and show what each context column "
                    "was worth.")
    ap.add_argument("--sample", required=True,
                    help="a folder from btp/51_make_sample.py")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds between windows. 2 makes it readable.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--explain-every", type=int, default=0,
                    help="print the per-column breakdown every N windows. "
                         "0 prints only the summary at the end.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    print("=" * 74)
    print("Stress inference, with the contribution of each context column")
    print("=" * 74)

    model, scaler, b = load_bundle(args.sample)
    cols = list(b["signal_columns"])
    ctx_cols = list(b.get("context_columns", []))
    fitted = b.get("context_fitted")
    ctx_dim = int(b.get("context_dim", 0))
    window = int(b["window"])
    stride = int(b.get("stride", window // 2))

    print("sample   : %s" % args.sample)
    fp = model_footprint(model, args.sample)
    print("footprint: %.2f MB on disk, %.2f MB of weights, %s parameters"
          % (fp["disk_mb"], fp["weights_mb"], format(fp["params"], ",")))
    if fp["params_frozen"]:
        print("           %s of them frozen (the encoder)"
              % format(fp["params_frozen"], ","))
    print("channels : %d  %s" % (len(cols), cols))
    print("context  : %d columns -> %d values" % (len(ctx_cols), ctx_dim))
    print("window   : %d s, stride %d s" % (window, stride))
    if ctx_dim > 0 and fitted is None:
        raise SystemExit("this model uses context but the fitted encoders "
                         "were not saved with it. Retrain with --save-model "
                         "using the current btp/34_train_users.py.")
    print()

    # ---------------- signals ----------------
    sig = pd.read_csv(os.path.join(args.sample, "signals.csv"))
    sig[TIME_COL] = pd.to_datetime(sig[TIME_COL], errors="coerce")
    sig = sig.dropna(subset=[TIME_COL]).sort_values(TIME_COL)
    sig = sig.reset_index(drop=True)

    arr = sig[cols].to_numpy(dtype=float)
    arr = pd.DataFrame(arr).interpolate(limit_direction="both") \
            .fillna(0.0).to_numpy()
    tidx = sig[TIME_COL]

    starts = list(range(0, len(arr) - window + 1, stride))
    X = np.stack([arr[s:s + window].T for s in starts]).astype(np.float32)
    stamps = [tidx.iloc[s] for s in starts]

    flat = X.transpose(0, 2, 1).reshape(-1, X.shape[1])
    X = scaler.transform(flat).reshape(X.shape[0], X.shape[2],
                                       X.shape[1]).transpose(0, 2, 1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0,
                      neginf=0.0).astype(np.float32)

    print("windows  : %d  (%.1f hours)" % (len(X), len(X) * stride / 3600.0))
    print("scaled   : mean %.2f, sd %.2f, range %.1f to %.1f"
          % (X.mean(), X.std(), X.min(), X.max()))

    # ---------------- context ----------------
    ctx_path = os.path.join(args.sample, "context.csv")
    have_ctx = ctx_dim > 0 and os.path.exists(ctx_path)
    C = None

    if have_ctx:
        cdf = pd.read_csv(ctx_path)
        cdf[TIME_COL] = pd.to_datetime(cdf[TIME_COL], errors="coerce")
        cdf = cdf.dropna(subset=[TIME_COL]).sort_values(TIME_COL)

        win = pd.DataFrame({TIME_COL: stamps}).sort_values(TIME_COL)
        C = pd.merge_asof(win, cdf, on=TIME_COL, direction="nearest",
                          tolerance=pd.Timedelta(seconds=stride))
        n_ctx = int(C[ctx_cols].notna().any(axis=1).sum())
        print("context  : %d of %d windows have it" % (n_ctx, len(C)))

        # windows with no context get the ordinary value, so the model is
        # not fed blanks it never saw
        usual = ordinary_values(cdf, ctx_cols)
        for c in ctx_cols:
            C[c] = C[c].fillna(usual[c])
        Cenc = encode_context(C, fitted)
    else:
        print("context  : none, so no contributions to report")
        Cenc = np.zeros((len(X), 1), dtype=np.float32)
        usual = {}

    # ---------------- truth, if it came along ----------------
    truth = None
    tpath = os.path.join(args.sample, "truth.csv")
    if os.path.exists(tpath):
        truth = pd.read_csv(tpath)
        truth[TIME_COL] = pd.to_datetime(truth[TIME_COL], errors="coerce")
        print("truth    : %d labelled windows, %d survey answers"
              % (len(truth), truth["response_time"].nunique()))
    if not HAVE_PSUTIL:
        print()
        print("psutil is not installed, so processor use and memory cannot")
        print("be measured. pip install psutil to get those columns.")
    print()

    # the first reading from psutil is meaningless, so take one and drop it
    if HAVE_PSUTIL:
        psutil.cpu_percent(interval=None)
    mem_before = rss_mb()

    print("-" * 86)
    print("  %5s  %8s  %-6s %7s %8s %8s %7s %8s"
          % ("win", "time", "state", "score", "uncert", "ms", "cpu %", "MB"))
    print("-" * 86)

    def score(xb, cb):
        with torch.no_grad():
            p = torch.softmax(model(xb, cb), dim=1).numpy()[0]
        return p, float(p[1] + 2 * p[2])

    n = len(X) if args.limit is None else min(args.limit, len(X))
    rows, contrib = [], {c: [] for c in ctx_cols}
    last, lat, cpus, mems = None, [], [], []

    for i in range(n):
        xb = torch.tensor(X[i:i + 1])
        cb = torch.tensor(Cenc[i:i + 1]) if have_ctx else None

        t0 = time.perf_counter()
        p, s = score(xb, cb)
        ms = (time.perf_counter() - t0) * 1000.0
        lat.append(ms)
        cpu = psutil.cpu_percent(interval=None) if HAVE_PSUTIL else float("nan")
        mem = rss_mb()
        cpus.append(cpu)
        mems.append(mem)

        pred = int(p.argmax())
        when = stamps[i].strftime("%H:%M:%S")

        # turn each column off in turn and see how far the score moves
        deltas = {}
        if have_ctx:
            for c in ctx_cols:
                row = C.iloc[[i]].copy()
                row[c] = usual[c]
                _, s2 = score(xb, torch.tensor(encode_context(row, fitted)))
                deltas[c] = s - s2
                contrib[c].append(abs(deltas[c]))

        rec = {"window": i, "time": when,
               "P_Low": round(float(p[0]), 4),
               "P_Medium": round(float(p[1]), 4),
               "P_High": round(float(p[2]), 4),
               "predicted": LABEL_NAMES[pred],
               "stress_score": round(s, 3),
               "uncertainty": round(float(1 - p.max()), 3),
               "latency_ms": round(ms, 3),
               "cpu_percent": round(cpu, 1),
               "memory_mb": round(mem, 1)}
        for c, v in deltas.items():
            rec["d_" + c] = round(v, 4)
        rows.append(rec)

        show = (pred != last) or (args.explain_every
                                  and i % args.explain_every == 0)
        if show:
            print("  %5d  %8s  %-6s %7.2f %8.2f %8.1f %7.0f %8.0f%s"
                  % (i, when, LABEL_NAMES[pred], s, 1 - p.max(), ms,
                     cpu, mem, "" if pred == last else "  <-"))
            if deltas and args.explain_every:
                top = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:4]
                for c, v in top:
                    if abs(v) < 0.001:
                        continue
                    bar = "#" * min(int(abs(v) * 40), 30)
                    print("          %-16s %+6.3f  %s"
                          % (c, v, bar))
            last = pred

        if args.sleep > 0:
            time.sleep(args.sleep)

    print("-" * 86)
    print()

    r = pd.DataFrame(rows)
    mem_after = rss_mb()

    print("=" * 74)
    print("WHAT IT COSTS TO RUN")
    print("=" * 74)
    print("  windows scored    %d, each covering %d seconds of signal"
          % (len(r), window))
    print()
    print("  latency per window")
    print("    average         %.2f ms" % np.mean(lat))
    print("    median          %.2f ms" % np.median(lat))
    print("    slowest         %.2f ms   (the first one, before anything is "
          "warm)" % np.max(lat))
    # the first window pays for warm-up, so quote the rest separately
    if len(lat) > 1:
        print("    slowest after the first   %.2f ms" % np.max(lat[1:]))
    print()
    print("    A window covers %d seconds and takes %.1f ms to score, so one"
          % (window, np.mean(lat)))
    print("    processor could keep up with about %d devices at once."
          % int(window * 1000 / max(np.mean(lat), 0.001)))
    print()

    if HAVE_PSUTIL:
        c = [x for x in cpus if not np.isnan(x)]
        print("  processor use")
        print("    average         %.1f %%" % np.mean(c))
        print("    highest         %.1f %%" % np.max(c))
        print("    (per core, across the whole machine, so other work counts)")
        print()
        print("  memory")
        print("    before loading  %.1f MB" % mem_before)
        print("    while running   %.1f MB" % np.mean(mems))
        print("    highest         %.1f MB" % np.max(mems))
        print("    the model itself %.2f MB of weights, %.2f MB on disk"
              % (fp["weights_mb"], fp["disk_mb"]))
        print()
        print("    What a device needs: %.0f MB of memory to run it and"
              % np.max(mems))
        print("    %.1f MB of storage to hold it." % fp["disk_mb"])
    else:
        print("  processor use and memory were not measured.")
        print("  pip install psutil, then run again.")
    print()

    print("  predictions       %s" % dict(r["predicted"].value_counts()))
    print("stress score   : mean %.2f, median %.2f, range %.2f to %.2f"
          % (r["stress_score"].mean(), r["stress_score"].median(),
             r["stress_score"].min(), r["stress_score"].max()))
    print()

    if have_ctx:
        print("=" * 74)
        print("WHAT EACH CONTEXT COLUMN WAS WORTH")
        print("=" * 74)
        print("How far the score moved, on average, when that column alone")
        print("was replaced by its ordinary value for the day. The score")
        print("runs 0 to 2, so 0.10 is five percent of the range.")
        print()
        summary = sorted(((c, float(np.mean(v)) if v else 0.0,
                           float(np.max(v)) if v else 0.0)
                          for c, v in contrib.items()),
                         key=lambda t: -t[1])
        print("  %-18s %9s %9s  %s" % ("column", "average", "largest", ""))
        print("  " + "-" * 60)
        for c, mean_d, max_d in summary:
            bar = "#" * min(int(mean_d * 100), 32)
            print("  %-18s %9.4f %9.4f  %s" % (c, mean_d, max_d, bar))
        print()

        flat_cols = [c for c, m, _ in summary if m < 0.001]
        if flat_cols:
            print("  %s moved nothing. Check the sample: a column that never"
                  % ", ".join(flat_cols))
            print("  changes during the day cannot change the answer, and")
            print("  that is the right result rather than a fault.")
            print()

        total = sum(m for _, m, _ in summary)
        if total < 0.01:
            print("  Every column is near zero. For this model and this day,")
            print("  the answer comes from the body signals and the context")
            print("  is barely being used.")
            print()

    if truth is not None and len(truth):
        print("=" * 74)
        print("AGAINST WHAT THE PERSON REPORTED")
        print("=" * 74)
        m = pd.merge_asof(
            r.assign(t=[stamps[i] for i in r["window"]]).sort_values("t"),
            truth[[TIME_COL, "LABEL_NAME", "STAI6"]].sort_values(TIME_COL),
            left_on="t", right_on=TIME_COL, direction="nearest",
            tolerance=pd.Timedelta(seconds=stride)).dropna(
                subset=["LABEL_NAME"])
        if len(m):
            short = {"LowStress": "Low", "MediumStress": "Medium",
                     "HighStress": "High"}
            m["true"] = m["LABEL_NAME"].map(short)
            agree = float((m["predicted"] == m["true"]).mean())
            print("  %d windows had a survey answer nearby" % len(m))
            print("  the prediction matched it %.0f%% of the time"
                  % (100 * agree))
            print()
            print(pd.crosstab(m["true"], m["predicted"]).to_string())
            print()
            print("  One day of one person. This shows the pipeline working,")
            print("  not how well the model works. Across five people, held")
            print("  out one at a time, balanced accuracy was 0.286 against")
            print("  a chance level of 0.333.")
            print()

    if args.out:
        r.to_csv(args.out, index=False)
        print("saved %s" % args.out)

        # the cost figures on their own, for a report
        cost = os.path.splitext(args.out)[0] + "_cost.csv"
        pd.DataFrame([{
            "sample": args.sample,
            "windows": len(r),
            "window_seconds": window,
            "channels": len(cols),
            "context_columns": len(ctx_cols),
            "context_values": ctx_dim,
            "latency_mean_ms": round(float(np.mean(lat)), 3),
            "latency_median_ms": round(float(np.median(lat)), 3),
            "latency_max_ms": round(float(np.max(lat)), 3),
            "latency_max_after_first_ms": (round(float(np.max(lat[1:])), 3)
                                           if len(lat) > 1 else None),
            "cpu_mean_percent": (round(float(np.nanmean(cpus)), 1)
                                 if HAVE_PSUTIL else None),
            "cpu_max_percent": (round(float(np.nanmax(cpus)), 1)
                                if HAVE_PSUTIL else None),
            "memory_mean_mb": (round(float(np.nanmean(mems)), 1)
                               if HAVE_PSUTIL else None),
            "memory_max_mb": (round(float(np.nanmax(mems)), 1)
                              if HAVE_PSUTIL else None),
            "model_weights_mb": round(fp["weights_mb"], 3),
            "model_disk_mb": round(fp["disk_mb"], 3),
            "parameters": fp["params"],
        }]).to_csv(cost, index=False)
        print("saved %s  (latency, processor and memory, for a report)"
              % cost)


if __name__ == "__main__":
    main()
