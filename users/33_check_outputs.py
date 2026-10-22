"""
33_check_outputs.py

WHY THIS EXISTS
    We deleted the raw files for user18, 25, 26 and 27 after deriving their
    context, without checking that the 1 Hz signal channels had also been
    built. They had not. AdaStress needs those channels, because the frozen
    encoder reads the signal, so the four users have to be downloaded again.

    This script is the check that should have run before any deletion. It
    lists, per user, which outputs exist and which do not, and says plainly
    whether the raw folder is safe to remove.

WHAT EACH OUTPUT IS FOR

    data/processed/user_6ch/<user>.csv
        The six channels WESAD and EmbracePlus both record, averaged to 1 Hz.
        This is what the frozen encoder reads. Without it there is no
        AdaStress pipeline, only context.

    data/processed/context/<user>_context.csv
        The derived context at 30 seconds: PR, RESP, activity, EDA, posture,
        skin temperature, clock. This is what the gate reads.

    data/processed/chronos_tokens/<user>_*.npz
        Optional. Only needed for the Chronos line of work, which is separate
        from AdaStress.

HOW TO RUN
    From the AdaStress_BTP folder:
        python3 btp/33_check_outputs.py

    Run it before deleting any raw folder.
"""

import os
import glob
import argparse

RAW_DIR = "data/raw/embraceplus"
SIGNAL_DIR = "data/processed/user_6ch"
CONTEXT_DIR = "data/processed/context"
TOKEN_DIR = "data/processed/chronos_tokens"

# The users with survey answers. Only these can be trained or tested on.
LABELLED_USERS = ["user17", "user18", "user25", "user26", "user27"]


def folder_size_gb(path):
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total / 1e9


def main():
    ap = argparse.ArgumentParser(
        description="Check every user has the outputs needed before any raw "
                    "folder is deleted.")
    ap.add_argument("--users", default=",".join(LABELLED_USERS))
    args = ap.parse_args()

    users = [u.strip() for u in args.users.split(",") if u.strip()]

    print("=" * 74)
    print("OUTPUT CHECK")
    print("=" * 74)
    print("Run this before deleting any raw folder.")
    print()
    print("  signals  the 6 channels at 1 Hz, read by the frozen encoder")
    print("  context  the derived context at 30 s, read by the gate")
    print("  tokens   Chronos tokens, optional, separate line of work")
    print()
    print("  %-9s %9s %9s %9s %9s   %s"
          % ("user", "raw GB", "signals", "context", "tokens", "verdict"))
    print("  " + "-" * 68)

    rows = []
    for u in users:
        raw = os.path.join(RAW_DIR, u)
        raw_gb = folder_size_gb(raw)
        has_raw = os.path.isdir(raw) and raw_gb > 0.001

        sig = os.path.exists(os.path.join(SIGNAL_DIR, "%s.csv" % u))
        ctx = os.path.exists(os.path.join(CONTEXT_DIR, "%s_context.csv" % u))
        tok = len(glob.glob(os.path.join(TOKEN_DIR, "%s_*.npz" % u))) > 0

        if sig and ctx:
            verdict = "safe to delete raw" if has_raw else "complete"
        elif has_raw:
            missing = [n for n, ok in (("signals", sig), ("context", ctx))
                       if not ok]
            verdict = "DO NOT DELETE, missing %s" % " and ".join(missing)
        else:
            missing = [n for n, ok in (("signals", sig), ("context", ctx))
                       if not ok]
            verdict = "RAW GONE and missing %s, re-download" % \
                      " and ".join(missing)

        print("  %-9s %9.1f %9s %9s %9s   %s"
              % (u, raw_gb,
                 "yes" if sig else "NO",
                 "yes" if ctx else "NO",
                 "yes" if tok else "-",
                 verdict))
        rows.append({"user": u, "raw_gb": raw_gb, "raw": has_raw,
                     "signals": sig, "context": ctx, "tokens": tok})

    print()

    need_dl = [r["user"] for r in rows
               if not r["raw"] and not (r["signals"] and r["context"])]
    hold = [r["user"] for r in rows
            if r["raw"] and not (r["signals"] and r["context"])]
    ready = [r["user"] for r in rows if r["signals"] and r["context"]]

    print("=" * 74)
    print("WHAT TO DO")
    print("=" * 74)
    if ready:
        print("  ready for training : %s" % ", ".join(ready))
    if hold:
        print("  raw still present, finish processing before deleting:")
        print("     %s" % ", ".join(hold))
        for u in hold:
            r = next(x for x in rows if x["user"] == u)
            if not r["signals"]:
                print("     python3 btp/25_build_user_6ch.py    # then check again")
            if not r["context"]:
                print("     python3 btp/30_derive_context.py --user %s "
                      "--drop-redundant" % u)
    if need_dl:
        print("  must be downloaded again: %s" % ", ".join(need_dl))
        print("     use btp/24b_download_remaining_users.ipynb")
        print("     then, for each one, in this order:")
        print("       python3 btp/25_build_user_6ch.py")
        print("       python3 btp/33_check_outputs.py      # confirm first")
        print("       rm -rf data/raw/embraceplus/<user>   # only then")
    print()

    n_ok = len(ready)
    print("%d of %d labelled users are ready." % (n_ok, len(users)))
    if n_ok < 2:
        print("Leave-one-user-out needs at least two, and really all five.")
    print()


if __name__ == "__main__":
    main()
