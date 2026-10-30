"""
53_power_measure.py

WHAT THIS DOES
    Measures how much power the inference actually costs on a Jetson, by
    measuring the board three times:

        idle before    the board doing nothing            X
        running        the board doing the inference      Y
        idle after     the board doing nothing again      X2

    and reporting Y - X.

    The second idle reading is the point of the exercise. A board that has
    warmed up draws more than a cold one, so if X2 is well above X then part
    of what looks like the cost of inference is really the cost of the board
    getting hot. Measuring once would hide that.

WHERE THE READINGS COME FROM
    Two ways, and it uses whichever is available.

    jtop, from the jetson-stats package. A Python interface, gives each
    power rail separately and the total. Install with:

        sudo pip install jetson-stats
        sudo systemctl restart jtop.service
        (then log out and back in)

    tegrastats, which NVIDIA ships with the board. A command that prints a
    line of text every interval, with the rails in it. Nothing to install.

    If neither is there the script says so and stops, rather than reporting
    a number it did not measure.

WHAT IT REPORTS
    average watts in each phase, and the difference
    energy per window, in millijoules, which is the number that matters
        for a battery
    the power mode the board was in, because a reading without it means
        nothing: an Orin at 15 W and the same board at 7 W are different
        machines

HOW TO RUN
    On the Jetson, from the folder holding the demo:

        python3 53_power_measure.py \\
            --cmd "python3 52_inference_demo.py --sample sample_data_user25_2026-09-16" \\
            --idle-seconds 30

    Close everything else first. The script measures the whole board, so a
    browser open in the background goes into the reading.
"""

import os
import re
import sys
import json
import time
import shutil
import argparse
import subprocess
import threading

import numpy as np


# ---------------------------------------------------------------------------
# reading the power
# ---------------------------------------------------------------------------
class JtopSampler:
    """Power through the jetson-stats package.

    Gives each rail separately, so the breakdown can be reported rather than
    only a total.
    """

    name = "jtop (jetson-stats)"

    def __init__(self):
        from jtop import jtop
        self._jtop = jtop
        self._j = None

    def start(self):
        self._j = self._jtop()
        self._j.start()

    def read(self):
        """Total board milliwatts, and each rail."""
        if not self._j.ok():
            return None, {}
        p = self._j.power
        # jetson-stats changed shape between versions, so handle both
        if isinstance(p, dict) and "tot" in p:
            tot = p["tot"]
            total = tot.get("power", tot.get("cur", 0)) if isinstance(tot, dict) \
                else tot
            rails = {k: (v.get("power", v.get("cur", 0))
                         if isinstance(v, dict) else v)
                     for k, v in p.get("rail", {}).items()}
        else:
            total, rails = 0, {}
        return float(total), rails

    def stop(self):
        if self._j is not None:
            self._j.close()


class TegrastatsSampler:
    """Power by reading what tegrastats prints.

    Lines look like:
        RAM 3092/7850MB ... VDD_IN 4532mW/4210mW VDD_CPU_GPU_CV 1203mW/1150mW

    The first number of each pair is now, the second is a running average.
    We take the first.
    """

    name = "tegrastats"
    RAIL = re.compile(r"(VDD[_A-Z0-9]*|VIN[_A-Z0-9]*|POM[_A-Z0-9]*)\s+"
                      r"(\d+)mW/(\d+)mW")

    def __init__(self, interval_ms=200):
        self.interval = interval_ms
        self._p = None
        self._last = (None, {})
        self._stop = threading.Event()
        self._t = None

    def _pump(self):
        for line in self._p.stdout:
            if self._stop.is_set():
                break
            rails = {m.group(1): float(m.group(2))
                     for m in self.RAIL.finditer(line)}
            if rails:
                # prefer a rail that measures the whole board
                total = (rails.get("VDD_IN") or rails.get("VIN_SYS_5V0")
                         or rails.get("POM_5V_IN") or sum(rails.values()))
                self._last = (float(total), rails)

    def start(self):
        exe = shutil.which("tegrastats") or "/usr/bin/tegrastats"
        self._p = subprocess.Popen(
            [exe, "--interval", str(self.interval)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self._t = threading.Thread(target=self._pump, daemon=True)
        self._t.start()
        time.sleep(1.0)              # let the first line arrive

    def read(self):
        return self._last

    def stop(self):
        self._stop.set()
        if self._p:
            self._p.terminate()
            try:
                self._p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._p.kill()


def pick_sampler(prefer):
    """Use jtop if it is there, otherwise tegrastats."""
    if prefer in ("auto", "jtop"):
        try:
            import jtop                                  # noqa: F401
            return JtopSampler()
        except Exception as e:
            if prefer == "jtop":
                raise SystemExit("jtop was asked for but is not usable: %s\n"
                                 "  sudo pip install jetson-stats\n"
                                 "  sudo systemctl restart jtop.service\n"
                                 "  then log out and back in" % e)
    if prefer in ("auto", "tegrastats"):
        if shutil.which("tegrastats") or os.path.exists("/usr/bin/tegrastats"):
            return TegrastatsSampler()
    raise SystemExit(
        "No way to read the power on this machine.\n"
        "\n"
        "This needs a Jetson. Either:\n"
        "  sudo pip install jetson-stats      (then restart jtop.service)\n"
        "or use the tegrastats that NVIDIA ships with the board.\n"
        "\n"
        "On anything else there is no power rail to read, and a number\n"
        "made up here would be worse than no number.")


def board_info():
    """What the board is and which power mode it is in."""
    info = {}
    try:
        out = subprocess.run(["nvpmodel", "-q"], capture_output=True,
                             text=True, timeout=5).stdout
        m = re.search(r"NV Power Mode:\s*(\S+)", out)
        if m:
            info["power_mode"] = m.group(1)
        m = re.search(r"^(\d+)\s*$", out, re.M)
        if m:
            info["power_mode_id"] = m.group(1)
    except Exception:
        pass
    for path, key in (("/proc/device-tree/model", "board"),
                      ("/etc/nv_tegra_release", "jetpack")):
        try:
            info[key] = open(path).read().strip().strip("\x00").split("\n")[0]
        except Exception:
            pass
    return info


def sample_for(sampler, seconds, every, label):
    """Read the power repeatedly and keep every reading."""
    vals, rails_seen = [], {}
    end = time.time() + seconds
    while time.time() < end:
        total, rails = sampler.read()
        if total:
            vals.append(total)
            for k, v in rails.items():
                rails_seen.setdefault(k, []).append(v)
        time.sleep(every)
        done = seconds - (end - time.time())
        print("\r  %-14s %4.0f s  %6.0f mW" % (label, done,
                                               vals[-1] if vals else 0),
              end="", flush=True)
    print()
    return vals, rails_seen


def summarise(vals):
    if not vals:
        return {"n": 0, "mean_mw": float("nan"), "sd_mw": float("nan"),
                "min_mw": float("nan"), "max_mw": float("nan")}
    a = np.array(vals, dtype=float)
    return {"n": len(a), "mean_mw": float(a.mean()), "sd_mw": float(a.std()),
            "min_mw": float(a.min()), "max_mw": float(a.max())}


def main():
    ap = argparse.ArgumentParser(
        description="Measure what the inference costs in power, by reading "
                    "the board idle, then busy, then idle again.")
    ap.add_argument("--cmd", required=True,
                    help="the inference command to run and measure")
    ap.add_argument("--idle-seconds", type=float, default=30.0,
                    help="how long to watch the idle board. 30 is enough to "
                         "average out the noise; shorter is unreliable.")
    ap.add_argument("--settle-seconds", type=float, default=5.0,
                    help="pause before each idle reading, so the previous "
                         "phase has finished affecting things")
    ap.add_argument("--sample-every", type=float, default=0.2)
    ap.add_argument("--windows", type=int, default=None,
                    help="how many windows the command scores. Given this, "
                         "the energy per window is reported, which is the "
                         "figure that matters for a battery.")
    ap.add_argument("--source", choices=["auto", "jtop", "tegrastats"],
                    default="auto")
    ap.add_argument("--voltage", type=float, default=5.0,
                    help="supply voltage, for turning energy into mAh. "
                         "Charge depends on it, so the figure is meaningless "
                         "without it. The Orin Nano dev kit takes 5 V over "
                         "USB-C or 19 V from the barrel jack; VDD_IN is the "
                         "module rail, usually 5 V. Set it to what your "
                         "board actually uses.")
    ap.add_argument("--out", default="power_measurement.json")
    args = ap.parse_args()

    print("=" * 74)
    print("Power cost of the inference")
    print("=" * 74)

    info = board_info()
    for k, v in info.items():
        print("  %-14s %s" % (k, v))
    if "power_mode" not in info:
        print("  power mode could not be read. Record it by hand with")
        print("  nvpmodel -q: the same board at 15 W and at 7 W gives")
        print("  completely different numbers.")
    print()

    sampler = pick_sampler(args.source)
    print("reading from : %s" % sampler.name)
    print("command      : %s" % args.cmd)
    print()
    print("Close everything else. This measures the whole board, so a")
    print("browser left open goes into the reading.")
    print()

    sampler.start()
    try:
        # ---- idle, before ----
        print("1. the board doing nothing")
        time.sleep(args.settle_seconds)
        idle1, rails1 = sample_for(sampler, args.idle_seconds,
                                   args.sample_every, "idle before")
        x = summarise(idle1)
        print("   %.0f mW, give or take %.0f" % (x["mean_mw"], x["sd_mw"]))
        print()

        # ---- running ----
        print("2. running the inference")
        vals, rails2 = [], {}
        proc = subprocess.Popen(args.cmd, shell=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        t0 = time.time()
        while proc.poll() is None:
            total, rails = sampler.read()
            if total:
                vals.append(total)
                for k, v in rails.items():
                    rails2.setdefault(k, []).append(v)
            time.sleep(args.sample_every)
            print("\r  %-14s %4.0f s  %6.0f mW"
                  % ("running", time.time() - t0, vals[-1] if vals else 0),
                  end="", flush=True)
        print()
        elapsed = time.time() - t0
        out = proc.stdout.read()
        y = summarise(vals)
        print("   %.0f mW, give or take %.0f, over %.1f s"
              % (y["mean_mw"], y["sd_mw"], elapsed))
        if proc.returncode != 0:
            print()
            print("   the command exited with %d. Its last lines:"
                  % proc.returncode)
            for line in out.strip().split("\n")[-6:]:
                print("     %s" % line)
            print()
            print("   The power figures below describe a failed run, so they")
            print("   are not usable.")
        print()

        # ---- idle, after ----
        print("3. the board doing nothing again")
        time.sleep(args.settle_seconds)
        idle2, _ = sample_for(sampler, args.idle_seconds, args.sample_every,
                              "idle after")
        x2 = summarise(idle2)
        print("   %.0f mW, give or take %.0f" % (x2["mean_mw"], x2["sd_mw"]))
    finally:
        sampler.stop()

    # ---- what it cost ----
    drift = x2["mean_mw"] - x["mean_mw"]

    # Which idle reading to subtract.
    #
    # Both are meant to be the same board doing nothing, so the one that
    # held steadier is the better measurement. A reading that swung wildly
    # had something else running in it, and subtracting that would hide
    # part of the cost rather than reveal it.
    if x["sd_mw"] > 2.5 * max(x2["sd_mw"], 1.0):
        base, base_name = x2, "idle after"
        baseline_note = (
            "The first idle reading varied by %.0f mW while the second "
            "varied by only %.0f, so something was still running during "
            "the first. The second is used as the baseline."
            % (x["sd_mw"], x2["sd_mw"]))
    elif x2["sd_mw"] > 2.5 * max(x["sd_mw"], 1.0):
        base, base_name = x, "idle before"
        baseline_note = (
            "The second idle reading varied by %.0f mW while the first "
            "varied by only %.0f, so the first is used."
            % (x2["sd_mw"], x["sd_mw"]))
    else:
        both = (x["mean_mw"] + x2["mean_mw"]) / 2.0
        base = {"mean_mw": both, "sd_mw": max(x["sd_mw"], x2["sd_mw"])}
        base_name = "the two idle readings averaged"
        baseline_note = ("Both idle readings held about as steady, so their "
                         "average is the baseline.")

    cost = y["mean_mw"] - base["mean_mw"]

    print()
    print("=" * 74)
    print("WHAT THE INFERENCE COST")
    print("=" * 74)
    print("  %-22s %10.0f mW" % ("idle before  (X)", x["mean_mw"]))
    print("  %-22s %10.0f mW" % ("running      (Y)", y["mean_mw"]))
    print("  %-22s %10.0f mW   (varied by %.0f)"
          % ("idle after   (X2)", x2["mean_mw"], x2["sd_mw"]))
    print("  " + "-" * 50)
    print("  %-22s %10.0f mW   = %.2f W"
          % ("the inference", cost, cost / 1000.0))
    print("  baseline used: %s" % base_name)
    print()

    print("  how much to trust that:")
    print("    %s" % baseline_note)
    print("    the baseline itself varied by %.0f mW" % base["sd_mw"])
    if abs(drift) > abs(cost) * 0.25 and abs(cost) > 1 and \
            abs(x["sd_mw"] - x2["sd_mw"]) < 100:
        print("    the two idle readings differ by %.0f mW even though both"
              % abs(drift))
        print("    held steady, which means the board genuinely changed")
        print("    temperature. Let it cool and measure again.")
    print()

    per_window_s = energy_mj = energy_mah = avg_mw = None
    if args.windows and args.windows > 0 and elapsed > 0:
        per_window_s = elapsed / args.windows
        energy_mj = cost / 1000.0 * per_window_s * 1000.0

        # Energy to charge. A milliamp-hour is 3.6 joules at one volt, so
        # charge is energy divided by the voltage and by 3.6. That means
        # mAh depends on the supply voltage and is not a property of the
        # work alone, which is why the voltage is printed beside it.
        energy_mah = energy_mj / 1000.0 / args.voltage / 3.6

        print("  per window")
        print("    %d windows in %.1f s, so %.3f s each"
              % (args.windows, elapsed, per_window_s))
        print("    %.2f mJ of extra energy each" % energy_mj)
        print("    %.5f mAh at %.1f V" % (energy_mah, args.voltage))
        print()
        # a wearable scoring one window every 30 s
        duty = min(per_window_s / 30.0, 1.0)
        avg_mw = cost * duty
        avg_mah_per_hour = avg_mw / args.voltage        # mA drawn on average
        print("    Scoring one window every 30 seconds, as a worn device")
        print("    would, the inference is busy %.1f%% of the time and adds"
              % (100 * duty))
        print("    about %.0f mW on average, which is %.1f mA at %.1f V."
              % (avg_mw, avg_mah_per_hour, args.voltage))
        print()
        print("    over a day of that: %.1f mAh for the inference"
              % (avg_mah_per_hour * 24))
        idle_ma = base["mean_mw"] / args.voltage
        print("    over the same day:  %.0f mAh just for the board being on"
              % (idle_ma * 24))
        if idle_ma > 0:
            print()
            print("    So the inference is %.1f%% of what the board uses."
                  % (100 * avg_mw / base["mean_mw"]))
            print("    On this hardware the idle draw is what matters, not")
            print("    the model.")
        print()

    # ---- which rails moved ----
    if rails2 and rails1:
        print("  which rails changed")
        print("    %-20s %10s %10s %10s" % ("rail", "idle", "running",
                                            "change"))
        for k in sorted(set(rails1) | set(rails2)):
            a = float(np.mean(rails1[k])) if rails1.get(k) else 0.0
            b = float(np.mean(rails2[k])) if rails2.get(k) else 0.0
            print("    %-20s %9.0f %10.0f %+10.0f" % (k, a, b, b - a))
        print()

    json.dump({"board": info, "source": sampler.name, "command": args.cmd,
               "idle_before": x, "running": y, "idle_after": x2,
               "inference_mw": cost, "idle_drift_mw": drift,
               "baseline_used": base_name,
               "voltage_v": args.voltage,
               "elapsed_s": elapsed, "windows": args.windows,
               "per_window_s": per_window_s,
               "energy_per_window_mj": energy_mj,
               "energy_per_window_mah": energy_mah,
               "duty_cycle_mw_at_30s": avg_mw,
               "rails_idle": {k: float(np.mean(v))
                              for k, v in rails1.items()},
               "rails_running": {k: float(np.mean(v))
                                 for k, v in rails2.items()}},
              open(args.out, "w"), indent=2)
    print("saved %s" % args.out)


if __name__ == "__main__":
    main()
