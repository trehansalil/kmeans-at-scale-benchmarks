"""Single-job guard for the shared Mac mini (stdlib only, runs ON the Mac).

Runs ONE command at a time, never touches any process it did not start.
Kills only its own process group when any of these trips:
  - wall-clock cap (--max-min)
  - job RSS above --max-rss-gb (sum over its process group)
  - system free memory below --kill-mem % for --kill-samples samples in a row
  - swap growth above --kill-swap-mb since start while free mem is low
  - sampler stall (a metric command hangs)
  - no progress: the heartbeat file (--beat) not touched for --stall-min minutes
Writes one line per event to guard.log and a final line
    GUARD_END name=<name> status=<done|failed|killed> rc=<rc> reason=<...> elapsed_s=<s>
so a Monitor can match on GUARD_END / GUARD_KILL. A done-marker (--done) is written on rc 0 only.
Usage: python guard.py --name X --max-min 20 --max-rss-gb 12 --beat run/X.beat --done run/X.done -- cmd args...
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time


def sh(cmd, timeout=12):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except subprocess.TimeoutExpired:
        return None


def group_rss_gb(pgid):
    out = sh(f"ps -o rss= -g {pgid}")
    if out is None:
        return None
    return sum(int(x) for x in out.split() if x.isdigit()) / 1048576.0


def mem_free_pct():
    out = sh("memory_pressure | grep 'free percentage'")
    if out is None:
        return None
    r = re.search(r"(\d+)%", out)
    return float(r.group(1)) if r else None


def swap_mb():
    out = sh("sysctl -n vm.swapusage")
    r = re.search(r"used = ([\d.]+)M", out or "")
    return float(r.group(1)) if r else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-min", type=float, required=True)
    ap.add_argument("--max-rss-gb", type=float, default=12.0)
    ap.add_argument("--kill-mem", type=float, default=20.0)
    ap.add_argument("--kill-swap-mb", type=float, default=4000.0)
    ap.add_argument("--kill-samples", type=int, default=4)
    ap.add_argument("--stall-min", type=float, default=10.0)
    ap.add_argument("--beat", default=None)
    ap.add_argument("--done", default=None)
    ap.add_argument("--log", default="guard.log")
    ap.add_argument("--sample-s", type=float, default=5.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    if not cmd:
        sys.exit("no command")

    def glog(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(a.log, "a") as f:
            f.write(line + "\n")

    t_start = time.time()
    proc = subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL)
    glog(f"GUARD_START name={a.name} pid={proc.pid} cap={a.max_min}min rss<={a.max_rss_gb}GB cmd={' '.join(cmd)[:160]}")
    base_swap = swap_mb()
    low_mem = 0
    reason = None
    last_beat_seen = t_start
    last_report = 0.0
    while proc.poll() is None:
        t0 = time.time()
        el = (t0 - t_start) / 60
        rss = group_rss_gb(proc.pid)
        free = mem_free_pct()
        sw = swap_mb()
        if rss is None or free is None:
            reason = "sampler stall (metric command timed out)"
        if el > a.max_min:
            reason = f"wall-clock cap {a.max_min} min"
        if rss is not None and rss > a.max_rss_gb:
            reason = f"rss {rss:.1f}GB > cap {a.max_rss_gb}GB"
        low_mem = low_mem + 1 if free is not None and free < a.kill_mem else 0
        if low_mem >= a.kill_samples:
            reason = f"free memory {free}% < {a.kill_mem}% for {low_mem} samples"
        if sw is not None and base_swap is not None and sw - base_swap > a.kill_swap_mb and free is not None and free < 30:
            reason = f"swap grew {sw - base_swap:.0f}MB with free mem {free}%"
        if a.beat and os.path.exists(a.beat):
            last_beat_seen = max(last_beat_seen, os.path.getmtime(a.beat))
        if a.beat and (t0 - last_beat_seen) / 60 > a.stall_min:
            reason = f"no heartbeat for {a.stall_min} min (hung?)"
        if t0 - last_report > 60:
            glog(f"m el={el:.1f}min rss={'-' if rss is None else round(rss, 2)}GB free={free}% swap={sw}MB")
            last_report = t0
        if reason:
            glog(f"GUARD_KILL name={a.name} reason={reason}")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                time.sleep(3)
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
            glog(f"GUARD_END name={a.name} status=killed rc={proc.returncode} reason={reason} elapsed_s={time.time() - t_start:.0f}")
            sys.exit(3)
        time.sleep(max(0.0, a.sample_s - (time.time() - t0)))

    rc = proc.returncode
    status = "done" if rc == 0 else "failed"
    if rc == 0 and a.done:
        open(a.done, "w").write(time.strftime("%F %T") + "\n")
    glog(f"GUARD_END name={a.name} status={status} rc={rc} reason=exit elapsed_s={time.time() - t_start:.0f}")
    sys.exit(0 if rc == 0 else 2)


if __name__ == "__main__":
    main()
