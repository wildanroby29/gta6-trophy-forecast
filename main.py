"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Reads each case's Exophase history and forecasts, for every trophy, five
quantiles (5/25/50/75/95%) of its earn rate at the case's target, scored by
pinball loss on the logit scale.

Model, per trophy, in logit space (the scale the score uses):
  1. Refresh-aware window. Exophase refreshes its counts on its own schedule:
     a refresh shows up as a sudden jump in players_tracked with most rates
     dropping (the new players have earned little yet). Reads before the most
     recent refresh sit on a different player base, so the trend fit uses
     only reads after it (and at most the last 24 hours).
  2. Trend. A recency-weighted least-squares line of logit(rate) on
     ln(1 + hours since launch 2026-11-19T00:00Z) -- the same damped clock
     the house curve baseline uses, but over the last 24 hours with a short
     (8h) recency half-life: the process saturates, so the recent slope is
     the honest one and older reads overstate the move. The forecast is the
     line read at the target, pulled 40% back toward the last read (slope
     damping: the slope is estimated from quantized, correlated reads, the
     process saturates, and the horizon is days).
  3. Structure. Story trophies nest: Chapter 2 cannot be earned without
     Chapter 1, and so on down the chain to the final mission and 100%
     completion; the platinum cannot exceed the rarest trophy. The medians
     are forced to respect that order (cumulative minimum along the chain),
     which the per-trophy baselines do not do.
  4. Spread. An honest prediction interval, not a fixed width: the
     random-walk base (0.5 * sqrt(horizon in days), as the persistence
     baseline) combined with the regression's own prediction error
     (residual scatter plus slope uncertainty over the extrapolation).
     A trophy whose recent reads are noisy or whose slope is uncertain gets
     a wider band; a flat, stable one gets a narrow band.

Fallbacks: fewer than 3 usable reads, or a flat history, fall back to
persistence (the starter's formula). A trophy stuck at 100% is forecast
pinned at the top with a small downside tail. Everything is deterministic
(stdlib only, no randomness, no network) and fast.
"""

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model` finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
SIGMA = 0.5  # random-walk spread, logit units per sqrt day (the starter's)

LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
# Tuned on synthetic final-like cases (3 scenarios: base/slow/fast): the
# process saturates, so the recent slope overstates the move -- a short,
# strongly recency-weighted window and a damped extrapolation win. The
# environment overrides exist only for offline tuning and are never set by
# the grader. (A counts-based variant -- fitting earn counts and players
# separately -- and a saturating time transform were both tested and
# rejected: the counts variant lost badly to this model in every scenario,
# and the saturating clock tied it, so the simpler log clock stays.)
WINDOW_HOURS = float(os.environ.get("TROPHY_WINDOW_HOURS", "24.0"))
RECENCY_HALF_LIFE = float(os.environ.get("TROPHY_HALF_LIFE_HOURS", "8.0"))
REFRESH_JUMP = 1.15       # a players_tracked jump of 15%+ in one read = a refresh
SLOPE_DAMP = float(os.environ.get("TROPHY_SLOPE_DAMP", "0.60"))
SPREAD_FLOOR = 0.25       # logit units
SPREAD_CAP = 3.0          # logit units


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x * 10000) / 10000 rounds it."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_launch(dt: datetime) -> float:
    return (dt - LAUNCH).total_seconds() / 3600.0


def clock_x(hours: float) -> float:
    """The time transform the trend is linear in: ln(1+hours)."""
    return math.log1p(max(hours, 0.0))


def persistence_quantiles(rate: float, horizon_days: float):
    """The starter's forecast: last rate, random-walk spread."""
    spread = SIGMA * math.sqrt(horizon_days)
    lg = logit(rate if rate is not None else 0)
    return [round4(inv_logit(lg + z * spread)) for z in Z]


def weighted_line(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_sd, se_pred_fn).

    se_pred_fn(x0) is the standard error of a prediction at x0 (residual sd
    times sqrt(1 + 1/n_eff + (x0 - xbar_w)^2 / Sxx_w)).
    """
    sw = sum(ws)
    if sw <= 0:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    n = len(xs)
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    # Weighted residual variance; use n-2 degrees of freedom where possible.
    dof = max(n - 2, 1)
    var = sum(w * r * r for r, w in zip(resid, ws)) / sw * n / dof
    resid_sd = math.sqrt(max(var, 0.0))

    def se_pred(x0: float) -> float:
        lev = 1.0 + 1.0 / max(n, 1) + (x0 - xbar) ** 2 / sxx
        return resid_sd * math.sqrt(max(lev, 0.0))

    return a, b, resid_sd, se_pred


def find_chain(trophies):
    """Story chain and platinum, found by description/tier, not by id.

    Returns (chain_ids_in_order, platinum_id_or_None). The chain is the
    chapter trophies in order, then the final mission, then 100% completion.
    """
    chain = []
    chapters = []
    final_mission = None
    completion = None
    platinum = None
    for t in trophies:
        desc = (t.get("description") or "").strip()
        name = (t.get("name") or "").strip()
        m = re.match(r"Complete Chapter (\d+)\.?", desc)
        if m:
            chapters.append((int(m.group(1)), t["id"]))
        elif desc.startswith("Complete the final mission"):
            final_mission = t["id"]
        elif desc.startswith("Attain 100% Completion"):
            completion = t["id"]
        if (t.get("tier") or "").lower() == "platinum":
            platinum = t["id"]
    chapters.sort()
    chain = [tid for _, tid in chapters]
    if final_mission is not None:
        chain.append(final_mission)
    if completion is not None:
        chain.append(completion)
    return chain, platinum


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    history = case.get("history") or []
    if not history:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    at = parse_time(case["at"])
    target = parse_time(case["target"])
    x_target = clock_x(hours_since_launch(target))

    # --- Reads, oldest first, with time, players and rates. ---
    reads = []
    for r in history:
        try:
            when = parse_time(r.get("slot") or r["at"])
        except Exception:
            continue
        rates = r.get("rates") or []
        if len(rates) < n_tr:
            rates = list(rates) + [None] * (n_tr - len(rates))
        reads.append({
            "t": hours_since_launch(when),
            "age_h": max((at - when).total_seconds() / 3600.0, 0.0),
            "n": r.get("players_tracked"),
            "rates": rates[:n_tr],
        })
    if not reads:
        raise ValueError(f"{name} has no usable reads")
    reads.sort(key=lambda r: r["t"])

    # --- Most recent refresh: the last read whose players_tracked jumped. ---
    refresh_idx = -1
    for k in range(1, len(reads)):
        n0, n1 = reads[k - 1]["n"], reads[k]["n"]
        if isinstance(n0, (int, float)) and isinstance(n1, (int, float)) and n0 > 0:
            if n1 / n0 >= REFRESH_JUMP:
                refresh_idx = k
    # Trend window: after the refresh, and at most the last WINDOW_HOURS.
    window_start_t = reads[-1]["t"] - WINDOW_HOURS
    start = 0
    if refresh_idx >= 0:
        start = refresh_idx
    while start < len(reads) and reads[start]["t"] < window_start_t:
        start += 1
    window = reads[start:]
    if len(window) < 3:
        window = reads[-3:] if len(reads) >= 3 else reads

    last = reads[-1]
    chain, platinum = find_chain(trophies)
    idx_by_id = {t["id"]: i for i, t in enumerate(trophies)}

    medians = [None] * n_tr
    spreads = [None] * n_tr
    pinned_top = [False] * n_tr

    for i, trophy in enumerate(trophies):
        last_rate = last["rates"][i]
        if last_rate is None:
            # No rate in the last read: fall back to the newest non-null one.
            last_rate = next((r["rates"][i] for r in reversed(reads) if r["rates"][i] is not None), 0.0)

        if last_rate is not None and last_rate >= 99.99:
            # Stuck at the top: pin it there, with only a small downside tail.
            pinned_top[i] = True
            medians[i] = 99.99
            spreads[i] = 0.35
            continue

        xs, ys, ws = [], [], []
        for r in window:
            rate = r["rates"][i]
            if rate is None or rate <= 0:
                continue
            xs.append(clock_x(r["t"]))
            ys.append(logit(rate))
            # Recency weight, times a reliability weight from the player base.
            w = 0.5 ** (r["age_h"] / RECENCY_HALF_LIFE)
            if isinstance(r["n"], (int, float)) and r["n"] > 0:
                w *= math.sqrt(r["n"]) / 10.0
            ws.append(w)
        fit = weighted_line(xs, ys, ws) if len(xs) >= 3 else None
        last_lg = logit(last_rate if last_rate is not None else 0)
        if fit is None:
            medians[i] = inv_logit(last_lg)
            spreads[i] = SIGMA * math.sqrt(horizon_days)
            continue
        a, b, resid_sd, se_pred = fit
        line_at_target = a + b * x_target
        # Damped extrapolation: move from the last read toward the line.
        med_lg = last_lg + SLOPE_DAMP * (line_at_target - last_lg)
        medians[i] = inv_logit(med_lg)
        rw = SIGMA * math.sqrt(horizon_days)
        s = math.sqrt(rw * rw + se_pred(x_target) ** 2)
        spreads[i] = min(SPREAD_CAP, max(SPREAD_FLOOR, s))

    # --- Nesting constraints on the medians (rate space, cumulative min). ---
    if len(chain) >= 2:
        run_min = 100.0
        for tid in chain:
            i = idx_by_id.get(tid)
            if i is None or pinned_top[i]:
                continue
            run_min = min(run_min, medians[i])
            medians[i] = run_min
    if platinum is not None:
        pi = idx_by_id.get(platinum)
        if pi is not None and not pinned_top[pi]:
            others = [medians[i] for i in range(n_tr) if i != pi and not pinned_top[i]]
            if others:
                medians[pi] = min(medians[pi], min(others))

    rows = []
    for i, trophy in enumerate(trophies):
        if pinned_top[i]:
            q = [99.0, 99.5, 99.9, 99.99, 99.99]
            rows.append({"id": trophy["id"], "q": q})
            continue
        med_lg = logit(medians[i])
        s = spreads[i]
        q = [round4(inv_logit(med_lg + z * s)) for z in Z]
        # Keep the quantiles non-decreasing and inside [0, 100].
        q = [min(100.0, max(0.0, v)) for v in q]
        for k in range(1, 5):
            if q[k] < q[k - 1]:
                q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main() -> None:
    os.umask(0o022)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")
        os.chmod(out, 0o644)


if __name__ == "__main__":
    main()
