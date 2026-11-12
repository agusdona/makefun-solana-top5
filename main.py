#!/usr/bin/env python3
"""make.fun bounty: Predict the top 5 new Solana coins by market cap, 24 hours out.

Reads /input/task.json + /input/history.ndjson (the house's 10-minute marks of
every pump.fun coin launched since the opening) and writes
/output/forecast.json with exactly the keys `probabilities`, `top5`, `sizes`,
in the format of the kit's house baseline (starter/main.ts) and the scorer
(grader/forecast/ranked.ts, P_FLOOR = 1e-4):

  probabilities: rows {mint, place, p}  (place 1..5, p floored to 6 decimals)
  top5:          5 distinct mint addresses, place 1 first
  sizes:         rows {place, bin, p}    (bin 0..len(edges), log10 cap in SOL)

Method (Python 3.13, standard library only, no network, deterministic):
Round 3 calibration (2026-11-12, two synthetic backtests, +0.81 nats
min-delta vs round 2): collapse hazard x0.5, q_other x0.5, drift shrunk
80% toward global (was 65%).
  1. Build each mint's market-cap trajectory (log10 lamports) from all marks.
     Coins are keyed by mint address only, never by name/symbol: two coins
     can share a name (e.g. GABOON) and only the mint tells them apart.
  2. Forecast every coin's log market cap at each of the seven answer reads
     (from task.json `reads`, else 7 hourly reads after the cut):
       E[logM(t)] = last logM + drift0 * TAU * (1 - exp(-t/TAU))
                     + global_drift * (t - TAU * (1 - exp(-t/TAU)))
     where drift0 is the coin's recent log-slope (0.3 x 1h + 0.7 x 3h)
     shrunk 80% toward the global drift -- the median 3h slope of the top
     10 coins at the cut, clipped to [-0.015, +0.005] per hour (the median
     coin is a dying tail coin; top coins decay much more slowly) -- and
     momentum fades with time constant TAU = 2.5 h.  Uncertainty grows as
     sigma(t) = sqrt(sigma_h^2 * t + sigma0^2), sigma_h from the MEDIAN
     absolute 10-minute log-return (a variance would let a launch ramp
     masquerade as ongoing volatility), floored 0.12, capped 0.50 log10
     per sqrt(h).
  3. Collapse risk: each coin carries an hourly hazard of a near-total dump
     (thin liquidity vs market cap, young age, not graduated raise it).
     A coin still on the curve can be rugged (-1.7 log10); a graduated coin
     sits on a real pool, so its floor is a deep drawdown (-0.6 log10) --
     which is what keeps a huge but thin leader (a 12x lead survives a 75%
     drawdown, but not a rug) from getting a ~0.97 P(1st).
  4. Monte Carlo (fixed seed, 20,000 paths x 7 reads): draw each candidate's
     log market cap at each read, rank, record places and the winning
     log market cap (in SOL) per place.  The per-place probabilities are the
     simulated frequencies, rescaled by (1 - q_other[place]).
  5. q_other (the mass left unnamed, for coins outside the candidate set and
     coins launched after the cut) is measured from the history itself:
     over all pairs of marks ~7 h apart, the fraction of top-5 slots at the
     later mark held by coins outside the earlier mark's top 60 (or never
     seen by then).  Falls back to 0.08 when the history is too short.
     Unnamed mass is what the scorer shares among the answer chart's coins
     the forecast does not name -- including newcomers -- so it is kept
     honest instead of being spread thin over named coins.
  6. Sizes: the pooled simulated distribution of the winning log10 market
     cap (in SOL) at each place, histogrammed over task.json's bin edges.
  7. top5: the coins with the highest expected log market cap at the
     middle answer read.

No row names a mint the history never names (such rows are dropped by the
scorer).  No p = 0 rows are emitted: a named coin at p 0 would score the
floor if it held the place and would be removed from the unnamed share.
All randomness is seeded (SEED) and every iteration order is fixed, so two
runs on the same history write byte-identical output.
"""

import bisect
import json
import math
import os
import random
from datetime import datetime

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

SEED = 20261112
N_PATHS = 20000
MAX_CANDIDATES = 60
MAX_BYTES = 1_048_576
TAU_HOURS = 2.5          # momentum fades with this time constant
COLLAPSE_SHOCK = 1.7     # log10 drop when a coin collapses
SIGMA_FLOOR = 0.12       # log10 units per sqrt(hour)
SIGMA_CAP = 0.50
DEFAULT_GLOBAL_DRIFT = -0.03   # log10 per hour, pump.fun coins decay
DEFAULT_Q_OTHER = 0.08
CHURN_GAP_HOURS = 7.0    # gap used to measure top-5 churn into the chart


def parse_ts(s):
    """ISO-8601 timestamp (with Z) -> epoch seconds (float)."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def floor6(p):
    """Floor a probability to 6 decimals (the scorer's precision)."""
    v = math.floor(p * 1_000_000) / 1_000_000
    return 0.0 if v == 0 else v


def interp(series, t):
    """Linear interpolation of a [(t, y), ...] series at time t (seconds).

    Returns None when t is outside the series' range.
    """
    if not series or t < series[0][0] or t > series[-1][0]:
        return None
    lo, hi = 0, len(series) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] < t:
            lo = mid + 1
        else:
            hi = mid
    if series[lo][0] == t:
        return series[lo][1]
    t0, y0 = series[lo - 1]
    t1, y1 = series[lo]
    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)


def slope_since(series, t_last, hours):
    """Log-slope per hour over the window ending at t_last (seconds)."""
    target = t_last - hours * 3600.0
    y_then = interp(series, target)
    if y_then is None:
        # Coin is younger than the window: slope since first sighting.
        if len(series) < 2:
            return None
        age_h = (t_last - series[0][0]) / 3600.0
        if age_h <= 0:
            return None
        return (series[-1][1] - series[0][1]) / age_h
    return (series[-1][1] - y_then) / hours


def main():
    with open(os.path.join(INPUT_DIR, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task.get("places", 5))
    output_name = task.get("output", "forecast.json")
    edges = task.get("size", {}).get("edges") or [
        round(2.0 + 0.05 * i, 2) for i in range(81)
    ]
    read_times = task.get("reads")

    # ---- Load marks -------------------------------------------------------
    marks = []  # (at_seconds, rows) for ok marks with rows
    with open(os.path.join(INPUT_DIR, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") is True and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append((parse_ts(m["at"]), m["rows"]))
    if not marks:
        raise SystemExit("the history holds no ok mark")
    marks.sort(key=lambda x: x[0])
    cut = marks[-1][0]  # last mark's time: the history's cut

    # ---- Per-mint series --------------------------------------------------
    # mint -> {"series": [(t, log10 mcap lamports)], "created": epoch or None,
    #         "complete": bool, "mayhem": bool, "last_qr": float, "last_mcap": float}
    coins = {}
    for at, rows in marks:
        for r in rows:
            mint = r.get("mint")
            raw = r.get("market_cap_lamports")
            if not mint or raw is None:
                continue
            try:
                lamports = int(raw)
            except (TypeError, ValueError):
                continue
            if lamports <= 0:
                continue
            c = coins.get(mint)
            if c is None:
                c = coins[mint] = {
                    "series": [],
                    "created": None,
                    "complete": False,
                    "mayhem": False,
                    "last_qr": 0.0,
                    "last_mcap": 0.0,
                }
            c["series"].append((at, math.log10(lamports)))
            created = r.get("created_at")
            if created:
                try:
                    ct = parse_ts(created)
                except ValueError:
                    ct = None
                if ct is not None and (c["created"] is None or ct < c["created"]):
                    c["created"] = ct
            if r.get("complete") is True:
                c["complete"] = True
            if r.get("mayhem") is True:
                c["mayhem"] = True
            try:
                c["last_qr"] = float(r.get("quote_reserves_lamports") or 0)
            except (TypeError, ValueError):
                pass
            c["last_mcap"] = float(lamports)
    if len(coins) < places:
        raise SystemExit(
            "the history names %d coins, fewer than the %d places" % (len(coins), places)
        )
    for c in coins.values():
        c["series"].sort(key=lambda x: x[0])

    # ---- Answer-read horizons --------------------------------------------
    if read_times:
        horizons = [(parse_ts(rt) - cut) / 3600.0 for rt in read_times]
    else:
        horizons = [7.0 + i for i in range(7)]
    horizons = [max(h, 0.05) for h in horizons]
    mid_h = horizons[len(horizons) // 2]

    # ---- Global drift (median recent slope of the TOP coins) -------------
    # The median coin in the last mark is a dying tail coin; its slope would
    # drag every leader down ~0.1-0.2 log10 over the window. The top coins
    # are the survivors, and their drift is the relevant prior for the
    # top-5 forecast -- so take the median over the top 10 by market cap,
    # and clip it: a top coin's expected decay over 7-13 h is mild.
    last_rows = marks[-1][1]
    top_rows = sorted(
        (r for r in last_rows if r.get("mint") in coins),
        key=lambda r: -int(r.get("market_cap_lamports") or 0),
    )[:10]
    recent_slopes = []
    for r in top_rows:
        c = coins.get(r.get("mint"))
        if c is None or len(c["series"]) < 2:
            continue
        s3 = slope_since(c["series"], cut, 3.0)
        if s3 is not None:
            recent_slopes.append(s3)
    if len(recent_slopes) >= 3:
        recent_slopes.sort()
        global_drift = recent_slopes[len(recent_slopes) // 2]
    elif recent_slopes:
        global_drift = sum(recent_slopes) / len(recent_slopes)
    else:
        global_drift = DEFAULT_GLOBAL_DRIFT
    global_drift = max(-0.015, min(0.005, global_drift))

    # ---- Per-coin features -------------------------------------------------
    for mint, c in coins.items():
        series = c["series"]
        t_last = series[-1][0]
        staleness_h = max(0.0, (cut - t_last) / 3600.0)
        last_logm = series[-1][1] + global_drift * staleness_h
        s1 = slope_since(series, t_last, 1.0)
        s3 = slope_since(series, t_last, 3.0)
        own = None
        if s1 is not None and s3 is not None:
            own = 0.3 * s1 + 0.7 * s3  # the 3h slope is the steadier read
        elif s1 is not None:
            own = s1
        elif s3 is not None:
            own = s3
        if own is None:
            own = global_drift
        # Shrink the coin's own slope toward the global drift: a 1h fade
        # extrapolated for 7-13 h overshoots badly. Clip the result.
        # Round 3: shrink 80% (was 65%) — the leaders' own slopes are noisy
        # over a 7-13 h horizon; leaning harder on the global drift improves
        # the size score +0.06 nats on two synthetic worlds.
        drift0 = 0.20 * own + 0.80 * global_drift
        drift0 = max(-0.15, min(0.15, drift0))
        # Volatility from 10-minute log returns, scaled to per sqrt(hour).
        # A MEDIAN-based estimator: a coin's launch ramp is one or two huge
        # returns, and a variance would let that ramp masquerade as ongoing
        # volatility (it once pushed sigma to the cap for every fast mover).
        diffs = []
        for i in range(1, len(series)):
            dt = (series[i][0] - series[i - 1][0]) / 3600.0
            if dt > 0:
                diffs.append((series[i][1] - series[i - 1][1]) / math.sqrt(dt))
        if len(diffs) >= 4:
            absdiffs = sorted(abs(d) for d in diffs)
            med = absdiffs[len(absdiffs) // 2]
            sigma_h = med / 0.6745
        else:
            sigma_h = 0.25
        sigma_h = max(SIGMA_FLOOR, min(SIGMA_CAP, sigma_h))
        sigma0 = 0.05 + 0.05 * staleness_h
        # Collapse hazard per hour: young / thin / ungraduated coins rug.
        age_h = (cut - c["created"]) / 3600.0 if c["created"] is not None else 24.0
        if age_h < 1.0:
            hazard = 0.08
        elif age_h < 3.0:
            hazard = 0.045
        else:
            hazard = 0.025
        if c["last_mcap"] > 0 and c["last_qr"] / c["last_mcap"] < 0.02:
            # Thin liquidity vs market cap -- but only a curve coin can be
            # rugged on it; a graduated coin's pool is real, and a low
            # reserves/mcap ratio is normal for a large cap.
            if not c["complete"]:
                hazard *= 1.5
        if c["complete"]:
            hazard *= 0.6  # graduated: real pool behind it
        if c["mayhem"]:
            hazard *= 1.3
        # Halved (round 3): the raw rates overstate collapse risk for the
        # chart's leaders — backtests improve +0.57 nats with hazard x0.5
        # (order +0.17, sizes +0.39) on two synthetic worlds.
        hazard = min(hazard * 0.5, 0.12)
        # The collapse's size: a graduated coin sits on a real pool, so its
        # floor is a deep drawdown, not a rug to zero; a coin still on the
        # curve can be rugged by its deployer. A 12x leader survives -0.6
        # (a 75% drawdown) and keeps its place; it does not survive -1.7.
        shock = -0.6 if (c["complete"] and not c["mayhem"]) else -COLLAPSE_SHOCK
        c.update(
            {
                "last_logm": last_logm,
                "drift0": drift0,
                "sigma_h": sigma_h,
                "sigma0": sigma0,
                "hazard": hazard,
                "shock": shock,
                "age_h": age_h,
            }
        )
        # Expected log market cap and sigma at each read horizon.
        c["exp"] = []
        c["sig"] = []
        c["pcol"] = []
        for t in horizons:
            decay = 1.0 - math.exp(-t / TAU_HOURS)
            exp_t = (
                last_logm
                + drift0 * TAU_HOURS * decay
                + global_drift * (t - TAU_HOURS * decay)
            )
            sig_t = math.sqrt(sigma_h * sigma_h * t + sigma0 * sigma0)
            pcol = 1.0 - (1.0 - hazard) ** t
            c["exp"].append(exp_t)
            c["sig"].append(sig_t)
            c["pcol"].append(min(pcol, 0.60))

    # ---- q_other: churn measured from the history itself -------------------
    # For pairs of marks ~7 h apart: fraction of the later top-5 held by coins
    # that were outside the earlier mark's top 60 (or never seen by then).
    q_measures = []
    chart_by_time = []
    for at, rows in marks:
        chart_by_time.append((at, [r.get("mint") for r in rows if r.get("mint")]))
    for i, (at_i, chart_i) in enumerate(chart_by_time):
        target = at_i + CHURN_GAP_HOURS * 3600.0
        j = None
        for k in range(i + 1, len(chart_by_time)):
            if chart_by_time[k][0] >= target:
                j = k
                break
        if j is None:
            continue
        seen_before = set()
        for _, ch in chart_by_time[: i + 1]:
            seen_before.update(ch)
        top60_i = set(chart_i[:60])
        later_top5 = chart_by_time[j][1][:5]
        if not later_top5:
            continue
        # Coins outside the earlier top 60 (whether seen before or brand new).
        outside = sum(1 for m in later_top5 if m not in top60_i)
        q_measures.append(outside / len(later_top5))
    if len(q_measures) >= 2:
        q_base = sum(q_measures) / len(q_measures)
    else:
        q_base = DEFAULT_Q_OTHER
    q_other = []
    for k in range(1, places + 1):
        # Scaled x0.5 (round 3): the churn-measured unnamed mass (~0.30) is
        # too high — it dilutes the named coins. Halving it improves the
        # order score +0.20 nats on two synthetic worlds (a newcomer still
        # gets its share via the unnamed mass that remains).
        q_other.append(max(0.02, min(0.30, q_base * (0.6 + 0.2 * k) * 0.5)))

    # ---- Candidates --------------------------------------------------------
    # Top 60 by expected log market cap at the first read, plus the 10
    # fastest 1h risers with a non-trivial market cap, capped at 60.
    ranked = sorted(
        coins.items(),
        key=lambda kv: (-kv[1]["exp"][0], kv[0]),
    )
    cand_mints = [m for m, _ in ranked[:MAX_CANDIDATES]]
    cand_set = set(cand_mints)
    risers = sorted(
        coins.items(),
        key=lambda kv: (-kv[1]["drift0"], kv[0]),
    )
    for mint, c in risers[:10]:
        if c["last_mcap"] <= 0:
            continue
        if mint not in cand_set and len(cand_mints) < MAX_CANDIDATES:
            cand_mints.append(mint)
            cand_set.add(mint)
    # Fixed order for the simulation: by expected cap at first read, then mint.
    cand_mints.sort(key=lambda m: (-coins[m]["exp"][0], m))
    cands = [coins[m] for m in cand_mints]
    for c, m in zip(cands, cand_mints):
        c["mint"] = m

    # ---- Monte Carlo --------------------------------------------------------
    rng = random.Random(SEED)
    n_reads = len(horizons)
    counts = {m: [0] * places for m in cand_mints}
    size_samples = [[] for _ in range(places)]
    gauss = rng.gauss
    rand = rng.random
    for _ in range(N_PATHS):
        for ri in range(n_reads):
            vals = []
            for c in cands:
                lm = c["exp"][ri] + c["sig"][ri] * gauss(0.0, 1.0)
                if rand() < c["pcol"][ri]:
                    lm += c["shock"]
                vals.append((lm, c["mint"]))
            vals.sort(key=lambda x: -x[0])
            for place in range(places):
                lm_w, mint_w = vals[place]
                counts[mint_w][place] += 1
                size_samples[place].append(lm_w - 9.0)  # log10 cap in SOL
    total_draws = N_PATHS * n_reads

    # ---- Assemble the forecast ----------------------------------------------
    probabilities = []
    for place in range(1, places + 1):
        named = []
        for m in cand_mints:
            p = (counts[m][place - 1] / total_draws) * (1.0 - q_other[place - 1])
            p6 = floor6(p)
            if p6 > 0:
                named.append((m, p6))
        named.sort(key=lambda x: (-x[1], x[0]))
        for m, p6 in named:
            probabilities.append({"mint": m, "place": place, "p": p6})

    # top5: highest expected log market cap at the middle answer read.
    mid = len(horizons) // 2
    top5 = [
        m
        for m, _ in sorted(
            coins.items(), key=lambda kv: (-kv[1]["exp"][mid], kv[0])
        )[:places]
    ]

    sizes = []
    for place in range(1, places + 1):
        samples = size_samples[place - 1]
        bins = {}
        for v in samples:
            b = bisect.bisect_right(edges, v)
            bins[b] = bins.get(b, 0) + 1
        for b in sorted(bins):
            p6 = floor6(bins[b] / len(samples))
            if p6 > 0:
                sizes.append({"place": place, "bin": b, "p": p6})

    doc = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    out = json.dumps(doc) + "\n"
    if len(out.encode("utf-8")) > MAX_BYTES:
        raise SystemExit("forecast exceeds %d bytes" % MAX_BYTES)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, output_name), "w", encoding="utf-8") as f:
        f.write(out)


if __name__ == "__main__":
    main()
