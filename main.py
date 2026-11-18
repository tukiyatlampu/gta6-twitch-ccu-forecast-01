"""Hourly Twitch-viewer forecasts for a game release day.

An input case holds the first hour's viewer print together with the
hourly curves of earlier comparable launches. For each later hour the
program fits a robust regression of the log ratio to the hour-1 print
on the hour-1 log print across those comparables, then derives the
spread from held-out residuals of the same comparables. The result is
five quantiles per hour for hours 2..24. Standard library only, no
network access, and fully deterministic: identical input bytes always
produce identical output bytes.
"""

import json
import math
import os

# Tuned constants, fixed from the practice launches. Both the fitted
# lines and the spreads adapt to each case while it runs; no constant
# here encodes an answer for any particular launch.
RIDGE_EQUIV_OBS = 1.1
RIDGE_FLOOR_SD = 0.35
HUBER_LIMIT = 1.8
HUBER_FLOOR = 0.15
HUBER_ROUNDS = 3
NEIGHBOUR_PULL = 0.12
RESID_FOLDS = 7
LOCAL_MIX = 0.30
KERNEL_WIDTH = 1.25
EDGE_USE = 0.55
EDGE_CAP = 0.90
FULL_WEIGHT_N = 8
DRIFT_SIGMA = 0.15
SPREAD_FLOOR = 0.55
TAIL_CAP = 2.8
TAIL_EPS = 1e-6
FORECAST_HOURS = tuple(range(2, 25))
STD_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
STD_ZSCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
CUM_DAYS = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
DAYS_IN_MONTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
MICROS_PER_SEC = 1_000_000
MICROS_PER_HOUR = 3600 * MICROS_PER_SEC
Q_CEILING = 1_000_000_000.0
LOG_CEILING = math.log(Q_CEILING)
LOG_FLOOR = math.log(1e-6)

# ---- scalar helpers ----
def as_list(maybe):
    """Keep lists as they are; anything else behaves like an empty list."""
    if isinstance(maybe, list):
        return maybe
    return ()
def as_number(maybe):
    """Coerce a real int or float to a finite float; anything else is None."""
    if type(maybe) not in (int, float):
        return None  # strings, bools, nulls and containers are not prints
    try:  # float() raises on ints too large for a double
        converted = float(maybe)
    except OverflowError as _big:
        return None  # magnitude beyond the float range
    if math.isfinite(converted):
        return converted
    return None

# ---- timestamps ----
def to_micros(text):
    """Parse an ISO-8601 timestamp into integer microseconds since year 1.

    Accepts a 'T', 't' or space separator, optional seconds and
    fraction, and a 'Z' or numeric zone offset. Anything malformed
    yields None so callers fall back instead of crashing on corrupt
    case data.
    """
    shape_ok = (
        isinstance(text, str)
        and len(text) >= 16
        and text[4] == "-"
        and text[7] == "-"
        and text[10] in "Tt "
        and text[13] == ":"
    )
    if not shape_ok:
        return None  # not shaped like an ISO timestamp at all
    try:  # every int() below can reject a non-numeric field
        yy = int(text[0:4])
        mo = int(text[5:7])
        dd = int(text[8:10])
        hh = int(text[11:13])
        mi = int(text[14:16])
        leap = yy % 4 == 0 and (yy % 100 != 0 or yy % 400 == 0)
        if yy < 1 or mo < 1 or mo > 12 or hh < 0 or hh >= 24 or mi < 0 or mi >= 60:
            return None  # a date or time field is out of range
        if dd < 1 or dd > DAYS_IN_MONTH[mo - 1] + (leap and mo == 2):
            return None  # impossible day of the month
        cursor = 16
        sec = 0
        frac = 0
        if text[cursor:cursor + 1] == ":":
            sec = int(text[cursor + 1:cursor + 3])
            if len(text) < cursor + 3 or sec < 0 or sec >= 60:
                return None  # seconds field missing or out of range
            cursor += 3
        if text[cursor:cursor + 1] in (".", ","):
            begin = cursor + 1
            cursor = begin
            while cursor < len(text) and "0" <= text[cursor] <= "9":
                cursor += 1
            if cursor == begin:
                return None  # fraction marker with no digits after it
            frac = int(text[begin:min(cursor, begin + 6)].ljust(6, "0"))
        zone = text[cursor:]
        shift = 0
        if zone and zone not in ("Z", "z"):
            if zone[0] not in ("+", "-"):
                return None  # a zone offset must start with a sign
            zlen = len(zone)
            if zlen == 6 and zone[3] == ":":
                zh = int(zone[1:3])
                zm = int(zone[4:6])
            elif zlen == 5:
                zh = int(zone[1:3])
                zm = int(zone[3:5])
            elif zlen == 3:
                zh = int(zone[1:3])
                zm = 0
            else:
                return None  # unrecognised zone shape
            if zh < 0 or zh >= 24 or zm < 0 or zm >= 60:
                return None  # zone offset out of range
            shift = (60 * zh + zm) if zone[0] == "+" else -(60 * zh + zm)
        prior = yy - 1
        days = 365 * prior + prior // 4 - prior // 100 + prior // 400
        days += CUM_DAYS[mo - 1] + (leap and mo > 2) + dd - 1
        whole = ((days * 24 + hh) * 3600 + (mi - shift) * 60 + sec) * MICROS_PER_SEC
        return whole + frac
    except ValueError as _bad:
        return None  # a field was not numeric

# ---- reading cases ----
def hourly_averages(rows, start, count):
    """Average viewer prints into consecutive hour buckets from a stamp.

    Rows with bad timestamps, out-of-range buckets or non-numeric
    viewers are skipped; buckets with no valid row stay None. The
    running mean keeps its sign-aware update so corrupt opposite-sign
    prints cannot overflow the subtraction.
    """
    sums = [0.0] * count
    seen = [0] * count
    if start is None:
        return [None] * count
    for rec in as_list(rows):
        when = to_micros(rec.get("from")) if isinstance(rec, dict) else None
        if when is None:
            continue
        slot = (when - start) // MICROS_PER_HOUR
        if 0 <= slot < count:
            val = as_number(rec.get("viewers"))
            if val is not None:
                seen[slot] += 1
                n = seen[slot]
                prev = sums[slot]
                if n == 1:
                    sums[slot] = val
                elif (prev >= 0.0) == (val >= 0.0):
                    sums[slot] = prev + (val - prev) / n
                else:
                    sums[slot] = prev * (1.0 - 1.0 / n) + val / n
    out = []
    for total, n in zip(sums, seen):
        out.append(total if n else None)
    return out
def anchor_print(case):
    """Recover the hour-1 print and whether a known row provided it.

    Preference order: a valid known row for hour 1, then the first
    bucket of the hourly history, then the latest dated history print
    at or before the checkpoint (an undated print is the last resort).
    Prints below 1 count as 1; with nothing usable the anchor is 1.
    """
    for rec in as_list(case.get("known")):
        if isinstance(rec, dict) and as_number(rec.get("hour")) == 1:
            val = as_number(rec.get("viewers"))
            if val is not None:
                return max(1.0, val), True
    start = to_micros(case.get("day_start"))
    cut = to_micros(case.get("checkpoint"))
    if start is None and cut is not None:
        start = cut - MICROS_PER_HOUR
    if cut is None and start is not None:
        cut = start + MICROS_PER_HOUR
    first = hourly_averages(case.get("history_hourly"), start, 1)[0]
    if first is not None:
        return max(1.0, first), False
    best_stamp = None
    best_val = None
    dateless = None
    for rec in as_list(case.get("history")):
        if isinstance(rec, dict):
            val = as_number(rec.get("viewers"))
            if val is not None:
                when = to_micros(rec.get("at"))
                if when is None:
                    dateless = val
                elif (cut is None or when <= cut) and (best_stamp is None or when >= best_stamp):
                    best_stamp = when
                    best_val = val
    picked = best_val if best_val is not None else dateless
    if picked is None:
        return 1.0, False
    return max(1.0, picked), False
def extract_curves(raw):
    """Turn raw comparables into sorted (log hour-1, log-ratio) curves.

    A comparable needs a parseable day start and a valid hour-1 bucket
    plus at least one later valid bucket. Sorting keeps the fold
    assignment and every floating-point sum independent of the order
    comparables happen to arrive in.
    """
    def sort_key(curve):
        base, ratios = curve
        return (base, tuple(-math.inf if r is None else r for r in ratios))
    found = []
    for cand in as_list(raw):
        t0 = to_micros(cand.get("day_start")) if isinstance(cand, dict) else None
        if t0 is None:
            continue
        buckets = hourly_averages(cand.get("hourly"), t0, 24)
        if buckets[0] is None:
            continue
        base = math.log(max(1.0, buckets[0]))
        ratios = []
        for val in buckets[1:]:
            if val is None:
                ratios.append(None)
            else:
                ratios.append(math.log(max(1.0, val)) - base)
        if any(rv is not None for rv in ratios):
            found.append((base, tuple(ratios)))
    found.sort(key=sort_key)
    return found
def midpoint(vals):
    """Middle value of a non-empty sample; mean of the central pair if even."""
    ranked = sorted(vals)
    n = len(ranked)
    half = n // 2
    if n % 2:
        return ranked[half]
    return 0.5 * (ranked[half - 1] + ranked[half])

# ---- robust regression ----
def ridge_fit(points, weights, penalty):
    """Weighted straight-line fit with a ridge penalty on the slope."""
    total_w = sum(weights)
    mean_x = sum(w * px for (px, _py), w in zip(points, weights)) / total_w
    mean_y = sum(w * py for (_px, py), w in zip(points, weights)) / total_w
    sxx = 0.0
    sxy = 0.0
    for (px, py), w in zip(points, weights):
        dx = px - mean_x
        sxx += w * dx * dx
        sxy += w * dx * (py - mean_y)
    slope = sxy / (sxx + penalty)
    return mean_y - slope * mean_x, slope
def fit_one_hour(pts):
    """Fit one hour's ratio line, with Huber reweighting when data is ample.

    The ridge penalty scales with the predictor variance so hours with
    little spread in hour-1 prints stay shrunk toward a flat line. The
    robust scale is fixed from the first fit; only the line itself is
    reweighted afterwards.
    """
    n = len(pts)
    mean_x = sum(px for px, _py in pts) / n
    var_x = sum((px - mean_x) ** 2 for px, _py in pts) / n
    penalty = RIDGE_EQUIV_OBS * max(var_x, RIDGE_FLOOR_SD ** 2)
    flat = [1.0] * n
    a, b = ridge_fit(pts, flat, penalty)
    if n >= FULL_WEIGHT_N:
        errs = [py - (a + b * px) for px, py in pts]
        middle = midpoint(errs)
        spread = midpoint([abs(e - middle) for e in errs])
        limit = HUBER_LIMIT * max(HUBER_FLOOR, 1.4826 * spread)
        for _round in range(HUBER_ROUNDS):
            wts = [1.0 if abs(e) <= limit else limit / abs(e) for e in errs]
            a, b = ridge_fit(pts, wts, penalty)
            errs = [py - (a + b * px) for px, py in pts]
    return a, b, pts[0][0], pts[-1][0], n
def fit_all_hours(curves):
    """Fit every forecast hour, then smooth each line toward its neighbours.

    Only the line coefficients are smoothed; each hour keeps its own
    training range and sample count, which the forecast uses for the
    extrapolation clamp and the model-versus-persistence blend.
    """
    per_hour = [[] for _ in FORECAST_HOURS]
    for base, ratios in curves:
        for col, ratio in enumerate(ratios):
            if ratio is not None:
                per_hour[col].append((base, ratio))
    raw = []
    for pts in per_hour:
        if pts:
            raw.append(fit_one_hour(pts))
        else:
            raw.append(None)
    done = []
    for col, line in enumerate(raw):
        if line is None:
            done.append(None)
            continue
        a, b, lo, hi, n = line
        aa = a
        bb = b
        for other in (col - 1, col + 1):
            if 0 <= other < len(raw) and raw[other] is not None:
                na, nb, _lo, _hi, nn = raw[other]
                w = NEIGHBOUR_PULL * min(1.0, nn / n)
                aa += w * (na - a)
                bb += w * (nb - b)
        done.append((aa, bb, lo, hi, n))
    return done
def ratio_at(line, x):
    """Evaluate a fitted line, allowing only damped extrapolation."""
    a, b, lo, hi, _n = line
    clamped = min(hi, max(lo, x))
    push = EDGE_USE * (x - clamped)
    push = max(-EDGE_CAP, min(EDGE_CAP, push))
    return a + b * (clamped + push)

# ---- held-out calibration ----
def crossfit_errors(curves):
    """Collect held-out residuals with rank-stratified folds.

    Each fold refits on the other folds and scores the held-out curves,
    so the calibration residuals never come from a fit that saw the
    curve being scored. A fixed fold count keeps this cheap.
    """
    out = [[] for _ in FORECAST_HOURS]
    folds = min(RESID_FOLDS, len(curves))
    if folds < 2:
        return out
    for fold in range(folds):
        train = []
        for rank, row in enumerate(curves):
            if rank % folds != fold:
                train.append(row)
        fits = fit_all_hours(train)
        held = curves[fold::folds]
        for base, ratios in held:
            for col, ratio in enumerate(ratios):
                if ratio is not None and fits[col] is not None:
                    out[col].append((base, ratio - ratio_at(fits[col], base)))
    return out
def error_cdf(errors, target):
    """Build a weighted empirical CDF of residuals near a target print.

    Weights mix a uniform share with a Gaussian kernel over the hour-1
    log print, so calibration leans on comparables whose opening hour
    looked like this case's. Plotting positions put equal-weight
    samples at i/(n+1).
    """
    if not errors:
        return []
    n = len(errors)
    kernels = []
    for src, _err in errors:
        kernels.append(math.exp(-0.5 * ((src - target) / KERNEL_WIDTH) ** 2))
    weights = [1.0 / n] * n
    ksum = sum(kernels)
    if ksum > 0.0:
        weights = []
        for k in kernels:
            weights.append((1.0 - LOCAL_MIX) / n + LOCAL_MIX * k / ksum)
    ranked = []
    for (_src, err), w in zip(errors, weights):
        ranked.append((err, w))
    ranked.sort()
    wsum = sum(w for _err, w in ranked)
    mean_w = wsum / n
    curve = []
    accum = 0.0
    for val, w in ranked:
        p = (accum + 0.5 * (w + mean_w)) / (wsum + mean_w)
        curve.append((p, val))
        accum += w
    return curve
def cdf_value_at(curve, level):
    """Invert a plotting-position CDF by linear interpolation."""
    plo, vlo = curve[0]
    if level <= plo:
        return vlo
    for k in range(1, len(curve)):
        phi, vhi = curve[k]
        if level <= phi:
            return vlo + (vhi - vlo) * ((level - plo) / (phi - plo))
        plo, vlo = phi, vhi
    return vlo

# ---- forecast assembly ----
def levels_of(case):
    """Read the case's quantile levels, repairing bad entries in place."""
    raw = as_list(case.get("levels"))
    if len(raw) != len(STD_LEVELS):
        return STD_LEVELS
    picked = []
    for item, fallback in zip(raw, STD_LEVELS):
        p = as_number(item)
        picked.append(p if p is not None and 0.0 <= p <= 1.0 else fallback)
    return tuple(picked)
def zscore(prob):
    """Standard normal quantile; exact table values for standard levels."""
    for lvl, zval in zip(STD_LEVELS, STD_ZSCORES):
        if prob == lvl:
            return zval
    clamped = max(TAIL_EPS, min(1.0 - TAIL_EPS, prob))
    a = -10.0
    b = 10.0
    for _step in range(48):
        m = (a + b) * 0.5
        below = 0.5 * math.erfc(-m / math.sqrt(2.0)) < clamped
        a = m if below else a
        b = b if below else m
    return (a + b) * 0.5
def counts_row(log_vals):
    """Convert log quantiles to viewer counts: monotone, capped, positive."""
    row = []
    running = -math.inf
    for val in log_vals:
        running = max(running, val)
        row.append(Q_CEILING if running >= LOG_CEILING else min(Q_CEILING, math.exp(max(LOG_FLOOR, running))))
    return row
def build_forecast(case):
    """Forecast one case: model quantiles blended with persistence.

    Quantiles are computed in ascending level order so the monotone
    repair inside counts_row runs the right way, then reordered to
    match the case's own level order exactly. When the anchor did not
    come from a valid known row the comparables are ignored and the
    forecast falls back to persistence alone. Hours with too few
    comparables blend toward persistence in proportion to their
    sample count, and every tail is floored by a share of the
    persistence spread and capped in log distance from the median.
    """
    case = case if isinstance(case, dict) else {}
    anchor, trusted = anchor_print(case)
    x = math.log(anchor)
    levels = levels_of(case)
    rank = sorted(range(len(levels)), key=lambda i: (levels[i], i))
    asc = tuple(levels[i] for i in rank)
    scores = tuple(zscore(p) for p in asc)
    curves = extract_curves(case.get("comparables")) if trusted else []
    if curves:
        fits = fit_all_hours(curves)
        calib = crossfit_errors(curves)
    else:
        fits = [None] * len(FORECAST_HOURS)
        calib = None
    rows = []
    for slot, hour in enumerate(FORECAST_HOURS):
        sigma = DRIFT_SIGMA * math.sqrt(hour - 1)
        base_offsets = [z * sigma for z in scores]
        line = fits[slot]
        if line is None:
            logs = [x + off for off in base_offsets]
        else:
            _a, _b, lo, hi, n = line
            blend = min(1.0, n / FULL_WEIGHT_N)
            curve = error_cdf(calib[slot], min(hi, max(lo, x)))
            shift = cdf_value_at(curve, 0.5) if curve else 0.0
            center = ratio_at(line, x) + shift
            logs = []
            for p, z, off in zip(asc, scores, base_offsets):
                if curve:
                    delta = cdf_value_at(curve, p) - shift
                else:
                    delta = off
                floor = SPREAD_FLOOR * abs(off)
                if z == 0.0:
                    delta = 0.0
                elif z < 0.0:
                    delta = -min(TAIL_CAP, max(floor, -delta))
                else:
                    delta = min(TAIL_CAP, max(floor, delta))
                mixed = (1.0 - blend) * off + blend * (center + delta)
                logs.append(x + mixed)
        asc_q = counts_row(logs)
        out_q = [0.0] * len(levels)
        for place, i in enumerate(rank):
            out_q[i] = asc_q[place]
        rows.append({"hour": hour, "q": out_q})
    return {"hours": rows}

# ---- program entry ----
def load_case(path):
    """Read one case file; bad files degrade to an empty case.

    The utf-8-sig codec discards a leading byte-order mark, without
    which a BOM'd file would fail JSON parsing and lose its hour-1
    print. Only file and parse errors are caught here, never model
    errors.
    """
    try:  # file and parse errors degrade to an empty case
        with open(path, "r", encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (OSError, ValueError, RecursionError) as _exc:
        return dict()
def run_all():
    """Forecast every .in file in the input dir into the output dir."""
    src_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    dst_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(dst_dir, exist_ok=True)
    try:  # a missing input dir means there is nothing to forecast
        entries = sorted(os.listdir(src_dir))
    except OSError as _missing:
        entries = []
    for name in entries:
        if name[-3:] != ".in":
            continue
        case = load_case(os.path.join(src_dir, name))
        result = build_forecast(case)
        payload = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        target = os.path.join(dst_dir, name[:-3] + ".out")
        with open(target, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
if __name__ == "__main__": run_all()
