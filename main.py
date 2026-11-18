import json
import math
import os


# ============================== CONFIG ==============================

# Ridge penalty in equivalent observations at the hour's predictor variance.
RIDGE_ROWS = 1.1
# Minimum log-viewer standard deviation used to set the ridge penalty.
RIDGE_MIN_X_SD = 0.35
# Huber cutoff in robust residual standard deviations.
HUBER_CUTOFF = 1.8
# Minimum robust residual scale used when computing Huber weights.
HUBER_MIN_SCALE = 0.15
# Number of reweighted regression passes after the initial ridge fit.
HUBER_PASSES = 3
# Smoothing weight for each available immediately adjacent hour.
ADJACENT_WEIGHT = 0.12
# Number of rank-stratified folds used to collect held-out residuals.
CALIBRATION_FOLDS = 7
# Fraction of residual weight assigned to the local rather than pooled sample.
LOCAL_RESIDUAL_FRACTION = 0.30
# Gaussian kernel bandwidth for local calibration, in log hour-1 viewers.
LOCAL_BANDWIDTH = 1.25
# Fraction of the distance outside the training range used in the regression.
EXTRAPOLATION_FRACTION = 0.55
# Maximum effective predictor displacement beyond either training edge.
MAX_EXTRAPOLATION_LOG = 0.90
# Number of usable comparable pairs needed for full model weight.
FULL_MODEL_COUNT = 8
# Persistence log standard deviation per square root hour of lead time.
PERSISTENCE_SIGMA = 0.15
# Minimum model spread as a fraction of the persistence spread.
SPREAD_FLOOR_FRACTION = 0.55
# Maximum log-quantile distance on either side of the model median.
MAX_TAIL_LOG_DISTANCE = 2.8
# Probability clipping for normal scores at nonstandard endpoint levels.
NORMAL_LEVEL_EPSILON = 1e-6

# =========================== END CONFIG =============================

HOURS = tuple(range(2, 25))
DEFAULT_LEVELS = (0.05, 0.25, 0.50, 0.75, 0.95)
DEFAULT_Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
MONTH_START = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
MONTH_LENGTH = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
SECOND = 1_000_000
HOUR = 3600 * SECOND
MAX_Q = 1_000_000_000.0
LOG_MAX_Q = math.log(MAX_Q)
LOG_MIN_Q = math.log(1e-6)


def array(value):
    return value if isinstance(value, list) else ()


def number(value):
    if type(value) not in (int, float):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def utc_stamp(value):
    """Convert an ISO timestamp to integer microseconds on a common UTC axis."""
    if (
        not isinstance(value, str)
        or len(value) < 16
        or value[4] != "-"
        or value[7] != "-"
        or value[10] not in "Tt "
        or value[13] != ":"
    ):
        return None
    try:
        year = int(value[:4])
        month = int(value[5:7])
        day = int(value[8:10])
        hour = int(value[11:13])
        minute = int(value[14:16])
        if not (1 <= year and 1 <= month <= 12 and
                0 <= hour < 24 and 0 <= minute < 60):
            return None
        leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
        if not 1 <= day <= MONTH_LENGTH[month - 1] + (leap and month == 2):
            return None

        pos = 16
        sec = micros = 0
        if value[pos:pos + 1] == ":":
            sec = int(value[pos + 1:pos + 3])
            if len(value) < pos + 3 or not 0 <= sec < 60:
                return None
            pos += 3
        if value[pos:pos + 1] in (".", ","):
            start = pos + 1
            pos = start
            while pos < len(value) and "0" <= value[pos] <= "9":
                pos += 1
            if pos == start:
                return None
            micros = int(value[start:min(pos, start + 6)].ljust(6, "0"))

        zone = value[pos:]
        offset = 0
        if zone not in ("", "Z", "z"):
            if zone[0] not in "+-":
                return None
            if len(zone) == 6 and zone[3] == ":":
                zh, zm = int(zone[1:3]), int(zone[4:6])
            elif len(zone) == 5:
                zh, zm = int(zone[1:3]), int(zone[3:5])
            elif len(zone) == 3:
                zh, zm = int(zone[1:3]), 0
            else:
                return None
            if not (0 <= zh < 24 and 0 <= zm < 60):
                return None
            offset = (1 if zone[0] == "+" else -1) * (60 * zh + zm)

        y = year - 1
        days = 365 * y + y // 4 - y // 100 + y // 400
        days += MONTH_START[month - 1] + (leap and month > 2) + day - 1
        return ((days * 24 + hour) * 3600 + (minute - offset) * 60 + sec) * SECOND + micros
    except ValueError:
        return None


def hourly_means(rows, start, count):
    """Bucket valid samples by 'from'; missing buckets stay missing."""
    means = [0.0] * count
    counts = [0] * count
    if start is not None:
        for row in array(rows):
            if not isinstance(row, dict):
                continue
            stamp = utc_stamp(row.get("from"))
            if stamp is None:
                continue
            idx = (stamp - start) // HOUR
            if not 0 <= idx < count:
                continue
            v = number(row.get("viewers"))
            if v is None:
                continue

            counts[idx] += 1
            n = counts[idx]
            old = means[idx]
            if n == 1:
                means[idx] = v
            elif (old >= 0.0) == (v >= 0.0):
                means[idx] = old + (v - old) / n
            else:
                # Avoid overflowing v - old for corrupt opposite-sign inputs.
                means[idx] = old * (1.0 - 1.0 / n) + v / n

    return [v if n else None for v, n in zip(means, counts)]


def hour_one(case):
    """Return the anchor print and whether it came from a valid known row."""
    for row in array(case.get("known")):
        if not isinstance(row, dict) or number(row.get("hour")) != 1:
            continue
        v = number(row.get("viewers"))
        if v is not None:
            return max(1.0, v), True

    start = utc_stamp(case.get("day_start"))
    checkpoint = utc_stamp(case.get("checkpoint"))
    if start is None and checkpoint is not None:
        start = checkpoint - HOUR
    if checkpoint is None and start is not None:
        checkpoint = start + HOUR

    v = hourly_means(case.get("history_hourly"), start, 1)[0]
    if v is not None:
        return max(1.0, v), False

    latest = None
    latest_value = None
    undated_value = None
    for row in array(case.get("history")):
        if not isinstance(row, dict):
            continue
        v = number(row.get("viewers"))
        if v is None:
            continue
        stamp = utc_stamp(row.get("at"))
        if stamp is None:
            undated_value = v
        elif (checkpoint is None or stamp <= checkpoint) and (
            latest is None or stamp >= latest
        ):
            latest, latest_value = stamp, v

    v = latest_value if latest_value is not None else undated_value
    return (max(1.0, v) if v is not None else 1.0), False


def comparable_curves(raw):
    curves = []
    for entry in array(raw):
        if not isinstance(entry, dict):
            continue
        start = utc_stamp(entry.get("day_start"))
        if start is None:
            continue
        values = hourly_means(entry.get("hourly"), start, 24)
        if values[0] is None:
            continue
        x = math.log(max(1.0, values[0]))
        ratios = tuple(
            None if v is None else math.log(max(1.0, v)) - x
            for v in values[1:]
        )
        if any(r is not None for r in ratios):
            curves.append((x, ratios))

    # Keep folds and floating-point summations independent of input ordering.
    curves.sort(key=lambda row: (
        row[0], tuple(-math.inf if r is None else r for r in row[1])
    ))
    return curves


def median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def ridge_line(points, weights, ridge):
    weight = sum(weights)
    mx = sum(w * x for (x, _), w in zip(points, weights)) / weight
    my = sum(w * y for (_, y), w in zip(points, weights)) / weight
    xx = xy = 0.0
    for (x, y), w in zip(points, weights):
        dx = x - mx
        xx += w * dx * dx
        xy += w * dx * (y - my)
    slope = xy / (xx + ridge)
    return my - slope * mx, slope


def fit_hour(points):
    n = len(points)
    mx = sum(x for x, _ in points) / n
    var_x = sum((x - mx) ** 2 for x, _ in points) / n
    ridge = RIDGE_ROWS * max(var_x, RIDGE_MIN_X_SD ** 2)
    intercept, slope = ridge_line(points, [1.0] * n, ridge)

    if n >= FULL_MODEL_COUNT:
        residuals = [y - (intercept + slope * x) for x, y in points]
        center = median(residuals)
        mad = median(abs(r - center) for r in residuals)
        cutoff = HUBER_CUTOFF * max(HUBER_MIN_SCALE, 1.4826 * mad)
        # Fix the robust scale after the initial fit; reweight only the line.
        for _ in range(HUBER_PASSES):
            weights = [
                1.0 if abs(r) <= cutoff else cutoff / abs(r)
                for r in residuals
            ]
            intercept, slope = ridge_line(points, weights, ridge)
            residuals = [y - (intercept + slope * x) for x, y in points]

    return intercept, slope, points[0][0], points[-1][0], n


def fit_curves(curves):
    by_hour = [[] for _ in HOURS]
    for x, ratios in curves:
        for idx, y in enumerate(ratios):
            if y is not None:
                by_hour[idx].append((x, y))

    raw = [fit_hour(points) if points else None for points in by_hour]
    smooth = []
    for idx, fit in enumerate(raw):
        if fit is None:
            smooth.append(None)
            continue
        a, b, lo, hi, n = fit
        # Smooth the lines, not the training range or the sample count.
        ai, bi = a, b
        for neighbor in (idx - 1, idx + 1):
            if 0 <= neighbor < len(raw) and raw[neighbor] is not None:
                na, nb, _, _, nn = raw[neighbor]
                w = ADJACENT_WEIGHT * min(1.0, nn / n)
                ai += w * (na - a)
                bi += w * (nb - b)
        smooth.append((ai, bi, lo, hi, n))
    return smooth


def fitted_ratio(fit, x):
    a, b, lo, hi, _ = fit
    edge = min(hi, max(lo, x))
    extension = EXTRAPOLATION_FRACTION * (x - edge)
    extension = max(-MAX_EXTRAPOLATION_LOG, min(MAX_EXTRAPOLATION_LOG, extension))
    return a + b * (edge + extension)


def held_out_residuals(curves):
    """Use a fixed number of fits rather than refitting once per comparable."""
    residuals = [[] for _ in HOURS]
    k = min(CALIBRATION_FOLDS, len(curves))
    if k < 2:
        return residuals
    for fold in range(k):
        training = [row for rank, row in enumerate(curves) if rank % k != fold]
        fits = fit_curves(training)
        for x, ratios in curves[fold::k]:
            for idx, y in enumerate(ratios):
                if y is not None and fits[idx] is not None:
                    residuals[idx].append((x, y - fitted_ratio(fits[idx], x)))
    return residuals


def residual_distribution(residuals, target):
    if not residuals:
        return []
    n = len(residuals)
    kernels = [
        math.exp(-0.5 * ((source - target) / LOCAL_BANDWIDTH) ** 2)
        for source, _ in residuals
    ]
    total = sum(kernels)
    if total > 0.0:
        weights = [
            (1.0 - LOCAL_RESIDUAL_FRACTION) / n
            + LOCAL_RESIDUAL_FRACTION * k / total
            for k in kernels
        ]
    else:
        weights = [1.0 / n] * n

    ranked = sorted((r, w) for (_, r), w in zip(residuals, weights))
    total = sum(w for _, w in ranked)
    average = total / n
    cdf = []
    mass = 0.0
    for value, weight in ranked:
        # At equal weights, these are i/(n+1) plotting positions.
        p = (mass + 0.5 * (weight + average)) / (total + average)
        cdf.append((p, value))
        mass += weight
    return cdf


def empirical_quantile(cdf, level):
    p0, v0 = cdf[0]
    if level <= p0:
        return v0
    for i in range(1, len(cdf)):
        p1, v1 = cdf[i]
        if level <= p1:
            return v0 + (v1 - v0) * ((level - p0) / (p1 - p0))
        p0, v0 = p1, v1
    return v0


def case_levels(case):
    raw = array(case.get("levels"))
    if len(raw) != len(DEFAULT_LEVELS):
        return DEFAULT_LEVELS
    levels = []
    for item, default in zip(raw, DEFAULT_LEVELS):
        p = number(item)
        levels.append(p if p is not None and 0.0 <= p <= 1.0 else default)
    return tuple(levels)


def normal_score(level):
    for p, z in zip(DEFAULT_LEVELS, DEFAULT_Z):
        if level == p:
            return z
    p = max(NORMAL_LEVEL_EPSILON, min(1.0 - NORMAL_LEVEL_EPSILON, level))
    lo, hi = -10.0, 10.0
    for _ in range(48):
        mid = (lo + hi) * 0.5
        if 0.5 * math.erfc(-mid / math.sqrt(2.0)) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) * 0.5


def quantile_row(log_quantiles):
    values = []
    previous = -math.inf
    for value in log_quantiles:
        # Cumulative max in log space is the same monotone fix on counts.
        previous = max(previous, value)
        if previous >= LOG_MAX_Q:
            values.append(MAX_Q)
        else:
            values.append(min(MAX_Q, math.exp(max(LOG_MIN_Q, previous))))
    return values


def forecast(case):
    if not isinstance(case, dict):
        case = {}
    p1, known = hour_one(case)
    x = math.log(p1)
    levels = case_levels(case)
    # Compute quantiles in ascending level order (so the monotone repair in
    # quantile_row runs the right way), then reorder to the case's order.
    # For the standard levels this order is the identity: no behavior change.
    order = sorted(range(len(levels)), key=lambda i: (levels[i], i))
    asc_levels = tuple(levels[i] for i in order)
    scores = tuple(normal_score(p) for p in asc_levels)

    curves = comparable_curves(case.get("comparables")) if known else []
    fits = fit_curves(curves) if curves else [None] * len(HOURS)
    calibration = held_out_residuals(curves) if curves else None

    output = []
    for idx, hour in enumerate(HOURS):
        sigma = PERSISTENCE_SIGMA * math.sqrt(hour - 1)
        persistence = [z * sigma for z in scores]
        fit = fits[idx]
        if fit is None:
            logs = [x + offset for offset in persistence]
        else:
            _, _, lo, hi, n = fit
            blend = min(1.0, n / FULL_MODEL_COUNT)
            cdf = residual_distribution(calibration[idx], min(hi, max(lo, x)))
            correction = empirical_quantile(cdf, 0.5) if cdf else 0.0
            center = fitted_ratio(fit, x) + correction
            logs = []
            for p, z, fallback in zip(asc_levels, scores, persistence):
                if cdf:
                    delta = empirical_quantile(cdf, p) - correction
                else:
                    delta = fallback

                floor = SPREAD_FLOOR_FRACTION * abs(fallback)
                if z < 0.0:
                    delta = -min(MAX_TAIL_LOG_DISTANCE, max(floor, -delta))
                elif z > 0.0:
                    delta = min(MAX_TAIL_LOG_DISTANCE, max(floor, delta))
                else:
                    delta = 0.0

                ratio = (1.0 - blend) * fallback + blend * (center + delta)
                logs.append(x + ratio)

        q_asc = quantile_row(logs)
        q = [0.0] * len(levels)
        for pos, i in enumerate(order):
            q[i] = q_asc[pos]
        output.append({"hour": hour, "q": q})
    return {"hours": output}


def read_case(path):
    # Recover from bad files here, not from programming errors in the model.
    # utf-8-sig tolerates a BOM; without it a BOM'd file would parse as
    # invalid JSON and the whole case would fall back to a print of 1.
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError, RecursionError):
        return {}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(output_dir, exist_ok=True)
    try:
        names = sorted(os.listdir(input_dir))
    except OSError:
        # No readable input directory: nothing to forecast, exit cleanly.
        return
    for name in names:
        if not name.endswith(".in"):
            continue
        case = read_case(os.path.join(input_dir, name))
        result = forecast(case)
        # Fixed keys and 115 bounded floats keep the payload far below 64 KiB.
        text = json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n"
        path = os.path.join(output_dir, name[:-3] + ".out")
        with open(path, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
