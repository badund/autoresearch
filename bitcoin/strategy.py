"""
BTC trading strategy. This is the file the agent modifies.
Exp79: Trailing stop 3250 tracking lows
"""

import sys
import time

import numpy as np
import pandas as pd

_dot = np.dot
_maximum = np.maximum

# Global state for hysteresis
_last_position = 0.0
_bar_count = 0
_t_start = 0.0
_bars_since_exit = 999
_best_price = 0.0

# --- Cached full-length column arrays (extracted from DataFrame internals) ---
_cached_close = None
_cached_high = None
_cached_low = None
_cached_open = None
_cached_ob10 = None
_cached_dur = None
_cached_upticks = None
_cached_downticks = None

# --- Precomputed linreg constants ---
_x28 = np.arange(28, dtype=float) - 13.5
_den28 = float(np.sum(_x28 ** 2))  # 1641.0
_inv_den28 = 1.0 / _den28

_x55 = np.arange(55, dtype=float) - 27.0
_den55 = float(np.sum(_x55 ** 2))  # 6930.0
_inv_den55 = 1.0 / _den55


def _extract_full_col(bars, col_name):
    """Extract the full underlying numpy array for a column."""
    v = bars[col_name].values
    base = v.base
    if base is None:
        return v
    if base.ndim == 1:
        return base
    if base.ndim == 2:
        v_stride = v.strides[0]
        v_ptr = v.ctypes.data
        base_ptr = base.ctypes.data
        offset = v_ptr - base_ptr
        if v_stride == base.strides[0]:
            col_idx = offset // base.strides[1] if base.strides[1] else 0
            return base[:, col_idx]
        elif v_stride == base.strides[1]:
            row_idx = offset // base.strides[0] if base.strides[0] else 0
            return base[row_idx, :]
    return v


def _sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    return 0


def _clamp(x, lo, hi):
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def strategy(bars: pd.DataFrame) -> float:
    """
    Linreg + all filters + 6-bar cooldown after exit.
    """
    global _last_position, _bar_count, _t_start, _bars_since_exit, _best_price
    global _cached_close, _cached_high, _cached_low
    global _cached_open, _cached_ob10, _cached_dur
    global _cached_upticks, _cached_downticks
    n = len(bars)
    _bar_count += 1
    if n < 80:
        return 0.0

    # --- Cache full column arrays on first call ---
    if _cached_close is None:
        _t_start = time.perf_counter()
        _cached_close = _extract_full_col(bars, "close")
        _cached_high = _extract_full_col(bars, "high")
        _cached_low = _extract_full_col(bars, "low")
        _cached_open = _extract_full_col(bars, "open")
        _cached_ob10 = _extract_full_col(bars, "ob10")
        _cached_dur = _extract_full_col(bars, "duration")
        _cached_upticks = _extract_full_col(bars, "upticks")
        _cached_downticks = _extract_full_col(bars, "downticks")

    # Slice to current bar count (zero-copy numpy views)
    close = _cached_close[:n]
    high = _cached_high[:n]
    low = _cached_low[:n]
    open_px = _cached_open[:n]
    ob10 = _cached_ob10[:n]
    duration = _cached_dur[:n]
    upticks = _cached_upticks[:n]
    downticks = _cached_downticks[:n]

    # Track cooldown
    _bars_since_exit += 1

    # Volatility regime check
    recent_range = (high[-20:] - low[-20:]).sum() / 20.0
    longer_range = (high[-60:] - low[-60:]).sum() / 60.0
    vol_ratio = recent_range / longer_range if longer_range > 1e-9 else 1.0

    # Linreg slopes
    y28 = close[-28:]
    y28_mean = y28.sum() / 28.0
    slope_short = _dot(_x28, y28) * _inv_den28
    slope_short = slope_short / y28_mean if abs(y28_mean) > 1e-9 else 0.0

    y55 = close[-55:]
    y55_mean = y55.sum() / 55.0
    slope_long = _dot(_x55, y55) * _inv_den55
    slope_long = slope_long / y55_mean if abs(y55_mean) > 1e-9 else 0.0

    ss = _sign(slope_short)
    sl = _sign(slope_long)

    if ss != sl:
        if _last_position < -0.4 and sl < 0:
            return _clamp(_last_position, -1.0, 1.0)
        else:
            was_in_position = abs(_last_position) > 0.1
            _last_position *= 0.6
            if abs(_last_position) < 0.05:
                _last_position = 0.0
                if was_in_position:
                    _bars_since_exit = 0
            return _clamp(_last_position, -1.0, 1.0)

    signal = (0.5 * slope_short + 0.5 * slope_long) * 750.0

    # Informed flow check
    lookback = min(n, 10)
    impact_arr = close[-lookback:] - open_px[-lookback:]
    price_impact = impact_arr.sum() / lookback
    ob_bias = ob10[-lookback:].sum() / (lookback * 50.0) - 1.0
    if abs(price_impact) > 0.01:
        informed = _sign(price_impact) * ob_bias
    else:
        informed = 0.0
    if informed < -0.2:
        signal *= 0.65

    # Tick imbalance confirmation for shorts
    up10 = upticks[-10:].sum()
    dn10 = downticks[-10:].sum()
    tick_total = up10 + dn10
    if tick_total > 0:
        tick_imbal = (dn10 - up10) / tick_total
        if signal < 0 and tick_imbal > 0.05:
            signal *= 1.2
        elif signal < 0 and tick_imbal < -0.05:
            signal *= 0.7

    # Vol gate
    if vol_ratio < 0.8:
        signal = 0.0

    # Short-only
    if signal > 0:
        signal = 0.0

    # Cooldown: block new entries within 6 bars of last exit
    if abs(_last_position) < 0.1 and _bars_since_exit < 6:
        signal = 0.0

    # Trailing stop for shorts
    cur_close = close[-1]
    cur_low = low[-1]
    if _last_position < -0.1:
        if _best_price == 0.0 or cur_low < _best_price:
            _best_price = cur_low
        retrace = cur_close - _best_price
        if retrace > 3250.0:
            signal = 0.0  # force exit

    target = _clamp(signal, -0.5, 0.0)

    if abs(target - _last_position) < 0.4:
        target = _last_position

    # Track entry for trailing stop
    if abs(_last_position) < 0.1 and target < -0.1:
        _best_price = cur_close

    # Detect exit for cooldown tracking
    if abs(_last_position) > 0.1 and abs(target) < 0.1:
        _bars_since_exit = 0
        _best_price = 0.0

    _last_position = target
    return _clamp(target, -1.0, 1.0)


EXECUTION = {
    "mode": "market_close",
}

if __name__ == "__main__":
    _last_position = 0.0
    _bar_count = 0
    _t_start = 0.0
    _bars_since_exit = 999
    _best_price = 0.0
    _cached_close = None
    _cached_high = None
    _cached_low = None
    _cached_open = None
    _cached_ob10 = None
    _cached_dur = None
    _cached_upticks = None
    _cached_downticks = None
    from prepare import evaluate
    evaluate(strategy, execution=EXECUTION)
    elapsed = time.perf_counter() - _t_start
    bars_sec = int(_bar_count / elapsed) if elapsed > 0 else 0
    print(f"strategy_bars:      {_bar_count}", file=sys.stderr)
    print(f"strategy_time:      {elapsed:.3f}s", file=sys.stderr)
    print(f"strategy_throughput: {bars_sec} bars/sec", file=sys.stderr)
