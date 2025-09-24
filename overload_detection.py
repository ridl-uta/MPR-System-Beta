"""Overload detection helpers used by read_pdu_power."""

from __future__ import annotations

from collections import deque
from datetime import timedelta
from typing import Deque, Optional


def make_simple_overload_ctx(
    sample_period_s: float,
    threshold_w: float,
    # hysteresis_w: float = 20.0,
    min_over_s: int = 10,# convert to time 
    cooldown_s: int = 30,# contvert to time
    spike_threshold_w: Optional[float] = None,
    spike_margin_w: float = 50.0, # should be based on percentage of capacity, any value
    spike_persistence_s: float = 1.0,
    ramp_fast_window_s: float = 5.0,
    ramp_slow_window_s: float = 60.0,
    ramp_delta_w: float = 40.0,
    ramp_reset_w: float = 10.0,
    ramp_slope_w: float = 15.0,
    ramp_slope_reset_w: float = 5.0,
    handled_window_s: float = 10.0,
    enable_handled: bool = True,
    peak_window_s: Optional[float] = None,
):
    """Return an initialized context dictionary for overload detection."""

    sample_period_s = float(sample_period_s)
    threshold_w = float(threshold_w)

    spike_threshold = (
        float(spike_threshold_w) if spike_threshold_w is not None else threshold_w + float(spike_margin_w)
    )
    spike_required_samples = max(1, int(round(spike_persistence_s / sample_period_s)))

    fast_window = max(1, int(round(ramp_fast_window_s / sample_period_s)))
    slow_window = max(fast_window, int(round(ramp_slow_window_s / sample_period_s)))
    handled_window = max(1, int(round(handled_window_s / sample_period_s)))

    slope_reset = float(ramp_slope_reset_w)
    slope_trigger = float(ramp_slope_w)
    if slope_reset >= slope_trigger:
        slope_reset = slope_trigger * 0.5

    peak_window = peak_window_s if peak_window_s is not None else handled_window_s
    peak_window = max(sample_period_s, float(peak_window))
    peak_window_samples = max(1, int(round(peak_window / sample_period_s)))

    return {
        'T_sample': sample_period_s,
        'T_high': threshold_w,
        'T_low': threshold_w - float(hysteresis_w),
        'min_hysteresis': float(hysteresis_w),
        'required_high_samples': max(1, int(round(min_over_s / sample_period_s))),
        'required_low_samples': max(1, int(round(cooldown_s / sample_period_s))),
        'spike_threshold': spike_threshold,
        'spike_required_samples': spike_required_samples,
        'spike_counter': 0,
        'spike_active': False,
        'ramp_fast_window': fast_window,
        'ramp_slow_window': slow_window,
        'ramp_delta_w': float(ramp_delta_w),
        'ramp_reset_w': float(ramp_reset_w),
        'ramp_slope_w': slope_trigger,
        'ramp_slope_reset_w': slope_reset,
        'handled_required_samples': handled_window,
        'handled_enabled': bool(enable_handled),
        'fast_samples': deque(),
        'slow_samples': deque(),
        'fast_sum': 0.0,
        'slow_sum': 0.0,
        'ramp_active': False,
        'ramp_slope_active': False,
        'previous_fast_avg': None,
        'peak_window_size': peak_window_samples,
        'peak_window_values': deque(),
        'peak_window_max_queue': deque(),
        'peak_window_max': 0.0,
        'peak_window_values_raw': deque(),
        'peak_window_max_queue_raw': deque(),
        'peak_window_max_raw': 0.0,
        'state': 'NORMAL',
        'consecutive_high_samples': 0,
        'consecutive_low_samples': 0,
        'consecutive_mid_samples': 0,
        'over_start': None,
        'seg_sum': 0.0,
        'seg_count': 0,
        'seg_peak': 0.0,
        'handled_since': None,
    }


def simple_overload_update(ctx, ts, watts):
    """Advance the overload detector with one sample and emit events when needed."""

    sample = float(watts)

    fast_samples: Deque[float] = ctx['fast_samples']
    slow_samples: Deque[float] = ctx['slow_samples']
    peak_values: Deque[float] = ctx['peak_window_values']
    peak_max_queue: Deque[float] = ctx['peak_window_max_queue']
    peak_window_size: int = ctx['peak_window_size']
    peak_values_raw: Deque[float] = ctx['peak_window_values_raw']
    peak_max_queue_raw: Deque[float] = ctx['peak_window_max_queue_raw']

    fast_samples.append(sample)
    ctx['fast_sum'] += sample
    if len(fast_samples) > ctx['ramp_fast_window']:
        ctx['fast_sum'] -= fast_samples.popleft()

    slow_samples.append(sample)
    ctx['slow_sum'] += sample
    if len(slow_samples) > ctx['ramp_slow_window']:
        ctx['slow_sum'] -= slow_samples.popleft()

    peak_values.append(sample)
    while peak_max_queue and peak_max_queue[-1] < sample:
        peak_max_queue.pop()
    peak_max_queue.append(sample)
    if len(peak_values) > peak_window_size:
        removed = peak_values.popleft()
        if peak_max_queue and removed == peak_max_queue[0]:
            peak_max_queue.popleft()
    ctx['peak_window_max'] = peak_max_queue[0] if peak_max_queue else sample

    raw_sample = ctx.get('last_raw_sample', sample)
    peak_values_raw.append(raw_sample)
    while peak_max_queue_raw and peak_max_queue_raw[-1] < raw_sample:
        peak_max_queue_raw.pop()
    peak_max_queue_raw.append(raw_sample)
    if len(peak_values_raw) > peak_window_size:
        removed_raw = peak_values_raw.popleft()
        if peak_max_queue_raw and removed_raw == peak_max_queue_raw[0]:
            peak_max_queue_raw.popleft()
    ctx['peak_window_max_raw'] = peak_max_queue_raw[0] if peak_max_queue_raw else raw_sample

    fast_count = min(len(fast_samples), ctx['ramp_fast_window'])
    slow_count = min(len(slow_samples), ctx['ramp_slow_window'])
    fast_avg = ctx['fast_sum'] / fast_count if fast_count else sample
    slow_avg = ctx['slow_sum'] / slow_count if slow_count else sample

    ramp_event_info = None
    ramp_predicted_info = None
    slow_window_ready = slow_count >= ctx['ramp_slow_window']

    prev_fast_avg = ctx['previous_fast_avg']
    if prev_fast_avg is None:
        slope = 0.0
    else:
        slope = (fast_avg - prev_fast_avg) / ctx['T_sample']
    ctx['previous_fast_avg'] = fast_avg

    if slow_window_ready:
        if slope >= ctx['ramp_slope_w'] and not ctx['ramp_slope_active']:
            ctx['ramp_slope_active'] = True
            ramp_predicted_info = {
                'fast_avg': fast_avg,
                'slow_avg': slow_avg,
                'slope_w_per_s': slope,
            }
        elif ctx['ramp_slope_active'] and slope <= ctx['ramp_slope_reset_w']:
            ctx['ramp_slope_active'] = False

        delta_w = fast_avg - slow_avg
        if delta_w >= ctx['ramp_delta_w'] and not ctx['ramp_active']:
            ctx['ramp_active'] = True
            ramp_event_info = {
                'fast_avg': fast_avg,
                'slow_avg': slow_avg,
                'delta_watts': delta_w,
            }
        elif ctx['ramp_active'] and delta_w <= ctx['ramp_reset_w']:
            ctx['ramp_active'] = False

    spike_event_info = None
    if sample >= ctx['spike_threshold']:
        ctx['spike_counter'] += 1
    else:
        ctx['spike_counter'] = 0
        ctx['spike_active'] = False

    if ctx['spike_counter'] >= ctx['spike_required_samples'] and not ctx['spike_active']:
        ctx['spike_active'] = True
        spike_event_info = {
            'watts': sample,
            'threshold': ctx['spike_threshold'],
            'duration_s': ctx['spike_counter'] * ctx['T_sample'],
        }

    T_high = ctx['T_high']
    T_low = ctx['T_low']
    consecutive_high = ctx['consecutive_high_samples']
    state = ctx['state']

    if sample >= T_high:
        consecutive_high += 1
    else:
        consecutive_high = 0
    ctx['consecutive_high_samples'] = consecutive_high

    between = T_low <= sample < T_high

    if state == 'NORMAL':
        if consecutive_high >= ctx['required_high_samples']:
            ctx['state'] = 'OVERLOAD'
            backdated_seconds = (ctx['required_high_samples'] - 1) * ctx['T_sample']
            ctx['over_start'] = ts - timedelta(seconds=backdated_seconds)
            ctx['seg_sum'] = 0.0
            ctx['seg_count'] = 0
            ctx['seg_peak'] = 0.0
            ctx['consecutive_mid_samples'] = 0
            ctx['consecutive_low_samples'] = 0
            return 'OVERLOAD_START', None

        if ramp_event_info:
            return 'RAMP_WARNING', ramp_event_info
        if ramp_predicted_info:
            return 'RAMP_PREDICTED', ramp_predicted_info
        if spike_event_info:
            return 'SPIKE_WARNING', spike_event_info
        return None, None

    ctx['seg_sum'] += sample
    ctx['seg_count'] += 1
    if sample > ctx['seg_peak']:
        ctx['seg_peak'] = sample

    if sample <= T_low:
        ctx['consecutive_low_samples'] += 1
    else:
        ctx['consecutive_low_samples'] = 0

    handled_enabled = ctx.get('handled_enabled', True)

    if state == 'OVERLOAD' and handled_enabled:
        if between:
            ctx['consecutive_mid_samples'] += 1
        else:
            ctx['consecutive_mid_samples'] = 0

        if ctx['consecutive_mid_samples'] >= ctx['handled_required_samples']:
            ctx['state'] = 'OVERLOAD_HANDLED'
            ctx['handled_since'] = ts
            return 'OVERLOAD_HANDLED', {'watts': sample}

    elif state == 'OVERLOAD_HANDLED' and handled_enabled:
        if sample >= T_high:
            ctx['state'] = 'OVERLOAD'
            ctx['consecutive_mid_samples'] = 0
        elif not between:
            ctx['consecutive_mid_samples'] = 0
        else:
            ctx['consecutive_mid_samples'] += 1

    if ctx['consecutive_low_samples'] >= ctx['required_low_samples']:
        cooldown_seconds = (ctx['required_low_samples'] - 1) * ctx['T_sample']
        end = ts - timedelta(seconds=cooldown_seconds)
        avg = ctx['seg_sum'] / max(1, ctx['seg_count'])
        info = {
            'start': ctx['over_start'],
            'end': end,
            'duration_s': ctx['seg_count'] * ctx['T_sample'],
            'avg_watts': avg,
            'peak_watts': ctx['seg_peak'],
        }

        ctx['state'] = 'NORMAL'
        ctx['over_start'] = None
        ctx['consecutive_high_samples'] = 0
        ctx['consecutive_low_samples'] = 0
        ctx['consecutive_mid_samples'] = 0
        ctx['handled_since'] = None
        ctx['T_low'] = ctx['T_high'] - ctx['min_hysteresis']
        return ('OVERLOAD_END', info), info

    if ramp_event_info:
        return 'RAMP_WARNING', ramp_event_info
    if ramp_predicted_info:
        return 'RAMP_PREDICTED', ramp_predicted_info
    if spike_event_info:
        return 'SPIKE_WARNING', spike_event_info
    return None, None
