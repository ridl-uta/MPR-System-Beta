#!/usr/bin/env python3
"""Plot sample totals with overload detection events highlighted."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from overload_detection import LoadShedder, make_simple_overload_ctx, simple_overload_update

EventRecord = Tuple[datetime, str, Dict[str, float]]


def load_samples(path: Path) -> List[Tuple[datetime, float]]:
    with path.open() as f:
        reader = csv.DictReader(f)
        return [
            (datetime.fromisoformat(row["timestamp"]), float(row["total_watts"]))
            for row in reader
        ]


def run_detection(samples: List[Tuple[datetime, float]], args) -> Tuple[List[EventRecord], List[float], List[float], List[float], List[str]]:
    ctx = make_simple_overload_ctx(
        sample_period_s=args.sample_period,
        threshold_w=args.threshold,
        hysteresis_w=args.hysteresis,
        min_over_s=args.min_over,
        cooldown_s=args.cooldown,
        spike_threshold_w=args.spike_threshold,
        spike_margin_w=args.spike_margin,
        spike_persistence_s=args.spike_persistence,
        ramp_fast_window_s=args.ramp_fast_window,
        ramp_slow_window_s=args.ramp_slow_window,
        ramp_delta_w=args.ramp_delta,
        ramp_reset_w=args.ramp_reset,
        ramp_slope_w=args.ramp_slope,
        ramp_slope_reset_w=args.ramp_slope_reset,
        handled_window_s=args.handled_window,
        enable_handled=False,
    )

    shedder = None
    if args.shed_watts > 0 or args.shed_margin > 0:
        shedder = LoadShedder(
            threshold_w=args.threshold,
            sample_period_s=args.sample_period,
            margin_w=args.shed_margin,
            activation_delay_s=args.shed_delay,
            max_reduction_w=args.shed_watts if args.shed_watts > 0 else None,
        )
        ctx['handled_enabled'] = True

    events: List[EventRecord] = []
    effective_totals: List[float] = []
    raw_totals: List[float] = []
    lows: List[float] = []
    states: List[str] = []

    for ts, watts in samples:
        raw_totals.append(watts)
        effective = shedder.adjust(watts) if shedder else watts
        effective_totals.append(effective)

        if shedder:
            peak_raw = ctx.get('peak_window_max_raw', watts)
            reduction = max(0.0, peak_raw - effective)
            if shedder.active and reduction > 0:
                base_low = ctx['T_high'] - ctx['min_hysteresis']
                ctx['T_low'] = min(max(0.0, ctx['T_high'] - reduction), base_low)
            else:
                ctx['T_low'] = ctx['T_high'] - ctx['min_hysteresis']

        ctx['last_raw_sample'] = watts
        lows.append(ctx['T_low'])

        event, info = simple_overload_update(ctx, ts, effective)
        if shedder:
            shedder.handle_event(event)
        states.append(ctx['state'])
        if event is None:
            continue
        if isinstance(event, tuple):
            label, payload = event
            payload = dict(payload)
            payload.setdefault('watts', watts)
            payload.setdefault('effective_watts', effective)
            events.append((ts, label, payload))
        else:
            payload = dict(info or {})
            payload.setdefault('watts', watts)
            payload.setdefault('effective_watts', effective)
            events.append((ts, event, payload))

    return events, effective_totals, raw_totals, lows, states


def plot(samples: List[Tuple[datetime, float]], events: List[EventRecord], effective: List[float], raw: List[float], lows: List[float], states: List[str], args, output: Path) -> None:
    times = [ts for ts, _ in samples]
    raw_watts = raw
    eff_watts = effective

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(times, raw_watts, label="Raw Total", color="steelblue")

    has_shedding = any(abs(r - e) > 1e-6 for r, e in zip(raw_watts, eff_watts))
    if has_shedding:
        ax.plot(times, eff_watts, label="Effective Total", color="seagreen", linestyle="--")

    index_map = {ts: idx for idx, ts in enumerate(times)}

    colors = {
        'RAMP_PREDICTED': 'orange',
        'RAMP_WARNING': 'darkorange',
        'SPIKE_WARNING': 'red',
        'OVERLOAD_START': 'crimson',
        'OVERLOAD_END': 'green',
    }
    markers = {
        'RAMP_PREDICTED': 'v',
        'RAMP_WARNING': '^',
        'SPIKE_WARNING': 'X',
        'OVERLOAD_START': 's',
        'OVERLOAD_END': 'o',
    }

    segments: List[Tuple[datetime, datetime, str]] = []
    if times and states:
        seg_start = times[0]
        current_state = states[0]
        for idx_state in range(1, len(times)):
            if states[idx_state] != current_state:
                segments.append((seg_start, times[idx_state], current_state))
                seg_start = times[idx_state]
                current_state = states[idx_state]
        segments.append((seg_start, times[-1], current_state))

    for ts, label, data in events:
        if label == 'OVERLOAD_HANDLED':
            continue
        color = colors.get(label, 'black')
        marker = markers.get(label, 'o')
        text = label
        if label == 'SPIKE_WARNING':
            text = f"SPIKE {data.get('watts', 0):.0f}W"
        elif label == 'RAMP_WARNING':
            text = f"RAMP Δ{data.get('delta_watts', 0):.0f}W"
        elif label == 'RAMP_PREDICTED':
            text = f"PRED slope {data.get('slope_w_per_s', 0):.1f}"
        elif label == 'OVERLOAD_END':
            text = f"END {data.get('duration_s', 0):.0f}s"

        idx = index_map.get(ts)
        series = eff_watts if has_shedding else raw_watts
        yval = data.get('effective_watts', data.get('watts', series[idx] if idx is not None else None))
        if idx is None:
            continue
        ax.scatter(ts, yval,
                   color=color, marker=marker, s=80, zorder=5)
        ax.annotate(
            text,
            xy=(ts, yval),
            xytext=(6, 12),
            textcoords='offset points',
            color=color,
            fontsize=9,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.6, lw=0),
        )

    ax.axhline(args.threshold, color='grey', linestyle='--', linewidth=1, label=f"T_high {args.threshold:.0f}W")
    base_low = max(0.0, args.threshold - args.hysteresis)

    max_reduction = max((r - e) for r, e in zip(raw_watts, eff_watts)) if has_shedding else 0.0
    if max_reduction > 1e-6:
        dynamic_low = max(0.0, args.threshold - max_reduction)
        ax.axhline(dynamic_low, color='silver', linestyle=':', linewidth=1.2,
                   label=f"T_low handled {dynamic_low:.0f}W (max reduction {max_reduction:.0f}W)")
    else:
        ax.axhline(base_low, color='lightgrey', linestyle='--', linewidth=1, label=f"T_low {base_low:.0f}W")

    state_colors = {
        'OVERLOAD': ('#ffe0e0', 'Overload window'),
        'OVERLOAD_HANDLED': ('#fff4cc', 'Handled window'),
    }

    used_state_labels = set()
    for start, end, state in segments:
        color_label = state_colors.get(state)
        if not color_label:
            continue
        color, label = color_label
        span_label = label if label not in used_state_labels else None
        ax.axvspan(start, end, color=color, alpha=0.25, label=span_label)
        used_state_labels.add(label)

    used_transition_labels = set()
    for idx in range(1, len(segments)):
        prev_state = segments[idx - 1][2]
        next_state = segments[idx][2]
        if prev_state == next_state:
            continue
        x = segments[idx][0]
        transition_label = f"{prev_state} → {next_state}"
        label = transition_label if transition_label not in used_transition_labels else None
        ax.axvline(x, color='lightcoral', linestyle='--', linewidth=1, label=label)
        used_transition_labels.add(transition_label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Watts")
    ax.set_title("Overload Detection Events")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

    fig.autofmt_xdate()

    fig.savefig(output, dpi=150, bbox_inches='tight')
    plt.close(fig)


def parse_args():
    ap = argparse.ArgumentParser(description="Plot overload detection events over time")
    ap.add_argument("csv", type=Path)
    ap.add_argument("--output", type=Path, default=Path("overload_events.png"))
    ap.add_argument("--sample-period", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=880.0)
    ap.add_argument("--hysteresis", type=float, default=20.0)
    ap.add_argument("--min-over", type=float, default=10.0)
    ap.add_argument("--cooldown", type=float, default=30.0)
    ap.add_argument("--spike-threshold", type=float, default=None)
    ap.add_argument("--spike-margin", type=float, default=50.0)
    ap.add_argument("--spike-persistence", type=float, default=1.0)
    ap.add_argument("--ramp-fast-window", type=float, default=5.0)
    ap.add_argument("--ramp-slow-window", type=float, default=60.0)
    ap.add_argument("--ramp-delta", type=float, default=40.0)
    ap.add_argument("--ramp-reset", type=float, default=10.0)
    ap.add_argument("--ramp-slope", type=float, default=15.0)
    ap.add_argument("--ramp-slope-reset", type=float, default=5.0)
    ap.add_argument("--handled-window", type=float, default=10.0)
    ap.add_argument("--shed-watts", type=float, default=0.0)
    ap.add_argument("--shed-margin", type=float, default=0.0)
    ap.add_argument("--shed-delay", type=float, default=5.0)
    return ap.parse_args()


def main():
    args = parse_args()
    samples = load_samples(args.csv)
    events, effective, raw, lows, states = run_detection(samples, args)
    plot(samples, events, effective, raw, lows, states, args, args.output)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
