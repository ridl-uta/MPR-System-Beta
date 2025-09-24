#!/usr/bin/env python3
"""Offline harness for exercising simple_overload_update on sample data."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from overload_detection import make_simple_overload_ctx, simple_overload_update, LoadShedder


def load_samples(path: Path):
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            timestamp = datetime.fromisoformat(row["timestamp"])
            watts = float(row["total_watts"])
            yield timestamp, watts


def main():
    ap = argparse.ArgumentParser(description="Replay recorded totals through the overload detector")
    ap.add_argument("csv", type=Path, help="CSV with timestamp,total_watts columns")
    ap.add_argument("--sample-period", type=float, default=1.0, help="Seconds per sample (default 1)")
    ap.add_argument("--threshold", type=float, default=900.0, help="High-water overload threshold in watts")
    ap.add_argument("--hysteresis", type=float, default=20.0, help="Hysteresis band below threshold")
    ap.add_argument("--min-over", type=float, default=10.0, help="Seconds above threshold required to trigger")
    ap.add_argument("--cooldown", type=float, default=30.0, help="Seconds below threshold required to clear")
    ap.add_argument("--spike-threshold", type=float, default=None, help="Explicit spike threshold (defaults to threshold + margin)")
    ap.add_argument("--spike-margin", type=float, default=50.0, help="Margin above threshold to define spike threshold")
    ap.add_argument("--spike-persistence", type=float, default=1.0, help="Seconds a spike must persist before warning")
    ap.add_argument("--ramp-fast-window", type=float, default=5.0, help="Seconds for fast average window")
    ap.add_argument("--ramp-slow-window", type=float, default=60.0, help="Seconds for slow average window")
    ap.add_argument("--ramp-delta", type=float, default=40.0, help="Watts difference to raise ramp warning")
    ap.add_argument("--ramp-reset", type=float, default=10.0, help="Watts difference to clear ramp warning state")
    ap.add_argument("--ramp-slope", type=float, default=15.0, help="Slope in W/s that triggers ramp prediction")
    ap.add_argument("--ramp-slope-reset", type=float, default=5.0, help="Slope in W/s needed to clear ramp prediction")
    ap.add_argument("--handled-window", type=float, default=10.0, help="Seconds between thresholds before declaring handled")
    ap.add_argument("--shed-watts", type=float, default=0.0, help="Maximum watts to shed; 0 means unlimited")
    ap.add_argument("--shed-margin", type=float, default=0.0, help="Maintain this headroom below the high threshold")
    ap.add_argument("--shed-delay", type=float, default=5.0, help="Seconds to wait after overload start before shedding")
    args = ap.parse_args()

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

    shedder = LoadShedder(
        threshold_w=args.threshold,
        sample_period_s=args.sample_period,
        margin_w=args.shed_margin,
        activation_delay_s=args.shed_delay,
        max_reduction_w=args.shed_watts if args.shed_watts > 0 else None,
    ) if (args.shed_watts > 0 or args.shed_margin > 0) else None

    if shedder:
        ctx['handled_enabled'] = True

    for ts, watts in load_samples(args.csv):
        effective_watts = shedder.adjust(watts) if shedder else watts
        ctx['last_raw_sample'] = watts

        if shedder:
            peak_raw = ctx.get('peak_window_max_raw', watts)
            reduction = max(0.0, peak_raw - effective_watts)
            if shedder.active and reduction > 0:
                base_low = ctx['T_high'] - ctx['min_hysteresis']
                ctx['T_low'] = min(max(0.0, ctx['T_high'] - reduction), base_low)
            else:
                ctx['T_low'] = ctx['T_high'] - ctx['min_hysteresis']

        event, info = simple_overload_update(ctx, ts, effective_watts)
        if shedder:
            shedder.handle_event(event)
        if event == "RAMP_PREDICTED":
            info_dict = info or {}
            print(
                f"{ts.isoformat()} RAMP_PREDICTED fast_avg={info_dict.get('fast_avg', 0):.1f}W "
                f"slow_avg={info_dict.get('slow_avg', 0):.1f}W "
                f"slope={info_dict.get('slope_w_per_s', 0):.1f}W/s"
            )
        elif event == "OVERLOAD_HANDLED":
            info_dict = info or {}
            print(
                f"{ts.isoformat()} OVERLOAD_HANDLED effective={effective_watts:.1f}W "
                f"raw={watts:.1f}W"
            )
        elif event == "SPIKE_WARNING":
            info_dict = info or {}
            print(
                f"{ts.isoformat()} SPIKE_WARNING  value={info_dict.get('watts', watts):.1f}W "
                f"threshold={info_dict.get('threshold', args.threshold):.1f}W"
            )
        elif event == "RAMP_WARNING":
            info_dict = info or {}
            print(
                f"{ts.isoformat()} RAMP_WARNING  fast_avg={info_dict.get('fast_avg', 0):.1f}W "
                f"slow_avg={info_dict.get('slow_avg', 0):.1f}W "
                f"delta={info_dict.get('delta_watts', 0):.1f}W"
            )
        elif event == "OVERLOAD_START":
            print(f"{ts.isoformat()} OVERLOAD_START total={watts:.1f}W")
        elif isinstance(event, tuple) and event[0] == "OVERLOAD_END":
            info_dict = info or {}
            print(
                f"{ts.isoformat()} OVERLOAD_END   duration={info_dict.get('duration_s', 0):.1f}s "
                f"avg={info_dict.get('avg_watts', 0):.1f}W peak={info_dict.get('peak_watts', 0):.1f}W"
            )


if __name__ == "__main__":
    main()
