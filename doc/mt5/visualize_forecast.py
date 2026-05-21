"""Visualize TimesFM forecast results as an interactive HTML dashboard.

Reads forecast_results.json and the corresponding covariate CSVs to plot
recent price history + forecast with confidence bands for all instruments.

Run:
    python mt5/scripts/visualize_forecast.py
    python mt5/scripts/visualize_forecast.py --history 120 --output dashboard.html
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import COV_DIR, DATA_DIR, INSTRUMENTS, TIMEFRAME


def load_results(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_dashboard(results: list[dict], history_bars: int) -> go.Figure:
    n = len(results)
    cols = 2
    rows = (n + 1) // 2

    # Build subplot titles with direction arrows
    titles = []
    for r in results:
        last = r["last_close"]
        end = r["point_forecast"][-1]
        pct = (end / last - 1) * 100
        arrow = "\u2191" if pct > 0 else "\u2193"
        titles.append(f"{r['instrument']}  {arrow} {abs(pct):.2f}%")

    fig = make_subplots(
        rows=rows, cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.04,
        horizontal_spacing=0.05,
    )

    tf_label = TIMEFRAME.replace("TIMEFRAME_", "")

    for idx, r in enumerate(results):
        row = idx // cols + 1
        col = idx % cols + 1
        name = r["instrument"]
        horizon = r["horizon"]
        last_dt = pd.Timestamp(r["last_datetime"])

        # Load historical prices for context
        cov_path = COV_DIR / f"{name}_{tf_label}_covariates.csv"
        hist_dts = []
        hist_close = []
        if cov_path.exists():
            df = pd.read_csv(cov_path, parse_dates=["datetime"])
            tail = df.tail(history_bars)
            hist_dts = tail["datetime"].tolist()
            hist_close = tail["close"].values.tolist()

        # Build forecast datetime axis (hourly steps from last_datetime)
        fc_dts = pd.date_range(
            start=last_dt + pd.Timedelta(hours=1),
            periods=horizon,
            freq="h",
        ).tolist()

        point = r["point_forecast"]
        q10 = r["quantile_10"]
        q90 = r["quantile_90"]

        last_close = r["last_close"]
        pct = (point[-1] / last_close - 1) * 100
        color = "rgb(0,180,80)" if pct > 0 else "rgb(220,50,50)"
        band_color = "rgba(0,180,80,0.15)" if pct > 0 else "rgba(220,50,50,0.15)"

        # Historical price line
        if hist_close:
            fig.add_trace(go.Scatter(
                x=hist_dts, y=hist_close,
                mode="lines",
                line=dict(color="rgb(100,100,100)", width=1.2),
                name="History",
                showlegend=(idx == 0),
                legendgroup="history",
            ), row=row, col=col)

        # Connection line from last history point to first forecast
        fig.add_trace(go.Scatter(
            x=[last_dt, fc_dts[0]],
            y=[last_close, point[0]],
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            showlegend=False,
        ), row=row, col=col)

        # 90% confidence band (fill between q10 and q90)
        fig.add_trace(go.Scatter(
            x=fc_dts + fc_dts[::-1],
            y=q90 + q10[::-1],
            fill="toself",
            fillcolor=band_color,
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI",
            showlegend=(idx == 0),
            legendgroup="ci",
        ), row=row, col=col)

        # Point forecast line
        fig.add_trace(go.Scatter(
            x=fc_dts, y=point,
            mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=3),
            name="Forecast",
            showlegend=(idx == 0),
            legendgroup="forecast",
        ), row=row, col=col)

        # Horizontal reference line at last close
        fig.add_hline(
            y=last_close, row=row, col=col,
            line_dash="dash", line_color="rgba(150,150,150,0.5)",
            annotation_text=f"{last_close:.5g}",
            annotation_position="bottom left",
            annotation_font_size=9,
        )

    fig.update_layout(
        title=dict(
            text="TimesFM 2.5 — Forecast Dashboard",
            font=dict(size=22),
        ),
        height=320 * rows,
        width=1400,
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.01,
            xanchor="center", x=0.5,
        ),
        margin=dict(t=80, b=30, l=50, r=30),
    )

    # Clean up axes
    fig.update_xaxes(showgrid=False, tickformat="%b %d\n%H:%M")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(80,80,80,0.3)")

    return fig


def build_summary_table(results: list[dict]) -> go.Figure:
    """Build a summary table ranking instruments by conviction."""
    rows_data = []
    for r in results:
        last = r["last_close"]
        end_pt = r["point_forecast"][-1]
        pct = (end_pt / last - 1) * 100
        q10_end = r["quantile_10"][-1]
        q90_end = r["quantile_90"][-1]
        band_width = abs(q90_end - q10_end)
        band_pct = band_width / last * 100
        direction = "LONG" if pct > 0 else "SHORT"
        # Conviction: signal strength relative to uncertainty
        conviction = abs(pct) / band_pct * 100 if band_pct > 0 else 0

        rows_data.append({
            "instrument": r["instrument"],
            "direction": direction,
            "pct_change": pct,
            "band_pct": band_pct,
            "conviction": conviction,
            "last_close": last,
            "end_forecast": end_pt,
        })

    # Sort by conviction descending
    rows_data.sort(key=lambda x: x["conviction"], reverse=True)

    colors = ["rgb(0,180,80)" if r["direction"] == "LONG" else "rgb(220,50,50)" for r in rows_data]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=["Rank", "Instrument", "Signal", "Expected %", "90% Band %", "Conviction", "Last Close", "Forecast End"],
            fill_color="rgb(40,40,40)",
            font=dict(color="white", size=13),
            align="center",
        ),
        cells=dict(
            values=[
                list(range(1, len(rows_data) + 1)),
                [r["instrument"] for r in rows_data],
                [r["direction"] for r in rows_data],
                [f"{r['pct_change']:+.3f}%" for r in rows_data],
                [f"{r['band_pct']:.3f}%" for r in rows_data],
                [f"{r['conviction']:.1f}" for r in rows_data],
                [f"{r['last_close']:.5g}" for r in rows_data],
                [f"{r['end_forecast']:.5g}" for r in rows_data],
            ],
            fill_color=[
                ["rgb(30,30,30)"] * len(rows_data),
                ["rgb(30,30,30)"] * len(rows_data),
                colors,
                ["rgb(30,30,30)"] * len(rows_data),
                ["rgb(30,30,30)"] * len(rows_data),
                ["rgb(30,30,30)"] * len(rows_data),
                ["rgb(30,30,30)"] * len(rows_data),
                ["rgb(30,30,30)"] * len(rows_data),
            ],
            font=dict(color="white", size=12),
            align="center",
            height=28,
        ),
    )])

    fig.update_layout(
        title="Signal Summary — Ranked by Conviction (signal strength / uncertainty)",
        height=60 + 28 * len(rows_data) + 80,
        width=1200,
        template="plotly_dark",
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


def main():
    p = argparse.ArgumentParser(description="Visualize forecast results")
    p.add_argument("--input", default=str(DATA_DIR / "forecast_results.json"))
    p.add_argument("--output", default=str(DATA_DIR / "forecast_dashboard.html"))
    p.add_argument("--history", type=int, default=120, help="Historical bars to show before forecast")
    args = p.parse_args()

    results = load_results(Path(args.input))
    print(f"Loaded {len(results)} instrument forecasts")

    # Build both figures
    dashboard = build_dashboard(results, args.history)
    summary = build_summary_table(results)

    # Write single HTML with both
    output_path = Path(args.output)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'>")
        f.write("<title>TimesFM Forecast Dashboard</title></head><body style='background:#1a1a1a;margin:0;padding:20px'>")
        f.write(summary.to_html(full_html=False, include_plotlyjs="cdn"))
        f.write("<hr style='border-color:#333;margin:30px 0'>")
        f.write(dashboard.to_html(full_html=False, include_plotlyjs=False))
        f.write("</body></html>")

    print(f"Dashboard saved to {output_path}")
    print(f"Open in browser: file:///{output_path.resolve()}")


if __name__ == "__main__":
    main()
