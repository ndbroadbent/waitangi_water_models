#!/usr/bin/env python3
"""Plot Waitangi River flow and water level from NRC Hilltop API.

Shows historical data with caching - use --refresh to fetch fresh data.
"""

import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from waitangi.data.nrc_hilltop import NRCHilltopClient


def plot_river_data(days: int = 7, refresh: bool = False):
    """Plot flow and stage data for the last N days."""
    client = NRCHilltopClient()

    print(f"Fetching {days} days of data (refresh={refresh})...")

    flow = client.get_flow_data(days=days, refresh=refresh)
    stage = client.get_stage_data(days=days, refresh=refresh)

    cache_status = "from cache" if flow.from_cache else "fresh from API"
    print(f"Data source: {cache_status}")
    print(f"Flow: {len(flow.points)} points, Stage: {len(stage.points)} points")

    if not flow.points or not stage.points:
        print("No data available!")
        return

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # Plot flow
    ax1.plot(flow.timestamps, flow.values, "b-", linewidth=1, alpha=0.8)
    ax1.fill_between(flow.timestamps, flow.values, alpha=0.3)
    ax1.set_ylabel(f"Flow ({flow.units})", fontsize=12)
    ax1.set_title(f"Waitangi River at Waimate North Rd - Last {days} Days", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)

    # Add flow stats
    ax1.axhline(y=flow.mean_value, color="r", linestyle="--", alpha=0.5, label=f"Mean: {flow.mean_value:.3f} {flow.units}")
    ax1.legend(loc="upper right")

    # Annotations for flow
    latest_flow = flow.latest
    if latest_flow:
        ax1.annotate(
            f"Latest: {latest_flow.value:.3f} {flow.units}\n{latest_flow.timestamp.strftime('%Y-%m-%d %H:%M')}",
            xy=(latest_flow.timestamp, latest_flow.value),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

    # Plot stage
    ax2.plot(stage.timestamps, stage.values, "g-", linewidth=1, alpha=0.8)
    ax2.fill_between(stage.timestamps, stage.values, alpha=0.3, color="green")
    ax2.set_ylabel(f"Water Level ({stage.units})", fontsize=12)
    ax2.set_xlabel("Date", fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Add stage stats
    ax2.axhline(y=stage.mean_value, color="r", linestyle="--", alpha=0.5, label=f"Mean: {stage.mean_value:.0f} {stage.units}")
    ax2.legend(loc="upper right")

    # Format x-axis
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)

    # Add data source info
    fetched_str = flow.fetched_at.strftime("%Y-%m-%d %H:%M")
    fig.text(
        0.99, 0.01,
        f"Data: NRC Hilltop API | Fetched: {fetched_str} | {'Cached' if flow.from_cache else 'Fresh'}",
        ha="right", va="bottom", fontsize=8, color="gray",
    )

    plt.tight_layout()

    # Save figure
    output_path = f"waitangi_river_{days}d.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")

    # Print summary
    print(f"\n=== Flow Summary ===")
    print(f"  Range: {flow.min_value:.3f} - {flow.max_value:.3f} {flow.units}")
    print(f"  Mean:  {flow.mean_value:.3f} {flow.units}")
    print(f"  Latest: {latest_flow.value:.3f} {flow.units} at {latest_flow.timestamp}")

    print(f"\n=== Stage Summary ===")
    print(f"  Range: {stage.min_value:.0f} - {stage.max_value:.0f} {stage.units}")
    print(f"  Mean:  {stage.mean_value:.0f} {stage.units}")
    if stage.latest:
        print(f"  Latest: {stage.latest.value:.0f} {stage.units} at {stage.latest.timestamp}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot Waitangi River flow and water level")
    parser.add_argument("--days", type=int, default=7, help="Number of days of data (default: 7)")
    parser.add_argument("--refresh", action="store_true", help="Force refresh from API (default: use cache)")
    args = parser.parse_args()

    plot_river_data(days=args.days, refresh=args.refresh)


if __name__ == "__main__":
    main()
