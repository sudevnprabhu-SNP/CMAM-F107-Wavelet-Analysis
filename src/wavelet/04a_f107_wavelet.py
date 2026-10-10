"""
Wavelet analysis of daily solar F10.7 index (January 1979–June 2010).

Place this file at:
    src/wavelet/04d_f107_wavelet.py

Expected project layout:
    data/raw/f107/F107_daily_1979_2010.csv
    wave_python/waveletFunctions.py

Outputs:
    results/figures/04d_f107_wavelet_1979_2010.png
    results/tables/04d_f107_wavelet_global_spectrum.csv

Assignment requirements retained:
- Daily F10.7 observations
- Do NOT subtract the mean from F10.7
- Morlet wavelet, omega0 = 6
- 95% significance level
- Cone of Influence (COI)
- Period axis in days
- Display power as log10(power)

Note: A continuous daily time series is required by the wavelet transform.
If a small number of dates are missing/invalid, this script reindexes to a
daily calendar and linearly interpolates those missing values. It does not
subtract the F10.7 mean.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ---------------------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------------------
# This script lives in <project_root>/src/wavelet/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAVELET_LIBRARY_DIR = PROJECT_ROOT / "wave_python"
DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "f107"
    / "F107_daily_1979_2010.csv"
)
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

if not WAVELET_LIBRARY_DIR.is_dir():
    raise FileNotFoundError(
        f"Wavelet library folder not found:\n{WAVELET_LIBRARY_DIR}\n"
        "Expected wave_python/ at the project root."
    )

if not DATA_FILE.is_file():
    raise FileNotFoundError(
        f"F10.7 CSV not found:\n{DATA_FILE}\n"
        "Check that F107_daily_1979_2010.csv is in data/raw/f107/."
    )

sys.path.insert(0, str(WAVELET_LIBRARY_DIR))
from waveletFunctions import wavelet, wave_signif

print("Wavelet library imported successfully.")


# ---------------------------------------------------------------------
# 2. LOAD AND VALIDATE F10.7 DATA
# ---------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
df.columns = [str(column).strip().lower() for column in df.columns]

# Support common variants in column names without silently guessing values.
date_candidates = ["date", "time", "datetime"]
f107_candidates = ["f107", "f10.7", "f10_7", "adjusted_f107", "f107_index"]

date_column = next((name for name in date_candidates if name in df.columns), None)
f107_column = next((name for name in f107_candidates if name in df.columns), None)

if date_column is None or f107_column is None:
    raise ValueError(
        "Could not identify the date and F10.7 columns in the CSV.\n"
        f"Columns found: {list(df.columns)}\n"
        "Expected columns similar to 'date' and 'f107'."
    )

df = df[[date_column, f107_column]].copy()
df.columns = ["date", "f107"]
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["f107"] = pd.to_numeric(df["f107"], errors="coerce")
df = df.dropna(subset=["date"])
df["date"] = df["date"].dt.floor("D")
df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

# Treat non-positive values as invalid for this solar index series.
df.loc[~np.isfinite(df["f107"]) | (df["f107"] <= 0), "f107"] = np.nan
df = df.set_index("date")

if df["f107"].notna().sum() < 100:
    raise ValueError("Too few valid F10.7 observations to perform wavelet analysis.")

daily_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
daily = df["f107"].reindex(daily_index)

missing_before_fill = int(daily.isna().sum())
if missing_before_fill:
    # Interpolate only short gaps; avoid silently filling long missing intervals.
    missing_mask = daily.isna()
    groups = (missing_mask != missing_mask.shift()).cumsum()
    gap_lengths = missing_mask.groupby(groups).sum()
    longest_gap = int(gap_lengths.max()) if len(gap_lengths) else 0
    if longest_gap > 10:
        raise ValueError(
            f"The daily series contains a gap of {longest_gap} consecutive days. "
            "Inspect the source data before interpolating."
        )
    daily = daily.interpolate(method="time", limit=10, limit_area="inside")

if daily.isna().any():
    raise ValueError(
        "Missing values remain after interpolation. Check the first/last dates "
        "and source data before running the wavelet transform."
    )

time = daily.index
f107 = daily.to_numpy(dtype=float)

print("\n========================================")
print("F10.7 DATA INFORMATION")
print("========================================")
print(f"Input file: {DATA_FILE}")
print(f"Daily samples used: {len(f107):,}")
print(f"Date range: {time[0].date()} to {time[-1].date()}")
print(f"Missing/invalid daily values filled: {missing_before_fill}")
print(f"Mean F10.7: {np.mean(f107):.4f}")
print(f"Minimum F10.7: {np.min(f107):.4f}")
print(f"Maximum F10.7: {np.max(f107):.4f}")
print("Mean subtraction: NOT PERFORMED")


# ---------------------------------------------------------------------
# 3. WAVELET SETTINGS
# ---------------------------------------------------------------------
dt = 1.0        # one day per sample
dj = 0.25
s0 = 2.0 * dt
mother = "MORLET"
param = 6       # Morlet omega0 = 6
pad = 1

j1 = int(np.floor(np.log2(len(f107) * dt / s0) / dj))

print("\n========================================")
print("WAVELET PARAMETERS")
print("========================================")
print(f"Sampling interval dt: {dt:g} day")
print(f"Scale spacing dj: {dj}")
print(f"Smallest scale s0: {s0:g} days")
print(f"j1: {j1}")
print(f"Mother wavelet: {mother}")
print(f"Morlet omega0: {param}")


# ---------------------------------------------------------------------
# 4. WAVELET TRANSFORM AND POWER
# ---------------------------------------------------------------------
wave, period, scale, coi = wavelet(
    f107, dt, pad, dj, s0, j1, mother, param
)

power = np.abs(wave) ** 2

if power.shape != (len(period), len(f107)):
    raise ValueError(
        f"Unexpected wavelet power shape {power.shape}; expected "
        f"({len(period)}, {len(f107)}). Check waveletFunctions.py."
    )

print("\nWavelet transform completed.")
print(f"Wavelet power shape: {power.shape}")
print(f"Period range: {period.min():.3f} to {period.max():.3f} days")


# ---------------------------------------------------------------------
# 5. 95% SIGNIFICANCE
# ---------------------------------------------------------------------
# F10.7 is used as supplied; no mean subtraction is performed.
lag1 = float(np.corrcoef(f107[:-1], f107[1:])[0, 1])
if not np.isfinite(lag1):
    raise ValueError("Could not calculate lag-1 autocorrelation.")

signif = wave_signif(
    f107,
    dt=dt,
    scale=scale,
    sigtest=0,
    lag1=lag1,
    siglvl=0.95,
    dof=-1,
    mother=mother,
    param=param,
)
signif = np.asarray(signif).reshape(-1)

if len(signif) != len(period):
    raise ValueError(
        f"Significance array length {len(signif)} does not match "
        f"number of periods {len(period)}."
    )

significance_ratio = power / signif[:, np.newaxis]
log_power = np.log10(np.maximum(power, np.finfo(float).tiny))

print(f"Lag-1 autocorrelation: {lag1:.6f}")
print("95% significance spectrum calculated.")


# The classic waveletFunctions implementation returns COI in time units
# (days when dt=1), so do not multiply it by dt a second time.
coi_days = np.asarray(coi, dtype=float)


# ---------------------------------------------------------------------
# 6. SELECT DISPLAY PERIOD RANGE
# ---------------------------------------------------------------------
period_min = 2.0
period_max = 4000.0
period_mask = (period >= period_min) & (period <= period_max)

if not np.any(period_mask):
    raise ValueError("No wavelet periods fall within the selected display range.")

period_plot = period[period_mask]
log_power_plot = log_power[period_mask, :]
significance_ratio_plot = significance_ratio[period_mask, :]


# ---------------------------------------------------------------------
# 7. PLOT: LOG10 WAVELET POWER, SIGNIFICANCE, AND COI
# ---------------------------------------------------------------------
# Requested RGB-like progression: blue -> cyan -> green -> yellow -> red.
rgb_cmap = LinearSegmentedColormap.from_list(
    "solar_rgb",
    ["blue", "cyan", "lime", "yellow", "red"],
    N=256,
)

x = np.arange(len(time))

fig, ax = plt.subplots(figsize=(16, 9))

mesh = ax.contourf(
    x,
    period_plot,
    log_power_plot,
    levels=50,
    cmap=rgb_cmap,
    extend="both",
)

# Black contour where wavelet power reaches the 95% significance level.
ax.contour(
    x,
    period_plot,
    significance_ratio_plot,
    levels=[1.0],
    colors="black",
    linewidths=1.5,
)

# COI boundary. The curve marks the edge-effect limit.
ax.plot(
    x,
    coi_days,
    color="white",
    linewidth=2,
    linestyle="--",
    label="Cone of Influence (COI)",
)

# Shade periods longer than the COI boundary, where edge effects matter.
ax.fill_between(
    x,
    coi_days,
    period_max,
    color="white",
    alpha=0.35,
    linewidth=0,
)

ax.set_yscale("log")
# Match the earlier plot: short periods at the top, long periods at the bottom.
ax.set_ylim(period_min, period_max)

period_ticks = [
    tick for tick in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]
    if period_min <= tick <= period_max
]
ax.set_yticks(period_ticks)
ax.set_yticklabels([str(tick) for tick in period_ticks])

ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Period (days)", fontsize=13)
ax.set_title(
    "Wavelet Power Spectrum of Daily F10.7 Solar Index\n"
    f"{time[0].year}–{time[-1].year}",
    fontsize=16,
    pad=15,
)

# Label every two years using the actual daily date index.
tick_positions = []
tick_labels = []
for year in range(int(time[0].year), int(time[-1].year) + 1, 2):
    location = np.flatnonzero((time.year == year) & (time.month == 1) & (time.day == 1))
    if location.size:
        tick_positions.append(int(location[0]))
        tick_labels.append(str(year))

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha="right")
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)

cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
cbar.set_label(r"$\log_{10}$(Wavelet Power)", fontsize=12)

ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()

figure_path = FIGURES_DIR / "04a_f107_wavelet_1979_2010.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved figure: {figure_path}")


# ---------------------------------------------------------------------
# 8. SAVE GLOBAL WAVELET SPECTRUM
# ---------------------------------------------------------------------
# Mean power across time at each period. Save original power, not log power.
global_power = np.mean(power, axis=1)
global_significance = signif

global_df = pd.DataFrame({
    "period_days": period,
    "global_wavelet_power": global_power,
    "significance_95": global_significance,
    "above_95_percent_significance": global_power >= global_significance,
})
table_path = TABLES_DIR / "04a_f107_wavelet_global_spectrum.csv"
global_df.to_csv(table_path, index=False)

print(f"Saved global wavelet spectrum: {table_path}")
print("\nF10.7 wavelet analysis complete.")
