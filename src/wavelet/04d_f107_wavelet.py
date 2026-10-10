"""
Wavelet analysis of daily F10.7 solar index, January 2000–December 2005.

Place this script at:
    src/wavelet/04f_f107_wavelet_2000_2005.py

Input:
    data/raw/f107/F107_daily_1979_2010.csv
    wave_python/waveletFunctions.py

Outputs:
    results/figures/04f_f107_wavelet_2000_2005.png
    results/tables/04f_f107_wavelet_global_spectrum_2000_2005.csv

Method follows the original F10.7 script:
- Filter the original daily series to 2000-01-01 through 2005-12-31.
- Do NOT subtract the F10.7 mean.
- Morlet wavelet with omega0 = 6.
- 95% red-noise significance level.
- Plot the cone of influence (COI).
- Period axis is in days.
- Display wavelet power as log10(power).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ============================================================
# 1. FIND PROJECT FOLDER AND SET PATHS
# ============================================================
# This script is inside <project>/src/wavelet/, so parents[2] is project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

WAVELET_DIR = PROJECT_ROOT / "wave_python"
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

if not WAVELET_DIR.is_dir():
    raise FileNotFoundError(f"Wavelet library folder not found: {WAVELET_DIR}")
if not DATA_FILE.is_file():
    raise FileNotFoundError(f"F10.7 input CSV not found: {DATA_FILE}")

sys.path.insert(0, str(WAVELET_DIR))
from waveletFunctions import wavelet, wave_signif

print("Wavelet library imported successfully!")


# ============================================================
# 2. LOAD DAILY F10.7 DATA
# ============================================================
df = pd.read_csv(DATA_FILE)
df.columns = [str(col).strip().lower() for col in df.columns]

if "date" not in df.columns or "f107" not in df.columns:
    raise ValueError(
        "Input CSV must contain 'date' and 'f107' columns. "
        f"Found columns: {list(df.columns)}"
    )

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["f107"] = pd.to_numeric(df["f107"], errors="coerce")
df = df.dropna(subset=["date", "f107"]).sort_values("date")
df = df.drop_duplicates(subset="date", keep="last")

# Extract exactly six calendar years: 1 Jan 2000 through 31 Dec 2005 inclusive.
start_date = pd.Timestamp("2000-01-01")
end_date = pd.Timestamp("2005-12-31")
df = df.loc[(df["date"] >= start_date) & (df["date"] < end_date + pd.Timedelta(days=1))].copy()
df = df.reset_index(drop=True)

if df.empty:
    raise ValueError("No F10.7 observations were found between 2000 and 2005.")

# A wavelet transform assumes regular sampling. Reindex to a daily calendar and
# fill only short internal gaps (up to 10 consecutive days), without mean removal.
df = df.set_index("date")
daily_index = pd.date_range(start=start_date, end=end_date, freq="D")
daily = df["f107"].reindex(daily_index)

missing_count = int(daily.isna().sum())
if missing_count:
    missing_mask = daily.isna()
    groups = (missing_mask != missing_mask.shift()).cumsum()
    gap_lengths = missing_mask.groupby(groups).sum()
    longest_gap = int(gap_lengths.max()) if len(gap_lengths) else 0
    if longest_gap > 10:
        raise ValueError(
            f"Found a gap of {longest_gap} consecutive days in 2000–2005. "
            "Inspect the source data rather than automatically filling a long gap."
        )
    daily = daily.interpolate(method="time", limit=10, limit_area="inside")

if daily.isna().any():
    raise ValueError(
        "Missing values remain at the start/end or inside the selected period. "
        "Check the source CSV before running the wavelet analysis."
    )

time = daily.index
f107 = daily.to_numpy(dtype=float)

if len(f107) < 100:
    raise ValueError(f"Too few daily samples ({len(f107)}) for wavelet analysis.")

print("\n========================================")
print("F10.7 DATA INFORMATION — SELECTED PERIOD")
print("========================================")
print(f"Input file: {DATA_FILE}")
print(f"Time range: {time[0].date()} to {time[-1].date()}")
print(f"Daily observations: {len(f107):,}")
print(f"Missing daily values filled: {missing_count}")
print(f"Mean F10.7: {np.mean(f107):.4f}")
print(f"Minimum F10.7: {np.min(f107):.4f}")
print(f"Maximum F10.7: {np.max(f107):.4f}")
print("Mean subtraction: NOT PERFORMED")


# ============================================================
# 3. SAMPLING INTERVAL AND WAVELET PARAMETERS
# ============================================================
dt = 1.0       # daily sampling, in days
pad = 1
dj = 0.25
s0 = 2.0 * dt
j1 = int(np.floor(np.log2(len(f107) * dt / s0) / dj))
mother = "MORLET"
param = 6      # Morlet omega0 = 6

print("\n========================================")
print("WAVELET PARAMETERS")
print("========================================")
print(f"Mother wavelet: {mother}")
print(f"Morlet parameter omega0: {param}")
print(f"dt: {dt} day")
print(f"dj: {dj}")
print(f"s0: {s0}")
print(f"j1: {j1}")


# ============================================================
# 4. PERFORM WAVELET TRANSFORM
# ============================================================
wave, period, scale, coi = wavelet(
    f107, dt, pad, dj, s0, j1, mother, param
)
power = np.abs(wave) ** 2

if power.shape != (len(period), len(f107)):
    raise ValueError(
        f"Unexpected wavelet power shape {power.shape}; expected "
        f"({len(period)}, {len(f107)})."
    )

print("\nWavelet transform completed.")
print(f"Wavelet shape: {wave.shape}")
print(f"Number of periods: {len(period)}")
print(f"Period range: {period.min():.3f}–{period.max():.3f} days")


# ============================================================
# 5. CALCULATE 95% SIGNIFICANCE
# ============================================================
lag1 = float(np.corrcoef(f107[:-1], f107[1:])[0, 1])
if not np.isfinite(lag1):
    raise ValueError("Could not calculate lag-1 autocorrelation.")

signif = np.asarray(
    wave_signif(
        f107,
        dt=dt,
        scale=scale,
        sigtest=0,
        lag1=lag1,
        siglvl=0.95,
        dof=-1,
        mother=mother,
        param=param,
    ),
    dtype=float,
).squeeze()

if signif.ndim == 0:
    signif = np.full(len(period), float(signif))
elif signif.size == len(period):
    signif = signif.reshape(-1)
else:
    raise ValueError(
        f"Unexpected significance shape {signif.shape} for {len(period)} periods."
    )

if np.any(~np.isfinite(signif)) or np.any(signif <= 0):
    raise ValueError("Significance values must be finite and positive.")

significance_ratio = power / signif[:, np.newaxis]

# Logarithm is used only for display; significance is calculated on original power.
log_power = np.log10(np.maximum(power, np.finfo(float).tiny))

# Classic waveletFunctions returns COI in time units. dt=1 day here.
coi_days = np.asarray(coi, dtype=float).squeeze()
if coi_days.size != len(f107):
    raise ValueError(f"COI length {coi_days.size} does not match data length {len(f107)}.")

print(f"Lag-1 autocorrelation: {lag1:.6f}")
print("95% significance level calculated.")


# ============================================================
# 6. SELECT DISPLAY PERIOD RANGE
# ============================================================
period_min = 2.0
period_max = 2000.0
period_mask = (period >= period_min) & (period <= period_max)

if not np.any(period_mask):
    raise ValueError("No calculated periods fall in the selected display range.")

period_plot = period[period_mask]
log_power_plot = log_power[period_mask, :]
significance_ratio_plot = significance_ratio[period_mask, :]


# ============================================================
# 7. PLOT WAVELET POWER, SIGNIFICANCE, AND COI
# ============================================================
rgb_cmap = LinearSegmentedColormap.from_list(
    "RGB_wavelet",
    ["blue", "cyan", "lime", "yellow", "red"],
    N=256,
)

x = np.arange(len(time))
fig, ax = plt.subplots(figsize=(16, 9))

power_plot = ax.contourf(
    x,
    period_plot,
    log_power_plot,
    levels=50,
    cmap=rgb_cmap,
    extend="both",
)

# Black contour where power/significance = 1.
ax.contour(
    x,
    period_plot,
    significance_ratio_plot,
    levels=[1.0],
    colors="black",
    linewidths=1.5,
)

# COI is plotted in days, not multiplied by dt a second time.
ax.plot(
    x,
    coi_days,
    color="white",
    linewidth=2,
    linestyle="--",
    label="Cone of Influence (COI)",
)

# Shade periods LONGER than the COI boundary, where edge effects matter.
# This matches the original script's ascending period axis.
ax.fill_between(
    x,
    coi_days,
    period_max,
    color="white",
    alpha=0.35,
    linewidth=0,
)

# Preserve the original script's axis convention: short periods at bottom,
# long periods at top.
ax.set_yscale("log")
ax.set_ylim(period_min, period_max)

period_ticks = [
    tick for tick in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    if period_min <= tick <= period_max
]
ax.set_yticks(period_ticks)
ax.set_yticklabels([str(tick) for tick in period_ticks])

ax.set_xlabel("Year", fontsize=13)
ax.set_ylabel("Period (days)", fontsize=13)
ax.set_title(
    "Wavelet Power Spectrum of Daily F10.7 Solar Index\n"
    "January 2000 – December 2005",
    fontsize=16,
    pad=15,
)

tick_positions = []
tick_labels = []
for year in range(2000, 2006):
    matching = np.flatnonzero((time.year == year) & (time.month == 1) & (time.day == 1))
    if matching.size:
        tick_positions.append(int(matching[0]))
        tick_labels.append(str(year))

ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels)
ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.25)

cbar = fig.colorbar(power_plot, ax=ax, pad=0.02)
cbar.set_label(r"$\log_{10}$(Wavelet Power)", fontsize=12)

ax.legend(loc="upper right", framealpha=0.9)
fig.tight_layout()

figure_path = FIGURES_DIR / "04d_f107_wavelet_2000_2005.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved figure: {figure_path}")


# ============================================================
# 8. SAVE GLOBAL WAVELET SPECTRUM
# ============================================================
global_power = np.mean(power, axis=1)
global_table = pd.DataFrame(
    {
        "period_days": period,
        "global_wavelet_power": global_power,
        "significance_95": signif,
        "above_95_percent_significance": global_power >= signif,
    }
)
table_path = TABLES_DIR / "04d_f107_wavelet_global_spectrum_2000_2005.csv"
global_table.to_csv(table_path, index=False)

print(f"Saved global wavelet spectrum: {table_path}")
print("\nF10.7 wavelet analysis for 2000–2005 complete.")
