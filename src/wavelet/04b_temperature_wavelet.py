"""
Assignment 4: Wavelet analysis of monthly CMAM temperature perturbations.

Place this script at:
    src/wavelet/04b_temperature_wavelet.py

Input:
    data/processed/temperature_perturbation.csv
    (created by src/temperature/02_prepare_monthly_series.py)

Expected CSV columns:
    date
    temperature_perturbation_K
The older column name "temperature_perturbation" is also accepted.

Outputs:
    results/figures/02_temperature_wavelet_2005_2010.png
    results/tables/02_temperature_wavelet_global_spectrum.csv

Method:
    - Monthly observations; dt = 365.2425 / 12 days.
    - Use the supplied perturbation directly (do not subtract its mean again).
    - Morlet wavelet with omega0 = 6.
    - 95% red-noise significance and cone of influence (COI).
    - Actual wavelet power with logarithmic colour normalization.
    - Period axis in days, with short periods at the top and long periods
      at the bottom, matching the earlier wavelet plot convention.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.ticker import LogLocator, LogFormatterMathtext


# ---------------------------------------------------------------------
# 1. PROJECT PATHS
# ---------------------------------------------------------------------
# Script is expected inside <project_root>/src/wavelet/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAVELET_DIR = PROJECT_ROOT / "wave_python"
INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "temperature_perturbation.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

if not WAVELET_DIR.is_dir():
    raise FileNotFoundError(
        f"Wavelet library folder not found:\n{WAVELET_DIR}\n"
        "Expected wave_python/ at the project root."
    )
if not INPUT_FILE.is_file():
    raise FileNotFoundError(
        f"Temperature perturbation CSV not found:\n{INPUT_FILE}\n"
        "Run src/temperature/02_prepare_monthly_series.py first."
    )

sys.path.insert(0, str(WAVELET_DIR))
try:
    from waveletFunctions import wavelet, wave_signif
except ImportError as exc:
    raise ImportError(
        f"Could not import waveletFunctions from {WAVELET_DIR}"
    ) from exc

print("Wavelet library imported successfully.")


# ---------------------------------------------------------------------
# 2. LOAD AND VALIDATE MONTHLY TEMPERATURE DATA
# ---------------------------------------------------------------------
df = pd.read_csv(INPUT_FILE)
df.columns = [str(column).strip() for column in df.columns]

if "date" not in df.columns:
    raise ValueError(
        f"CSV must contain a 'date' column. Found: {list(df.columns)}"
    )

# Support both the new column name and the older name used in previous files.
candidate_columns = [
    "temperature_perturbation_K",
    "temperature_perturbation",
]
temperature_column = next(
    (column for column in candidate_columns if column in df.columns),
    None,
)
if temperature_column is None:
    raise ValueError(
        "Could not find the temperature perturbation column. Expected one of "
        f"{candidate_columns}; found columns: {list(df.columns)}"
    )

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df[temperature_column] = pd.to_numeric(
    df[temperature_column], errors="coerce"
)
df = df.dropna(subset=["date", temperature_column])
df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

if len(df) < 24:
    raise ValueError(
        f"Only {len(df)} valid monthly samples were found. "
        "Check the CSV before performing wavelet analysis."
    )

# Check for consecutive monthly observations. The CMAM series should contain
# January 2005 through June 2010 (66 samples).
month_index = df["date"].dt.to_period("M")
expected_months = pd.period_range(
    start=month_index.iloc[0],
    end=month_index.iloc[-1],
    freq="M",
)
if not np.array_equal(month_index.astype(str).to_numpy(), expected_months.astype(str)):
    raise ValueError(
        "The temperature series is not a continuous monthly series. "
        "Check for missing or duplicated months before wavelet analysis."
    )

dates = pd.DatetimeIndex(df["date"])
temperature = df[temperature_column].to_numpy(dtype=float)

if not np.all(np.isfinite(temperature)):
    raise ValueError("Temperature perturbation contains non-finite values.")

print("\n" + "=" * 55)
print("CMAM TEMPERATURE PERTURBATION DATA")
print("=" * 55)
print(f"Input file: {INPUT_FILE}")
print(df.head())
print(f"Observations: {len(temperature)}")
print(f"Time range: {dates[0].date()} to {dates[-1].date()}")
print(f"Column used: {temperature_column}")
print(f"Mean perturbation: {np.mean(temperature):.8f} K")
print(f"Minimum perturbation: {np.min(temperature):.4f} K")
print(f"Maximum perturbation: {np.max(temperature):.4f} K")


# ---------------------------------------------------------------------
# 3. SAMPLING INTERVAL AND WAVELET PARAMETERS
# ---------------------------------------------------------------------
# Monthly data must not be treated as daily observations.
dt = 365.2425 / 12.0
pad = 1
dj = 0.25
s0 = 2.0 * dt
mother = "MORLET"
param = 6  # Morlet omega0

j1 = int(np.floor(np.log2(len(temperature) * dt / s0) / dj))

print("\n" + "=" * 55)
print("WAVELET PARAMETERS")
print("=" * 55)
print(f"Sampling: monthly")
print(f"dt: {dt:.6f} days")
print(f"dj: {dj}")
print(f"s0: {s0:.6f} days")
print(f"j1: {j1}")
print(f"Mother wavelet: {mother}")
print(f"Morlet omega0: {param}")


# ---------------------------------------------------------------------
# 4. WAVELET TRANSFORM AND POWER
# ---------------------------------------------------------------------
wave, period, scale, coi = wavelet(
    temperature, dt, pad, dj, s0, j1, mother, param
)
power = np.abs(wave) ** 2

if power.shape != (len(period), len(temperature)):
    raise ValueError(
        f"Unexpected wavelet shape {power.shape}; expected "
        f"({len(period)}, {len(temperature)})."
    )

print("\nWavelet transform completed.")
print(f"Wavelet shape: {wave.shape}")
print(f"Period range: {period.min():.3f} to {period.max():.3f} days")


# ---------------------------------------------------------------------
# 5. 95% SIGNIFICANCE TEST
# ---------------------------------------------------------------------
lag1 = float(np.corrcoef(temperature[:-1], temperature[1:])[0, 1])
if not np.isfinite(lag1):
    raise ValueError("Could not estimate lag-1 autocorrelation.")

signif = np.asarray(
    wave_signif(
        temperature,
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

# For sigtest=0, Torrence & Compo implementations normally return one
# significance threshold per scale or a single scalar. Handle either form.
if signif.ndim == 0:
    significance_by_period = np.full(len(period), float(signif))
elif signif.size == len(period):
    significance_by_period = signif.reshape(-1)
else:
    raise ValueError(
        "Unexpected significance output shape "
        f"{np.shape(signif)} for {len(period)} periods."
    )

if np.any(~np.isfinite(significance_by_period)) or np.any(significance_by_period <= 0):
    raise ValueError("Significance thresholds must be finite and positive.")

significance_ratio = power / significance_by_period[:, np.newaxis]

print(f"Lag-1 autocorrelation: {lag1:.6f}")
print("95% significance thresholds calculated.")


# The classic waveletFunctions implementation returns COI in time units.
# Since dt is expressed in days, the returned COI is already in days.
coi_days = np.asarray(coi, dtype=float).squeeze()
if coi_days.size != len(temperature):
    raise ValueError(
        f"COI length {coi_days.size} does not match sample count {len(temperature)}."
    )


# ---------------------------------------------------------------------
# 6. SELECT PERIODS FOR DISPLAY
# ---------------------------------------------------------------------
period_min = 60.0
period_max = 730.0
period_mask = (period >= period_min) & (period <= period_max)

if not np.any(period_mask):
    raise ValueError("No wavelet periods fall within the requested 60–730 day range.")

period_plot = period[period_mask]
power_plot = power[period_mask, :]
significance_ratio_plot = significance_ratio[period_mask, :]
significance_plot = significance_by_period[period_mask]


# ---------------------------------------------------------------------
# 7. PLOT WAVELET POWER, SIGNIFICANCE, AND COI
# ---------------------------------------------------------------------
positive_power = power_plot[np.isfinite(power_plot) & (power_plot > 0)]
if positive_power.size == 0:
    raise ValueError("No positive wavelet power is available to plot.")

vmin = float(np.min(positive_power))
vmax = float(np.max(positive_power))
if np.isclose(vmin, vmax):
    vmax = vmin * (1.0 + 1e-6)

# Match the requested blue -> cyan -> green -> yellow -> red gradient.
rgb_cmap = LinearSegmentedColormap.from_list(
    "wavelet_rgb",
    ["blue", "cyan", "lime", "yellow", "red"],
    N=256,
)

fig, ax = plt.subplots(figsize=(14, 8))

power_plot_obj = ax.contourf(
    dates,
    period_plot,
    power_plot,
    levels=np.logspace(np.log10(vmin), np.log10(vmax), 50),
    norm=LogNorm(vmin=vmin, vmax=vmax),
    cmap=rgb_cmap,
    extend="both",
)

# Black contour marks power/significance = 1 (95% threshold).
ax.contour(
    dates,
    period_plot,
    significance_ratio_plot,
    levels=[1.0],
    colors="black",
    linewidths=1.2,
)

# COI boundary. In this plot, periods longer than the COI boundary
# are edge-effect affected. Keep COI values in days; do not multiply by dt.
ax.plot(
    dates,
    coi_days,
    color="white",
    linestyle="--",
    linewidth=1.8,
    label="Cone of Influence (COI)",
)

ax.fill_between(
    dates,
    coi_days,
    period_max,
    color="white",
    alpha=0.35,
    linewidth=0,
)

ax.set_yscale("log")
# Match the earlier wavelet figure: short periods at the TOP,
# long periods at the BOTTOM. This requires an inverted numeric y-axis.
ax.set_ylim(period_min, period_max)

period_ticks = [60, 90, 120, 180, 240, 365, 540, 730]
ax.set_yticks(period_ticks)
ax.set_yticklabels([str(value) for value in period_ticks])

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Period (days)", fontsize=12)
ax.set_title(
    "Wavelet Power Spectrum of CMAM Temperature Perturbation\n"
    "January 2005 – June 2010",
    fontsize=15,
    fontweight="bold",
    pad=12,
)

# Label January of each year on the x-axis.
year_ticks = pd.date_range(
    dates[0].to_period("Y").start_time,
    dates[-1].to_period("Y").start_time,
    freq="YS",
)
year_ticks = year_ticks[(year_ticks >= dates[0]) & (year_ticks <= dates[-1])]
ax.set_xticks(year_ticks)
ax.set_xticklabels([str(date.year) for date in year_ticks])

# Reference line for annual periodicity.
ax.axhline(
    365,
    color="white",
    linestyle=":",
    linewidth=1.3,
    label="365 days",
)

ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.25)

cbar = fig.colorbar(power_plot_obj, ax=ax, pad=0.02)
cbar.set_label(r"$\log_{10}$(Wavelet Power)", fontsize=11)
cbar.locator = LogLocator(base=10)
cbar.formatter = LogFormatterMathtext(base=10)
cbar.update_ticks()

ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
fig.tight_layout()

figure_path = FIGURES_DIR / "04b_temperature_wavelet_global_spectrum.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved figure: {figure_path}")


# ---------------------------------------------------------------------
# 8. SAVE GLOBAL WAVELET SPECTRUM
# ---------------------------------------------------------------------
global_power = np.mean(power, axis=1)
global_table = pd.DataFrame(
    {
        "period_days": period,
        "global_wavelet_power": global_power,
        "significance_95": significance_by_period,
        "above_95_percent_significance": global_power >= significance_by_period,
    }
)
table_path = TABLES_DIR / "04b_temperature_wavelet_global_spectrum.csv"
global_table.to_csv(table_path, index=False)

print(f"Saved global wavelet spectrum: {table_path}")
print("\nTemperature wavelet analysis complete.")
