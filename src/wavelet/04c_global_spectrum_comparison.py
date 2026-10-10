"""
03. COMPARISON OF PERIODICITIES
F10.7 Solar Index vs CMAM Temperature Perturbation
Assignment 4

Comparison interval: January 2005–June 2010.
F10.7: daily, dt=1 day; original F10.7 values are used (no mean subtraction).
Temperature: monthly perturbation; dt=365.2425/12 days.
Morlet wavelet omega0=6. Global wavelet power is compared over 60–730 days.

Expected location:
    src/solar/03_compare_temperature_f107.py

Inputs:
    data/raw/f107/F107_daily_1979_2010.csv
    data/processed/temperature_perturbation.csv

Outputs:
    results/figures/03_periodicity_comparison_2005_2010.png
    results/tables/03_periodicity_comparison_global_spectra.csv
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================
# This file belongs in <project_root>/src/solar/.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAVELET_DIR = PROJECT_ROOT / "wave_python"
F107_FILE = PROJECT_ROOT / "data" / "raw" / "f107" / "F107_daily_1979_2010.csv"
TEMPERATURE_FILE = PROJECT_ROOT / "data" / "processed" / "temperature_perturbation.csv"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

for required_path in (WAVELET_DIR, F107_FILE, TEMPERATURE_FILE):
    if not required_path.exists():
        raise FileNotFoundError(
            f"Required project item was not found:\n{required_path}\n\n"
            "Check the project folder layout and the input filenames."
        )

sys.path.insert(0, str(WAVELET_DIR))
from waveletFunctions import wavelet

print("Wavelet library imported successfully.")


# ============================================================
# 2. SETTINGS
# ============================================================
COMMON_START = pd.Timestamp("2005-01-01")
COMMON_END = pd.Timestamp("2010-06-30")

DJ = 0.25
PAD = 1
MOTHER = "MORLET"
MORLET_OMEGA0 = 6

PERIOD_MIN_DAYS = 60.0
PERIOD_MAX_DAYS = 730.0
DAYS_PER_MONTH = 365.2425 / 12.0


# ============================================================
# 3. LOAD AND VALIDATE F10.7 DATA
# ============================================================
f107_df = pd.read_csv(F107_FILE)
f107_df.columns = [str(c).strip().lower() for c in f107_df.columns]

if "date" not in f107_df.columns:
    raise ValueError(f"F10.7 CSV must contain 'date'. Found: {list(f107_df.columns)}")

f107_col = next(
    (c for c in ("f107", "f10.7", "f10_7", "adjusted_f107", "f107_index")
     if c in f107_df.columns),
    None,
)
if f107_col is None:
    raise ValueError(
        "Could not identify the F10.7 column. "
        f"Available columns: {list(f107_df.columns)}"
    )

f107_df = f107_df[["date", f107_col]].rename(columns={f107_col: "f107"})
f107_df["date"] = pd.to_datetime(f107_df["date"], errors="coerce")
f107_df["f107"] = pd.to_numeric(f107_df["f107"], errors="coerce")
f107_df = f107_df.dropna(subset=["date"])
f107_df["date"] = f107_df["date"].dt.floor("D")
f107_df = f107_df.sort_values("date").drop_duplicates("date", keep="last")
f107_df.loc[
    (~np.isfinite(f107_df["f107"])) |
    (f107_df["f107"] <= 0) |
    (f107_df["f107"] >= 400),
    "f107"
] = np.nan

f107_common = f107_df[
    (f107_df["date"] >= COMMON_START) &
    (f107_df["date"] <= COMMON_END)
].copy()

if f107_common.empty:
    raise ValueError("No F10.7 observations found in the common date interval.")

# Reindex to a complete daily grid so the wavelet transform receives regular
# daily samples. Only short internal gaps (up to 5 days) are interpolated.
f107_common = f107_common.set_index("date")["f107"]
daily_dates = pd.date_range(COMMON_START, COMMON_END, freq="D")
f107_daily = f107_common.reindex(daily_dates)
missing_daily = int(f107_daily.isna().sum())

if missing_daily:
    is_missing = f107_daily.isna()
    group_id = is_missing.ne(is_missing.shift()).cumsum()
    gap_lengths = is_missing.groupby(group_id).sum()
    longest_gap = int(gap_lengths.max()) if not gap_lengths.empty else 0
    if longest_gap > 5:
        raise ValueError(
            f"F10.7 common-period series has a missing gap of {longest_gap} days. "
            "Inspect the source data rather than automatically filling a long gap."
        )
    f107_daily = f107_daily.interpolate(method="time", limit=5, limit_area="inside")

if f107_daily.isna().any():
    raise ValueError(
        "Missing F10.7 values remain at the start/end or after interpolation. "
        "Check the source CSV."
    )

f107 = f107_daily.to_numpy(dtype=float)
f107_dates = f107_daily.index

# Per assignment instruction: do not subtract the F10.7 mean.
print("\nF10.7 common-period data")
print(f"  Date range: {f107_dates[0].date()} to {f107_dates[-1].date()}")
print(f"  Daily samples: {len(f107)}")
print(f"  Missing values filled: {missing_daily}")
print(f"  Mean: {np.mean(f107):.4f}; minimum: {np.min(f107):.4f}; maximum: {np.max(f107):.4f}")
print("  Mean subtraction: NOT performed")


# ============================================================
# 4. LOAD AND VALIDATE TEMPERATURE PERTURBATION
# ============================================================
temperature_df = pd.read_csv(TEMPERATURE_FILE)
temperature_df.columns = [str(c).strip() for c in temperature_df.columns]

if "date" not in temperature_df.columns:
    raise ValueError(
        f"Temperature CSV must contain 'date'. Found: {list(temperature_df.columns)}"
    )

temperature_candidates = (
    "temperature_perturbation_K",
    "temperature_perturbation",
)
temperature_col = next(
    (c for c in temperature_candidates if c in temperature_df.columns),
    None,
)
if temperature_col is None:
    raise ValueError(
        "Could not find the temperature perturbation column. "
        f"Expected one of {temperature_candidates}; found {list(temperature_df.columns)}"
    )

temperature_df["date"] = pd.to_datetime(temperature_df["date"], errors="coerce")
temperature_df[temperature_col] = pd.to_numeric(
    temperature_df[temperature_col], errors="coerce"
)
temperature_df = temperature_df.dropna(subset=["date", temperature_col])
temperature_df = (
    temperature_df.sort_values("date")
    .drop_duplicates("date", keep="last")
    .reset_index(drop=True)
)
temperature_common = temperature_df[
    (temperature_df["date"] >= COMMON_START) &
    (temperature_df["date"] <= COMMON_END)
].copy()

if len(temperature_common) < 24:
    raise ValueError(
        f"Only {len(temperature_common)} monthly temperature samples found "
        "in the common period; expected approximately 66."
    )

temperature_dates = pd.DatetimeIndex(temperature_common["date"])
temperature = temperature_common[temperature_col].to_numpy(dtype=float)

if not np.all(np.isfinite(temperature)):
    raise ValueError("Temperature perturbation contains non-finite values.")

# Validate continuous monthly spacing.
month_labels = temperature_dates.to_period("M")
expected_months = pd.period_range(
    month_labels[0], month_labels[-1], freq="M"
)
if not np.array_equal(month_labels.astype(str), expected_months.astype(str)):
    raise ValueError("Temperature data are not a continuous monthly series.")

print("\nTemperature common-period data")
print(f"  Date range: {temperature_dates[0].date()} to {temperature_dates[-1].date()}")
print(f"  Monthly samples: {len(temperature)}")
print(f"  Input column: {temperature_col}")
print(
    f"  Mean perturbation: {np.mean(temperature):.6f} K; "
    f"minimum: {np.min(temperature):.4f} K; maximum: {np.max(temperature):.4f} K"
)
print("  Perturbation mean is not subtracted again.")


# ============================================================
# 5. WAVELET PARAMETERS
# ============================================================
dt_f107 = 1.0
s0_f107 = 2.0 * dt_f107
j1_f107 = int(np.floor(np.log2(len(f107) * dt_f107 / s0_f107) / DJ))

dt_temperature = DAYS_PER_MONTH
s0_temperature = 2.0 * dt_temperature
j1_temperature = int(
    np.floor(np.log2(len(temperature) * dt_temperature / s0_temperature) / DJ)
)

print("\nWavelet parameters")
print(f"  Mother wavelet: {MOTHER}; Morlet omega0={MORLET_OMEGA0}; dj={DJ}")
print(f"  F10.7: dt={dt_f107:.6f} day, s0={s0_f107:.6f}, j1={j1_f107}")
print(
    f"  Temperature: dt={dt_temperature:.6f} days, "
    f"s0={s0_temperature:.6f}, j1={j1_temperature}"
)


# ============================================================
# 6. WAVELET TRANSFORMS
# ============================================================
wave_f107, period_f107, scale_f107, coi_f107 = wavelet(
    f107, dt_f107, PAD, DJ, s0_f107, j1_f107, MOTHER, MORLET_OMEGA0
)
wave_temperature, period_temperature, scale_temperature, coi_temperature = wavelet(
    temperature, dt_temperature, PAD, DJ, s0_temperature,
    j1_temperature, MOTHER, MORLET_OMEGA0
)

power_f107 = np.abs(wave_f107) ** 2
power_temperature = np.abs(wave_temperature) ** 2

print("\nWavelet transforms completed")
print(f"  F10.7 power shape: {power_f107.shape}")
print(f"  Temperature power shape: {power_temperature.shape}")


# ============================================================
# 7. GLOBAL WAVELET POWER AND COMMON PERIOD RANGE
# ============================================================
# Average ORIGINAL power over time first. Do not normalize log(power) by
# its maximum: log values can be negative and that is not a meaningful
# relative-power normalization.
global_power_f107 = np.mean(power_f107, axis=1)
global_power_temperature = np.mean(power_temperature, axis=1)

mask_f107 = (
    (period_f107 >= PERIOD_MIN_DAYS) &
    (period_f107 <= PERIOD_MAX_DAYS)
)
mask_temperature = (
    (period_temperature >= PERIOD_MIN_DAYS) &
    (period_temperature <= PERIOD_MAX_DAYS)
)

period_f107_display = np.asarray(period_f107[mask_f107], dtype=float)
period_temperature_display = np.asarray(period_temperature[mask_temperature], dtype=float)
f107_global_display = np.asarray(global_power_f107[mask_f107], dtype=float)
temperature_global_display = np.asarray(global_power_temperature[mask_temperature], dtype=float)

if not len(period_f107_display) or not len(period_temperature_display):
    raise ValueError("No global-spectrum periods found within 60–730 days.")

# Normalize each ORIGINAL global-power spectrum by its own maximum within
# the selected range. This compares relative periodicity strength despite
# different physical units and sampling intervals.
if np.max(f107_global_display) <= 0 or np.max(temperature_global_display) <= 0:
    raise ValueError("Global wavelet power must be positive for normalization.")

f107_normalized = f107_global_display / np.max(f107_global_display)
temperature_normalized = (
    temperature_global_display / np.max(temperature_global_display)
)

f107_peak_index = int(np.argmax(f107_normalized))
temperature_peak_index = int(np.argmax(temperature_normalized))
f107_dominant_period = float(period_f107_display[f107_peak_index])
temperature_dominant_period = float(period_temperature_display[temperature_peak_index])
period_difference = abs(f107_dominant_period - temperature_dominant_period)

print("\nDominant periods in selected 60–730 day range")
print(
    f"  F10.7: {f107_dominant_period:.2f} days "
    f"({f107_dominant_period / DAYS_PER_MONTH:.2f} months)"
)
print(
    f"  Temperature: {temperature_dominant_period:.2f} days "
    f"({temperature_dominant_period / DAYS_PER_MONTH:.2f} months)"
)
print(f"  Absolute difference: {period_difference:.2f} days")
print(
    "  Interpretation: different dominant periods do not establish or rule out "
    "a causal solar influence."
)


# ============================================================
# 8. SAVE GLOBAL SPECTRA
# ============================================================
f107_spectrum_df = pd.DataFrame({
    "period_days": period_f107_display,
    "f107_global_power": f107_global_display,
    "f107_normalized_global_power": f107_normalized,
})
temperature_spectrum_df = pd.DataFrame({
    "period_days": period_temperature_display,
    "temperature_global_power": temperature_global_display,
    "temperature_normalized_global_power": temperature_normalized,
})

spectrum_table = pd.merge(
    f107_spectrum_df,
    temperature_spectrum_df,
    on="period_days",
    how="outer",
).sort_values("period_days")

# Keep each spectrum's native period grid; values are not interpolated onto
# a shared grid because their sampling intervals differ.
spectrum_file = TABLES_DIR / "03_periodicity_comparison_global_spectra.csv"
spectrum_table.to_csv(spectrum_file, index=False)
print(f"\nSaved global spectra table: {spectrum_file}")


# ============================================================
# 9. PLOT COMPARISON
# ============================================================
fig, ax = plt.subplots(figsize=(13, 7))

ax.plot(
    period_f107_display,
    f107_normalized,
    linewidth=2,
    label="F10.7 Solar Index",
)
ax.plot(
    period_temperature_display,
    temperature_normalized,
    linewidth=2,
    label="CMAM Temperature Perturbation",
)

ax.axvline(
    365.2425,
    linestyle=":",
    linewidth=1.5,
    label="365-day period",
)
ax.axvline(
    f107_dominant_period,
    linestyle="--",
    linewidth=1,
    alpha=0.7,
)
ax.axvline(
    temperature_dominant_period,
    linestyle="--",
    linewidth=1,
    alpha=0.7,
)

ax.set_xlabel("Period (days)", fontsize=12)
ax.set_ylabel("Normalized Global Wavelet Power", fontsize=12)
ax.set_title(
    "Comparison of Periodicities: F10.7 and CMAM Temperature Perturbation\n"
    "Common interval: January 2005–June 2010",
    fontsize=14,
    pad=12,
)
ax.set_xlim(PERIOD_MIN_DAYS, PERIOD_MAX_DAYS)
ax.set_xticks([60, 90, 120, 180, 240, 365, 540, 730])
ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.3)
ax.legend(loc="best", framealpha=0.9)
fig.tight_layout()

figure_file = FIGURES_DIR / "04c_periodicity_comparison_2005_2010.png"
fig.savefig(figure_file, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved comparison figure: {figure_file}")
print("\nPeriodicity comparison completed.")
