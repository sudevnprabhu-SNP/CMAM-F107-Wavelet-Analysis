"""
3D wavelet-power comparison:
F10.7 solar radio flux vs CMAM temperature perturbation.

Save as:
    src/wavelet/04e_3d_wavelet_comparison.py

Inputs:
    data/raw/f107/F107_daily_1979_2010.csv
    data/processed/temperature_perturbation.csv
    wave_python/waveletFunctions.py

The CMAM temperature series available in this project covers January 2005
through June 2010. Therefore, the comparison uses that common interval.
Daily F10.7 data are averaged to monthly means before comparison, so both
signals have the same monthly sampling interval.

Important:
- F10.7 mean is NOT subtracted.
- Temperature perturbation is used as supplied; no extra de-meaning.
- Morlet wavelet, omega0=6.
- Power is normalized independently for visualization because the signals
  have different physical units. This is a relative-power comparison, not
  a comparison of absolute physical power.
- Periods are shown in days.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ============================================================
# 1. PROJECT ROOT AND WAVELET LIBRARY
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WAVELET_DIR = PROJECT_ROOT / "wave_python"

F107_FILE = PROJECT_ROOT / "data" / "raw" / "f107" / "F107_daily_1979_2010.csv"
TEMPERATURE_FILE = PROJECT_ROOT / "data" / "processed" / "temperature_perturbation.csv"

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

if not WAVELET_DIR.is_dir():
    raise FileNotFoundError(f"Wavelet library folder not found: {WAVELET_DIR}")
if not F107_FILE.is_file():
    raise FileNotFoundError(f"F10.7 CSV not found: {F107_FILE}")
if not TEMPERATURE_FILE.is_file():
    raise FileNotFoundError(f"Temperature perturbation CSV not found: {TEMPERATURE_FILE}")

sys.path.insert(0, str(WAVELET_DIR))
from waveletFunctions import wavelet

print("Wavelet library imported successfully!")


# ============================================================
# 2. LOAD AND VALIDATE F10.7 DAILY DATA
# ============================================================
f107_df = pd.read_csv(F107_FILE)
f107_df.columns = [str(c).strip().lower() for c in f107_df.columns]

if not {"date", "f107"}.issubset(f107_df.columns):
    raise ValueError(
        f"F10.7 CSV needs 'date' and 'f107' columns; found {list(f107_df.columns)}"
    )

f107_df["date"] = pd.to_datetime(f107_df["date"], errors="coerce")
f107_df["f107"] = pd.to_numeric(f107_df["f107"], errors="coerce")
f107_df = f107_df.dropna(subset=["date", "f107"]).sort_values("date")
f107_df = f107_df.drop_duplicates("date", keep="last")
f107_df = f107_df.loc[f107_df["f107"] > 0].copy()

# Daily source is converted to monthly means so it matches the monthly CMAM series.
f107_df["month"] = f107_df["date"].dt.to_period("M")
f107_monthly = (
    f107_df.groupby("month", as_index=False)["f107"]
    .mean()
)
f107_monthly["date"] = f107_monthly["month"].dt.to_timestamp()
f107_monthly = f107_monthly[["date", "f107"]]


# ============================================================
# 3. LOAD AND VALIDATE TEMPERATURE PERTURBATION
# ============================================================
temperature_df = pd.read_csv(TEMPERATURE_FILE)
temperature_df.columns = [str(c).strip() for c in temperature_df.columns]

if "date" not in temperature_df.columns:
    raise ValueError(
        f"Temperature CSV needs a 'date' column; found {list(temperature_df.columns)}"
    )

temperature_candidates = [
    "temperature_perturbation_K",
    "temperature_perturbation",
]
temperature_column = next(
    (c for c in temperature_candidates if c in temperature_df.columns), None
)
if temperature_column is None:
    raise ValueError(
        "Could not find temperature perturbation column. "
        f"Expected one of {temperature_candidates}; found {list(temperature_df.columns)}"
    )

temperature_df["date"] = pd.to_datetime(temperature_df["date"], errors="coerce")
temperature_df[temperature_column] = pd.to_numeric(
    temperature_df[temperature_column], errors="coerce"
)
temperature_df = temperature_df.dropna(
    subset=["date", temperature_column]
).sort_values("date")
temperature_df = temperature_df.drop_duplicates("date", keep="last")
temperature_df["month"] = temperature_df["date"].dt.to_period("M")
temperature_df["date"] = temperature_df["month"].dt.to_timestamp()
temperature_df = temperature_df[["date", temperature_column]].rename(
    columns={temperature_column: "temperature_perturbation"}
)


# ============================================================
# 4. MERGE COMMON MONTHS (JANUARY 2005–JUNE 2010)
# ============================================================
data = pd.merge(f107_monthly, temperature_df, on="date", how="inner")
data = data.loc[
    (data["date"] >= pd.Timestamp("2005-01-01"))
    & (data["date"] <= pd.Timestamp("2010-06-01"))
].sort_values("date").reset_index(drop=True)

if len(data) < 24:
    raise ValueError(
        f"Only {len(data)} common monthly observations were found. "
        "Check both input CSV files. At least 24 months are required here."
    )

expected_months = pd.period_range(
    data["date"].iloc[0].to_period("M"),
    data["date"].iloc[-1].to_period("M"),
    freq="M",
)
actual_months = data["date"].dt.to_period("M")
if not np.array_equal(actual_months.astype(str).to_numpy(), expected_months.astype(str).to_numpy()):
    raise ValueError(
        "The merged series has missing months. Check the source files before wavelet analysis."
    )

time = pd.DatetimeIndex(data["date"])
f107 = data["f107"].to_numpy(dtype=float)
temperature = data["temperature_perturbation"].to_numpy(dtype=float)

if not (np.isfinite(f107).all() and np.isfinite(temperature).all()):
    raise ValueError("Merged series contains non-finite values.")

print("\n========================================")
print("COMMON DATASET INFORMATION")
print("========================================")
print(f"F10.7 source: {F107_FILE}")
print(f"Temperature source: {TEMPERATURE_FILE}")
print(f"Common monthly observations: {len(data)}")
print(f"Common interval: {time[0].date()} to {time[-1].date()}")
print(f"F10.7 monthly mean range: {f107.min():.3f}–{f107.max():.3f}")
print(f"Temperature perturbation range: {temperature.min():.3f}–{temperature.max():.3f} K")
print("F10.7 mean subtraction: NOT PERFORMED")
print("Temperature re-centering: NOT PERFORMED")


# ============================================================
# 5. WAVELET PARAMETERS AND TRANSFORMS
# ============================================================
# Both input series are monthly; express sampling interval in days.
dt = 365.2425 / 12.0
pad = 1
dj = 0.25
s0 = 2.0 * dt
mother = "MORLET"
param = 6

j1 = int(np.floor(np.log2(len(data) * dt / s0) / dj))

print("\n========================================")
print("WAVELET PARAMETERS")
print("========================================")
print(f"dt: {dt:.6f} days/month")
print(f"dj: {dj}")
print(f"s0: {s0:.6f} days")
print(f"j1: {j1}")
print(f"Mother wavelet: {mother}; omega0: {param}")

wave_f107, period_f107, scale_f107, coi_f107 = wavelet(
    f107, dt, pad, dj, s0, j1, mother, param
)
wave_temperature, period_temperature, scale_temperature, coi_temperature = wavelet(
    temperature, dt, pad, dj, s0, j1, mother, param
)

if not np.allclose(period_f107, period_temperature):
    raise ValueError("The two wavelet transforms returned different period grids.")

power_f107 = np.abs(wave_f107) ** 2
power_temperature = np.abs(wave_temperature) ** 2
period = np.asarray(period_f107, dtype=float)

if power_f107.shape != power_temperature.shape:
    raise ValueError(
        f"Wavelet power shapes differ: {power_f107.shape} vs {power_temperature.shape}"
    )

print(f"F10.7 wavelet shape: {wave_f107.shape}")
print(f"Temperature wavelet shape: {wave_temperature.shape}")
print(f"Calculated period range: {period.min():.3f}–{period.max():.3f} days")


# ============================================================
# 6. SELECT PERIODS AND NORMALIZE POWER FOR DISPLAY
# ============================================================
period_min = 60.0
period_max = 730.0
period_mask = (period >= period_min) & (period <= period_max)

if not np.any(period_mask):
    raise ValueError("No wavelet periods lie within the requested 60–730 day range.")

period_display = period[period_mask]
power_f107_display = power_f107[period_mask, :]
power_temperature_display = power_temperature[period_mask, :]

max_f107 = float(np.max(power_f107_display))
max_temperature = float(np.max(power_temperature_display))
if max_f107 <= 0 or max_temperature <= 0:
    raise ValueError("Cannot normalize zero or negative maximum wavelet power.")

power_f107_normalized = power_f107_display / max_f107
power_temperature_normalized = power_temperature_display / max_temperature

# Global normalized spectra are also saved for reproducibility.
global_f107 = np.mean(power_f107, axis=1)
global_temperature = np.mean(power_temperature, axis=1)
global_f107_normalized = global_f107 / np.max(global_f107[period_mask])
global_temperature_normalized = global_temperature / np.max(global_temperature[period_mask])


# ============================================================
# 7. PREPARE 3D COORDINATE GRIDS
# ============================================================
time_index = np.arange(len(time))
X, Y = np.meshgrid(time_index, period_display)
Z_f107 = power_f107_normalized
Z_temperature = power_temperature_normalized

print("\n========================================")
print("3D PLOT INFORMATION")
print("========================================")
print(f"Time-axis points: {len(time_index)}")
print(f"Displayed periods: {len(period_display)}")
print(f"F10.7 normalized power range: {Z_f107.min():.3f}–{Z_f107.max():.3f}")
print(f"Temperature normalized power range: {Z_temperature.min():.3f}–{Z_temperature.max():.3f}")


# ============================================================
# 8. CREATE 3D FIGURE
# ============================================================
fig = plt.figure(figsize=(17, 10))
ax = fig.add_subplot(111, projection="3d")

surface_f107 = ax.plot_surface(
    X, Y, Z_f107,
    cmap="viridis",
    linewidth=0,
    antialiased=True,
    alpha=0.70,
    rcount=len(period_display),
    ccount=len(time_index),
)

surface_temperature = ax.plot_surface(
    X, Y, Z_temperature,
    cmap="plasma",
    linewidth=0,
    antialiased=True,
    alpha=0.60,
    rcount=len(period_display),
    ccount=len(time_index),
)

ax.set_xlim(0, len(time_index) - 1)
ax.set_ylim(period_min, period_max)
ax.set_zlim(0, 1)
ax.set_box_aspect((4.0, 1.8, 2.2))

# Label every 6 months.
tick_positions = np.arange(0, len(time), 6)
tick_labels = [time[i].strftime("%b %Y") for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=9)

period_ticks = [60, 90, 120, 180, 240, 365, 540, 730]
ax.set_yticks([t for t in period_ticks if period_min <= t <= period_max])
ax.set_xlabel("Time", fontsize=12, labelpad=14)
ax.set_ylabel("Period (days)", fontsize=12, labelpad=14)
ax.set_zlabel("Normalized wavelet power", fontsize=12, labelpad=12)
ax.set_title(
    "3D Comparison of Wavelet Power\n"
    "F10.7 Solar Index vs CMAM Temperature Perturbation\n"
    "Common interval: January 2005–June 2010",
    fontsize=15,
    pad=22,
)

legend_elements = [
    Patch(facecolor="green", edgecolor="black", alpha=0.70, label="F10.7 Solar Index"),
    Patch(facecolor="purple", edgecolor="black", alpha=0.60, label="CMAM Temperature Perturbation"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
ax.view_init(elev=30, azim=-125)
ax.grid(True, alpha=0.25)

fig.tight_layout()
figure_path = FIGURES_DIR / "04e_3d_wavelet_comparison_2005_2010.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()
print(f"Saved 3D comparison figure: {figure_path}")


# ============================================================
# 9. SAVE GLOBAL SPECTRA
# ============================================================
global_table = pd.DataFrame(
    {
        "period_days": period,
        "f107_global_power": global_f107,
        "temperature_global_power": global_temperature,
        "f107_global_power_normalized": global_f107_normalized,
        "temperature_global_power_normalized": global_temperature_normalized,
    }
)
table_path = TABLES_DIR / "04e_3d_wavelet_comparison_global_spectra.csv"
global_table.to_csv(table_path, index=False)
print(f"Saved global spectra table: {table_path}")
print("\n3D wavelet comparison complete.")
