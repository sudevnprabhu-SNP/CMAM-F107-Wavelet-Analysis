# ============================================================
# 01. WAVELET ANALYSIS OF DAILY F10.7 SOLAR RADIO FLUX
# ============================================================
#
# Assignment 4
#
# Requirements:
#   1. OMNIWeb F10.7 data: 1979–2010
#   2. Daily F10.7 data
#   3. DO NOT subtract the mean
#   4. Morlet wavelet
#   5. Period in DAYS
#   6. Wavelet power expressed as log10(power)
#   7. 95% significance level
#   8. Cone of Influence (COI)
#
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 2. PROJECT PATH
# ============================================================

project_folder = Path(__file__).resolve().parent

# Add wavelet library folder
sys.path.append(str(project_folder / "wave_python"))


# ============================================================
# 3. IMPORT WAVELET LIBRARY
# ============================================================

try:

    from waveletFunctions import wavelet, wave_signif

    print("Wavelet library imported successfully!")

except ImportError as e:

    print("ERROR: Could not import wavelet library.")
    print(e)
    sys.exit()


# ============================================================
# 4. DATA FILE
# ============================================================

data_file = (
    project_folder
    / "data"
    / "F107_daily_1979_2010.csv"
)


# ============================================================
# 5. LOAD DATA
# ============================================================

df = pd.read_csv(data_file)

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# Sort chronologically
df = df.sort_values("date")

# Reset index
df = df.reset_index(drop=True)


# ============================================================
# 6. CLEAN F10.7 DATA
# ============================================================
#
# OMNI datasets may contain placeholder / invalid values
# such as 999.9.
#
# Real F10.7 values should be within a reasonable range.
#
# We remove:
#   <= 0
#   >= 400
#
# This also removes 999.9.
# ============================================================

before_cleaning = len(df)

df = df[
    (df["f107"] > 0)
    &
    (df["f107"] < 400)
]

df = df.dropna(subset=["f107"])

df = df.drop_duplicates(subset=["date"])

df = df.sort_values("date")

df = df.reset_index(drop=True)

after_cleaning = len(df)

removed_points = before_cleaning - after_cleaning


# ============================================================
# 7. PRINT DATA INFORMATION
# ============================================================

print()
print("=" * 40)
print("F10.7 DATA INFORMATION")
print("=" * 40)

print(df.head())

print()
print("Number of data points =", len(df))

print(
    "Time range =",
    df["date"].min(),
    "to",
    df["date"].max()
)

print()
print("Removed invalid / missing records =", removed_points)


# ============================================================
# 8. EXTRACT F10.7
# ============================================================

f107 = df["f107"].values.astype(float)


# ============================================================
# 9. F10.7 INPUT INFORMATION
# ============================================================

print()
print("=" * 40)
print("F10.7 INPUT")
print("=" * 40)

print("Mean F10.7 =", np.mean(f107))

print("Minimum F10.7 =", np.min(f107))

print("Maximum F10.7 =", np.max(f107))

print("Mean subtraction = NOT PERFORMED")


# ============================================================
# 10. WAVELET PARAMETERS
# ============================================================

dt = 1.0

# Frequency resolution
dj = 0.25

# Starting scale
s0 = 2 * dt

# Morlet wavelet
mother = "MORLET"

# Morlet omega0
param = 6


# Number of scales
j1 = int(
    np.floor(
        np.log2(
            len(f107) * dt / s0
        )
        / dj
    )
)


# ============================================================
# 11. PRINT WAVELET PARAMETERS
# ============================================================

print()
print("=" * 40)
print("WAVELET PARAMETERS")
print("=" * 40)

print("Mother wavelet =", mother)

print("Morlet parameter =", param)

print("dt =", dt, "day")

print("dj =", dj)

print("s0 =", s0)

print("j1 =", j1)


# ============================================================
# 12. WAVELET TRANSFORM
# ============================================================

print()
print("=" * 40)
print("WAVELET TRANSFORM")
print("=" * 40)


wave, period, scale, coi = wavelet(
    f107,
    dt,
    pad=1,
    dj=dj,
    s0=s0,
    J1=j1,
    mother=mother,
    param=param
)


print("Wavelet analysis completed!")

print("Wavelet shape =", wave.shape)

print("Number of periods =", len(period))

print(
    "Minimum period =",
    np.min(period),
    "days"
)

print(
    "Maximum period =",
    np.max(period),
    "days"
)


# ============================================================
# 13. WAVELET POWER
# ============================================================

power = np.abs(wave) ** 2


print()
print("=" * 40)
print("WAVELET POWER")
print("=" * 40)

print("Power shape =", power.shape)

print("Minimum power =", np.min(power))

print("Maximum power =", np.max(power))


# ============================================================
# 14. LAG-1 AUTOCORRELATION
# ============================================================
#
# The red-noise significance test requires lag-1
# autocorrelation.
# ============================================================

lag1 = np.corrcoef(
    f107[:-1],
    f107[1:]
)[0, 1]


# ============================================================
# 15. 95% SIGNIFICANCE LEVEL
# ============================================================

print()
print("=" * 40)
print("SIGNIFICANCE INFORMATION")
print("=" * 40)

print(
    "Lag-1 autocorrelation =",
    lag1
)

print("Confidence level = 95%")


# ============================================================
# 16. CALCULATE SIGNIFICANCE
# ============================================================
#
# IMPORTANT:
#
# Your installed version of wave_signif() returns ONE value,
# not two values.
#
# Therefore:
#
#     signif = wave_signif(...)
#
# instead of:
#
#     signif, fft_theor = wave_signif(...)
# ============================================================

signif = wave_signif(
    f107,
    dt=dt,
    scale=scale,
    sigtest=0,
    lag1=lag1,
    siglvl=0.95,
    dof=-1,
    mother=mother,
    param=param
)


print("95% significance calculated successfully!")


# ============================================================
# 17. SIGNIFICANCE RATIO
# ============================================================
#
# Ratio:
#
#     Wavelet Power / Significance Level
#
# Values:
#
#   > 1  -> significant at 95%
#   = 1  -> significance boundary
#   < 1  -> not significant
#
# ============================================================

significance_ratio = (
    power
    /
    signif[:, np.newaxis]
)


# ============================================================
# 18. LOG10 WAVELET POWER
# ============================================================
#
# Sir specifically requested wavelet power as log10.
#
# We calculate:
#
#     log10(Power)
#
# A very small value is added only to avoid log10(0).
# ============================================================

log_power = np.log10(
    power + 1e-12
)


# ============================================================
# 19. CONE OF INFLUENCE
# ============================================================
#
# COI is returned by the wavelet library in scale units.
#
# Since dt = 1 day:
#
#     COI in days = COI * dt
#
# ============================================================

coi_days = coi * dt


print()
print("=" * 40)
print("CONE OF INFLUENCE")
print("=" * 40)

print(
    "COI minimum =",
    np.min(coi_days),
    "days"
)

print(
    "COI maximum =",
    np.max(coi_days),
    "days"
)


# ============================================================
# 20. SELECT DISPLAY PERIOD RANGE
# ============================================================
#
# The complete wavelet calculation contains periods up to
# approximately 10000 days.
#
# For a 1979–2010 record, we display:
#
#       2 days → 4000 days
#
# This includes:
#
#   short-term variations
#   ~27-day solar rotation
#   ~100-day variations
#   ~1-year variations
#   multi-year solar-cycle variations
#
# ============================================================

min_period = 2

max_period = 4000


period_mask = (
    (period >= min_period)
    &
    (period <= max_period)
)


plot_period = period[period_mask]

plot_log_power = log_power[period_mask, :]

plot_significance_ratio = (
    significance_ratio[period_mask, :]
)


# ============================================================
# 21. CREATE TIME AXIS
# ============================================================

dates = df["date"].values


# ============================================================
# 22. CREATE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(15, 8)
)


# ============================================================
# 23. WAVELET POWER CONTOUR
# ============================================================

levels = 100

contour = ax.contourf(
    dates,
    plot_period,
    plot_log_power,
    levels=levels,
    cmap="viridis",
    extend="both"
)


# ============================================================
# 24. 95% SIGNIFICANCE CONTOUR
# ============================================================
#
# The contour is drawn where:
#
#     power / significance = 1
#
# Therefore the enclosed region represents
# statistically significant wavelet power at 95%.
# ============================================================

ax.contour(
    dates,
    plot_period,
    plot_significance_ratio,
    levels=[1],
    colors="black",
    linewidths=1.2
)


# ============================================================
# 25. CONE OF INFLUENCE
# ============================================================
#
# White dashed line = COI boundary.
#
# Region above the COI curve is affected by edge effects.
# ============================================================

ax.plot(
    dates,
    coi_days,
    color="white",
    linestyle="--",
    linewidth=1.5,
    label="Cone of Influence"
)


# ============================================================
# 26. SHADE REGION OUTSIDE COI
# ============================================================
#
# We shade periods outside the reliable region.
#
# Because the y-axis is logarithmic, use fill_between.
# ============================================================

upper_period = np.full(
    len(dates),
    max_period
)

coi_for_fill = np.minimum(
    coi_days,
    max_period
)

ax.fill_between(
    dates,
    coi_for_fill,
    upper_period,
    color="white",
    alpha=0.35
)


# ============================================================
# 27. Y-AXIS: LOG PERIOD
# ============================================================

ax.set_yscale("log")


# ============================================================
# 28. PERIOD TICKS
# ============================================================

period_ticks = [
    2,
    5,
    10,
    20,
    50,
    100,
    200,
    500,
    1000,
    2000,
    4000
]


ax.set_yticks(period_ticks)

ax.set_yticklabels(
    [str(x) for x in period_ticks]
)


# ============================================================
# 29. AXIS LABELS
# ============================================================

ax.set_xlabel(
    "Year",
    fontsize=13
)

ax.set_ylabel(
    "Period (days)",
    fontsize=13
)


# ============================================================
# 30. TITLE
# ============================================================

ax.set_title(
    "Wavelet Power Spectrum of Daily F10.7 Solar Radio Flux\n"
    "OMNIWeb Data: 1979–2010",
    fontsize=15,
    fontweight="bold"
)


# ============================================================
# 31. COLORBAR
# ============================================================

cbar = fig.colorbar(
    contour,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    r"$\log_{10}$(Wavelet Power)",
    fontsize=12
)


# ============================================================
# 32. 12-MONTH PERIOD REFERENCE
# ============================================================
#
# 1 year ≈ 365 days
#
# This is useful because annual periodicity may appear
# in the solar index.
# ============================================================

ax.axhline(
    365,
    color="red",
    linestyle=":",
    linewidth=1.2,
    label="365 days"
)


# ============================================================
# 33. LEGEND
# ============================================================

ax.legend(
    loc="upper right",
    fontsize=10
)


# ============================================================
# 34. GRID
# ============================================================

ax.grid(
    True,
    which="both",
    alpha=0.15
)


# ============================================================
# 35. X-AXIS LIMITS
# ============================================================

ax.set_xlim(
    dates[0],
    dates[-1]
)


# ============================================================
# 36. LAYOUT
# ============================================================

plt.tight_layout()


# ============================================================
# 37. SAVE FIGURE
# ============================================================

output_folder = (
    project_folder
    / "results"
)

output_folder.mkdir(
    exist_ok=True
)


output_file = (
    output_folder
    / "01_F107_wavelet_1979_2010.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)


# ============================================================
# 38. DISPLAY
# ============================================================

plt.show()


# ============================================================
# 39. FINAL MESSAGE
# ============================================================

print()
print("=" * 60)
print("F10.7 WAVELET ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)

print()
print("Data period:")
print(
    df["date"].min(),
    "to",
    df["date"].max()
)

print()
print("Number of valid observations:")
print(len(f107))

print()
print("95% significance:")
print("Calculated")

print()
print("Cone of Influence:")
print("Included")

print()
print("Wavelet power:")
print("log10(Power)")

print()
print("Period unit:")
print("Days")

print()
print("Mean subtraction:")
print("NOT PERFORMED")

print()
print("Figure saved to:")
print(output_file)

print()
print("============================================================")