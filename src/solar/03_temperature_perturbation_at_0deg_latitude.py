"""
Prepare the monthly CMAM temperature-perturbation time series at ~120 km,
0 degrees latitude, for subsequent wavelet analysis.

Place this script at:
    src/temperature/02_prepare_monthly_series.py

Input:
    data/raw/cmam/2005.nc ... data/raw/cmam/2010.nc

Outputs:
    data/processed/temperature_perturbation.csv
    data/processed/temperature_120km_equator_monthly.csv
    results/figures/02_temperature_perturbation_120km_equator.png

Important:
- The altitude relation is an approximate scale-height model.
- Pressure values are converted to hPa before applying the formula.
- The CMAM files use a CF calendar; this script preserves those dates while
  processing and writes ISO-like date strings to CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Paths and settings
# ---------------------------------------------------------------------
# Expected location: <project_root>/src/temperature/02_prepare_monthly_series.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CMAM_DIR = PROJECT_ROOT / "data" / "raw" / "cmam"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2005, 2011))

P_REF_HPA = 0.1
H_REF_KM = 95.0
SCALE_HEIGHT_KM = 8.0
TARGET_ALTITUDE_KM = 120.0
TARGET_LATITUDE_DEG = 0.0


# ---------------------------------------------------------------------
# 2. Pressure conversion helper
# ---------------------------------------------------------------------
def pressure_to_hpa(values, units):
    """Convert plev values to hPa using the NetCDF units attribute."""
    values = np.asarray(values, dtype=float)
    unit = (units or "").strip().lower().replace(" ", "")

    if unit in {"pa", "pascal", "pascals"}:
        return values / 100.0
    if unit in {"hpa", "hectopascal", "hectopascals", "mbar", "millibar"}:
        return values
    if unit in {"kpa", "kilopascal", "kilopascals"}:
        return values * 10.0

    if not unit:
        finite = values[np.isfinite(values)]
        if finite.size and np.nanmax(finite) <= 2000:
            print("WARNING: plev units missing; assuming hPa.")
            return values
        raise ValueError(
            "plev units are missing and values may be in Pa. "
            "Check the NetCDF metadata before continuing."
        )

    raise ValueError(f"Unsupported pressure units {units!r}.")


def date_strings(time_coord):
    """Convert datetime/cftime coordinates to YYYY-MM-DD strings."""
    # xarray's dt.strftime accessor supports CFTime coordinates.
    return np.asarray(time_coord.dt.strftime("%Y-%m-%d").values, dtype=str)


# ---------------------------------------------------------------------
# 3. Read yearly files and extract monthly temperature at the target point
# ---------------------------------------------------------------------
monthly_series = []
selected_altitudes = []

for year in YEARS:
    file_path = CMAM_DIR / f"{year}.nc"
    print(f"\nProcessing {year}: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Missing input file: {file_path}\n"
            "Expected files in data/raw/cmam/."
        )

    with xr.open_dataset(file_path) as ds:
        for variable in ("ta", "plev", "lat", "lon"):
            if variable not in ds.variables:
                raise KeyError(f"{file_path.name} does not contain {variable!r}.")

        pressure = pressure_to_hpa(
            ds["plev"].values,
            ds["plev"].attrs.get("units", ""),
        )

        if np.any(pressure <= 0):
            raise ValueError(f"Non-positive pressure value in {file_path.name}.")

        # Approximate altitude; P_REF_HPA and pressure are both in hPa.
        altitude_km = H_REF_KM + SCALE_HEIGHT_KM * np.log(P_REF_HPA / pressure)
        altitude_index = int(np.nanargmin(np.abs(altitude_km - TARGET_ALTITUDE_KM)))
        actual_altitude = float(altitude_km[altitude_index])
        selected_altitudes.append(actual_altitude)

        print(f"  plev units: {ds['plev'].attrs.get('units', 'not specified')}")
        print(f"  Requested altitude: {TARGET_ALTITUDE_KM:.2f} km")
        print(f"  Selected model altitude: {actual_altitude:.2f} km")

        # Select the pressure level closest to the requested altitude.
        ta_at_level = ds["ta"].isel(plev=altitude_index)

        # Average over longitude, then interpolate across latitude to 0 degrees.
        temperature_by_latitude = ta_at_level.mean(dim="lon", skipna=True)
        temperature_equator = temperature_by_latitude.interp(
            lat=TARGET_LATITUDE_DEG
        )

        print(f"  Interpolated latitude: {float(temperature_equator['lat'].values):.2f}°")

        # Monthly means; retain xarray's CFTime calendar.
        monthly = temperature_equator.resample(time="MS").mean(skipna=True)
        monthly_series.append(monthly.load())
        print(f"  Number of months: {monthly.sizes['time']}")

if not np.allclose(selected_altitudes, selected_altitudes[0], rtol=0, atol=1e-6):
    raise ValueError("Selected altitude differs between yearly files; check plev coordinates.")

actual_altitude = selected_altitudes[0]

all_temperature = xr.concat(monthly_series, dim="time").sortby("time")

# Guard against duplicate timestamps at annual file boundaries.
time_values = all_temperature["time"].values
if len(time_values) != len(np.unique(time_values)):
    _, first_indices = np.unique(time_values, return_index=True)
    all_temperature = all_temperature.isel(time=np.sort(first_indices))

if all_temperature.sizes["time"] != 66:
    print(
        f"WARNING: expected 66 monthly samples for Jan 2005–Jun 2010; "
        f"found {all_temperature.sizes['time']}."
    )

print("\n===== COMBINED MONTHLY DATA =====")
print(all_temperature)
print(f"Total monthly samples: {all_temperature.sizes['time']}")
print(f"Selected altitude: {actual_altitude:.2f} km")
print(f"Target latitude: {TARGET_LATITUDE_DEG:.1f}°")


# ---------------------------------------------------------------------
# 4. Reference mean and perturbation
# ---------------------------------------------------------------------
reference_temperature = all_temperature.mean(dim="time", skipna=True)
temperature_perturbation = all_temperature - reference_temperature

print("\n===== REFERENCE TEMPERATURE =====")
print(f"{float(reference_temperature.values):.4f} K")

print("\n===== MONTHLY TEMPERATURE PERTURBATIONS =====")
dates = date_strings(temperature_perturbation["time"])
perturbation_values = np.asarray(temperature_perturbation.values, dtype=float)

for date, value in zip(dates, perturbation_values):
    print(f"{date:10s} : {value:+.3f} K")

mean_perturbation = float(temperature_perturbation.mean(skipna=True).values)
print("\n===== PERTURBATION CHECK =====")
print(f"Mean perturbation over all months: {mean_perturbation:.10f} K")


# ---------------------------------------------------------------------
# 5. Save processed CSV files
# ---------------------------------------------------------------------
# This is the main wavelet input: one monthly observation per row.
wavelet_csv = PROCESSED_DIR / "temperature_perturbation.csv"
wavelet_df = pd.DataFrame({
    "date": dates,
    "temperature_perturbation_K": perturbation_values,
})
wavelet_df.to_csv(wavelet_csv, index=False)

# A more descriptive companion file with raw monthly temperature and metadata.
details_csv = PROCESSED_DIR / "temperature_120km_equator_monthly.csv"
details_df = pd.DataFrame({
    "date": dates,
    "temperature_K": np.asarray(all_temperature.values, dtype=float),
    "reference_temperature_K": np.full(
        len(dates), float(reference_temperature.values), dtype=float
    ),
    "temperature_perturbation_K": perturbation_values,
})
details_df.to_csv(details_csv, index=False)

print("\nSaved wavelet input:", wavelet_csv)
print("Saved monthly detail table:", details_csv)


# ---------------------------------------------------------------------
# 6. Plot the monthly temperature perturbation
# ---------------------------------------------------------------------
x = np.arange(len(perturbation_values))

fig, ax = plt.subplots(figsize=(16, 7))
ax.plot(x, perturbation_values, marker="o", linewidth=1.5)
ax.axhline(0, color="black", linewidth=1)

ax.set_xlabel("Time")
ax.set_ylabel("Temperature Perturbation (K)")
ax.set_title(
    "Monthly Temperature Perturbation\n"
    f"Altitude ≈ {actual_altitude:.2f} km, latitude = {TARGET_LATITUDE_DEG:.1f}°\n"
    "CMAM30, January 2005 – June 2010"
)

tick_positions = np.arange(0, len(x), 3)
tick_labels = [dates[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha="right")
ax.grid(True, alpha=0.3)
fig.tight_layout()

figure_path = FIGURES_DIR / "03_temperature_perturbation_120km_equator.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:", figure_path)
print("\nTemperature preparation complete. Use data/processed/temperature_perturbation.csv for the temperature wavelet analysis.")
