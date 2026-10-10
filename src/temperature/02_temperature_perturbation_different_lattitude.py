"""
CMAM Assignment: Temperature perturbation versus latitude near 120 km.

Place this script at:
    CMAM-F107-Wavelet-Analysis/src/temperature/05_latitude_perturbation_120km.py

Expected input files:
    CMAM-F107-Wavelet-Analysis/data/raw/cmam/2005.nc
    ...
    CMAM-F107-Wavelet-Analysis/data/raw/cmam/2010.nc

Generated outputs:
    data/processed/temperature_120km_monthly_by_latitude.csv
    data/processed/temperature_120km_annual_perturbation_by_latitude.csv
    results/figures/05_temperature_perturbation_vs_latitude_120km.png

Scientific method:
    1. Approximate altitude from pressure using:
       z = h_ref + H * ln(P_ref / p)
    2. Select the pressure level closest to 120 km.
    3. Average temperature over longitude, leaving a temperature profile
       at each latitude.
    4. Calculate monthly means.
    5. At each latitude, calculate the mean over all monthly samples.
    6. Subtract that reference mean to obtain perturbations.
    7. Average the monthly perturbations by year and plot against latitude.

Important: The altitude formula is approximate. P_ref and p must have the
same pressure units. This script converts plev to hPa using its units metadata.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# 1. Project paths and settings
# ---------------------------------------------------------------------
# This script is expected inside <project>/src/temperature/.
# parents[0] = src/temperature, parents[1] = src, parents[2] = project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

CMAM_DIR = PROJECT_ROOT / "data" / "raw" / "cmam"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

YEARS = list(range(2005, 2011))

TARGET_ALTITUDE_KM = 120.0
P_REF_HPA = 0.1
H_REF_KM = 95.0
SCALE_HEIGHT_KM = 8.0


# ---------------------------------------------------------------------
# 2. Convert pressure values to hPa
# ---------------------------------------------------------------------
def convert_pressure_to_hpa(values, units):
    """Convert pressure values to hPa based on NetCDF units metadata."""
    unit = (units or "").strip().lower().replace(" ", "")
    values = np.asarray(values, dtype=float)

    if unit in {"pa", "pascal", "pascals"}:
        return values / 100.0
    if unit in {
        "hpa", "hectopascal", "hectopascals",
        "mbar", "millibar", "millibars",
    }:
        return values
    if unit in {"kpa", "kilopascal", "kilopascals"}:
        return values * 10.0

    if not unit:
        # The user's original formula uses P_ref = 0.1. In the absence of
        # metadata, only assume hPa if values are consistent with hPa.
        finite_values = values[np.isfinite(values)]
        if finite_values.size and np.nanmax(finite_values) <= 2000:
            print("WARNING: plev units are missing; assuming hPa.")
            return values
        raise ValueError(
            "plev units are missing and values may be in Pa. "
            "Inspect ds['plev'].attrs['units'] before proceeding."
        )

    raise ValueError(
        f"Unsupported pressure units {units!r}. "
        "Add the appropriate conversion before running."
    )


def get_years(time_coord):
    """Extract years from xarray coordinates or pandas Series, including CFTime objects."""
    # Pandas Series and object arrays containing cftime dates do not always
    # support .dt, so read the year attribute from each date object.
    values = getattr(time_coord, "values", time_coord)
    return np.asarray([int(value.year) for value in values], dtype=int)


# ---------------------------------------------------------------------
# 3. Read each NetCDF file and calculate monthly temperature by latitude
# ---------------------------------------------------------------------
monthly_profiles = []
selected_altitudes = []

for year in YEARS:
    file_path = CMAM_DIR / f"{year}.nc"
    print(f"\nProcessing {year}: {file_path}")

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Input file not found:\n{file_path}\n\n"
            "Expected layout: data/raw/cmam/2005.nc through 2010.nc."
        )

    # Load monthly results before closing the NetCDF file.
    with xr.open_dataset(file_path) as ds:
        required = {"ta", "plev", "lat", "lon"}
        missing = required.difference(ds.variables)
        if missing:
            raise KeyError(
                f"{file_path.name} is missing required variables: {sorted(missing)}"
            )

        pressure_units = ds["plev"].attrs.get("units", "")
        pressure_hpa = convert_pressure_to_hpa(
            ds["plev"].values,
            pressure_units,
        )

        if np.any(pressure_hpa <= 0):
            raise ValueError(f"Non-positive pressure found in {file_path.name}.")

        # Approximate altitude; pressure and P_ref are both in hPa.
        altitude_km = H_REF_KM + SCALE_HEIGHT_KM * np.log(
            P_REF_HPA / pressure_hpa
        )

        altitude_index = int(
            np.nanargmin(np.abs(altitude_km - TARGET_ALTITUDE_KM))
        )
        actual_altitude_km = float(altitude_km[altitude_index])
        selected_altitudes.append(actual_altitude_km)

        print(f"  plev units: {pressure_units or 'not specified'}")
        print(f"  Selected altitude: {actual_altitude_km:.2f} km")
        print(f"  ta units: {ds['ta'].attrs.get('units', 'not specified')}")

        # Select the closest pressure level and average over longitude.
        ta_at_level = ds["ta"].isel(plev=altitude_index)

        if "time" not in ta_at_level.dims:
            raise ValueError(f"'ta' in {file_path.name} has no time dimension.")

        temperature_by_latitude = ta_at_level.mean(
            dim="lon",
            skipna=True,
        )

        monthly = temperature_by_latitude.resample(time="MS").mean(skipna=True)
        monthly_profiles.append(monthly.load())

        print(f"  Number of monthly samples: {monthly.sizes['time']}")

# Check that every year used the same altitude.
if not np.allclose(selected_altitudes, selected_altitudes[0], atol=1e-6, rtol=0):
    raise ValueError(
        "Selected altitudes differ between files. Check that all files use "
        "the same plev coordinate."
    )

# Concatenate all years into one time series.
all_temperature = xr.concat(monthly_profiles, dim="time").sortby("time")

# Remove duplicate timestamps if there is any overlap at year boundaries.
time_values = all_temperature.time.values
if len(time_values) != len(np.unique(time_values)):
    _, first_indices = np.unique(time_values, return_index=True)
    all_temperature = all_temperature.isel(time=np.sort(first_indices))

actual_altitude = selected_altitudes[0]

print("\n===== COMBINED MONTHLY DATA =====")
print(all_temperature)
print("Shape (time, latitude):", all_temperature.shape)
print("Number of months:", all_temperature.sizes["time"])
print("First timestamp:", all_temperature.time.values[0])
print("Last timestamp:", all_temperature.time.values[-1])
print(f"Selected altitude: {actual_altitude:.2f} km")


# ---------------------------------------------------------------------
# 4. Reference mean and temperature perturbation
# ---------------------------------------------------------------------
# The reference mean is calculated separately at each latitude across all
# monthly samples (2005 through the available part of 2010).
reference_temperature = all_temperature.mean(dim="time", skipna=True)

temperature_perturbation = all_temperature - reference_temperature

print("\n===== REFERENCE TEMPERATURE BY LATITUDE =====")
for lat, temp in zip(
    all_temperature["lat"].values,
    reference_temperature.values,
):
    print(f"Latitude {lat:8.2f}° : {temp:8.2f} K")

print("\n===== PERTURBATION CHECK =====")
print("The time mean of the perturbation should be approximately zero.")
perturbation_check = temperature_perturbation.mean(dim="time", skipna=True)

for lat, value in zip(
    temperature_perturbation["lat"].values,
    perturbation_check.values,
):
    print(f"Latitude {lat:8.2f}° : {value:.10f} K")


# ---------------------------------------------------------------------
# 5. Annual average perturbation at each latitude
# ---------------------------------------------------------------------
annual_perturbation = temperature_perturbation.resample(
    time="YS"
).mean(skipna=True)


# ---------------------------------------------------------------------
# 6. Save processed CSV files
# ---------------------------------------------------------------------
# Monthly temperature: rows are months, columns are latitude values.
monthly_df = all_temperature.to_dataframe(name="temperature_K").reset_index()
monthly_wide = monthly_df.pivot(
    index="time",
    columns="lat",
    values="temperature_K",
)
monthly_wide.columns = [
    f"lat_{float(lat):g}_deg" for lat in monthly_wide.columns
]
monthly_wide.index.name = "date"
monthly_csv = PROCESSED_DIR / "temperature_120km_monthly_by_latitude.csv"
monthly_wide.to_csv(monthly_csv, index=True)

# Annual perturbation: rows are years, columns are latitude values.
annual_df = annual_perturbation.to_dataframe(
    name="temperature_perturbation_K"
).reset_index()
annual_df["year"] = get_years(annual_df["time"])

annual_wide = annual_df.pivot(
    index="year",
    columns="lat",
    values="temperature_perturbation_K",
)
annual_wide.columns = [
    f"lat_{float(lat):g}_deg" for lat in annual_wide.columns
]
annual_wide.index.name = "year"
annual_csv = (
    PROCESSED_DIR / "temperature_120km_annual_perturbation_by_latitude.csv"
)
annual_wide.to_csv(annual_csv, index=True)

print("\nSaved monthly temperature CSV:", monthly_csv)
print("Saved annual perturbation CSV:", annual_csv)


# ---------------------------------------------------------------------
# 7. Plot annual temperature perturbation versus latitude
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 7))

years_for_plot = get_years(annual_perturbation["time"])

for i, year in enumerate(years_for_plot):
    profile = annual_perturbation.isel(time=i)

    ax.plot(
        profile["lat"].values,
        profile.values,
        marker="o",
        markersize=4,
        linewidth=1.8,
        label=str(year),
    )

ax.axhline(0, color="black", linewidth=1.0, linestyle="--")
ax.set_xlabel("Latitude (degrees)")
ax.set_ylabel("Temperature Perturbation (K)")
ax.set_title(
    "Annual Temperature Perturbation vs Latitude\n"
    f"CMAM30, 2005–2010, altitude ≈ {actual_altitude:.2f} km"
)
ax.legend(title="Year", ncols=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()

figure_path = (
    FIGURES_DIR / "02_temperature_perturbation_vs_latitude_120km.png"
)
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.show()

print("\nSaved figure:", figure_path)
print("\nAnalysis complete.")
