"""
Assignment 1: CMAM Temperature Perturbation Analysis
====================================================

Loads CMAM temperature data for 2005–2010, computes monthly
horizontal-mean temperatures in approximate 95–150 km altitude
levels, calculates perturbations relative to the temporal mean at
each altitude, saves CSV outputs, and creates a contour plot.

Input:
    data/raw/cmam/YYYY.nc

Outputs:
    data/processed/temperature_perturbation.csv
    data/processed/temperature_monthly_mean.csv
    data/processed/temperature_reference_mean.csv
    results/figures/01_temperature_perturbation.png

Calendar note:
    The CMAM files use a no-leap (CFTime) calendar. The script keeps
    the original xarray time coordinates for calculations and converts
    only the monthly labels to Python dates for CSV and plotting.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CMAM_DIR = PROJECT_ROOT / "data" / "raw" / "cmam"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

YEARS = [2005, 2006, 2007, 2008, 2009, 2010]

ALTITUDE_MIN_KM = 95.0
ALTITUDE_MAX_KM = 150.0

# Approximate pressure-to-altitude relation:
# z = H_REF + SCALE_HEIGHT * ln(P_REF / pressure)
# This assumes pressure is in hPa and is an approximation only.
P_REF_HPA = 0.1
H_REF_KM = 95.0
SCALE_HEIGHT_KM = 8.0

TEMPERATURE_VARIABLE = "ta"
SHOW_PLOT = True


# ============================================================
# 2. RGB-STYLE COLOUR GRADIENT
# ============================================================

RGB_CMAP = LinearSegmentedColormap.from_list(
    "CMAM_RGB",
    [
        "#1428A0",  # blue
        "#087DCE",  # cyan-blue
        "#13C9B5",  # turquoise
        "#35C64A",  # green
        "#F2E632",  # yellow
        "#F99B22",  # orange
        "#D7191C",  # red
    ],
    N=256,
)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def timestamp_label(value):
    """Return a stable YYYY-MM-DD string for numpy/cftime dates."""
    return f"{int(value.year):04d}-{int(value.month):02d}-{int(value.day):02d}"


def timestamp_to_python_datetime(value):
    """
    Convert a timestamp to Python datetime for plotting only.

    CMAM uses a no-leap calendar. Monthly timestamps such as the first
    of each month map safely to ordinary dates for display; the source
    xarray time coordinates remain unchanged for calculations.
    """
    return datetime(
        int(value.year),
        int(value.month),
        int(value.day),
        int(getattr(value, "hour", 0)),
        int(getattr(value, "minute", 0)),
        int(getattr(value, "second", 0)),
    )


def wide_dataframe(data_array, value_prefix):
    """
    Convert a (time, altitude) DataArray to a CSV-friendly wide table.
    Avoids relying on Pandas conversion of CFTime objects.
    """
    time_values = data_array["time"].values
    altitude_values = data_array["altitude"].values
    values = np.asarray(data_array.values)

    frame = pd.DataFrame(values, columns=[
        f"{value_prefix}_{float(altitude):.2f}km"
        for altitude in altitude_values
    ])
    frame.insert(0, "date", [timestamp_label(t) for t in time_values])
    return frame


# ============================================================
# 4. CHECK INPUT FILES
# ============================================================

print("=" * 70)
print("CMAM TEMPERATURE PERTURBATION ANALYSIS")
print("=" * 70)
print(f"Project root: {PROJECT_ROOT}")
print(f"CMAM input folder: {CMAM_DIR}")

input_files = {year: CMAM_DIR / f"{year}.nc" for year in YEARS}
missing_files = [str(path) for path in input_files.values() if not path.is_file()]

if missing_files:
    raise FileNotFoundError(
        "Missing CMAM input file(s):\n"
        + "\n".join(missing_files)
        + "\nCheck data/raw/cmam/."
    )


# ============================================================
# 5. PROCESS EACH YEAR
# ============================================================

monthly_data = []
reference_altitude = None

for year in YEARS:
    file_path = input_files[year]

    print("\n" + "-" * 65)
    print(f"Processing {year}: {file_path.name}")

    with xr.open_dataset(file_path) as ds:
        required_variables = {TEMPERATURE_VARIABLE, "plev", "lat", "lon"}
        missing_variables = required_variables.difference(ds.variables)

        if missing_variables:
            raise KeyError(
                f"{file_path.name} is missing required variables: "
                f"{sorted(missing_variables)}"
            )

        ta = ds[TEMPERATURE_VARIABLE]
        required_dims = {"time", "plev", "lat", "lon"}

        if not required_dims.issubset(set(ta.dims)):
            raise ValueError(
                f"Unexpected dimensions for {TEMPERATURE_VARIABLE}: {ta.dims}"
            )

        print("Original dimensions:", dict(ta.sizes))
        print("Temperature units:", ta.attrs.get("units", "unknown"))

        # Pressure-to-altitude approximation.
        pressure = np.asarray(ds["plev"].values, dtype=float)
        pressure_units = str(ds["plev"].attrs.get("units", "")).lower()

        if not np.all(np.isfinite(pressure)) or np.any(pressure <= 0):
            raise ValueError(f"Invalid pressure levels in {file_path.name}.")

        # Convert Pa to hPa when metadata explicitly indicates Pa.
        if pressure_units in {"pa", "pascal", "pascals"}:
            pressure_hpa = pressure / 100.0
        else:
            # The dataset's values in this project appear to be hPa.
            # This is explicitly reported so units can be verified.
            pressure_hpa = pressure

        altitude_km = (
            H_REF_KM
            + SCALE_HEIGHT_KM * np.log(P_REF_HPA / pressure_hpa)
        )

        mask = (
            (altitude_km >= ALTITUDE_MIN_KM)
            & (altitude_km <= ALTITUDE_MAX_KM)
        )
        selected_indices = np.flatnonzero(mask)
        selected_altitude = altitude_km[mask]

        if selected_indices.size == 0:
            raise ValueError(
                f"No levels found between {ALTITUDE_MIN_KM} and "
                f"{ALTITUDE_MAX_KM} km in {file_path.name}. "
                "Check pressure units and the altitude approximation."
            )

        if reference_altitude is None:
            reference_altitude = selected_altitude.copy()
        elif (
            len(selected_altitude) != len(reference_altitude)
            or not np.allclose(
                selected_altitude, reference_altitude, rtol=0, atol=1e-5
            )
        ):
            raise ValueError(
                f"Selected altitude levels differ in {file_path.name}."
            )

        print(
            f"Selected approximate altitude range: "
            f"{selected_altitude.min():.2f}–"
            f"{selected_altitude.max():.2f} km"
        )
        print("Selected levels:", len(selected_indices))

        ta_selected = ta.isel(plev=selected_indices)

        # Area-weighted mean over latitude and longitude.
        latitude_weights = np.cos(np.deg2rad(ds["lat"]))
        horizontal_mean = (
            ta_selected
            .weighted(latitude_weights)
            .mean(dim=["lat", "lon"], skipna=True)
        )

        horizontal_mean = horizontal_mean.assign_coords(
            altitude=("plev", selected_altitude)
        )
        horizontal_mean = horizontal_mean.swap_dims({"plev": "altitude"})
        horizontal_mean = horizontal_mean.sortby("altitude")

        # Monthly means; preserve the source CFTime calendar.
        monthly = horizontal_mean.resample(time="MS").mean(skipna=True)
        monthly = monthly.dropna(dim="time", how="all")

        if monthly.sizes.get("time", 0) == 0:
            raise ValueError(f"No valid monthly values in {file_path.name}.")

        monthly_data.append(monthly.load())
        print("Monthly observations:", monthly.sizes["time"])
        print("Monthly shape:", monthly.shape)


# ============================================================
# 6. COMBINE MONTHLY DATA AND VALIDATE TIME
# ============================================================

print("\n" + "=" * 70)
print("COMBINING YEARLY DATA")
print("=" * 70)

all_monthly = xr.concat(
    monthly_data,
    dim="time",
    join="exact",
    combine_attrs="drop_conflicts",
).sortby("time")

time_values = all_monthly["time"].values
time_labels = [timestamp_label(value) for value in time_values]

if len(time_labels) != len(set(time_labels)):
    raise ValueError("Duplicate monthly dates found after combining yearly files.")

print("Calendar:", all_monthly.time.encoding.get("calendar", "stored as CFTime objects"))
print("First month:", time_labels[0])
print("Last month:", time_labels[-1])
print("Monthly observations:", len(time_values))
print("Combined shape:", all_monthly.shape)


# ============================================================
# 7. REFERENCE TEMPERATURE AND PERTURBATIONS
# ============================================================

reference_temperature = all_monthly.mean(dim="time", skipna=True)

temp_perturbation = all_monthly - reference_temperature
temp_perturbation.name = "temperature_perturbation"
temp_perturbation.attrs.update({
    "long_name": "CMAM temperature perturbation",
    "units": all_monthly.attrs.get("units", "K"),
    "description": (
        "Monthly temperature minus the temporal mean at each "
        "selected approximate altitude."
    ),
})

print("\n" + "=" * 70)
print("REFERENCE TEMPERATURE BY ALTITUDE")
print("=" * 70)

for i, altitude in enumerate(all_monthly["altitude"].values):
    value = float(reference_temperature.isel(altitude=i).values)
    print(f"{altitude:8.2f} km : {value:10.4f} K")

valid_values = temp_perturbation.values
valid_values = valid_values[np.isfinite(valid_values)]

if valid_values.size == 0:
    raise ValueError("No valid perturbation values were produced.")

print("\nPerturbation minimum:", f"{np.min(valid_values):.4f} K")
print("Perturbation maximum:", f"{np.max(valid_values):.4f} K")
print("Overall mean perturbation:", f"{np.mean(valid_values):.10f} K")

print("\nPerturbation mean check by altitude:")
perturbation_mean_by_altitude = temp_perturbation.mean(
    dim="time", skipna=True
)

for i, altitude in enumerate(temp_perturbation["altitude"].values):
    value = float(perturbation_mean_by_altitude.isel(altitude=i).values)
    print(f"{altitude:8.2f} km : {value:+.12e} K")


# ============================================================
# 8. SAVE PROCESSED CSV FILES
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED DATA")
print("=" * 70)

perturbation_csv = PROCESSED_DIR / "temperature_perturbation.csv"
monthly_mean_csv = PROCESSED_DIR / "temperature_monthly_mean.csv"
reference_csv = PROCESSED_DIR / "temperature_reference_mean.csv"

wide_dataframe(temp_perturbation, "perturbation").to_csv(
    perturbation_csv, index=False
)
wide_dataframe(all_monthly, "temperature").to_csv(
    monthly_mean_csv, index=False
)

reference_df = pd.DataFrame({
    "altitude_km": all_monthly["altitude"].values,
    "reference_temperature_K": reference_temperature.values,
})
reference_df.to_csv(reference_csv, index=False)

print("Saved:", perturbation_csv)
print("Saved:", monthly_mean_csv)
print("Saved:", reference_csv)


# ============================================================
# 9. PREPARE PLOT
# ============================================================

# Use original CFTime values for scientific data; convert only
# plotting coordinates to ordinary Python datetime objects.
dates = list(all_monthly["time"].values)
plot_dates = [timestamp_to_python_datetime(t) for t in dates]
date_numbers = mdates.date2num(plot_dates)

altitude_values = temp_perturbation["altitude"].values
Z_plot = np.asarray(temp_perturbation.values).T  # (altitude, time)

expected_shape = (len(altitude_values), len(plot_dates))
if Z_plot.shape != expected_shape:
    raise ValueError(
        f"Unexpected plot array shape {Z_plot.shape}; expected {expected_shape}."
    )

max_abs = float(np.nanmax(np.abs(Z_plot)))
if not np.isfinite(max_abs) or max_abs <= 0:
    raise ValueError("Perturbation values are zero or invalid; cannot plot.")

levels = np.linspace(-max_abs, max_abs, 41)

fig, ax = plt.subplots(figsize=(17, 8))

contour = ax.contourf(
    date_numbers,
    altitude_values,
    Z_plot,
    levels=levels,
    cmap=RGB_CMAP,
    extend="both",
)

# Zero-perturbation contour.
ax.contour(
    date_numbers,
    altitude_values,
    Z_plot,
    levels=[0],
    colors="black",
    linewidths=0.65,
    alpha=0.7,
)

colorbar = fig.colorbar(contour, ax=ax, pad=0.025)
colorbar.set_label("Temperature Perturbation, T(t,z) − T̄(z) (K)")
colorbar.ax.yaxis.set_major_locator(MaxNLocator(nbins=9))

ax.set_xlabel("Time")
ax.set_ylabel("Approximate altitude (km)")
ax.set_ylim(ALTITUDE_MIN_KM, ALTITUDE_MAX_KM)
ax.set_xlim(date_numbers[0], date_numbers[-1])

first_month = f"{dates[0].year:04d}-{dates[0].month:02d}"
last_month = f"{dates[-1].year:04d}-{dates[-1].month:02d}"

ax.set_title(
    "CMAM30 Temperature Perturbation\n"
    f"Reference: Mean Monthly Temperature at Each Altitude "
    f"({first_month} – {last_month})",
    fontsize=14,
    fontweight="bold",
    pad=12,
)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.25)


# ============================================================
# 10. INTERACTIVE INFORMATION BOX
# ============================================================

std_temperature = all_monthly.std(dim="time", skipna=True)

info = fig.text(
    0.755,
    0.50,
    "Move the mouse over the plot",
    ha="left",
    va="center",
    fontsize=9.5,
    family="monospace",
    bbox={
        "boxstyle": "round,pad=0.6",
        "facecolor": "white",
        "edgecolor": "gray",
        "alpha": 0.95,
    },
)


def mouse_move(event):
    """Display values at the nearest time and altitude."""
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return

    time_index = int(np.argmin(np.abs(date_numbers - event.xdata)))
    altitude_index = int(
        np.argmin(np.abs(altitude_values - event.ydata))
    )

    selected_time = dates[time_index]
    altitude_km = float(altitude_values[altitude_index])

    actual_temperature = float(
        all_monthly.isel(time=time_index, altitude=altitude_index).values
    )
    reference_value = float(
        reference_temperature.isel(altitude=altitude_index).values
    )
    perturbation_value = float(
        temp_perturbation.isel(time=time_index, altitude=altitude_index).values
    )
    standard_deviation = float(
        std_temperature.isel(altitude=altitude_index).values
    )

    info.set_text(
        f"Date             : {timestamp_label(selected_time)[:7]}\n"
        f"Altitude         : {altitude_km:.2f} km\n\n"
        f"Actual Temp.     : {actual_temperature:.2f} K\n"
        f"Actual Temp.     : {actual_temperature - 273.15:.2f} °C\n"
        f"Reference Mean   : {reference_value:.2f} K\n"
        f"Perturbation     : {perturbation_value:+.2f} K\n"
        f"Standard Dev.    : {standard_deviation:.2f} K"
    )
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", mouse_move)

fig.subplots_adjust(
    left=0.075,
    right=0.735,
    bottom=0.19,
    top=0.88,
)


# ============================================================
# 11. SAVE FIGURE AND FINISH
# ============================================================

figure_path = FIGURES_DIR / "01_temperature_perturbation.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")

print("\n" + "=" * 70)
print("PROCESSING COMPLETED")
print("=" * 70)
print("Years requested:", YEARS)
print("First month:", time_labels[0])
print("Last month:", time_labels[-1])
print("Monthly observations:", len(time_values))
print(
    f"Approximate altitude range: {altitude_values.min():.2f}–"
    f"{altitude_values.max():.2f} km"
)
print("Number of altitude levels:", len(altitude_values))
print("\nOutput files:")
print("1.", perturbation_csv)
print("2.", monthly_mean_csv)
print("3.", reference_csv)
print("4.", figure_path)

if SHOW_PLOT:
    plt.show()
else:
    plt.close(fig)
