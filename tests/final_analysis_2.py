import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

data_folder = "/Users/sudevnprabhu/Desktop/Future_Sudev/CUAP/CMAM_Temperature_Analysis/data"

years = [2005, 2006, 2007, 2008, 2009, 2010]

# pressure to altitude
P_ref = 0.1
h_ref = 95.0
H = 8.0

all_monthly = []

for year in years:

    file = f"{data_folder}/{year}.nc"

    print(f"\nProcessing {year}...")

    ds = xr.open_dataset(file)

    pressure = ds["plev"].values

    altitude = h_ref + H * np.log(P_ref / pressure)

    # select 95 to 150 km
    mask = (altitude >= 95) & (altitude <= 150)

    selected_altitude = altitude[mask]

    # temperature for selected levels
    ta_selected = ds["ta"].isel(plev=np.where(mask)[0])

    # latitude weighting
    weights = np.cos(np.deg2rad(ds["lat"]))

    ta_global = ta_selected.weighted(weights).mean(
        dim=["lat", "lon"]
    )

    # use altitude instead of pressure
    ta_global = ta_global.assign_coords(
        altitude=("plev", selected_altitude)
    )

    ta_global = ta_global.swap_dims({"plev": "altitude"})

    # monthly mean
    monthly = ta_global.resample(time="MS").mean()

    all_monthly.append(monthly)

    print("Months:", len(monthly.time))
    print("Shape:", monthly.shape)

    ds.close()

print("\n===== ALL FILES PROCESSED =====")

# combine all years
all_monthly = xr.concat(all_monthly, dim="time")

print("\n===== COMBINED DATA =====")
print(all_monthly)
print("Shape:", all_monthly.shape)

# reference temperature
mean_temperature = all_monthly.mean(dim="time")

print("\n===== REFERENCE TEMPERATURE =====")

for i in range(len(all_monthly.altitude)):
    print(f"{float(all_monthly.altitude[i]):8.2f} km : "
          f"{float(mean_temperature.isel(altitude=i)):.2f} K")


# temperature perturbation
temp_perturb = all_monthly - mean_temperature

print("\n===== TEMPERATURE PERTURBATION =====")
print(temp_perturb)

print("\n===== TIME RANGE =====")
print("First month:", all_monthly.time[0].values)
print("Last month :", all_monthly.time[-1].values)
print("Number of months:", len(all_monthly.time))

perturbation_check = temp_perturb.mean(dim="time")

print("\n===== PERTURBATION CHECK =====")

for i in range(len(temp_perturb.altitude)):
    print(f"{float(temp_perturb.altitude[i]):8.2f} km : "
          f"{float(perturbation_check.isel(altitude=i)):.10f} K")



plt.figure(figsize=(8, 6))

plt.plot(
    mean_temperature.values,
    all_monthly.altitude.values,
    marker="o"
)

plt.xlabel("Mean Temperature (K)")
plt.ylabel("Altitude (km)")
plt.title("CMAM30 Mean Temperature Profile (2005–2010)")

plt.ylim(95, 150)

plt.grid()
plt.tight_layout()
plt.show()


# final perturbation contour

X = np.arange(len(temp_perturb.time))
Y = temp_perturb.altitude.values
Z = temp_perturb.values

max_abs = np.max(np.abs(Z))

levels = np.linspace(
    -max_abs,
    max_abs,
    21
)

fig, ax = plt.subplots(figsize=(16, 8))

contour = ax.contourf(
    X,
    Y,
    Z.T,
    levels=levels,
    cmap="viridis"
)

cbar = fig.colorbar(
    contour,
    ax=ax,
    pad=0.02
)

cbar.set_label(
    "Temperature Perturbation, T - T̄(z) (K)"
)

ax.set_xlabel("Time")
ax.set_ylabel("Altitude (km)")

ax.set_title(
    "CMAM30 Temperature Perturbation\n"
    "Reference: Mean Temperature at Each Altitude, "
    "Jan 2005 – Jun 2010"
)

# time labels
tick_positions = np.arange(0, len(X), 3)

tick_labels = [
    str(t.dt.strftime("%b %Y").values)
    for t in temp_perturb.time.isel(time=tick_positions)
]

ax.set_xticks(tick_positions)
ax.set_xticklabels(
    tick_labels,
    rotation=45,
    ha="right"
)

ax.set_ylim(95, 150)

# standard deviation at each altitude
std_temperature = all_monthly.std(dim="time")


# overall statistics

actual_values = all_monthly.values
perturbation_values = temp_perturb.values

# highest actual temperature
max_index = np.unravel_index(
    np.argmax(actual_values),
    actual_values.shape
)

max_time_index = max_index[0]
max_altitude_index = max_index[1]

max_temperature = actual_values[max_index]
max_temperature_c = max_temperature - 273.15


# lowest actual temperature
min_index = np.unravel_index(
    np.argmin(actual_values),
    actual_values.shape
)

min_time_index = min_index[0]
min_altitude_index = min_index[1]

min_temperature = actual_values[min_index]
min_temperature_c = min_temperature - 273.15


# largest positive perturbation
max_perturb_index = np.unravel_index(
    np.argmax(perturbation_values),
    perturbation_values.shape
)

max_perturb = perturbation_values[max_perturb_index]


# largest negative perturbation
min_perturb_index = np.unravel_index(
    np.argmin(perturbation_values),
    perturbation_values.shape
)

min_perturb = perturbation_values[min_perturb_index]


# time labels for statistics
max_time = str(
    all_monthly.time.isel(time=max_time_index)
    .dt.strftime("%b %Y").values
)

min_time = str(
    all_monthly.time.isel(time=min_time_index)
    .dt.strftime("%b %Y").values
)

max_perturb_time = str(
    all_monthly.time.isel(time=max_perturb_index[0])
    .dt.strftime("%b %Y").values
)

min_perturb_time = str(
    all_monthly.time.isel(time=min_perturb_index[0])
    .dt.strftime("%b %Y").values
)


max_altitude = all_monthly.altitude.values[
    max_altitude_index
]

min_altitude = all_monthly.altitude.values[
    min_altitude_index
]

max_perturb_altitude = all_monthly.altitude.values[
    max_perturb_index[1]
]

min_perturb_altitude = all_monthly.altitude.values[
    min_perturb_index[1]
]


# information box
info = fig.text(
    0.76,
    0.64,
    "Move mouse over the plot",
    ha="left",
    va="center",
    fontsize=10,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)


# statistics box
stats = fig.text(
    0.76,
    0.25,
    "",
    ha="left",
    va="center",
    fontsize=9,
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        edgecolor="black"
    )
)


stats.set_text(
    f"DATASET STATISTICS\n"
    f"\n"
    f"Highest Temp\n"
    f"{max_temperature:.2f} K "
    f"({max_temperature_c:.2f} °C)\n"
    f"{max_time}, {max_altitude:.2f} km\n"
    f"\n"
    f"Lowest Temp\n"
    f"{min_temperature:.2f} K "
    f"({min_temperature_c:.2f} °C)\n"
    f"{min_time}, {min_altitude:.2f} km\n"
    f"\n"
    f"Largest + Perturbation\n"
    f"{max_perturb:+.2f} K\n"
    f"{max_perturb_time}, "
    f"{max_perturb_altitude:.2f} km\n"
    f"\n"
    f"Largest - Perturbation\n"
    f"{min_perturb:+.2f} K\n"
    f"{min_perturb_time}, "
    f"{min_perturb_altitude:.2f} km"
)


def mouse_move(event):

    if event.inaxes != ax:
        return

    if event.xdata is None or event.ydata is None:
        return

    # nearest time
    time_index = int(
        np.clip(
            round(event.xdata),
            0,
            len(X) - 1
        )
    )

    # nearest altitude
    altitude_index = np.argmin(
        np.abs(Y - event.ydata)
    )

    time_value = temp_perturb.time.isel(
        time=time_index
    )

    altitude_value = Y[altitude_index]

    perturbation = Z[
        time_index,
        altitude_index
    ]

    actual_temperature = all_monthly.values[
        time_index,
        altitude_index
    ]

    reference_temperature = mean_temperature.values[
        altitude_index
    ]

    std_value = std_temperature.values[
        altitude_index
    ]

    # z-score
    z_score = perturbation / std_value

    date_string = str(
        time_value.dt.strftime("%b %Y").values
    )

    actual_celsius = actual_temperature - 273.15

    info.set_text(
        f"Time          : {date_string}\n"
        f"Altitude      : {altitude_value:.2f} km\n"
        f"\n"
        f"Actual Temp   : {actual_temperature:.2f} K\n"
        f"              : {actual_celsius:.2f} °C\n"
        f"\n"
        f"Reference     : {reference_temperature:.2f} K\n"
        f"Perturbation  : {perturbation:+.2f} K\n"
        f"Std. Dev.     : {std_value:.2f} K\n"
        f"Z-score       : {z_score:+.2f}"
    )

    fig.canvas.draw_idle()


fig.canvas.mpl_connect(
    "motion_notify_event",
    mouse_move
)

plt.subplots_adjust(
    left=0.07,
    right=0.74,
    bottom=0.18,
    top=0.90
)

plt.show()