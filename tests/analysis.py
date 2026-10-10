import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

file = "/Users/sudevnprabhu/Desktop/Future_Sudev/CUAP/CMAM_Temperature_Analysis/data/2010.nc"

ds = xr.open_dataset(file)

print(ds)

pressure = ds["plev"].values


# pressure to altitude
P_ref = 0.1
h_ref = 95.0
H = 8.0

altitude = h_ref + H * np.log(P_ref / pressure)

# select 95 to 150 km
mask = (altitude >= 95) & (altitude <= 150)

selected_pressure = pressure[mask]
selected_altitude = altitude[mask]

print("\nSelected levels for 95–150 km")
print("--------------------------------")

for p, h in zip(selected_pressure, selected_altitude):
    print(f"{p:12.6g} Pa   {h:8.2f} km")


# temperature for selected levels
ta_selected = ds["ta"].isel(plev=np.where(mask)[0])

print("\n===== SELECTED TEMPERATURE DATA =====")
print(ta_selected)


# latitude weighting
weights = np.cos(np.deg2rad(ds["lat"]))

ta_global = ta_selected.weighted(weights).mean(dim=["lat", "lon"])

print("\n===== GLOBAL MEAN TEMPERATURE =====")
print(ta_global)


# temperature perturbation
mean_temperature = ta_global.mean(dim="time")
temp_perturb = ta_global - mean_temperature

print("\n===== TEMPERATURE PERTURBATION =====")
print(temp_perturb)

print("\n===== MEAN TEMPERATURE AT EACH ALTITUDE =====")

for i in range(len(selected_altitude)):
    print(f"{selected_altitude[i]:8.2f} km : "
          f"{mean_temperature.isel(plev=i).values:.2f} K")


# use altitude instead of pressure
temp_perturb = temp_perturb.assign_coords(
    altitude=("plev", selected_altitude)
)

temp_perturb = temp_perturb.swap_dims({"plev": "altitude"})

print("\n===== PERTURBATION WITH ALTITUDE =====")
print(temp_perturb)


# monthly mean
monthly_perturb = temp_perturb.resample(time="MS").mean()

print("\n===== MONTHLY PERTURBATION =====")
print(monthly_perturb)

print("\nNumber of months:", len(monthly_perturb.time))


# plot
X = np.arange(len(monthly_perturb.time))
Y = monthly_perturb.altitude.values
Z = monthly_perturb.values.T

plt.figure(figsize=(14, 7))

contour = plt.contourf(X, Y, Z, levels=20)

plt.colorbar(contour, label="Temperature Perturbation (K)")

plt.xlabel("Time")
plt.ylabel("Altitude (km)")
plt.title("CMAM30 Temperature Perturbation (2006)")

plt.xticks(
    X,
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

plt.ylim(95, 150)

plt.tight_layout()
plt.show()