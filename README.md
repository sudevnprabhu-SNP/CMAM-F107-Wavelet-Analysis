# CMAM–F10.7 Wavelet Analysis

**A reproducible analysis of solar radio flux variability and middle/upper-atmosphere temperature perturbations**

This repository investigates variability in the daily solar F10.7 radio-flux index and CMAM model air temperature, then compares their time-scale structure using continuous wavelet analysis. It is organized as a research project: data preparation, exploratory analysis, time–frequency analysis, comparison, figures, tables, and interpretation are connected in one workflow.

> **Scope and caution:** The available CMAM temperature series used here covers January 2005–June 2010. The F10.7 source spans 1979–June 2010. Comparisons between the two signals must use their common period and compatible sampling. Similar spectral peaks do not demonstrate that solar variability caused a temperature response.

## Research questions

1. How does CMAM temperature vary at approximately 120 km and the equator over January 2005–June 2010?
2. Which periodic components are visible in daily F10.7 data over the longer 1979–June 2010 record?
3. Which time scales are present in the CMAM monthly temperature-perturbation series?
4. How do the global wavelet spectra compare during the shared observation window?
5. How do significance thresholds and the cone of influence (COI) affect interpretation?

## Scientific workflow

```text
CMAM NetCDF (2005–2010)              Daily F10.7 CSV (1979–2010)
          │                                      │
          ▼                                      ▼
Pressure → approximate altitude             Validate dates/values
Select model level nearest 120 km           Preserve original F10.7
Longitude mean                               Daily wavelet analysis
Interpolate to 0° latitude                   (full record or selected years)
Monthly mean
          │                                      │
          ▼                                      │
Reference mean over available months            │
Temperature perturbation = T − reference mean   │
          │                                      │
          └────────── common period ─────────────┘
                            │
                            ▼
              Matched monthly comparison
              Morlet continuous wavelet transform
              Global spectra and visual comparison
```

## Repository layout

```text
CMAM-F107-Wavelet-Analysis/
├── data/
│   ├── raw/
│   │   ├── cmam/                 # 2005.nc ... 2010.nc (not committed by default)
│   │   └── f107/                 # source F10.7 CSV and source metadata
│   └── processed/                # generated CSVs
├── docs/
│   └── assignments/              # optional source instructions/notes
├── notebooks/
│   ├── 01_project_data_overview.ipynb
│   ├── 02_temperature_perturbation.ipynb
│   ├── 03_f107_wavelet_analysis.ipynb
│   ├── 04_temperature_wavelet_analysis.ipynb
│   └── 05_common_period_comparison.ipynb
├── results/
│   ├── figures/                  # generated plots
│   └── tables/                   # generated summary tables
├── src/
│   ├── temperature/
│   ├── solar/
│   └── wavelet/
├── tests/
├── wave_python/                  # wavelet implementation used by scripts
├── requirements.txt
└── README.md
```

The actual repository may contain additional scripts and files. Keep generated products in `data/processed/`, `results/figures/`, and `results/tables/` rather than scattering them through the project.

## Data sources and inputs

### CMAM model temperature

- Expected files: `data/raw/cmam/2005.nc` through `2010.nc`.
- Variable used: `ta` (air temperature, units K).
- Dimensions in the inspected files include time, pressure level (`plev`), latitude (`lat`), and longitude (`lon`).
- The files contain a CF non-standard calendar (`noleap`/`365_day` in the inspected workflow); preserve it when processing time coordinates.
- The current extraction selects the model pressure level nearest an **approximate** altitude of 120 km, averages over longitude, and interpolates the latitude profile to 0°.
- The altitude conversion is a simplified scale-height approximation:
  \[
  z = h_{\mathrm{ref}} + H\ln\left(\frac{P_{\mathrm{ref}}}{p}\right)
  \]
  with \(h_{\mathrm{ref}}=95\,\mathrm{km}\), \(H=8\,\mathrm{km}\), and \(P_{\mathrm{ref}}=0.1\,\mathrm{hPa}\). Pressure values must be converted to hPa before applying this equation. This is an approximate mapping, not a direct geometric-altitude coordinate.

### Solar F10.7 index

- Expected input: `data/raw/f107/F107_daily_1979_2010.csv`.
- Expected columns for the current scripts: `date` and `f107` (column names can be adapted if the source uses another schema).
- The series used in the prior run had 11,495 valid observations between 1979-01-01 and 2010-06-30, with 9 invalid/missing records removed. Confirm these counts against the current local CSV before reporting them as final.
- The main daily F10.7 wavelet analysis follows the project requirement **not to subtract the F10.7 mean**. A separate common-period comparison converts daily F10.7 to monthly means so it can be compared with the monthly CMAM series.

Respect the data providers' terms and cite the original sources in publications or derivative work. Large raw NetCDF files need not be committed to GitHub; document where they can be obtained instead.

## Environment setup

Use Python 3.10+ (the workflow has also been run in a Python 3.12 environment). From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS, use `python3` if `python` is not available. If `netCDF4` installation fails because of an HDF5 dependency, install the required system HDF5 library first or use a compatible conda environment. `xarray` can sometimes open NetCDF files through another installed backend, but the backend must support the source files.

## Run the analysis

Run scripts from the repository root so relative paths and project-root discovery behave predictably. For example:

```bash
python src/temperature/01_temperature_perturbation.py
python src/temperature/02_prepare_monthly_series.py
python src/wavelet/04d_f107_wavelet.py
python src/wavelet/04b_temperature_wavelet.py
python src/solar/03_compare_temperature_f107.py
python src/wavelet/04e_3d_wavelet_comparison.py
```

For the selected F10.7 interval January 2000–December 2005:

```bash
python src/wavelet/04f_f107_wavelet_2000_2005.py
```

Script names may differ if a local version has been renamed. Check the file names in `src/` before running. Run the temperature preparation step before temperature wavelet analysis, and verify that the input CSV column names match the selected script.

## Core methods

### Temperature perturbation

At each selected location, monthly temperature is calculated first. The reference temperature is the mean across the available monthly samples, and the perturbation is

\[
T'(t)=T(t)-\overline{T}.
\]

For the current 66-month series, the reference period is January 2005–June 2010. The time mean of the perturbation should be close to zero, apart from floating-point rounding.

### Continuous wavelet transform

The project uses the Torrence–Compo-style `waveletFunctions.py` implementation with a Morlet mother wavelet and \(\omega_0=6\). Wavelet power is

\[
P(t,s)=|W(t,s)|^2.
\]

The scripts estimate a 95% significance threshold using a lag-1 red-noise background and display the cone of influence. Power is plotted on a logarithmic scale. Significance should be tested on the original power, not on log-transformed values.

The daily F10.7 series uses \(dt=1\) day. The monthly temperature series uses \(dt=365.2425/12\approx30.436875\) days. Do not treat monthly values as daily samples.

### Comparing the two signals

For a fair comparison, use the shared interval January 2005–June 2010 and monthly sampling for both signals. Daily F10.7 is aggregated to monthly means. Because F10.7 and temperature power have different physical units, independently normalized spectra can compare *relative shape* but not absolute physical power. Report the normalization method clearly.

## Outcomes and preliminary inferences

The following are **preliminary results from previous local runs**, not a substitute for rerunning the full pipeline and checking the current input files.

- The inspected CMAM files yielded 66 monthly observations from January 2005 through June 2010.
- The selected pressure level mapped to approximately 120.75 km under the project's approximate pressure–altitude relation.
- The equatorial monthly temperature perturbation had a time mean close to zero by construction.
- In a prior analysis over the common interval, the strongest global wavelet-power feature was reported near **178 days** for the temperature perturbation and near **529 days** for F10.7. These values depend on the exact input series, period range, transform window, and global-spectrum calculation; verify them from the current output tables before quoting them as final findings.
- The 66-month temperature record is short for confidently resolving multi-year periodicities. Features near 365–530 days are represented by only a small number of cycles and may be strongly affected by the COI and finite-record effects.
- A spectral peak indicates variability at a time scale; it does not establish a physical causal connection between solar flux and CMAM temperature.
- Daily F10.7 and monthly temperature wavelet plots are useful for their respective signals, but a direct cross-signal comparison should use consistent sampling and the same common observation interval.

### How to interpret a wavelet figure

- **Colour field:** local wavelet power as a function of time and period.
- **Black contour:** region where power exceeds the configured 95% significance threshold, provided the code contours the power/significance ratio at 1.
- **COI curve:** separates regions less affected by edge effects from regions where edge effects are important. Interpret strong features outside the reliable side of the COI cautiously.
- **Global wavelet spectrum:** time-averaged power at each period. A maximum is a descriptive peak, not proof of a stable periodic cycle.
- **Logarithmic colour scale:** helps show weaker and stronger power together; the colourbar label should accurately indicate whether the displayed values are \(\log_{10}(P)\) or raw \(P\) with logarithmic normalization.

## Notebooks

The notebooks provide an explanatory, interactive route through the workflow. They complement the scripts; they do not replace them. Run them in numerical order. Each notebook checks for required files and explains what the output means.

1. `01_project_data_overview.ipynb` — inspect available files and schemas.
2. `02_temperature_perturbation.ipynb` — load and inspect the processed temperature perturbation series.
3. `03_f107_wavelet_analysis.ipynb` — perform a daily F10.7 wavelet transform.
4. `04_temperature_wavelet_analysis.ipynb` — analyse monthly temperature perturbations.
5. `05_common_period_comparison.ipynb` — compare global spectra over the common monthly interval.

## Reproducibility and quality checks

Before interpreting results:

- Verify input dates, units, missing values, duplicate dates, and monthly continuity.
- Verify that `plev` units are read from NetCDF metadata and converted correctly.
- Confirm that the same altitude level is selected for every CMAM year.
- Confirm that the temperature CSV contains 66 monthly rows for January 2005–June 2010.
- Confirm the wavelet transform uses the correct sampling interval and period units.
- Check that COI values are already in days before plotting; do not multiply by `dt` twice if the library already returns time units.
- Ensure the significance contour is calculated from original power and its matching significance threshold.
- Distinguish full-record F10.7 results from common-window results.
- Record Python and package versions, input data provenance, and script changes when producing a final figure.

## Limitations

1. The altitude conversion is approximate and depends on the assumed scale height.
2. The analysis covers one selected altitude and latitude; it is not a complete global or vertical climatology.
3. The temperature record has only 66 monthly points.
4. Missing-data interpolation can affect wavelet power; long gaps should not be silently filled.
5. Red-noise significance depends on assumptions, including the lag-1 background model.
6. Wavelet results near the COI are less reliable.
7. A relationship between two spectral peaks is not, by itself, evidence of causation.

## Citation and acknowledgement

When publishing or sharing results, cite the CMAM/CMIP data source, the F10.7 data provider (for example, the source documented alongside the downloaded OMNIWeb data), and the wavelet method/library used. Add exact dataset DOIs, access dates, and relevant papers once verified from the source metadata.

## License

Add a repository license before inviting reuse. Choose a license compatible with the data providers' terms and with the license of any included wavelet implementation. Do not assume that third-party code is automatically covered by your own repository license.
