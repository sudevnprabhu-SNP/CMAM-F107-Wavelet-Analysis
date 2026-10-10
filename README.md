# CMAM–F10.7 Wavelet Analysis

A Python-based scientific analysis of CMAM upper-atmospheric temperature perturbations and solar F10.7 radio flux using time-series processing and continuous wavelet analysis.

The project includes temperature-data preparation, solar-index analysis, wavelet power spectra, and comparisons of time-scale variability. It is intended as a reproducible research workflow; results should be interpreted with the limitations of the data and methods in mind.

> **Scientific caution:** Similar periodic features in two signals do not establish that solar variability caused a temperature response. The CMAM temperature series currently analysed covers January 2005–June 2010, while the available F10.7 series spans a longer interval. Direct comparisons should use the common observation period and compatible sampling.

## Research questions

- How do CMAM temperature perturbations vary near the selected altitude and latitude?
- What time scales appear in the daily solar F10.7 record?
- What time scales appear in the monthly CMAM temperature-perturbation record?
- How do global wavelet spectra compare over a shared time interval?
- How should significance thresholds and the cone of influence affect interpretation?

## Workflow

1. Obtain the required CMAM NetCDF files and F10.7 input data.
2. Inspect and prepare the input data.
3. Extract the CMAM air-temperature variable `ta` and calculate temperature perturbations.
4. Prepare the monthly temperature series.
5. Analyse F10.7 and temperature time series using the included wavelet routines.
6. Compare signals only over the common time window and with compatible sampling.
7. Inspect the figures and tables, including significance and cone-of-influence information.

## Data sources

### CMAM air temperature

The CMAM-Ext CMAM30-SD 6-hourly atmospheric air-temperature files are available from the official data index:

**[Download CMAM `ta` data — official index](https://climate-modelling.canada.ca/climatemodeldata/cmam/output/CMAM-Ext/CMAM30-SD/6hr/atmos/ta/index.shtml)**

For the current analysis, obtain the yearly files covering **2005–2010**. The 2010 file available in this dataset ends on **30 June 2010**, so the combined time series is expected to end in June 2010. File sizes and exact names are shown on the source page and may change; download the files from the official index rather than relying on renamed examples.

#### CMAM download and setup steps

1. Open the [official CMAM `ta` data index](https://climate-modelling.canada.ca/climatemodeldata/cmam/output/CMAM-Ext/CMAM30-SD/6hr/atmos/ta/index.shtml).
2. Locate the yearly NetCDF files for 2005, 2006, 2007, 2008, 2009, and 2010.
3. Download the files for your own analysis. They are large files, so allow sufficient disk space and download time.
4. Place them in `data/raw/cmam/`.
5. Use the exact downloaded filenames or rename them to match the filenames expected by the scripts. If you rename files, keep the names consistent with the code.
6. Check each file's variables and coordinates before running the analysis. The temperature variable used by this project is `ta`.
7. Preserve the dataset's non-standard calendar (such as `365_day`/`noleap`) when reading time coordinates; do not force it into an incompatible standard calendar.

The raw CMAM NetCDF files are **not included in this GitHub repository** because of their size. Users must download them separately and comply with the data provider's terms.

### Solar F10.7 radio flux

The project uses a daily F10.7 CSV prepared from the source data documented in the repository. The expected local path is:

`data/raw/f107/F107_daily_1979_2010.csv`

The current scripts may expect columns named `date` and `f107`; check the input schema and adapt the loader if needed. Preserve source metadata and cite the data provider when using or publishing results.

## Repository structure

```text
CMAM-F107-Wavelet-Analysis/
├── data/
│   ├── raw/
│   │   ├── cmam/          # Download CMAM NetCDF files here (not committed)
│   │   └── f107/          # F10.7 input and source metadata
│   └── processed/         # Prepared CSV data
├── docs/
│   └── assignments/
├── notebooks/
├── results/
│   ├── figures/
│   └── tables/
├── src/
│   ├── temperature/
│   ├── solar/
│   ├── wavelet/
│   └── utils/
├── tests/
├── wave_python/           # Included wavelet routines and their original notices
├── requirements.txt
└── README.md
```

The repository may include additional scripts and notebooks. Use the actual filenames present in the checked-out version.

## Environment setup

Python 3.10 or newer is recommended. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`. Some NetCDF installations may require a compatible backend or system HDF5 libraries. Follow any installation notes provided by your operating system and package manager.

## Running the analysis

Run commands from the repository root so relative paths resolve consistently. The current project tree includes the following scripts:

```bash
python src/temperature/01_temperature_perturbation.py
python src/temperature/02_prepare_monthly_series.py
python src/solar/03_compare_temperature_f107.py
python src/wavelet/04a_f107_wavelet.py
python src/wavelet/04b_temperature_wavelet.py
python src/wavelet/04c_global_spectrum_comparison.py
python src/wavelet/04d_f107_wavelet.py
python src/wavelet/04e_3d_wavelet_comparison.py
```

Run only the scripts relevant to your task, and inspect each script's input/output paths before executing it. Some scripts may depend on files generated by earlier steps. If a script in your local checkout has a different name, use the filename that actually exists in `src/`.

## Core methods

### Temperature perturbation

The temperature perturbation is calculated relative to the reference mean:

\[
T'(t,z)=T(t,z)-\overline{T}(z)
\]

The reference interval, selected altitude, latitude handling, and aggregation method should be recorded alongside the generated data. The pressure-to-altitude mapping used in this project is approximate; it should not be presented as a direct geometric-altitude measurement.

### Wavelet analysis

The project uses a Morlet continuous wavelet analysis, with the implementation and parameter choices documented in the relevant scripts. The analysis considers time-varying wavelet power, significance thresholds, the cone of influence, and global wavelet spectra.

Sampling intervals must match the data: the F10.7 daily analysis uses daily sampling, while the CMAM perturbation series is monthly. For a direct comparison, aggregate F10.7 to monthly values and use the shared January 2005–June 2010 interval. Do not compare raw power magnitudes across signals with different physical units unless the normalization is clearly defined.

## Interpretation and limitations

- The temperature series is short (66 monthly observations for January 2005–June 2010), limiting confidence in multi-year periodicities.
- Pressure-derived altitude is an approximation based on assumptions; document those assumptions.
- Missing values, interpolation, sampling interval, significance model, and cone-of-influence effects can affect wavelet results.
- A global-spectrum maximum is a descriptive feature of the analysed series, not proof of a stable physical cycle.
- Similar peaks do not prove a causal connection between solar radio flux and atmospheric temperature.
- Re-run the scripts on the current input files before quoting numerical findings as final results.

## Citation and third-party software

When sharing results, cite the CMAM data source, the F10.7 data provider, and the wavelet method/software used. Preserve the original copyright and license notices in third-party source files.

See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for third-party attribution and license notes. The project license applies only as described in the root `LICENSE` file; it does not override third-party software licenses or dataset terms.

## License

See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Check the original license headers in `wave_python/` before redistributing or modifying those files.
