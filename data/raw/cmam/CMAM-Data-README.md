# CMAM Raw Temperature Data

This directory is intended to hold the raw CMAM atmospheric temperature (`ta`) NetCDF files used by the **CMAM-F107-Wavelet-Analysis** project.

## Official data source

Download the files from the Canadian Centre for Climate Modelling and Analysis (CCCma) CMAM30-SD 6-hourly atmospheric temperature data page:

**Dataset page:** <https://climate-modelling.canada.ca/climatemodeldata/cmam/output/CMAM-Ext/CMAM30-SD/6hr/atmos/ta/index.shtml>

The source page lists the `r1i1p1` ensemble files. The annual files are approximately 993 MB each for full years; the listed 2010 file covers only part of the year and is approximately 493 MB.

## Files used by this project

For the analysis period used in this repository, download these files from the official page:

- `ta_6hrPlev_CMAM-Ext_CMAM30-SD_r1i1p1_2005010100-2005123118.nc`
- `ta_6hrPlev_CMAM-Ext_CMAM30-SD_r1i1p1_2006010100-2006123118.nc`
- `ta_6hrPlev_CMAM-Ext_CMAM30-SD_r1i1p1_2007010100-2007123118.nc`
- `ta_6hrPlev_CMAM-Ext_CMAM30-SD_r1i1p1_2008010100-2008123118.nc`
- `ta_6hrPlev_CMAM-Ext_CMAM30-SD_r1i1p1_2009010100-2009123118.nc`
- `ta_6hrPlev_CMAM-Ext_CMAM30-SD_r1i1p1_2010010100-2010063018.nc`

**Note:** The official page's 2010 file ends on 30 June 2010, not 31 December. This project should therefore describe the available 2010 coverage accurately and should not imply that the source file contains the second half of 2010.

## Download and setup steps

1. Open the official dataset page linked above.
2. Find the `r1i1p1` files for 2005 through 2010.
3. Download the six filenames listed above. These are large files, so ensure you have enough disk space and a stable internet connection.
4. Place the downloaded NetCDF files in this directory (`data/raw/cmam/`) within the project folder.
5. If your scripts expect short filenames such as `2005.nc`, either update the scripts to use the downloaded filenames or rename the files consistently. Do not keep two copies of these very large files unless needed.
6. Install the Python dependencies from the repository root using the instructions in the main `README.md`.
7. Run the temperature-processing scripts in the order documented in the main `README.md`. The raw files must be present before running scripts that read them.

## Why raw files are not stored on GitHub

These NetCDF files are very large, so they are intentionally excluded from the Git repository. Download them directly from the official source when reproducing the analysis. The repository may include smaller processed datasets and figures where appropriate.

## Data citation and attribution

The data are hosted by the **Canadian Centre for Climate Modelling and Analysis (CCCma), Environment and Climate Change Canada**. Please consult the source page and any associated terms or documentation for appropriate attribution and usage conditions when reusing the data.
