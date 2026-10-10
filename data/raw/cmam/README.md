# CMAM Raw Temperature Data

The raw CMAM NetCDF files are not stored in this GitHub repository because of their large size. Download them from the official source before running analyses that require the original temperature data.

## Official download page

[CMAM-Ext CMAM30-SD — 6-hourly atmospheric air temperature (`ta`)](https://climate-modelling.canada.ca/climatemodeldata/cmam/output/CMAM-Ext/CMAM30-SD/6hr/atmos/ta/index.shtml)

## Required period

The project uses yearly files for 2005–2010. The available 2010 file ends on 30 June 2010, so the current temperature series covers January 2005 through June 2010.

## Download and setup

1. Open the official CMAM `ta` data index above.
2. Locate and download the yearly NetCDF files for 2005, 2006, 2007, 2008, 2009, and 2010.
3. Place the downloaded files in this directory: `data/raw/cmam/`.
4. Keep the original filenames or rename them to the names expected by the scripts. If you rename files, make the same change in the relevant code.
5. Check that each file contains the `ta` variable and the expected time, pressure, latitude, and longitude coordinates.
6. Preserve the dataset's non-standard calendar (for example, `365_day` or `noleap`) when decoding times.

File names and sizes can vary by source listing. Use the names displayed on the official page and check the scripts before running the pipeline.

## Notes

- The files are large; ensure you have sufficient storage and download time.
- Follow the data provider's terms and cite the source in reports and publications.
- Do not commit the large raw NetCDF files to this repository.
