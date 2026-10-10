
from pathlib import Path
import shutil
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parent

# Create a backup of the original file layout before moving anything.
backup_dir = ROOT / "_organization_backup"
backup_dir.mkdir(exist_ok=True)

manifest_path = backup_dir / "original_layout.json"

if not manifest_path.exists():
    original_items = []
    for item in ROOT.iterdir():
        if item.name != backup_dir.name and item.name != "__pycache__":
            original_items.append({
                "name": item.name,
                "type": "directory" if item.is_dir() else "file"
            })

    manifest_path.write_text(
        json.dumps(original_items, indent=2),
        encoding="utf-8"
    )

# Target folders.
folders = [
    "src/temperature",
    "src/solar",
    "src/wavelet",
    "src/utils",
    "data/raw/cmam",
    "data/raw/f107",
    "data/processed",
    "notebooks",
    "results/figures",
    "results/tables",
    "docs/assignments",
    "tests",
    "wave_python",
]

for folder in folders:
    (ROOT / folder).mkdir(parents=True, exist_ok=True)


def move_file(filename, destination):
    """Move a file if it exists; do not overwrite an existing destination."""
    source = ROOT / filename
    target_dir = ROOT / destination
    target = target_dir / source.name

    if not source.exists():
        print(f"SKIP (not found): {filename}")
        return

    if source.resolve() == target.resolve():
        return

    if target.exists():
        print(f"SKIP (destination exists): {target.relative_to(ROOT)}")
        return

    shutil.move(str(source), str(target))
    print(f"MOVED: {filename} -> {target.relative_to(ROOT)}")


# Temperature perturbation and time-series preparation.
move_file(
    "01_temperature_perturbation.py",
    "src/temperature"
)
move_file(
    "02_prepare_monthly_series.py",
    "src/temperature"
)
move_file(
    "prepare_temperature.py",
    "src/temperature"
)

# Solar F10.7 preparation.
move_file("prepare_f107.py", "src/solar")

# Wavelet analysis scripts.
for filename in [
    "04a_f107_wavelet.py",
    "04b_temperature_wavelet.py",
    "04c_global_spectrum_comparison.py",
    "04d_f107_wavelet.py",
    "04e_3d_wavelet_comparison.py",
]:
    move_file(filename, "src/wavelet")

# Comparison script.
move_file(
    "03_compare_temperature_f107.py",
    "src/solar"
)

# Data files: large NetCDF files are kept locally, but organized.
for year in range(2005, 2011):
    move_file(
        f"data/{year}.nc",
        "data/raw/cmam"
    )

for filename in [
    "F107_daily_1979_2010.csv",
    "omni2_daily_xx0Z6ausMp.fmt.txt",
    "omni2_daily_xx0Z6ausMp.lst.txt",
]:
    move_file(filename, "data/raw/f107")
    move_file(f"data/{filename}", "data/raw/f107")

move_file(
    "data/temperature_perturbation.csv",
    "data/processed"
)

# Existing exploratory scripts. Preserve their names.
for filename in ["analysis.py", "final_analysis_2.py"]:
    move_file(filename, "tests")

# If the old test folder exists, move its contents before removing it.
old_test_dir = ROOT / "test"
if old_test_dir.exists():
    for item in list(old_test_dir.iterdir()):
        destination = ROOT / "tests" / item.name
        if destination.exists():
            print(f"SKIP (destination exists): {destination.name}")
        else:
            shutil.move(str(item), str(destination))
            print(f"MOVED: test/{item.name} -> tests/{item.name}")

    if not any(old_test_dir.iterdir()):
        old_test_dir.rmdir()

# Keep the wavelet implementation and its attribution together.
for filename in ["waveletAnalysis.py", "waveletFunctions.py"]:
    move_file(filename, "wave_python")

# Remove generated Python bytecode, not source code.
pycache = ROOT / "wave_python" / "__pycache__"
if pycache.exists():
    shutil.rmtree(pycache)
    print("REMOVED generated Python cache: wave_python/__pycache__")

# Preserve README and requirements at the repository root.
print("\nOrganization complete.")
print("Review the final tree and update script paths before running analyses.")
print("Backup manifest:", manifest_path.relative_to(ROOT))
