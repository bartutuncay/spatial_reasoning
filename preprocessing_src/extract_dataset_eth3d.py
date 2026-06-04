## Helper function to extract datasets for downloaded ETH3D datasets
# Note: download datasets in the 'datasets' folder before running

import os
import py7zr
import glob

location = 'relief' # 'test_meadow', 'relief_2', 'relief', 'pipes', 'office', 'anlieferung', 'break_room', 'terrains', 'test_terrace', 'test_playground'

out_dir = f'datasets_processed/{location}'
os.makedirs(out_dir,exist_ok=True)
zipped = glob.glob(f'datasets/{location}/*.7z')

for archive in zipped:
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(path=out_dir)