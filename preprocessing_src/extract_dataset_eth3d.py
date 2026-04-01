import os
import py7zr
import glob

out_dir = '../../../scratch/btuncay/cog/gnn_spatial_reasoning/datasets/anlieferung'
os.makedirs(out_dir,exist_ok=True)
zipped = glob.glob('gnn_spatial_reasoning/datasets/anlieferung/*.7z')

for archive in zipped:
    with py7zr.SevenZipFile(archive) as z:
        z.extractall(path=out_dir)