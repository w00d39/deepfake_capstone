"""
Compresses the frames folder into a zip archive.
Uses zipfile with ZIP_DEFLATED compression and tqdm for progress.
"""

import zipfile
import os
from pathlib import Path
from tqdm import tqdm

FRAMES_DIR = "/Volumes/Seagate/capstone/frames"
OUTPUT_ZIP = "/Volumes/Seagate/capstone/frames.zip"

def zip_frames():
    frames_path = Path(FRAMES_DIR)

    # collect all files to zip
    all_files = list(frames_path.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]

    print(f"Found {len(all_files)} files to compress.")

    with zipfile.ZipFile(OUTPUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for file in tqdm(all_files, desc="Zipping frames"):
            # store with relative path so the zip is self-contained
            arcname = file.relative_to(frames_path.parent)
            zf.write(file, arcname)

    zip_size_gb = os.path.getsize(OUTPUT_ZIP) / (1024 ** 3)
    print(f"\nDone! Archive saved to: {OUTPUT_ZIP}")
    print(f"Zip size: {zip_size_gb:.2f} GB")

if __name__ == "__main__":
    zip_frames()
