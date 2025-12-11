# extract_dataset.py
import tarfile
import os
from pathlib import Path

# Path to your tar.gz file
tar_file = "data/raw/speechocean762.tar.gz"

# Extract location
extract_to = "data/raw/"

print("Extracting speechocean762.tar.gz...")
print(f"From: {tar_file}")
print(f"To: {extract_to}")

try:
    with tarfile.open(tar_file, 'r:gz') as tar:
        tar.extractall(path=extract_to)
    
    print("\n✓ Extraction complete!")
    print("\nChecking extracted contents...")
    
    # List what was extracted
    raw_dir = Path(extract_to)
    for item in raw_dir.iterdir():
        print(f"  Found: {item.name}")
    
except Exception as e:
    print(f"✗ Error: {e}")