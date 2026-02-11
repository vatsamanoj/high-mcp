import os
import glob

base_dir = os.getcwd()
quota_dir = os.path.join(base_dir, "quotas")
pattern = os.path.join(quota_dir, "*.json")

print(f"Base Dir: {base_dir}")
print(f"Quota Dir: {quota_dir}")
print(f"Pattern: {pattern}")

files = glob.glob(pattern)
print(f"Files found: {files}")
