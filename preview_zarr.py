import zarr
import numpy as np
import sys
from pathlib import Path

def preview_zarr(store_path, show_data=True, max_elements=10):
    # Open the Zarr store
    try:
        store = zarr.open(store_path, mode='r')
    except Exception as e:
        print(f"Error opening Zarr store: {e}")
        return

    print("\n--- Zarr Store Preview ---")
    print(f"Path: {store_path}\n")

    def explore(group, prefix=''):
        for name, item in group.items():
            full_path = f"{prefix}/{name}".strip('/')
            if isinstance(item, zarr.Group):
                print(f"[Group]  {full_path}/")
                explore(item, full_path)
            elif isinstance(item, zarr.Array):
                print(f"[Array]  {full_path}")
                print(f"  shape: {item.shape}")
                print(f"  dtype: {item.dtype}")
                print(f"  chunks: {item.chunks}")
                print(f"  compressor: {item.compressor}\n")
                if show_data:
                    try:
                        preview_data = item[:max_elements]
                        print(f"  data (first {max_elements} elements):")
                        print(f"{preview_data}\n")
                    except Exception as e:
                        print(f"  Could not preview data: {e}\n")

    explore(store)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preview_zarr.py <zarr_path_or_url>")
        sys.exit(1)
    
    path = sys.argv[1]
    preview_zarr(path, max_elements=10)
