import numpy as np
import sys
from pathlib import Path

def preview_npy(file_path, show_data=True, max_elements=10):
    """
    Preview a .npy file: prints shape, dtype, and first few elements.
    
    Parameters:
        file_path (str or Path): Path to the .npy file
        show_data (bool): Whether to print actual array values
        max_elements (int): Maximum number of elements to preview
    """
    try:
        arr = np.load(file_path, allow_pickle=True)
    except Exception as e:
        print(f"Error loading .npy file: {e}")
        return

    print("\n--- .npy File Preview ---")
    print(f"Path: {file_path}\n")
    print(f"Shape: {arr.shape}")
    print(f"Dtype: {arr.dtype}\n")

    if show_data:
        np.set_printoptions(
            threshold=max_elements,
            linewidth=200,
            edgeitems=max_elements
        )
        try:
            print(f"Data preview (up to {max_elements} elements):")
            print(arr)
        except Exception as e:
            print(f"Could not preview data: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preview_npy.py <npy_file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    preview_npy(file_path, max_elements=128)