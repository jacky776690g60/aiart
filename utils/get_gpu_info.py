"""
Simple Script to test if GPU is available;
also show some general information about 
GPU on the current machine.
"""
from torch import cuda
from .progressbar import TermArtist

def get_gpu_info():
    DEVICE_IDX = cuda.current_device()
    print(f"GPU Availability:\t{cuda.is_available()}")
    print(f"Number of Devices:\t{cuda.device_count()}")
    print(f"Current Device:\t\t{DEVICE_IDX}")

    print(f"{TermArtist.GREEN}====> Device: {cuda.get_device_name()} <===={TermArtist.WHITE}")
    print(f"Properties:\t{cuda.get_device_properties(DEVICE_IDX)}")
    MAJOR_CAP, MINOR_CAP = cuda.get_device_capability(DEVICE_IDX)
    print(f"Capability:\tmajor:{MAJOR_CAP} minor:{MINOR_CAP}")


if __name__ == "__main__":
    get_gpu_info()