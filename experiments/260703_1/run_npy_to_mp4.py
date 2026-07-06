from expt_thu_eact_50_chl.npy_to_mp4 import npy_to_video
from pathlib import Path

if __name__ == "__main__":
    npy_to_video(Path(__file__).parent / "result" / "A0P8C0-2021_11_02_21_47_15_hw_filtered.npy", Path(__file__).parent / "result" / "A0P8C0_filtered.mp4")