# src/holoev-net-v-3/make_dataset.py
import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from expt_thu_eact_50_chl.config import HW_DENOISED_NOPSI_DATA_DIR

CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"

def events_to_robust_chsr(events, T_bins=224, H=260, W=346):
    """
    割り算を排除し、発散ノイズを防いだ堅牢な4チャネルCHSR
    """
    v_tensor = np.zeros((4, T_bins, H), dtype=np.float32)
    if len(events) == 0:
        return v_tensor
        
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]
    ps = events[:, 3].astype(np.int32)
    
    polarities = np.where(ps == 1, 1.0, -1.0)
    t_min, t_max = ts.min(), ts.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6
    
    t_indices = ((ts - t_min) / t_total * T_bins).astype(np.int32)
    t_indices = np.clip(t_indices, 0, T_bins - 1)
        
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, polarities, t_indices = xs[valid], ys[valid], ts[valid], polarities[valid], t_indices[valid]

    time_surface = np.zeros((H, W), dtype=np.float32)
    phi_x = np.sin(np.pi * np.arange(W) / W)
    
    for b in range(T_bins):
        bin_mask = (t_indices == b)
        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_ts = ts[bin_mask]
        bin_pols = polarities[bin_mask]
        
        if len(bin_xs) == 0:
            continue

        # Time Surfaceの更新
        norm_ts = (bin_ts - t_min) / t_total
        time_surface[bin_ys, bin_xs] = norm_ts
        
        # --- チャネル0: 極性マップ (+1/-1) ---
        np.add.at(v_tensor[0, b, :], bin_ys, bin_pols)
        
        # --- チャネル3: 論文準拠のホログラフィックマップ (位相加算) ---
        np.add.at(v_tensor[3, b, :], bin_ys, phi_x[bin_xs])
            
        # --- 速度（勾配）の計算 ---
        # OpenCVのSobelフィルタで生勾配をそのまま使用（割り算・逆数は使用しない）
        gx = cv2.Sobel(time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(time_surface, cv2.CV_32F, 0, 1, ksize=3)
        
        # --- チャネル1 & 2: イベントが発生した位置の生勾配のみを抽出して蓄積 ---
        # これにより、背景のノイズを完全に無視できます
        np.add.at(v_tensor[1, b, :], bin_ys, gy[bin_ys, bin_xs])
        np.add.at(v_tensor[2, b, :], bin_ys, gx[bin_ys, bin_xs])
        
    # ResNet向けにテンソル全体を標準化（Mean 0, Std 1）
    mean = np.mean(v_tensor)
    std = np.std(v_tensor) + 1e-5
    v_tensor = (v_tensor - mean) / std

    return v_tensor

def preprocess_and_save():
    print(f"🔍 大元データフォルダ（{HW_DENOISED_NOPSI_DATA_DIR}）内の全データファイルをスキャン中...")
    all_npy_paths = glob.glob(os.path.join(HW_DENOISED_NOPSI_DATA_DIR, "**", "*.npy"), recursive=True)
    file_map = {os.path.basename(p): p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個のデータファイルを見つけました。")

    for mode in ["train", "test"]:
        txt_file = Path(HW_DENOISED_NOPSI_DATA_DIR) / f"{mode}.txt"
        output_dir = PROCESSED_DIR / mode
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(txt_file):
            print(f"❌ {txt_file} が見つかりません。")
            return
            
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"\n--- HoloEv-Net-V2 (Robust Velocity) 用 【{mode.upper()}】 データのCHSR前処理を開始します ---")
        
        for idx, line in enumerate(tqdm(lines, desc=f"Processing {mode}")):
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map: continue
            filename_noext = os.path.splitext(filename_only)[0]
                
            events = np.load(file_map[filename_only])
            chsr_matrix = events_to_robust_chsr(events, T_bins=224, H=260, W=346)
            
            filename = f"{idx}_{filename_noext}_orig_label_{parts[1]}.npy"
            np.save(os.path.join(output_dir, filename), chsr_matrix)

if __name__ == "__main__":
    preprocess_and_save()
    print(f"\n🎉 全ての前処理が完了し、データが '{PROCESSED_DIR}' に固定保存されました！")