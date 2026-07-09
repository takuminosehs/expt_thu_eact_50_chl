import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from expt_thu_eact_50_chl.config import ORIGINAL_DATA_DIR

data_dir = ORIGINAL_DATA_DIR
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"

def events_to_robust_chsr_5ch(events, T_bins=224, H=260, W=346):
    """
    正負の極性を分離した、5チャネル構成の堅牢なCHSRテンソル生成
    """
    # 4チャネルから5チャネルへ拡張
    v_tensor = np.zeros((5, T_bins, H), dtype=np.float32)
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
        
        # 極性ごとにマスクを作成
        pos_mask = (bin_pols > 0)
        neg_mask = (bin_pols < 0)
        
        # --- チャネル0: 正の極性マップ (+1) ---
        if np.any(pos_mask):
            np.add.at(v_tensor[0, b, :], bin_ys[pos_mask], bin_pols[pos_mask])
        
        # --- チャネル1: 負の極性マップ (-1のまま蓄積) ---
        if np.any(neg_mask):
            np.add.at(v_tensor[1, b, :], bin_ys[neg_mask], bin_pols[neg_mask])
        
        # OpenCVのSobelフィルタで生勾配を計算
        gx = cv2.Sobel(time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(time_surface, cv2.CV_32F, 0, 1, ksize=3)
        
        # --- チャネル2 & 3: 生勾配 (旧チャネル1 & 2) ---
        np.add.at(v_tensor[2, b, :], bin_ys, gy[bin_ys, bin_xs])
        np.add.at(v_tensor[3, b, :], bin_ys, gx[bin_ys, bin_xs])
        
        # --- チャネル4: 論文準拠のホログラフィックマップ (旧チャネル3) ---
        np.add.at(v_tensor[4, b, :], bin_ys, phi_x[bin_xs])
            
    # 標準化（Mean 0, Std 1）
    mean = np.mean(v_tensor)
    std = np.std(v_tensor) + 1e-5
    v_tensor = (v_tensor - mean) / std

    return v_tensor

def preprocess_and_save():
    print(f"🔍 大元データフォルダ（{data_dir}）内の全データファイルをスキャン中...")
    all_npy_paths = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    file_map = {os.path.basename(p): p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個のデータファイルを見つけました。")

    for mode in ["train", "test"]:
        txt_file = Path(data_dir) / f"{mode}.txt"
        output_dir = PROCESSED_DIR / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not txt_file.exists():
            print(f"❌ {txt_file} が見つかりません。")
            return
            
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"\n--- HoloEv-Net-V4.2 (5-Ch 分離型) 用 【{mode.upper()}】 データのCHSR前処理を開始します ---")
        
        for idx, line in enumerate(tqdm(lines, desc=f"Processing {mode}")):
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map: continue
            filename_noext = os.path.splitext(filename_only)[0]
                
            events = np.load(file_map[filename_only])
            chsr_matrix = events_to_robust_chsr_5ch(events, T_bins=224, H=260, W=346)
            
            filename = f"{idx}_{filename_noext}_orig_label_{parts[1]}.npy"
            np.save(output_dir / filename, chsr_matrix)

if __name__ == "__main__":
    preprocess_and_save()
    print(f"\n🎉 全ての前処理が完了し、データが '{PROCESSED_DIR}' に固定保存されました！")