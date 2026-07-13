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

def process_event_file(events, T_bins_g=224, B_bins_l=4, H=260, W=346):
    """
    グローバル(4, 224, 260) と 新ローカル時空間ボクセル(4, 260, 346) を個別に生成
    """
    v_tensor_global = np.zeros((4, T_bins_g, H), dtype=np.float32)
    v_tensor_local = np.zeros((B_bins_l, H, W), dtype=np.float32)
    
    if len(events) == 0:
        return v_tensor_global, v_tensor_local
        
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]
    ps = events[:, 3].astype(np.int32)
    
    polarities = np.where(ps == 1, 1.0, -1.0)
    t_min, t_max = ts.min(), ts.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6
    
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, polarities = xs[valid], ys[valid], ts[valid], polarities[valid]
    
    # 1. グローバル側インデックス表現の計算 (224ビン)
    t_indices_g = ((ts - t_min) / t_total * T_bins_g).astype(np.int32)
    t_indices_g = np.clip(t_indices_g, 0, T_bins_g - 1)
    
    # 2. 新ローカル側ボクセルグリッドの計算 (4ビン)
    t_indices_l = ((ts - t_min) / t_total * B_bins_l).astype(np.int32)
    t_indices_l = np.clip(t_indices_l, 0, B_bins_l - 1)

    time_surface = np.zeros((H, W), dtype=np.float32)
    phi_x = np.sin(np.pi * np.arange(W) / W)
    
    # --- ① グローバルストリームの生成 (W軸を圧縮した4chマトリクス) ---
    for b in range(T_bins_g):
        mask_g = (t_indices_g == b)
        if not np.any(mask_g): continue
        
        b_xs, b_ys, b_ts, b_pols = xs[mask_g], ys[mask_g], ts[mask_g], polarities[mask_g]
        time_surface[b_ys, b_xs] = (b_ts - t_min) / t_total
        
        np.add.at(v_tensor_global[0, b, :], b_ys, b_pols)
        np.add.at(v_tensor_global[3, b, :], b_ys, phi_x[b_xs])
        
        gx = cv2.Sobel(time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(time_surface, cv2.CV_32F, 0, 1, ksize=3)
        
        np.add.at(v_tensor_global[1, b, :], b_ys, gy[b_ys, b_xs])
        np.add.at(v_tensor_global[2, b, :], b_ys, gx[b_ys, b_xs])

    # --- ② 新・ローカルストリームの生成 (空間解像度維持の4chボクセルグリッド) ---
    PADDING = 10
    for b in range(B_bins_l):
        mask_l = (t_indices_l == b)
        if not np.any(mask_l): continue
        
        b_xs, b_ys, b_pols = xs[mask_l], ys[mask_l], polarities[mask_l]
        
        # 密度クラスター抽出用バイナリマップ
        bin_binary = np.zeros((H, W), dtype=np.uint8)
        bin_binary[b_ys, b_xs] = 255
        
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_binary, connectivity=8)
        spatial_mask = np.zeros((H, W), dtype=np.float32)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < 8: continue
            cx, cy, cw, ch = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            
            x1, y1 = max(0, cx - PADDING), max(0, cy - PADDING)
            x2, y2 = min(W, cx + cw + PADDING), min(H, cy + ch + PADDING)
            spatial_mask[y1:y2, x1:x2] = 1.0
            
        # 背景を完全透過(0クリア)しつつ、極性値を空間2Dボクセルへ蓄積
        valid_cluster_mask = spatial_mask[b_ys, b_xs] == 1.0
        if np.any(valid_cluster_mask):
            np.add.at(v_tensor_local[b], (b_ys[valid_cluster_mask], b_xs[valid_cluster_mask]), b_pols[valid_cluster_mask])
            
    # 個別標準化
    std_g = np.std(v_tensor_global) + 1e-5
    v_tensor_global = (v_tensor_global - np.mean(v_tensor_global)) / std_g
    
    std_l = np.std(v_tensor_local) + 1e-5
    v_tensor_local = (v_tensor_local - np.mean(v_tensor_local)) / std_l

    return v_tensor_global, v_tensor_local

def main():
    all_npy_paths = glob.glob(str(data_dir / "**" / "*.npy"), recursive=True)
    file_map = {os.path.basename(p): p for p in all_npy_paths}
    print(f"Scanned {len(file_map)} source files.")

    for mode in ["train", "test"]:
        txt_file = data_dir / f"{mode}.txt"
        if not txt_file.exists(): continue
        
        # 指示通りのフォルダ構成を用意
        g_out_dir = PROCESSED_DIR / mode / "global"
        l_out_dir = PROCESSED_DIR / mode / "local"
        g_out_dir.mkdir(parents=True, exist_ok=True)
        l_out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"--- Generating {mode.upper()} Splits ---")
        for idx, line in enumerate(tqdm(lines)):
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map: continue
            filename_noext = os.path.splitext(filename_only)[0]
            
            events = np.load(file_map[filename_only])
            feat_g, feat_l = process_event_file(events)
            
            out_name = f"{idx}_{filename_noext}_label_{parts[1]}.npy"
            np.save(str(g_out_dir / out_name), feat_g)
            np.save(str(l_out_dir / out_name), feat_l)

if __name__ == "__main__":
    main()