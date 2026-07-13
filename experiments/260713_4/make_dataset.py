import os
import glob
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from expt_thu_eact_50_chl.config import ORIGINAL_DATA_DIR

data_dir = ORIGINAL_DATA_DIR
CURRENT_DIR = Path(__file__).parent.resolve()
# 実験フォルダ 260713_4 の直下に processed_data を出力
PROCESSED_DIR = CURRENT_DIR / "processed_data"

def events_to_twostream_tensor(events, T_bins=224, H=260, W=346):
    """
    グローバル4ch と クラスター抽出ローカル4ch を結合した計8chのテンソルを生成
    形状: (8, T_bins, H) -> (8, 224, 260)
    """
    # チャンネル数を 8 (0~3: グローバル, 4~7: ローカル) に拡張
    v_tensor = np.zeros((8, T_bins, H), dtype=np.float32)
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
    
    # クラスター切り出し時の周囲の「余裕（パディング）」ピクセル数
    PADDING = 10
    
    for b in range(T_bins):
        bin_mask = (t_indices == b)
        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_ts = ts[bin_mask]
        bin_pols = polarities[bin_mask]
        
        if len(bin_xs) == 0:
            continue

        # --- ベースとなる共通の Time Surface の更新 ---
        norm_ts = (bin_ts - t_min) / t_total
        time_surface[bin_ys, bin_xs] = norm_ts
        
        # ==========================================
        # ① グローバル・ストリームの蓄積 (チャンネル 0~3)
        # ==========================================
        np.add.at(v_tensor[0, b, :], bin_ys, bin_pols)
        np.add.at(v_tensor[3, b, :], bin_ys, phi_x[bin_xs])
        
        gx = cv2.Sobel(time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(time_surface, cv2.CV_32F, 0, 1, ksize=3)
        
        np.add.at(v_tensor[1, b, :], bin_ys, gy[bin_ys, bin_xs])
        np.add.at(v_tensor[2, b, :], bin_ys, gx[bin_ys, bin_xs])
        
        # ==========================================
        # ② ローカル・ストリームの抽出と蓄積 (チャンネル 4~7)
        # ==========================================
        # この時間ビンでイベントが発生した領域のバイナリマスクを作成
        bin_binary = np.zeros((H, W), dtype=np.uint8)
        bin_binary[bin_ys, bin_xs] = 255
        
        # 高速な連通成分ラベル付けにより、イベントの「塊（クラスター）」を検出
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_binary, connectivity=8)
        
        # ローカル領域として残すための空間マスク
        local_spatial_mask = np.zeros((H, W), dtype=np.float32)
        
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 8:  # 極小の孤立ノイズクラスターは形状を持たないため無視
                continue
                
            # クラスターのバウンディングボックスを取得
            cx = stats[i, cv2.CC_STAT_LEFT]
            cy = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            
            # 指示通り、周囲に少し余裕（PADDING）を持たせて範囲を拡張（画面外へのハミ出しをクリップ）
            x1 = max(0, cx - PADDING)
            y1 = max(0, cy - PADDING)
            x2 = min(W, cx + cw + PADDING)
            y2 = min(H, cy + ch + PADDING)
            
            # 余裕を持たせたエリアを有効化
            local_spatial_mask[y1:y2, x1:x2] = 1.0
            
        # 背景をクリア（0化）したローカル専用の幾何マップ群を生成
        local_time_surface = time_surface * local_spatial_mask
        local_bin_pols = bin_pols * local_spatial_mask[bin_ys, bin_xs]
        local_phi_x = phi_x[bin_xs] * local_spatial_mask[bin_ys, bin_xs]
        
        # ローカル側の生勾配を再計算
        gx_local = cv2.Sobel(local_time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy_local = cv2.Sobel(local_time_surface, cv2.CV_32F, 0, 1, ksize=3)
        
        # チャンネル 4~7 にローカル特徴を蓄積
        np.add.at(v_tensor[4, b, :], bin_ys, local_bin_pols)
        np.add.at(v_tensor[5, b, :], bin_ys, gy_local[bin_ys, bin_xs])
        np.add.at(v_tensor[6, b, :], bin_ys, gx_local[bin_ys, bin_xs])
        np.add.at(v_tensor[7, b, :], bin_ys, local_phi_x)
        
    # それぞれのストリーム独立で標準化（情報破壊を防ぐため個別に実施）
    mean_g, std_g = np.mean(v_tensor[:4]), np.std(v_tensor[:4]) + 1e-5
    v_tensor[:4] = (v_tensor[:4] - mean_g) / std_g
    
    mean_l, std_l = np.mean(v_tensor[4:]), np.std(v_tensor[4:]) + 1e-5
    v_tensor[4:] = (v_tensor[4:] - mean_l) / std_l

    return v_tensor

def preprocess_and_save():
    print(f"🔍 大元データフォルダ（{data_dir}）内の全データファイルをスキャン中...")
    all_npy_paths = glob.glob(str(data_dir / "**" / "*.npy"), recursive=True)
    file_map = {os.path.basename(p): p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個のデータファイルを見つけました。")

    for mode in ["train", "test"]:
        txt_file = data_dir / f"{mode}.txt"
        output_dir = PROCESSED_DIR / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not txt_file.exists():
            print(f"❌ {txt_file} が見つかりません。")
            return
            
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"\n--- 🧠 Two-Stream (Global/Local 8ch) 用 【{mode.upper()}】 前処理開始 ---")
        
        for idx, line in enumerate(tqdm(lines, desc=f"Processing {mode}")):
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map: continue
            filename_noext = os.path.splitext(filename_only)[0]
                
            events = np.load(file_map[filename_only])
            
            # グローバル4ch + クラスターローカル4ch の計8chを一括生成
            twostream_matrix = events_to_twostream_tensor(events, T_bins=224, H=260, W=346)
            filename = f"{idx}_{filename_noext}_orig_label_{parts[1]}.npy"
            np.save(str(output_dir / filename), twostream_matrix)

if __name__ == "__main__":
    preprocess_and_save()
    print(f"\n🎉 統合8ch前処理データが '{PROCESSED_DIR}' に固定保存されました！")