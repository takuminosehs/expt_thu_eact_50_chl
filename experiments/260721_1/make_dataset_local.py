import os
import glob
from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
from expt_thu_eact_50_chl import config

# 📂 パス・ディレクトリ設定 (config.pyおよびPathlib完全準拠)
ORIGINAL_DATA_DIR = config.ORIGINAL_DATA_DIR
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"

def spatio_temporal_filter(xs, ys, ts_sec, polarities, H=260, W=346, dt_ms=10.0, min_neighbors=2):
    """
    時空間相関フィルタ (STCF): 孤立した熱雑音イベントを除去する
    ※ ts_sec (秒単位) を用いて軽量な時間窓に分割して評価
    """
    t_min, t_max = ts_sec.min(), ts_sec.max()
    duration = t_max - t_min + 1e-6
    num_bins = int(np.ceil(duration / (dt_ms / 1000.0)))
    num_bins = max(3, num_bins)  # 最低3枚の時間窓を確保
    
    t_bins = ((ts_sec - t_min) / duration * (num_bins - 1)).astype(np.int32)
    
    # 3D時空間グリッド（バイナリ）の構築
    voxel = np.zeros((num_bins, H, W), dtype=np.uint8)
    voxel[t_bins, ys, xs] = 1
    
    # 各時間フレームにおける3x3空間近傍内のイベント数を計算
    neighbor_counts = np.zeros_like(voxel, dtype=np.uint8)
    for b in range(num_bins):
        neighbor_counts[b] = cv2.boxFilter(voxel[b], -1, (3, 3), normalize=False)
        
    # 前後の時間窓のカウントも累積（時空間3x3x3での近傍評価）
    total_counts = neighbor_counts.copy()
    total_counts[1:] += neighbor_counts[:-1]
    total_counts[:-1] += neighbor_counts[1:]
    
    # 閾値以上の近傍イベントを持つものだけを残す
    keep_mask = total_counts[t_bins, ys, xs] >= min_neighbors
    return xs[keep_mask], ys[keep_mask], ts_sec[keep_mask], polarities[keep_mask]

def generate_adaptive_4ch_local(xs, ys, ts, polarities, H=260, W=346):
    """
    動画長に依存しない全範囲適応型4チャネル表現を構築
    Ch1: 適応的 Time-Surface (動画全体をカバーする動的tau減衰)
    Ch2: TSの空間勾配強度 (細部エッジ強調)
    Ch3: 純粋なノーマルフロー X成分 (速度・方向)
    Ch4: 純粋なノーマルフロー Y成分 (速度・方向)
    """
    v_tensor_local = np.zeros((4, H, W), dtype=np.float32)
    
    if len(xs) == 0:
        return v_tensor_local
        
    # ─── 💡 タイムスタンプをマイクロ秒から秒にスケーリング ───
    ts_sec = ts / 1e6
        
    # 1. 時空間ノイズ除去 (STCF)
    xs, ys, ts_sec, polarities = spatio_temporal_filter(xs, ys, ts_sec, polarities, H, W)
    
    if len(xs) == 0:
        return v_tensor_local

    t_min, t_max = ts_sec.min(), ts_sec.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6

    # 各座標における最新のタイムスタンプと、イベント有無を記録
    t_last = np.zeros((H, W), dtype=np.float32)
    has_event = np.zeros((H, W), dtype=bool)
    
    t_last[ys, xs] = ts_sec
    has_event[ys, xs] = True

    # ─── 💡 適応的 tau (Adaptive Tau) の算定 ───
    # τ = T_total / 3.0 とすることで、最古イベントでも exp(-3.0) ≒ 0.05 の値が残り、
    # 全範囲の動作が途切れずに滑らかなグラデーションとして表現される
    tau_adaptive = max(t_total / 3.0, 1e-4)

    # ─── Ch1: 適応的 Time-Surface ───
    ts_decay = np.exp(-(t_max - t_last) / tau_adaptive)
    ts_decay[~has_event] = 0.0  # 未発生領域のクリア
    v_tensor_local[0] = ts_decay

    # ─── Ch2: TSの空間勾配強度 (Sobelエッジ) ───
    gx = cv2.Sobel(ts_decay, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ts_decay, cv2.CV_32F, 0, 1, ksize=3)
    ts_edge = np.sqrt(gx**2 + gy**2)
    v_tensor_local[1] = ts_edge

    # ─── Ch3 & Ch4: 純粋なノーマルフロー (時間減衰と分離保持) ───
    gx_raw = cv2.Sobel(t_last, cv2.CV_32F, 1, 0, ksize=3)
    gy_raw = cv2.Sobel(t_last, cv2.CV_32F, 0, 1, ksize=3)
    grad_sq = gx_raw**2 + gy_raw**2
    
    valid_grad = grad_sq > 1e-5
    flow_x = np.zeros_like(gx_raw)
    flow_y = np.zeros_like(gy_raw)
    
    # 勾配の逆数に比例する法線速度ベクトル
    flow_x[valid_grad] = gx_raw[valid_grad] / grad_sq[valid_grad]
    flow_y[valid_grad] = gy_raw[valid_grad] / grad_sq[valid_grad]
    
    # 速度外れ値のクリッピング
    flow_x = np.clip(flow_x, -10.0, 10.0)
    flow_y = np.clip(flow_y, -10.0, 10.0)
    
    flow_x[~has_event] = 0.0
    flow_y[~has_event] = 0.0
    
    v_tensor_local[2] = flow_x
    v_tensor_local[3] = flow_y

    # ─── チャネル単位での標準化 (平均0, 標準偏差1) ───
    for c in range(4):
        std = np.std(v_tensor_local[c]) + 1e-5
        v_tensor_local[c] = (v_tensor_local[c] - np.mean(v_tensor_local[c])) / std

    return v_tensor_local

def main():
    all_npy_paths = glob.glob(str(ORIGINAL_DATA_DIR / "**" / "*.npy"), recursive=True)
    file_map = {os.path.basename(p): Path(p) for p in all_npy_paths}
    print(f"Scanned {len(file_map)} source files.")

    for mode in ["train", "test"]:
        txt_file = ORIGINAL_DATA_DIR / f"{mode}.txt"
        if not txt_file.exists(): 
            continue
        
        # 保存先ディレクトリの作成 (PROCESSED_DIR / mode / local)
        l_out_dir = PROCESSED_DIR / mode / "local"
        l_out_dir.mkdir(parents=True, exist_ok=True)
        
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"--- Generating {mode.upper()} Splits (Adaptive 4ch Local Feature) ---")
        for idx, line in enumerate(tqdm(lines)):
            parts = line.strip().split()
            if len(parts) < 2: 
                continue
            
            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map: 
                continue
            filename_noext = os.path.splitext(filename_only)[0]
            
            # イベント配列の読み込み (shape: [N, 4])
            events = np.load(file_map[filename_only])
            xs = events[:, 0].astype(np.int32)
            ys = events[:, 1].astype(np.int32)
            ts = events[:, 2]
            ps = events[:, 3].astype(np.int32)
            
            polarities = np.where(ps == 1, 1.0, -1.0)
            
            # 4チャネル前処理テンソルの生成
            feat_l = generate_adaptive_4ch_local(xs, ys, ts, polarities)
            
            out_name = f"{idx}_{filename_noext}_label_{parts[1]}.npy"
            np.save(str(l_out_dir / out_name), feat_l)

if __name__ == "__main__":
    main()