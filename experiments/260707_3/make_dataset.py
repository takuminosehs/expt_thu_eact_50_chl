# src/holoev-net-v-3/make_dataset.py
import sys
import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path

# プロジェクトルートをパスに追加して config を確実に読み込む
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# config から指定されたデータソースをインポート
from expt_thu_eact_50_chl.config import HW_DENOISED_NOPSI_DATA_DIR

# ====================================================
# 実験設定（手動でここを書き換えて実験をコントロールします）
# ====================================================
NUM_CHANNELS = 3  # 🌟 3 または 4 に手動で切り替え
T_BINS = 224
H_MAX = 260
W_MAX = 346

# 保存先ディレクトリの定義 (プロジェクト構成に準拠)
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"


def events_to_robust_chsr(
    events: np.ndarray, 
    T_bins: int = T_BINS, 
    H: int = H_MAX, 
    W: int = W_MAX
) -> np.ndarray:
    """
    位置依存性を排除し、運動の本質（正負の勾配）をホログラフィックに埋め込む新型CHSR
    形状: [NUM_CHANNELS, T_bins, H]
    """
    v_tensor = np.zeros((NUM_CHANNELS, T_bins, H), dtype=np.float32)
    if len(events) == 0:
        return v_tensor
        
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]
    
    t_min, t_max = ts.min(), ts.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6
    
    t_indices = ((ts - t_min) / t_total * T_bins).astype(np.int32)
    t_indices = np.clip(t_indices, 0, T_bins - 1)
        
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, t_indices = xs[valid], ys[valid], ts[valid], t_indices[valid]

    time_surface = np.zeros((H, W), dtype=np.float32)
    phi_x = np.sin(np.pi * np.arange(W) / W)
    
    # 補完チャネル3用の前フレーム勾配保持バッファ（案A: 勾配の時間差分用）
    if NUM_CHANNELS == 4:
        prev_gx = np.zeros((H, W), dtype=np.float32)
        prev_gy = np.zeros((H, W), dtype=np.float32)
    
    for b in range(T_bins):
        bin_mask = (t_indices == b)
        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_ts = ts[bin_mask]
        
        if len(bin_xs) == 0:
            if NUM_CHANNELS == 4:
                # イベントがないBinでも前フレームの勾配状態はリセット（または維持）
                prev_gx.fill(0)
                prev_gy.fill(0)
            continue

        # Time Surfaceの更新
        norm_ts = (bin_ts - t_min) / t_total
        time_surface[bin_ys, bin_xs] = norm_ts
        
        # OpenCVのSobelフィルタで生勾配をそのまま抽出
        gx = cv2.Sobel(time_surface, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(time_surface, cv2.CV_32F, 0, 1, ksize=3)
        
        bin_gx = gx[bin_ys, bin_xs]
        bin_gy = gy[bin_ys, bin_xs]
        bin_phi = phi_x[bin_xs]
        
        # --- チャネル0: 正の向きを向いているX方向の勾配ホログラフィック ---
        pos_mask = bin_gx > 0
        if np.any(pos_mask):
            np.add.at(v_tensor[0, b, :], bin_ys[pos_mask], bin_gx[pos_mask] * bin_phi[pos_mask])
        
        # --- チャネル1: 負の向きを向いているX方向の勾配ホログラフィック（負のまま加算） ---
        neg_mask = bin_gx < 0
        if np.any(neg_mask):
            np.add.at(v_tensor[1, b, :], bin_ys[neg_mask], bin_gx[neg_mask] * bin_phi[neg_mask])
            
        # --- チャネル2: Y方向の勾配ホログラフィック ---
        np.add.at(v_tensor[2, b, :], bin_ys, bin_gy * bin_phi)
        
        # --- チャネル3: 加速度マップ（有り設定の場合のみ計算） ---
        if NUM_CHANNELS == 4:
            # 物理的加速度（前Binとの勾配変化の絶対値の和）
            accel_x = gx - prev_gx
            accel_y = gy - prev_gy
            bin_accel = np.abs(accel_x[bin_ys, bin_xs]) + np.abs(accel_y[bin_ys, bin_xs])
            
            np.add.at(v_tensor[3, b, :], bin_ys, bin_accel * bin_phi)
            
            # 次のBinのために現在の状態を保存
            prev_gx = gx.copy()
            prev_gy = gy.copy()
        
    # ResNet向けにテンソル全体を標準化
    mean = np.mean(v_tensor)
    std = np.std(v_tensor) + 1e-5
    v_tensor = (v_tensor - mean) / std

    return v_tensor


def preprocess_and_save():
    data_dir = HW_DENOISED_NOPSI_DATA_DIR
    print(f"🔍 ノイズ除去済みデータフォルダ（{data_dir}）内の全データファイルをスキャン中...")
    
    # Pathlibによる再帰的検索
    all_npy_paths = list(data_dir.rglob("*.npy"))
    file_map = {p.name: p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個のデータファイルを見つけました。")

    for mode in ["train", "test"]:
        txt_file = data_dir / f"{mode}.txt"
        output_dir = PROCESSED_DIR / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not txt_file.exists():
            print(f"❌ {txt_file} が見つかりません。オリジナルデータのパス、または配置を確認してください。")
            return
            
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"\n--- HoloEv-Net-V3 (手動設定: {NUM_CHANNELS}Ch) 【{mode.upper()}】 前処理を開始します ---")
        
        for idx, line in enumerate(tqdm(lines, desc=f"Processing {mode}")):
            parts = line.strip().split()
            if len(parts) < 2: 
                continue
            
            filename_only = Path(parts[0]).name
            if filename_only not in file_map: 
                continue
            filename_noext = Path(filename_only).stem
                
            events = np.load(file_map[filename_only])
            chsr_matrix = events_to_robust_chsr(events, T_bins=T_BINS, H=H_MAX, W=W_MAX)
            
            # ファイル名にチャネル数情報を付与して保存（学習時に識別しやすくするため）
            filename = f"{idx}_{filename_noext}_ch{NUM_CHANNELS}_label_{parts[1]}.npy"
            np.save(output_dir / filename, chsr_matrix)


if __name__ == "__main__":
    preprocess_and_save()
    print(f"\n🎉 全ての前処理が完了し、データが '{PROCESSED_DIR}' に固定保存されました！")