import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from expt_thu_eact_50_chl.config import HW_DENOISED_NOPSI_DATA_DIR

# --- 共有ライブラリ (.so) を正常にロードするためのパス設定 ---
CURRENT_DIR = Path(__file__).parent.resolve()
os.environ["LD_LIBRARY_PATH"] = str(CURRENT_DIR) + ":" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.append(str(CURRENT_DIR))

# ビルドしたC++/CUDA高速ノーマルフロー抽出器をインポート
from VecKM_flow import SliceNormalFlowEstimator

data_dir = HW_DENOISED_NOPSI_DATA_DIR
PROCESSED_DIR = CURRENT_DIR / "processed_data"
MODEL_DIR = CURRENT_DIR / "640x480_24ms_C64_k8"

def preprocess_with_normal_flow(events, estimator, T_bins=224, H=260, W=346):
    v_tensor = np.zeros((4, T_bins, H), dtype=np.float32)
    if len(events) == 0:
        return v_tensor
        
    events = events[np.argsort(events[:, 2])]
    
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2].copy()
    ps = events[:, 3].astype(np.int32)
    
    # 🌟【追加】タイムスタンプを「秒」に自動正規化
    max_t = ts.max()
    if max_t > 100000:
        ts_seconds = ts / 1000000.0
    elif max_t > 200:
        ts_seconds = ts / 1000.0
    else:
        ts_seconds = ts.copy()
    
    polarities = np.where(ps == 1, 1.0, -1.0)
    t_min, t_max = ts.min(), ts.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6
    
    t_indices = ((ts - t_min) / t_total * T_bins).astype(np.int32)
    t_indices = np.clip(t_indices, 0, T_bins - 1)
        
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ts, ts_seconds, polarities, t_indices = (
        xs[valid], ys[valid], ts[valid], ts_seconds[valid], polarities[valid], t_indices[valid]
    )

    if len(ts) == 0:
        return v_tensor

    phi_x = np.sin(np.pi * np.arange(W) / W)
    all_indices = np.arange(len(ts))
    t_radius = 0.012 
    
    for b in range(T_bins):
        bin_mask = (t_indices == b)
        if not np.any(bin_mask):
            continue

        bin_xs = xs[bin_mask]
        bin_ys = ys[bin_mask]
        bin_pols = polarities[bin_mask]
        
        np.add.at(v_tensor[0, b, :], bin_ys, bin_pols)
        np.add.at(v_tensor[3, b, :], bin_ys, phi_x[bin_xs])
        
        # 🌟【修正】秒単位のタイムスタンプを使って中心とコンテキストを計算
        bin_ts_sec_sub = ts_seconds[bin_mask]
        t_center_seconds = (bin_ts_sec_sub.min() + bin_ts_sec_sub.max()) / 2.0
        
        context_mask = (ts_seconds >= t_center_seconds - 0.030) & (ts_seconds <= t_center_seconds + 0.030)
        
        if np.sum(context_mask) > 450000:
            dists = np.abs(ts_seconds[context_mask] - t_center_seconds)
            threshold_dist = np.partition(dists, 450000)[450000]
            context_mask = context_mask & (np.abs(ts_seconds - t_center_seconds) <= threshold_dist)
            
        context_indices = all_indices[context_mask]
        context_events_txy = np.stack([ts_seconds[context_mask], xs[context_mask], ys[context_mask]], axis=1).astype(np.float32)
        context_events_txy = np.ascontiguousarray(context_events_txy)
        
        bin_global_indices = all_indices[bin_mask]
        target_indices = np.where(np.isin(context_indices, bin_global_indices))[0].astype(np.int32)
        
        if len(target_indices) == 0:
            continue
            
        try:
            flow = estimator.predict_flows(
                context_events_txy, context_events_txy.shape[0],
                target_indices, target_indices.shape[0],
                t_center_seconds, t_radius
            )
            
            global_target_indices = context_indices[target_indices]
            flow_ys = ys[global_target_indices]
            
            np.add.at(v_tensor[1, b, :], flow_ys, flow[:, 1]) 
            np.add.at(v_tensor[2, b, :], flow_ys, flow[:, 0]) 
            
        except Exception as e:
            pass
        
    mean = np.mean(v_tensor)
    std = np.std(v_tensor) + 1e-5
    v_tensor = (v_tensor - mean) / std

    return v_tensor

def preprocess_and_save():
    if not MODEL_DIR.exists():
        print(f"❌ 事前学習済みモデルフォルダ '{MODEL_DIR}' が見つかりません。")
        return

    print("🚀 SliceNormalFlowEstimator を初期化しています...")
    estimator = SliceNormalFlowEstimator(str(MODEL_DIR), 500000, 346, 260, 64, 8)

    print(f"🔍 大元データフォルダ（{data_dir}）内の全データファイルをスキャン中...")
    all_npy_paths = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    file_map = {os.path.basename(p): p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個 of データファイルを見つけました。")

    for mode in ["train", "test"]:
        txt_file = Path(data_dir) / f"{mode}.txt"
        output_dir = PROCESSED_DIR / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not txt_file.exists():
            print(f"❌ {txt_file} が見つかりません。")
            return
            
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"\n--- 【{mode.upper()}】 データの最先端 Normal Flow 前処理を開始します ---")
        
        for idx, line in enumerate(tqdm(lines, desc=f"Processing {mode}")):
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map: continue
            filename_noext = os.path.splitext(filename_only)[0]
                
            events = np.load(file_map[filename_only])
            chsr_matrix = preprocess_with_normal_flow(events, estimator, T_bins=224, H=260, W=346)
            
            filename = f"{idx}_{filename_noext}_orig_label_{parts[1]}.npy"
            np.save(output_dir / filename, chsr_matrix)

if __name__ == "__main__":
    preprocess_and_save()
    print(f"\n🎉 全ての前処理が完了し、ノーマルフロー埋め込みデータが '{PROCESSED_DIR}' に保存されました！")