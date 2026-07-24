# src/holoev-net-re/make_dataset.py
import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path
from expt_thu_eact_50_chl.config import ORIGINAL_DATA_DIR

CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"

def events_to_original_chsr(events, T_bins=224, H=260, W=346):
    """
    原著論文 HoloEv-Net (Sec 3.B) に基づく3チャネルのCHSR構築
    出力形状: [3, 224, 260]
    """
    v_tensor = np.zeros((3, T_bins, H), dtype=np.float32)
    if len(events) == 0:
        return v_tensor
        
    xs = events[:, 0].astype(np.int32)
    ys = events[:, 1].astype(np.int32)
    ts = events[:, 2]
    ps = events[:, 3].astype(np.int32)
    
    t_min, t_max = ts.min(), ts.max()
    t_total = t_max - t_min if t_max > t_min else 1e-6
    
    t_indices = ((ts - t_min) / t_total * T_bins).astype(np.int32)
    t_indices = np.clip(t_indices, 0, T_bins - 1)
        
    # 画面外のノイズイベントをガード
    valid = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    xs, ys, ps, t_indices = xs[valid], ys[valid], ps[valid], t_indices[valid]

    # --- チャネル0: Positive Density Map (極性が 1) ---
    # --- チャネル1: Negative Density Map (極性が 0 または -1) ---
    pos_mask = (ps == 1)
    neg_mask = ~pos_mask
    
    # 🌟 修正箇所: インデックスをタプル (T軸, H軸) として渡す
    np.add.at(v_tensor[0], (t_indices[pos_mask], ys[pos_mask]), 1.0)
    np.add.at(v_tensor[1], (t_indices[neg_mask], ys[neg_mask]), 1.0)
    
    # --- チャネル2: Holographic Map ---
    # x座標による位相エンコーディング: sin(pi * x_k / W)
    phi_x = np.sin(np.pi * xs / W)
    
    # 🌟 修正箇所: インデックスをタプル (T軸, H軸) として渡す
    np.add.at(v_tensor[2], (t_indices, ys), phi_x)
    
    return v_tensor

def preprocess_and_save():
    print(f"🔍 大元データフォルダ（{ORIGINAL_DATA_DIR}）内の全データファイルをスキャン中...")
    all_npy_paths = glob.glob(os.path.join(ORIGINAL_DATA_DIR, "**", "*.npy"), recursive=True)
    file_map = {os.path.basename(p): p for p in all_npy_paths}
    print(f"識別完了: {len(file_map)} 個のデータファイルを見つけました。")

    for mode in ["train", "test"]:
        txt_file = Path(ORIGINAL_DATA_DIR) / f"{mode}.txt"
        output_dir = PROCESSED_DIR / mode
        os.makedirs(output_dir, exist_ok=True)
        
        if not os.path.exists(txt_file):
            print(f"❌ {txt_file} が見つかりません。")
            return
            
        with open(txt_file, "r") as f:
            lines = f.readlines()
            
        print(f"\n--- HoloEv-Net 論文再現用 【{mode.upper()}】 データのCHSR前処理を開始します ---")
        
        for idx, line in enumerate(tqdm(lines, desc=f"Processing {mode}")):
            parts = line.strip().split()
            if len(parts) < 2: continue
            
            filename_only = os.path.basename(parts[0])
            if filename_only not in file_map: continue
                
            events = np.load(file_map[filename_only])
            chsr_matrix = events_to_original_chsr(events, T_bins=224, H=260, W=346)
            
            filename = f"sample_{idx}_orig_label_{parts[1]}.npy"
            np.save(os.path.join(output_dir, filename), chsr_matrix)

if __name__ == "__main__":
    preprocess_and_save()
    print(f"\n🎉 全ての前処理が完了し、データが '{PROCESSED_DIR}' に固定保存されました！")