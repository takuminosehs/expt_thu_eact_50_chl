import glob
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

from expt_thu_eact_50_chl import config

# 📂 パス・ディレクトリ設定 (config.pyおよびPathlib完全準拠)
ORIGINAL_DATA_DIR = config.ORIGINAL_DATA_DIR
CURRENT_DIR = Path(__file__).parent.resolve()
PROCESSED_DIR = CURRENT_DIR / "processed_data"


def spatio_temporal_filter(
    xs, ys, ts_sec, polarities, H=260, W=346, dt_ms=10.0, min_neighbors=2
):
  """時空間相関フィルタ (STCF): 孤立した熱雑音イベントを除去する"""
  if len(xs) == 0:
    return xs, ys, ts_sec, polarities

  t_min, t_max = ts_sec.min(), ts_sec.max()
  duration = t_max - t_min + 1e-6
  num_bins = int(np.ceil(duration / (dt_ms / 1000.0)))
  num_bins = max(3, num_bins)

  t_bins = ((ts_sec - t_min) / duration * (num_bins - 1)).astype(np.int32)

  voxel = np.zeros((num_bins, H, W), dtype=np.uint8)
  voxel[t_bins, ys, xs] = 1

  neighbor_counts = np.zeros_like(voxel, dtype=np.uint8)
  for b in range(num_bins):
    neighbor_counts[b] = cv2.boxFilter(voxel[b], -1, (3, 3), normalize=False)

  total_counts = neighbor_counts.copy()
  total_counts[1:] += neighbor_counts[:-1]
  total_counts[:-1] += neighbor_counts[1:]

  keep_mask = total_counts[t_bins, ys, xs] >= min_neighbors
  return (
      xs[keep_mask],
      ys[keep_mask],
      ts_sec[keep_mask],
      polarities[keep_mask],
  )


def generate_adaptive_2ch_local(
    xs, ys, ts, polarities, H=260, W=346, bin_dt_sec=0.010
):
  """入力: events [N, 4] (x, y, ts, p)

  出力: [2, 260, 346] テンソル
  Ch1: 末尾トリミング適用後の適応的 Time-Surface
  Ch2: ピークビンの極性維持イベントマップ (極性の正負符号付き)
  """
  v_tensor_local = np.zeros((2, H, W), dtype=np.float32)

  if len(xs) == 0:
    return v_tensor_local

  # 1. タイムスタンプの秒単位スケーリング
  max_t = ts.max()
  if max_t > 100000:
    ts_sec = ts / 1e6
  elif max_t > 200:
    ts_sec = ts / 1e3
  else:
    ts_sec = ts.copy()

  # 2. STCF ノイズ除去
  xs, ys, ts_sec, polarities = spatio_temporal_filter(
      xs, ys, ts_sec, polarities, H, W
  )
  if len(xs) == 0:
    return v_tensor_local

  t_min, t_max = ts_sec.min(), ts_sec.max()
  duration = t_max - t_min if t_max > t_min else 1e-6

  # 3. 時間ビン分割によるイベント密度・末尾トリミング判定
  num_bins = max(1, int(np.ceil(duration / bin_dt_sec)))
  bin_indices = np.clip(
      ((ts_sec - t_min) / duration * (num_bins - 1)).astype(np.int32),
      0,
      num_bins - 1,
  )

  bin_event_counts = np.bincount(bin_indices, minlength=num_bins)
  n_max = bin_event_counts.max()
  threshold_count = 0.1 * n_max

  # 末尾からスキャンし、ピークの1割未満の状態が続く区間を除外
  last_valid_bin = num_bins - 1
  while last_valid_bin > 0 and bin_event_counts[last_valid_bin] < threshold_count:
    last_valid_bin -= 1

  # トリミング後の有効範囲イベントを抽出
  valid_mask = bin_indices <= last_valid_bin
  xs_val = xs[valid_mask]
  ys_val = ys[valid_mask]
  ts_sec_val = ts_sec[valid_mask]

  if len(xs_val) == 0:
    return v_tensor_local

  t_min_val, t_cut_max = ts_sec_val.min(), ts_sec_val.max()
  t_total_val = t_cut_max - t_min_val if t_cut_max > t_min_val else 1e-6

  # ─── Ch1: 末尾トリミング適用後の適応的 Time-Surface ───
  t_last = np.zeros((H, W), dtype=np.float32)
  has_event = np.zeros((H, W), dtype=bool)
  t_last[ys_val, xs_val] = ts_sec_val
  has_event[ys_val, xs_val] = True

  tau_adaptive = max(t_total_val / 3.0, 1e-4)
  ts_decay = np.exp(-(t_cut_max - t_last) / tau_adaptive)
  ts_decay[~has_event] = 0.0
  v_tensor_local[0] = ts_decay

  # ─── Ch2: ピークビンの極性維持イベントマップ ───
  peak_bin_idx = np.argmax(bin_event_counts)
  peak_mask = bin_indices == peak_bin_idx

  peak_map = np.zeros((H, W), dtype=np.float32)
  if np.any(peak_mask):
    xs_peak = xs[peak_mask]
    ys_peak = ys[peak_mask]
    pols_peak = polarities[peak_mask]  # +1.0 または -1.0
    np.add.at(peak_map, (ys_peak, xs_peak), pols_peak)

  v_tensor_local[1] = peak_map

  # ─── チャネル単位での標準化 (平均0, 標準偏差1) ───
  for c in range(2):
    std = np.std(v_tensor_local[c]) + 1e-5
    v_tensor_local[c] = (v_tensor_local[c] - np.mean(v_tensor_local[c])) / std

  return v_tensor_local


def main():
  all_npy_paths = glob.glob(
      str(ORIGINAL_DATA_DIR / "**" / "*.npy"), recursive=True
  )
  file_map = {os.path.basename(p): Path(p) for p in all_npy_paths}
  print(f"スキャン完了: {len(file_map)} 個のデータファイル")

  for mode in ["train", "test"]:
    txt_file = ORIGINAL_DATA_DIR / f"{mode}.txt"
    if not txt_file.exists():
      continue

    l_out_dir = PROCESSED_DIR / mode / "local"
    l_out_dir.mkdir(parents=True, exist_ok=True)

    with open(txt_file, "r") as f:
      lines = f.readlines()

    print(
        f"--- Generating {mode.upper()} Splits (Adaptive 2ch: TS + Peak"
        " Frame) ---"
    )
    for idx, line in enumerate(tqdm(lines)):
      parts = line.strip().split()
      if len(parts) < 2:
        continue

      filename_only = os.path.basename(parts[0])
      if filename_only not in file_map:
        continue
      filename_noext = os.path.splitext(filename_only)[0]

      events = np.load(file_map[filename_only])
      xs = events[:, 0].astype(np.int32)
      ys = events[:, 1].astype(np.int32)
      ts = events[:, 2]
      ps = events[:, 3].astype(np.int32)

      polarities = np.where(ps == 1, 1.0, -1.0)

      feat_l = generate_adaptive_2ch_local(xs, ys, ts, polarities)

      out_name = f"{idx}_{filename_noext}_label_{parts[1]}.npy"
      np.save(str(l_out_dir / out_name), feat_l)


if __name__ == "__main__":
  main()