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


def generate_spatiotemporal_2ch_tensor(
    xs,
    ys,
    ts,
    polarities,
    T_bins=16,
    H=260,
    W=346,
    fine_bin_dt_sec=0.010,
):
  """入力: events [N, 4] (x, y, ts, p)

  出力: [2, 16, 260, 346] テンソル
  Ch1: スライスごとの短期 Time-Surface (16フレーム)
  Ch2: スライスごとの極性維持イベントマップ (+1/-1 符号付き, 16フレーム)
  """
  v_tensor_3d = np.zeros((2, T_bins, H, W), dtype=np.float32)

  if len(xs) == 0:
    return v_tensor_3d

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
    return v_tensor_3d

  t_min, t_max = ts_sec.min(), ts_sec.max()
  duration = t_max - t_min if t_max > t_min else 1e-6

  # 3. 末尾トリミング計算 (ピークの1割未満が続く動画末尾を除外)
  num_fine_bins = max(1, int(np.ceil(duration / fine_bin_dt_sec)))
  fine_bin_indices = np.clip(
      ((ts_sec - t_min) / duration * (num_fine_bins - 1)).astype(np.int32),
      0,
      num_fine_bins - 1,
  )

  bin_event_counts = np.bincount(fine_bin_indices, minlength=num_fine_bins)
  n_max = bin_event_counts.max()
  threshold_count = 0.1 * n_max

  last_valid_bin = num_fine_bins - 1
  while (
      last_valid_bin > 0 and bin_event_counts[last_valid_bin] < threshold_count
  ):
    last_valid_bin -= 1

  valid_mask = fine_bin_indices <= last_valid_bin
  xs_val = xs[valid_mask]
  ys_val = ys[valid_mask]
  ts_sec_val = ts_sec[valid_mask]
  polarities_val = polarities[valid_mask]

  if len(xs_val) == 0:
    return v_tensor_3d

  t_min_val, t_cut_max = ts_sec_val.min(), ts_sec_val.max()
  t_total_val = t_cut_max - t_min_val if t_cut_max > t_min_val else 1e-6

  # 4. 有効時間領域の T=16 均等スライス分割
  t_slice_indices = np.clip(
      ((ts_sec_val - t_min_val) / t_total_val * T_bins).astype(np.int32),
      0,
      T_bins - 1,
  )

  slice_duration = t_total_val / T_bins
  tau_slice = max(slice_duration / 3.0, 1e-4)

  # 5. 各タイムスライス (16フレーム) ごとの特徴抽出
  for t in range(T_bins):
    slice_mask = t_slice_indices == t
    if not np.any(slice_mask):
      continue

    xs_sub = xs_val[slice_mask]
    ys_sub = ys_val[slice_mask]
    ts_sub = ts_sec_val[slice_mask]
    pols_sub = polarities_val[slice_mask]

    t_sub_max = ts_sub.max()

    # --- Ch1: 該当スライス内での短期 Time-Surface ---
    t_last_sub = np.zeros((H, W), dtype=np.float32)
    has_event_sub = np.zeros((H, W), dtype=bool)
    t_last_sub[ys_sub, xs_sub] = ts_sub
    has_event_sub[ys_sub, xs_sub] = True

    ts_decay_sub = np.exp(-(t_sub_max - t_last_sub) / tau_slice)
    ts_decay_sub[~has_event_sub] = 0.0
    v_tensor_3d[0, t] = ts_decay_sub

    # --- Ch2: 該当スライス内の極性維持イベントマップ ---
    np.add.at(v_tensor_3d[1, t], (ys_sub, xs_sub), pols_sub)

  # 6. チャネル単位での標準化 (平均0, 標準偏差1)
  for c in range(2):
    std = np.std(v_tensor_3d[c]) + 1e-5
    v_tensor_3d[c] = (v_tensor_3d[c] - np.mean(v_tensor_3d[c])) / std

  return v_tensor_3d


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
        f"--- Generating {mode.upper()} Splits (3D Spatiotemporal [2, 16, 260,"
        " 346]) ---"
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

      feat_3d = generate_spatiotemporal_2ch_tensor(xs, ys, ts, polarities)

      out_name = f"{idx}_{filename_noext}_label_{parts[1]}.npy"
      np.save(str(l_out_dir / out_name), feat_3d)


if __name__ == "__main__":
  main()