import math
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from expt_thu_eact_50_chl import config
from expt_thu_eact_50_chl.utils import (
    calculate_topk_accuracy,
    save_best_model,
)

# ====================================================
# ─── ⚙️ 再現性のためのグローバル設定 ───
# ====================================================
SEED = 42


def seed_everything(seed=42):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


seed_everything(SEED)


def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)


# ====================================================
# ─── 📂 パス・ディレクトリ設定 (config.py準拠) ───
# ====================================================
PROJECT_ROOT = config.PROJECT_ROOT
CURRENT_DIR = Path(__file__).parent.resolve()

# 前処理データディレクトリ（スクリプト直下の processed_data を参照）
PROCESSED_DIR = CURRENT_DIR / "processed_data"

# ====================================================
# ─── 🎯 動的サブスペシャリスト設定 ───
# ====================================================
# Globalモデルが苦手とした31クラス
TARGET_CLASSES = [
    0,
    1,
    2,
    3,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    14,
    15,
    16,
    17,
    18,
    19,
    21,
    22,
    27,
    29,
    30,
    34,
    35,
    36,
    37,
    39,
    40,
    41,
    42,
    46,
]

NUM_TARGET_CLASSES = len(TARGET_CLASSES)
if NUM_TARGET_CLASSES <= 1:
  print(f"⚠️ 対象クラス数が {NUM_TARGET_CLASSES} のため学習をバイパスします。")
  exit(0)

# ─── ⚙️ 高速検証用ハイパーパラメータ ───
NUM_EPOCHS = 50
LR_INITIAL = 0.0005
BATCH_SIZE = 16

MODEL_SAVE_PATH = CURRENT_DIR / "best_r2plus1d_model.pth"
LOG_FILENAME = CURRENT_DIR / "r2plus1d_result.txt"


# ----------------------------------------------------
# ─── 📊 サブスペシャリスト専用 3D データセット定義 ───
# ----------------------------------------------------
class SubSpecialistLocal3DDataset(Dataset):
  """指定された TARGET_CLASSES のデータのみを抽出し、 3D テンソル [2, 16, 260, 346] を読み込むデータセット"""

  def __init__(self, mode="train", target_classes=None):
    self.local_dir = PROCESSED_DIR / mode / "local"
    if not self.local_dir.exists():
      raise FileNotFoundError(
          f"データディレクトリが見つかりません: {self.local_dir}"
      )

    if target_classes is None:
      raise ValueError("target_classes のリストを指定してください。")

    self.target_set = set(target_classes)

    # 順マッピング・逆マッピング辞書の動的生成
    self.src_to_local = {
        src_cls: local_idx
        for local_idx, src_cls in enumerate(sorted(target_classes))
    }
    self.local_to_src = {
        local_idx: src_cls for src_cls, local_idx in self.src_to_local.items()
    }

    all_files = sorted([p.name for p in self.local_dir.glob("*.npy")])
    self.file_names = []

    for f_name in all_files:
      label_str = f_name.split("_label_")[-1].split(".npy")[0]
      src_label = int(label_str.replace("A", ""))

      if src_label in self.target_set:
        self.file_names.append((f_name, src_label))

  def __len__(self):
    return len(self.file_names)

  def __getitem__(self, idx):
    f_name, src_label = self.file_names[idx]
    local_label = self.src_to_local[src_label]

    feat_3d = np.load(self.local_dir / f_name)  # 形状: [2, 16, 260, 346]

    ml = np.max(np.abs(feat_3d))
    if ml > 0:
      feat_3d = feat_3d / ml

    return (
        torch.tensor(feat_3d, dtype=torch.float32),
        torch.tensor(local_label, dtype=torch.long),
        torch.tensor(src_label, dtype=torch.long),
    )


# ----------------------------------------------------
# ─── 🧠 R(2+1)D (2D空間 + 1D時間 分離畳み込み) モジュール ───
# ----------------------------------------------------
class R2Plus1DConv(nn.Module):
  """3D 畳み込みを [1 x 3 x 3] 空間 Conv と [3 x 1 x 1] 時間 Conv に分解して処理するレイヤー"""

  def __init__(self, in_planes, planes, stride=1, dilation=1):
    super().__init__()
    # 分解パラメータ (中間の次元数を近似計算)
    inter_planes = int(
        (3 * in_planes * planes * 3 * 3) / (in_planes * 3 * 3 + 3 * planes)
    )
    inter_planes = max(1, inter_planes)

    # 空間畳み込み (1 x 3 x 3)
    self.spatial_conv = nn.Conv3d(
        in_planes,
        inter_planes,
        kernel_size=(1, 3, 3),
        stride=(1, stride, stride),
        padding=(0, dilation, dilation),
        dilation=(1, dilation, dilation),
        bias=False,
    )
    self.bn_spatial = nn.BatchNorm3d(inter_planes)
    self.relu_spatial = nn.ReLU(inplace=True)

    # 時間畳み込み (3 x 1 x 1)
    self.temporal_conv = nn.Conv3d(
        inter_planes,
        planes,
        kernel_size=(3, 1, 1),
        stride=(1, 1, 1),
        padding=(1, 0, 0),
        bias=False,
    )
    self.bn_temporal = nn.BatchNorm3d(planes)

  def forward(self, x):
    x = self.relu_spatial(self.bn_spatial(self.spatial_conv(x)))
    x = self.bn_temporal(self.temporal_conv(x))
    return x


class R2Plus1DBasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1, dilation=1):
    super().__init__()
    self.conv1 = R2Plus1DConv(
        in_planes, planes, stride=stride, dilation=dilation
    )
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = R2Plus1DConv(planes, planes, stride=1, dilation=dilation)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes * self.expansion:
      self.shortcut = nn.Sequential(
          nn.Conv3d(
              in_planes,
              planes * self.expansion,
              kernel_size=1,
              stride=(1, stride, stride),
              bias=False,
          ),
          nn.BatchNorm3d(planes * self.expansion),
      )

  def forward(self, x):
    out = self.relu(self.conv1(x))
    out = self.conv2(out)
    out += self.shortcut(x)
    out = self.relu(out)
    return out


# ----------------------------------------------------
# ─── 🧠 HighRes R(2+1)D ResNet18 QuarterWidth モデル ───
# ----------------------------------------------------
class HighResR2Plus1DResNet18QuarterWidth(nn.Module):
  """[B, 2, 16, 260, 346] の 3D テンソルを受け取り、

  R(2+1)D 畳み込みにより時空間特徴を軽量かつ高速に抽出するモデル。
  """

  def __init__(self, in_channels=2, num_classes=31):
    super().__init__()
    self.in_planes = 16

    # Stem (初期 R2Plus1D Conv)
    self.stem = R2Plus1DConv(in_channels, 16, stride=1, dilation=1)
    self.bn1 = nn.BatchNorm3d(16)
    self.relu = nn.ReLU(inplace=True)

    # チャネル数 QuarterWidth: [16, 32, 64, 128]
    self.layer1 = self._make_layer(
        R2Plus1DBasicBlock, 16, 2, stride=1, dilation=1
    )
    self.layer2 = self._make_layer(
        R2Plus1DBasicBlock, 32, 2, stride=2, dilation=2
    )
    self.layer3 = self._make_layer(
        R2Plus1DBasicBlock, 64, 2, stride=2, dilation=4
    )
    self.layer4 = self._make_layer(
        R2Plus1DBasicBlock, 128, 2, stride=2, dilation=8
    )

    self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
    self.fc = nn.Linear(128 * R2Plus1DBasicBlock.expansion, num_classes)

    self._init_custom_weights()

  def _make_layer(self, block, planes, num_blocks, stride, dilation):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for s in strides:
      layers.append(block(self.in_planes, planes, s, dilation))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def _init_custom_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv3d):
        nn.init.kaiming_normal_(
            m.weight, mode="fan_out", nonlinearity="relu"
        )
      elif isinstance(m, nn.BatchNorm3d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    nn.init.normal_(self.fc.weight, mean=0.0, std=0.01)
    nn.init.constant_(self.fc.bias, 0.0)

  def forward(self, x):
    # 入力 x: [B, 2, 16, 260, 346]
    out = self.relu(self.bn1(self.stem(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)  # [B, 128, 1, 1, 1]
    out = torch.flatten(out, 1)  # [B, 128]
    out = self.fc(out)  # [B, num_classes]
    return out


# ====================================================
# ─── 🏃 トレーニングメイン処理 (AMP + AdamW) ───
# ====================================================
if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"⚡ 使用デバイス: {device}")
  print(f"🎯 ターゲットクラス数: {NUM_TARGET_CLASSES} / 50")

  # R(2+1)D 軽量 QuarterWidth モデルの初期化
  model = HighResR2Plus1DResNet18QuarterWidth(
      in_channels=2, num_classes=NUM_TARGET_CLASSES
  ).to(device)

  train_dataset = SubSpecialistLocal3DDataset(
      mode="train", target_classes=TARGET_CLASSES
  )
  test_dataset = SubSpecialistLocal3DDataset(
      mode="test", target_classes=TARGET_CLASSES
  )

  print(f"📊 データ数 - Train: {len(train_dataset)}, Test: {len(test_dataset)}")

  g_init = torch.Generator()
  g_init.manual_seed(SEED)

  train_loader = DataLoader(
      train_dataset,
      batch_size=BATCH_SIZE,
      shuffle=True,
      num_workers=4,
      pin_memory=True,
      drop_last=True,
      worker_init_fn=seed_worker,
      generator=g_init,
  )
  test_loader = DataLoader(
      test_dataset,
      batch_size=BATCH_SIZE,
      shuffle=False,
      num_workers=4,
      pin_memory=True,
  )

  optimizer = torch.optim.AdamW(
      model.parameters(), lr=LR_INITIAL, weight_decay=0.01
  )
  scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

  best_test_acc1 = 0.0

  with LOG_FILENAME.open("w", encoding="utf-8") as f:
    f.write("=== Fast Verifier R(2+1)D ResNet18 Training Log ===\n")
    f.write(
        f"Target Classes ({NUM_TARGET_CLASSES} ch):"
        f" {sorted(TARGET_CLASSES)}\n\n"
    )

    for epoch in range(NUM_EPOCHS):
      cos_factor = (1 + math.cos(math.pi * epoch / NUM_EPOCHS)) / 2
      current_lr = LR_INITIAL * cos_factor
      for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

      # --- Training with AMP ---
      model.train()
      train_loss, train_total = 0.0, 0
      train_top1 = 0.0

      for x_3d, local_labels, _ in train_loader:
        x_3d, local_labels = x_3d.to(device), local_labels.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
          outputs = model(x_3d)
          loss = criterion(outputs, local_labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item() * local_labels.size(0)
        train_total += local_labels.size(0)
        acc1, _ = calculate_topk_accuracy(
            outputs,
            local_labels,
            topk=(1, 5 if NUM_TARGET_CLASSES >= 5 else 1),
        )
        train_top1 += acc1

      # --- Validation ---
      model.eval()
      test_total, test_top1 = 0, 0.0

      class_correct = np.zeros(NUM_TARGET_CLASSES)
      class_total = np.zeros(NUM_TARGET_CLASSES)

      with torch.no_grad():
        for x_3d, local_labels, src_labels in test_loader:
          x_3d, local_labels = x_3d.to(device), local_labels.to(device)

          with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(x_3d)

          test_total += local_labels.size(0)
          acc1, _ = calculate_topk_accuracy(
              outputs,
              local_labels,
              topk=(1, 5 if NUM_TARGET_CLASSES >= 5 else 1),
          )
          test_top1 += acc1

          preds = torch.argmax(outputs, dim=1)
          for p, l in zip(preds, local_labels):
            class_total[l.item()] += 1
            if p == l:
              class_correct[l.item()] += 1

      tr_acc = (train_top1 / train_total) * 100
      te_acc = (test_top1 / test_total) * 100

      status = (
          f"Epoch {epoch+1:03d} [LR: {current_lr:.6f}] -> Loss:"
          f" {train_loss/train_total:.4f} | Train: {tr_acc:.2f}% | ★Test:"
          f" {te_acc:.2f}%"
      )
      print(status)
      f.write(status + "\n")

      # 携帯 (Class 14) と敬礼 (Class 36) の精度モニタリング
      class_log = (
          "    🔍 [Target Class Accuracy Monitoring (R(2+1)D Verifier)]\n"
      )
      for src_cls in [14, 36]:
        if src_cls in train_dataset.src_to_local:
          local_idx = train_dataset.src_to_local[src_cls]
          tot = class_total[local_idx]
          acc = (
              (class_correct[local_idx] / tot * 100) if tot > 0 else 0.0
          )
          class_log += (
              f"       Original Class {src_cls:02d}: {acc:.2f}%"
              f" ({int(class_correct[local_idx])}/{int(tot)})\n"
          )
      print(class_log, end="")
      f.write(class_log)

      best_test_acc1 = save_best_model(
          model, te_acc, best_test_acc1, MODEL_SAVE_PATH
      )
      f.flush()